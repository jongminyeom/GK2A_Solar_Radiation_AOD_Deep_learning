import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import netCDF4 as nc
import os
import sys

# Device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ATMP = sys.argv[1]
RPRD = ATMP.upper()
rprd = ATMP.lower()


# 1. Set candidate hyperparameters
param_choices = {
    "d_token":         [32, 64, 128, 256],
    "n_heads":         [2, 4, 8],
    "n_layers":        [1, 2, 3],
    "dropout":         [0.0, 0.05, 0.1],
    "dim_feedforward": [64, 128, 256, 512, 1024],
    "learning_rate":   [1e-3, 5e-4, 3e-4],
    "batch_size":      [32, 64, 128, 256, 512],
    "optimizer":       ["adam", "adamw", "radam", "sgd"],
    "loss_fn":         ["mse", "mae", "huber", "logcosh", "l1l2"],
    "patience":        [10],
    "init_weights":    ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
}

n_runs   = 1000
n_epochs = 300

# 2. Load and prepare data
#file_path = "Matchup-INS_202101-202312.asc"
#df = pd.read_csv(file_path, delim_whitespace=True, header=None)
# Assume shape is (N, 24) where last column is target
#df.columns = [f"X{i}" for i in range(23)] + ["target"]


# 2. Load and prepare data
if RPRD == 'INS':
   nc_path = "Matchup_INSD_202101_202312.nc"

if RPRD == 'AWV':
   nc_path = "Matchup_AWVD_202101_202312.nc"

if RPRD == 'AOT':
   nc_path = "Matchup_AOTD_202101_202312.nc"

if not os.path.exists(nc_path):
    raise FileNotFoundError(f"{nc_path} 파일이 존재하지 않습니다.")

ds = nc.Dataset(nc_path)
var_data = ds.variables["MDL_VAR"][:].astype(np.float32)
ds.close()

# Assume shape is (N, 24) where last column is target

if RPRD == 'INS':
   df = pd.DataFrame(var_data, columns=[f"X{i}" for i in range(19)] + ["DUM1","DUM2","DUM3","DUM4","target"])
   X = df.drop(columns=["DUM1","DUM2","DUM3","DUM4","target"]).values.astype(np.float32)

if RPRD != 'INS':
   df = pd.DataFrame(var_data, columns=[f"X{i}" for i in range(23)] + ["target"])
   X = df.drop(columns=["target"]).values.astype(np.float32)

y = df["target"].values.astype(np.float32).reshape(-1, 1)

input_dim = X.shape[1]


# 3. Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 4. Scale data
x_scaler = StandardScaler().fit(X_train)
X_train = x_scaler.transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)
# StandardScaler outputs float64; cast back to float32 for torch/GPU
X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)


if RPRD == 'INS':
   y_scaler = 1000.0

if RPRD == 'AWV':
   y_scaler = 10.0

if RPRD == 'AOT':
   y_scaler = 1.0


y_train = y_train / y_scaler
y_val = y_val / y_scaler
y_test = y_test / y_scaler

# 5. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 6. Datasets only
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

# 7. FT-Transformer Model
class FeatureTokenizer(nn.Module):
    """Feature-wise tokenizer for tabular data.

    Turns x: (B, n_features) into tokens: (B, n_features, d_token) with
    learnable per-feature weights/bias: token_j = x_j * W_j + b_j.
    This naturally encodes column identity (feature ID) without positional encoding.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.empty(n_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """FT-Transformer for tabular regression WITHOUT CLS token.

    (B, n_features) -> feature tokens (B, n_features, d) -> TransformerEncoder
    -> pooling over tokens -> regression head -> (B, 1)
    """
    def __init__(self, num_features, d_token=32, n_heads=4, n_layers=2, dropout=0.05,
                 dim_feedforward=128, pooling='mean'):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.token_ln = nn.LayerNorm(d_token)
        self.pooling = pooling
        if pooling == 'attn':
            self.attn = nn.Linear(d_token, 1)
        else:
            self.attn = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )

    def forward(self, x):
        # x: (B, num_features)
        x = self.tokenizer(x)      # (B, num_features, d_token)
        x = self.token_ln(x)
        x = self.transformer(x)    # (B, num_features, d_token)

        if self.pooling == 'mean':
            pooled = x.mean(dim=1)  # (B, d_token)
        elif self.pooling == 'sum':
            pooled = x.sum(dim=1)
        elif self.pooling == 'attn':
            w = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (B, num_features)
            pooled = (x * w.unsqueeze(-1)).sum(dim=1)           # (B, d_token)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(pooled)   # (B, 1)

# 8. Custom Loss Functions
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target)))

class L1L2Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)

# 9. Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

results = []

#for run in range(n_runs):
for run in range(9,n_runs):
    # Random hyperparameter selection
    d_token = random.choice(param_choices["d_token"])
    n_heads = random.choice(param_choices["n_heads"])
    n_layers = random.choice(param_choices["n_layers"])
    dropout = random.choice(param_choices["dropout"])
    dim_feedforward = random.choice(param_choices["dim_feedforward"])
    lr = random.choice(param_choices["learning_rate"])
    batch_size = random.choice(param_choices["batch_size"])
    opt_type = random.choice(param_choices["optimizer"])
    loss_type = random.choice(param_choices["loss_fn"])
    patience = random.choice(param_choices["patience"])

    print(f"=== Training Run {run+1}/{n_runs} ===")
    print(f"Hyperparams: d_token={d_token}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}, dim_ff={dim_feedforward}, lr={lr}, batch_size={batch_size}, optimizer={opt_type}, loss={loss_type}, patience={patience}")

    # Dataloaders defined after batch_size is chosen
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
  
    model = FTTransformer(num_features=input_dim, d_token=d_token, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
                        dim_feedforward=dim_feedforward, pooling='mean').to(device)

    init_type = random.choice(param_choices["init_weights"])
    print(f"Weight init: {init_type}")
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, FeatureTokenizer):
            # Initialize tokenizer weight/bias (shape: [n_features, d_token])
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)


    model.apply(init_weights)
  
    if opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_type == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
      
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "mae":
        loss_fn = nn.L1Loss()
    elif loss_type == "huber":
        loss_fn = nn.SmoothL1Loss()
    elif loss_type == "logcosh":
        loss_fn = LogCoshLoss()
    elif loss_type == "l1l2":
        loss_fn = L1L2Loss(alpha=0.8)
      
    best_val_loss = float('inf')
    wait = 0

    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []

    for epoch in range(n_epochs):
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1} for run {run+1}")
            break
        model.train()
        train_preds, train_targets = [], []
        train_loss_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * xb.size(0)
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(yb.detach().cpu().numpy())

        train_loss = train_loss_total / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_rmses.append(train_rmse)

        model.eval()
        val_loss_total = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_total += loss_fn(pred, yb).item() * xb.size(0)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_rmses.append(val_rmse)
        val_r2 = r2_score(val_targets, val_preds)
        val_bias = np.mean(val_preds - val_targets)
        
        print(f"Run {run+1}, Epoch {epoch+1:03d}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}, Val R^2 = {val_r2:.4f}, Val Bias = {val_bias:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), f"best_model_run{run+1:03d}.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    # Evaluate on test set
    model.load_state_dict(torch.load(f"best_model_run{run+1:03d}.pt"))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    test_r2 = r2_score(targets, preds)
    test_rmse = np.sqrt(mean_squared_error(targets, preds))
    test_bias = np.mean(preds - targets)
    print(f"Test R^2 = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}, Test Bias = {test_bias:.4f}")

    row = {
        "run": f"{run+1:03d}",  
        "train_r2": r2_score(train_targets, train_preds),
        "train_rmse": train_rmse,
        "train_bias": np.mean(train_preds - train_targets),
        "val_r2": val_r2,
        "val_rmse": val_rmse,
        "val_bias": val_bias,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_bias": test_bias,
#        "epoch": epoch + 1,
        "d_token": d_token,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "dim_feedforward": dim_feedforward,
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": opt_type,
        "loss_fn": loss_type,
        "patience": patience,
        "init_weights": init_type
    }
    results.append(row)

    pd.DataFrame([row]).to_csv("training_results.csv", mode="a", header=not bool(run), index=False)

    # Visualization after each run
    plt.figure(figsize=(6, 4))
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Val RMSE')
    plt.title(f"RMSE per Epoch (Run {run+1:03d})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"rmse_curve_run{run+1:03d}.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Loss per Epoch (Run {run+1:03d})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curve_run{run+1:03d}.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.3)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted (Run {run+1:03d})")
    plt.tight_layout()
    plt.savefig(f"scatter_plot_run{run+1:03d}.png")
    plt.close()


# results_df = pd.DataFrame(results)
# results_df.to_csv("training_results.csv", index=False)
# print("Saved training results to training_results.csv")
