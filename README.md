# GK-2A DL Training Scripts (BPNN / Conv1D / FT-Transformer)

This repository contains training scripts used to develop deep learning models for GK-2A matchup datasets (NetCDF) for:
- **BPNN (DNN)** and **Conv1D (1D CNN)** using **TensorFlow/Keras**
- **FT-Transformer** using **PyTorch** (feature-tokenizer transformer for tabular predictors)

> These scripts were used for model benchmarking and operational feasibility studies on GK-2A solar radiation / AOD / related targets.

---

## Contents

- `cvpps_mdl_train.py`  
  TensorFlow/Keras training for **DNN(BPNN)** + **Conv1D** with random hyperparameter sampling, checkpointing, early stopping, and evaluation plots.  
  (Reads one input NetCDF file passed as a command-line argument.)

- `test_ft-transformer_v0007_rnd-param_fttrue_nocls.py`  
  PyTorch training for **FT-Transformer** with random hyperparameter search across multiple runs, early stopping, and per-run plots + CSV logging.  
  (Reads a fixed NetCDF filename based on a product code argument.)

---

## Requirements

### Common
- Python 3.8+ recommended
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `netCDF4`

### For BPNN / Conv1D (TensorFlow)
- `tensorflow` / `keras`

### For FT-Transformer (PyTorch)
- `torch` (CUDA optional but recommended)

---

## Data format (NetCDF)

Both scripts expect a NetCDF file with a variable named:

- `MDL_VAR` (2D array: **N samples × (features + target)**)

### 1) TensorFlow script (`cvpps_mdl_train.py`)
- Reads `MDL_VAR` as:
  - **X** = `MDL_VAR[:, 0:23]`  (23 predictors)
  - **y** = `MDL_VAR[:, 23]`    (target)
- A feature name list is present in the script (for reference):  
  `['jday','b01'..'b16','sza','saa','vza','vaa','lon','lat']`  
  but the script directly slices columns (0–22) rather than indexing by name.
- Data split:
  - 70% train, 20% validation, 10% test (via two `train_test_split` calls)
- Standardization:
  - X is standardized using mean/std computed from the concatenation of train+val+test
  - y is standardized similarly

### 2) PyTorch FT-Transformer script (`test_ft-transformer_v0007_rnd-param_fttrue_nocls.py`)
- Selects the NetCDF filename based on a product code argument:
  - `INS` → `Matchup_INSD_202101_202312.nc`
  - `AWV` → `Matchup_AWVD_202101_202312.nc`
  - `AOT` → `Matchup_AOTD_202101_202312.nc`
- Reads `MDL_VAR` as:
  - If `INS`: uses only the first 19 columns as predictors and drops 4 dummy columns
  - Else: uses 23 predictor columns
  - Target is the final column
- Data split:
  - 70% train, 15% val, 15% test
- Scaling:
  - X is standardized with `StandardScaler` fitted on train
  - y is scaled by a constant (`y_scaler`): INS=1000, AWV=10, AOT=1

---

## How to run

### A) Train BPNN (DNN) and Conv1D (TensorFlow)

```bash
python cvpps_mdl_train.py /path/to/your_matchup.nc
```

Outputs are saved under:
- `./callbacks/conv1d/`
- `./callbacks/dnn/`

Per model, it will create:
- `check_points/model_<name>_<run>.hdf5` (best checkpoint per run)
- `<name>_results.csv` (metrics per run + hyperparameters)
- loss plots (`*_loss.png`) and scatter plots (`*_train.png`, `*_valid.png`, `*_test.png`)
- `*_result_datset.npz` (saved predictions)

> Notes:
> - The script performs repeated random hyperparameter sampling (default `n_repeat=30`) for each model.
> - It uses custom utilities from `user_results.py` (`calc_evaluation`, `plot_results`), which must be available on `PYTHONPATH`.
> - It references local font paths (Helvetica) when saving plots; update or remove if not available.

### B) Train FT-Transformer (PyTorch)

```bash
python test_ft-transformer_v0007_rnd-param_fttrue_nocls.py AOT
# or INS / AWV
```

This script writes:
- `training_results.csv` (appended per run)
- `best_model_runXXX.pt` (best checkpoint per run)
- `rmse_curve_runXXX.png`, `loss_curve_runXXX.png`, `scatter_plot_runXXX.png`

> Notes:
> - Default random search is large (`n_runs=1000`, `n_epochs=300`). Consider lowering `n_runs` for quick tests.
> - The script starts from `run=9` (i.e., `for run in range(9, n_runs)`), which you may want to reset to `0` for a clean run.
> - Uses GPU automatically if available (`cuda`).

---

## Model details (high-level)

### BPNN / DNN (TensorFlow)
- Dense layers (3–6 layers sampled), BatchNorm, Activation (relu/selu), Dropout
- Loss: MSE
- Optimizer: RMSprop or Adam

### Conv1D (TensorFlow)
- Reshape to `(n_features, 1)`
- Conv1D → BatchNorm → Activation → MaxPool → Dropout (1–3 layers sampled)
- GlobalMaxPool1D → Dense(1)
- Loss: MSE

### FT-Transformer (PyTorch)
- Feature-wise tokenizer (per-feature weight+bias) → TransformerEncoder layers → pooling(mean) → regression head
- No CLS token (`nocls`)
- Randomized hyperparameters across:
  `d_token`, `n_heads`, `n_layers`, `dropout`, `dim_feedforward`, optimizer, loss, init, batch_size

---

## Reproducibility

- TensorFlow script uses fixed `random_state=1004` for splits, but hyperparameter choices are random.
- PyTorch script uses random hyperparameter draws and does not set a global seed by default.

For reproducible runs, consider setting:
- `random.seed(...)`, `np.random.seed(...)`, and framework-specific seeds.

---

## Common troubleshooting

- **Missing module `user_results`** (TensorFlow script):
  - Ensure `user_results.py` is in the same directory or on `PYTHONPATH`.

- **Font file not found** (TensorFlow loss plot):
  - Edit the font paths inside `cvpps_mdl_train.py` or remove the font properties.

- **NetCDF file not found** (PyTorch script):
  - The PyTorch script expects a fixed filename (e.g., `Matchup_AOTD_202101_202312.nc`) in the current working directory. Place the file there or modify `nc_path`.

- **CUDA / GPU issues**:
  - PyTorch uses `cuda` if available. If driver/CUDA mismatch occurs, force CPU by setting:
    `device = torch.device('cpu')`

---

## Suggested directory layout (NAS-friendly)

```
project_root/
  data/
    Matchup_INSD_202101_202312.nc
    Matchup_AWVD_202101_202312.nc
    Matchup_AOTD_202101_202312.nc
  scripts/
    cvpps_mdl_train.py
    test_ft-transformer_v0007_rnd-param_fttrue_nocls.py
    user_results.py
  outputs/
    callbacks/
    ft_transformer_runs/
```

---

## Citation / provenance

These scripts correspond to the training implementations in:
- TensorFlow DNN/Conv1D trainer: `cvpps_mdl_train.py`
- PyTorch FT-Transformer trainer: `test_ft-transformer_v0007_rnd-param_fttrue_nocls.py`

---

## Contact

Maintainer: (fill in)
- Name: Jongmin Yeom / AISEE Lab / Email: yeom.jongmin@gmail.com
