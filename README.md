# GK2A_Solar_Radiation_AOD_Deep_learning
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
