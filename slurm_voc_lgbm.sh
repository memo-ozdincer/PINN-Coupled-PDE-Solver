#!/bin/bash
#SBATCH --job-name=voc_lgbm_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/voc_lgbm_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/voc_lgbm_%j.err
#SBATCH --account=aip-aspuru

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load gcc/12.3 cuda/12.2 python/3.11

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Working directory
WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe/scalar_predictors"
cd $WORK_DIR

# Create logs directory if needed
mkdir -p /scratch/memoozd/ts-tools-scratch/dbe/logs

# Activate virtual environment
source ../venv/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Output directory with timestamp
OUT_DIR="$WORK_DIR/outputs_voc_lgbm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/models"

# ============================================================================
# CONFIGURATION
# ============================================================================
export HPO_TRIALS_LGBM=50
export HPO_TIMEOUT=3600

echo "Running Voc LGBM training..."
echo "Output directory: $OUT_DIR"
echo "HPO_TRIALS_LGBM: $HPO_TRIALS_LGBM"
echo "HPO_TIMEOUT: $HPO_TIMEOUT"
echo ""

# NOTE:
# - Voc LGBM uses physics features + Voc ceiling ratio target.
# - Jacobian regularization applies to Voc NN only (not LGBM).

python - <<PY
import json
from pathlib import Path
import numpy as np
import torch
import os

from config import COLNAMES, RANDOM_SEED, VAL_SPLIT, TEST_SPLIT
from data import load_raw_data, prepare_tensors, extract_targets_gpu, split_indices
from features import compute_all_physics_features, compute_voc_ceiling
from preprocessing import PVDataPreprocessor
from models.voc_lgbm import build_voc_model
from hpo import HPOConfig, DistributedHPO, get_best_configs_from_study

params_file = "LHS_parameters_m.txt"
iv_file = "IV_m.txt"
out_dir = Path(r"${OUT_DIR}")
models_dir = out_dir / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading data from {params_file} and {iv_file}")
params_df, iv_data = load_raw_data(params_file, iv_file)
params_raw = params_df.values.astype(np.float32)

params_tensor, iv_tensor, v_grid = prepare_tensors(params_df, iv_data, device)
targets = extract_targets_gpu(iv_tensor, v_grid)
targets_np = {k: v.cpu().numpy() for k, v in targets.items()}

physics_features = compute_all_physics_features(params_tensor).cpu().numpy()
voc_ceiling = compute_voc_ceiling(params_tensor).cpu().numpy()

train_idx, val_idx, test_idx = split_indices(len(params_df), VAL_SPLIT, TEST_SPLIT)

pre = PVDataPreprocessor(colnames=list(COLNAMES))
X_train = pre.fit_transform_params(params_raw[train_idx])
X_val = pre.transform_params(params_raw[val_idx])

X_train_ph = physics_features[train_idx]
X_val_ph = physics_features[val_idx]

voc_ceiling_train = voc_ceiling[train_idx]
voc_ceiling_val = voc_ceiling[val_idx]
voc_train = targets_np["Voc"][train_idx]
voc_val = targets_np["Voc"][val_idx]

# HPO for Voc LGBM
hpo_config = HPOConfig(
    n_trials_lgbm=int(os.environ["HPO_TRIALS_LGBM"]),
    timeout_per_model=int(os.environ["HPO_TIMEOUT"]),
)
engine = DistributedHPO(hpo_config)

X_train_voc = np.hstack([
    X_train, X_train_ph,
    np.log10(np.abs(voc_ceiling_train) + 1e-30).reshape(-1, 1)
])
X_val_voc = np.hstack([
    X_val, X_val_ph,
    np.log10(np.abs(voc_ceiling_val) + 1e-30).reshape(-1, 1)
])
y_train_voc = voc_train / (np.abs(voc_ceiling_train) + 1e-30)
y_val_voc = voc_val / (np.abs(voc_ceiling_val) + 1e-30)

voc_params, voc_study = engine.optimize_lgbm(
    X_train_voc, y_train_voc, X_val_voc, y_val_voc, "voc"
)
best_cfg = get_best_configs_from_study({"voc_lgbm": {"params": voc_params}})["voc_lgbm"]

print("Training final Voc LGBM with best params...")
model = build_voc_model(best_cfg)
model.fit(
    X_train, X_train_ph, voc_train, voc_ceiling_train,
    X_val, X_val_ph, voc_val, voc_ceiling_val
)

pred_val = model.predict(X_val, X_val_ph, voc_ceiling_val)
rmse = float(np.sqrt(np.mean((pred_val - voc_val) ** 2)))
mae = float(np.mean(np.abs(pred_val - voc_val)))
print(f"Validation Voc RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# Save artifacts
models_dir.mkdir(parents=True, exist_ok=True)
model.save(str(models_dir / "voc_lgbm.txt"))
pre.save(str(models_dir / "preprocessor.joblib"))

configs = {
    "voc_lgbm": best_cfg.__dict__,
    "voc_model_type": "lgbm",
}
with open(models_dir / "configs.json", "w") as f:
    json.dump(configs, f, indent=2)

metrics = {
    "voc_lgbm": {
        "rmse": rmse,
        "mae": mae,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
}
with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved model to {models_dir / 'voc_lgbm.txt'}")
print(f"Saved preprocessor to {models_dir / 'preprocessor.joblib'}")
print(f"Saved metrics to {out_dir / 'metrics.json'}")
PY

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
