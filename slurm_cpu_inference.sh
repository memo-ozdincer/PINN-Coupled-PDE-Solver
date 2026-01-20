#!/bin/bash
#SBATCH --job-name=scalar_pred_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_inference_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/dbe/logs/scalar_pred_inference_%j.err
#SBATCH --account=rrg-aspuru

# ============================================================================
# Scalar PV Predictors - CPU-only Batch Inference
# Use this for large-scale inference when GPU queue is full
# 192 cores provide fast inference even on CPU
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load gcc/12.3 python/3.11

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR="/scratch/memoozd/ts-tools-scratch/dbe"
cd $WORK_DIR
mkdir -p $WORK_DIR/logs

source venv/bin/activate

# Run inference on CPU
python scalar_predictors/inference.py \
    --models outputs/models \
    --input new_parameters.csv \
    --output predictions.csv \
    --device cpu

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
