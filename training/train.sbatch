#!/bin/bash
#SBATCH --job-name=g-substrate-train
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/train_%j.out
#SBATCH --error=outputs/train_%j.err

# ---- Environment ----
source setup_env.sh 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1

# ---- Configuration ----
CONFIG="${CONFIG:-training/configs/sft_multi_task.yaml}"

echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Train ----
llamafactory-cli train "$CONFIG"
