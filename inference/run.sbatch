#!/bin/bash
#SBATCH --job-name=g-substrate-eval
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/eval_%j.out
#SBATCH --error=outputs/eval_%j.err

# ---- Environment ----
# Modify these paths for your environment
source setup_env.sh 2>/dev/null || true

# Prevent thread oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Set HF cache to scratch to avoid quota issues (adjust path as needed)
# export HF_HOME=/path/to/scratch/hf_cache
# export TRANSFORMERS_CACHE=${HF_HOME}/hub

# ---- Required Environment Variables ----
# MODEL_PATH: path to model checkpoint or HuggingFace model ID
# TEST_FILE:  path to test JSON file
# These should be exported before sbatch submission:
#   export MODEL_PATH=/path/to/model
#   export TEST_FILE=/path/to/test.json
#   sbatch inference/run.sbatch

if [ -z "$MODEL_PATH" ] || [ -z "$TEST_FILE" ]; then
    echo "ERROR: MODEL_PATH and TEST_FILE must be set."
    echo "Usage: export MODEL_PATH=... TEST_FILE=... && sbatch inference/run.sbatch"
    exit 1
fi

# ---- Optional Environment Variables ----
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TEMPLATE="${TEMPLATE:-qwen3_vl_nothink}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# ---- GPU Info ----
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Run Pipeline ----
bash inference/run.sh \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --template "$TEMPLATE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
