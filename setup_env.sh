#!/bin/bash
# G-Substrate Environment Setup
# Usage: source setup_env.sh
#
# This script sets up the conda environment and installs dependencies.
# Run once to create the environment, then source to activate.

ENV_NAME="gsubstrate"

# ---- Create conda environment (if not exists) ----
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# ---- Activate ----
conda activate ${ENV_NAME}

# ---- Install dependencies ----
pip install -r requirements.txt

# ---- Optional: Set HuggingFace cache ----
# export HF_HOME=/path/to/hf_cache
# export TRANSFORMERS_CACHE=${HF_HOME}/hub

echo "Environment ready: ${ENV_NAME}"
