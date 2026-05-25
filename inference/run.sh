#!/bin/bash
# G-Substrate: Unified Inference + Evaluation Pipeline
# Usage: bash inference/run.sh --model_path <path> --test_file <path> [--output_dir <path>]
#
# Pipeline: infer → post-process (merge raw fields) → evaluate

set -euo pipefail

# ---- Parse arguments ----
MODEL_PATH=""
TEST_FILE=""
OUTPUT_DIR="./outputs"
TEMPLATE="qwen3_vl_nothink"
MAX_NEW_TOKENS=4096
TENSOR_PARALLEL_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --test_file) TEST_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --template) TEMPLATE="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$TEST_FILE" ]; then
    echo "Usage: bash inference/run.sh --model_path <path> --test_file <path>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

INFER_OUTPUT="${RUN_DIR}/predictions_raw.jsonl"
FIXED_OUTPUT="${RUN_DIR}/predictions.jsonl"
EVAL_OUTPUT="${RUN_DIR}/eval_results.json"

echo "============================================"
echo "G-Substrate Inference + Evaluation"
echo "============================================"
echo "Model:   $MODEL_PATH"
echo "Input:   $TEST_FILE"
echo "Output:  $RUN_DIR"
echo ""

# ---- Step 1: Inference ----
echo "[Step 1/3] Running vLLM inference..."
python "${SCRIPT_DIR}/infer.py" \
    --model_path "$MODEL_PATH" \
    --raw_file "$TEST_FILE" \
    --output_file "$INFER_OUTPUT" \
    --template "$TEMPLATE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --trust_remote_code

echo "[Step 1/3] Inference complete: $INFER_OUTPUT"

# ---- Step 2: Post-process ----
# Merge task/subtask/messages/images from raw test file back into predictions.
# The evaluator needs these fields for task routing and ground-truth comparison.
echo "[Step 2/3] Post-processing: merging raw fields into predictions..."

python - "$TEST_FILE" "$INFER_OUTPUT" "$FIXED_OUTPUT" << 'PYEOF'
import json, sys

raw_path, pred_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Load raw test data (JSON array)
with open(raw_path, "r", encoding="utf-8") as f:
    raw_items = json.load(f)

# Load predictions (JSONL)
pred_items = []
with open(pred_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pred_items.append(json.loads(line))

if len(raw_items) != len(pred_items):
    print(f"[WARN] Length mismatch: raw={len(raw_items)}, pred={len(pred_items)}")
    # Use min length to avoid crash
    n = min(len(raw_items), len(pred_items))
else:
    n = len(raw_items)

with open(out_path, "w", encoding="utf-8") as fw:
    for i in range(n):
        r, p = raw_items[i], pred_items[i]
        item = {
            "task":     r.get("task"),
            "subtask":  r.get("subtask"),
            "messages": r.get("messages"),
            "images":   r.get("images"),
            "prompt":   p.get("prompt"),
            "predict":  p.get("predict"),
            "label":    p.get("label"),
        }
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[OK] Merged {n} items -> {out_path}")
PYEOF

echo "[Step 2/3] Post-processing complete: $FIXED_OUTPUT"

# ---- Step 3: Evaluation ----
echo "[Step 3/3] Running evaluation..."
python "${SCRIPT_DIR}/evaluate.py" \
    --input "$FIXED_OUTPUT" \
    --output "$EVAL_OUTPUT"

echo ""
echo "============================================"
echo "Results: $EVAL_OUTPUT"
echo "============================================"
cat "$EVAL_OUTPUT"
