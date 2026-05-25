# G-Substrate: Graph is a Substrate Across Data Modalities

This repository contains the code and data for the paper:

> **Graph is a Substrate Across Data Modalities**
> Ziming Li, Xiaoming Wu, Zehong Wang, Jiazheng Li, Yijun Tian, Jinhe Bi, Yunpu Ma, Yanfang Ye, Chuxu Zhang
> *ICML 2026*

## Overview

G-Substrate introduces a representation-centric perspective where graph structure is treated as a **persistent structural substrate** that accumulates across heterogeneous data modalities and tasks. The framework comprises two mechanisms:

1. **Unified Structural Schema**: Ensures compatibility among graph representations across modalities and tasks.
2. **Interleaved Role-Based Training**: Exposes the same graph structure to multiple functional roles (generation and understanding) during learning.

## Repository Structure

```
G-Substrate/
├── config.yaml                   # Centralized configuration
├── setup_env.sh                  # Environment setup
├── requirements.txt              # Python dependencies
│
├── data_processing/              # Data transformation pipeline
│   ├── transform_sgg.py          # Scene graph → unified schema
│   ├── transform_mol.py          # Molecular graph → unified schema
│   ├── transform_nlgraph.py      # Graph algorithmic tasks → unified schema
│   ├── transform_event.py        # Event graph → unified schema
│   ├── generate_interleave.py    # Generate interleaved role-based tasks
│   └── utils/
│       ├── merge_shuffle.py      # Merge and shuffle datasets
│       └── analyze_graph_sizes.py
│
├── training/                     # Training with LLaMA-Factory
│   ├── configs/
│   │   ├── sft_multi_task.yaml   # Multi-task SFT config
│   │   └── sft_single.yaml      # Single-task SFT config
│   ├── dataset_info.json         # LLaMA-Factory dataset registry
│   └── train.sbatch              # SLURM training script
│
└── inference/                    # Inference + Evaluation
    ├── infer.py                  # vLLM inference
    ├── evaluate.py               # Unified evaluation router
    ├── run.sh                    # Single entry: infer → evaluate
    ├── run.sbatch                # SLURM version
    └── evaluators/               # Task-specific evaluators
        ├── graph_search_eval.py
        ├── mol_eval.py
        ├── event_graph_eval.py
        └── sgg_eval/
            ├── vg_sgg_eval.py
            ├── vg_metadata.json
            └── vg150_gt.pkl
```

## Setup

```bash
# 1. Create environment
source setup_env.sh

# 2. Install LLaMA-Factory
pip install llamafactory
```

## Data

### Download

Download the G-Substrate dataset from HuggingFace:

```bash
# TODO: Replace with your HuggingFace dataset URL
# huggingface-cli download <org>/G-Substrate-Data --local-dir ./data
```

### Visual Genome Images (for SGG)

Scene graph data references images via relative paths (e.g., `VG_100K/2321212.jpg`).

1. Download from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html):
   - [images.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) (VG_100K)
   - [images2.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (VG_100K_2)

2. Extract into an `images/` directory:
   ```
   images/
   ├── VG_100K/
   └── VG_100K_2/
   ```

3. Set `vg_images_dir` in `config.yaml` to the `images/` directory path so that relative paths resolve correctly.

### Data Processing (Optional)

To regenerate the unified schema datasets from raw source data:

```bash
# Scene Graph (from VG150 SFT data)
python data_processing/transform_sgg.py path/to/train.json path/to/test.json

# Molecular Graph (from Mol-Instructions)
python data_processing/transform_mol.py path/to/smiles_graph.json

# Graph Algorithmic (from NLGraph / GVLQA)
python data_processing/transform_nlgraph.py path/to/nlgraph_dir/

# Event Graph (from MAVEN-ERE, MATRES, HiEve)
python data_processing/transform_event.py path/to/train_ERE.json path/to/train_MATRES.json

# Generate interleaved role-based training data
python data_processing/generate_interleave.py \
    --sg_path data/train/scene_graph.json \
    --eg_path data/train/event_graph.json \
    --gs_path data/train/graph_search.json \
    --output_dir data/train/
```

## Training

Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for multi-task SFT.

```bash
# Multi-task training (G-Substrate)
llamafactory-cli train training/configs/sft_multi_task.yaml

# Single-task training (baseline)
llamafactory-cli train training/configs/sft_single.yaml

# SLURM submission
sbatch training/train.sbatch
```

## Inference & Evaluation

```bash
# Run inference + evaluation pipeline
bash inference/run.sh \
    --model_path path/to/checkpoint \
    --test_file data/test/scene_graph.json

# Or submit as SLURM job
export MODEL_PATH=path/to/checkpoint
export TEST_FILE=data/test/scene_graph.json
sbatch inference/run.sbatch
```

## Pre-trained Model

Download the G-Substrate checkpoint from HuggingFace:

```bash
# TODO: Replace with your HuggingFace model URL
# huggingface-cli download <org>/G-Substrate-Qwen3-VL-2B --local-dir ./model
```

**Tasks**: GAR (Graph Algorithmic Reasoning: CT/CD/SP/BM), MGD (Molecular Graph Description: BLEU-4/ROUGE-L), SGG (Scene Graph Generation: PCIs R@50), ERE (Event Relation Extraction: F1 on MA-S/MA-T/MA-C/HiE).

## Citation

```bibtex
@inproceedings{li2026gsubstrate,
  title={Graph is a Substrate Across Data Modalities},
  author={Li, Ziming and Wu, Xiaoming and Wang, Zehong and Li, Jiazheng and Tian, Yijun and Bi, Jinhe and Ma, Yunpu and Ye, Yanfang and Zhang, Chuxu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
