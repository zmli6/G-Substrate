# G-Substrate: Graph is a Substrate Across Data Modalities

<p align="center">
  <a href="https://huggingface.co/datasets/zmli/G-Substrate-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-G--Substrate--Data-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/zmli/G-Substrate-Qwen3-VL-2B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-G--Substrate--Qwen3--VL--2B-blue" alt="Model"></a>
  <a href="https://arxiv.org/abs/2601.22384"><img src="https://img.shields.io/badge/arXiv-2601.22384-b31b1b" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/ICML-2026-purple" alt="ICML 2026"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

This repository contains the code for the paper:

> **Graph is a Substrate Across Data Modalities**
> Ziming Li, Xiaoming Wu, Zehong Wang, Jiazheng Li, Yijun Tian, Jinhe Bi, Yunpu Ma, Yanfang Ye, Chuxu Zhang
> *ICML 2026*

## Overview

G-Substrate introduces a representation-centric perspective where graph structure is treated as a **persistent structural substrate** that accumulates across heterogeneous data modalities and tasks. The framework comprises two mechanisms:

1. **Unified Structural Schema**: Ensures compatibility among graph representations across modalities and tasks.
2. **Interleaved Role-Based Training**: Exposes the same graph structure to multiple functional roles (generation and understanding) during learning.

## Main Results

| Method | CT | CD | SP | BM | BLEU-4 | ROUGE-L | PCIs | MA-S | MA-T | MA-C | HiE |
|--------|----:|----:|----:|----:|-------:|--------:|-----:|-----:|-----:|-----:|----:|
| G-Substrate | 98.41 | 96.97 | 48.59 | 94.54 | 51.53 | 68.47 | 25.38 | 52.20 | 42.68 | 40.91 | 25.15 |

**Tasks**: GAR (Graph Algorithmic Reasoning: CT/CD/SP/BM), MGD (Molecular Graph Description: BLEU-4/ROUGE-L), SGG (Scene Graph Generation: PCIs R@50), ERE (Event Relation Extraction: F1 on MA-S/MA-T/MA-C/HiE).

## Repository Structure

```
G-Substrate/
├── data_processing/              # Data transformation pipeline
│   ├── transform_sgg.py          # Scene graph → unified schema
│   ├── transform_mol.py          # Molecular graph → unified schema
│   ├── transform_nlgraph.py      # Graph algorithmic tasks → unified schema
│   ├── transform_event.py        # Event graph → unified schema
│   └── generate_interleave.py    # Generate interleaved role-based tasks
│
├── training/                     # Training with LLaMA-Factory
│   ├── configs/                  # Multi-task & single-task SFT configs
│   ├── dataset_info.json         # LLaMA-Factory dataset registry
│   └── train.sbatch              # SLURM training script
│
└── inference/                    # Inference + Evaluation
    ├── infer.py                  # vLLM inference
    ├── evaluate.py               # Unified evaluation router
    ├── run.sh                    # Single entry: infer → evaluate
    └── evaluators/               # Task-specific evaluators
```

## Setup

```bash
# 1. Create environment
source setup_env.sh

# 2. Install LLaMA-Factory
pip install llamafactory
```

## Data

Download the dataset from HuggingFace:

```bash
huggingface-cli download zmli/G-Substrate-Data --repo-type dataset --local-dir ./data
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

Download the model from HuggingFace:

```bash
huggingface-cli download zmli/G-Substrate-Qwen3-VL-2B --local-dir ./model
```

## Citation

```bibtex
@inproceedings{li2026gsubstrate,
  title={Graph is a Substrate Across Data Modalities},
  author={Li, Ziming and Wu, Xiaoming and Wang, Zehong and Li, Jiazheng and Tian, Yijun and Bi, Jinhe and Ma, Yunpu and Ye, Yanfang and Zhang, Chuxu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
