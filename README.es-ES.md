

# G-Substrate: El Grafo es un Sustrato a través de las Modalidades de Datos

<p align="center">
  <a href="https://huggingface.co/datasets/zmli/G-Substrate-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-G--Substrate--Data-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/zmli/G-Substrate-Qwen3-VL-2B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-G--Substrate--Qwen3--VL--2B-blue" alt="Model"></a>
  <a href="https://arxiv.org/abs/2601.22384"><img src="https://img.shields.io/badge/arXiv-2601.22384-b31b1b" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/ICML-2026-purple" alt="ICML 2026"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

Este repositorio contiene el código del artículo:

> **El Grafo es un Sustrato a través de las Modalidades de Datos**
> Ziming Li, Xiaoming Wu, Zehong Wang, Jiazheng Li, Yijun Tian, Jinhe Bi, Yunpu Ma, Yanfang Ye, Chuxu Zhang
> *ICML 2026*

## Descripción General

G-Substrate introduce una perspectiva centrada en la representación, donde la estructura del grafo se trata como un **sustrato estructural persistente** que se acumula a través de modalidades de datos heterogéneas y tareas. El marco de trabajo comprende dos mecanismos:

1. **Esquema Estructural Unificado**: Garantiza la compatibilidad entre las representaciones de grafos en diferentes modalidades y tareas.
2. **Entrenamiento Intercalado Basado en Roles**: Expone la misma estructura de grafo a múltiples roles funcionales (generación y comprensión) durante el aprendizaje.

## Resultados Principales

| Method | CT | CD | SP | BM | BLEU-4 | ROUGE-L | PCIs | MA-S | MA-T | MA-C | HiE |
|--------|----:|----:|----:|----:|-------:|--------:|-----:|-----:|-----:|-----:|----:|
| G-Substrate | 98.41 | 96.97 | 48.59 | 94.54 | 51.53 | 68.47 | 25.38 | 52.20 | 42.68 | 40.91 | 25.15 |

**Tareas**: GAR (Razonamiento Algorítmico en Grafos: CT/CD/SP/BM), MGD (Descripción de Grafos Moleculares: BLEU-4/ROUGE-L), SGG (Generación de Grafos de Escena: PCIs R@50), ERE (Extracción de Relaciones de Eventos: F1 en MA-S/MA-T/MA-C/HiE).

## Estructura del Repositorio

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

## Configuración

```bash
# 1. Create environment
source setup_env.sh

# 2. Install LLaMA-Factory
pip install llamafactory
```

## Datos

Descarga el conjunto de datos desde HuggingFace:

```bash
huggingface-cli download zmli/G-Substrate-Data --repo-type dataset --local-dir ./data
```

### Imágenes de Visual Genome (para SGG)

Los datos del grafo de escena hacen referencia a las imágenes mediante rutas relativas (p. ej., `VG_100K/2321212.jpg`).

1. Descarga desde [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html):
   - [images.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) (VG_100K)
   - [images2.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (VG_100K_2)

2. Extrae en un directorio `images/`:
   ```
   images/
   ├── VG_100K/
   └── VG_100K_2/
   ```

3. Establece `vg_images_dir` en `config.yaml` a la ruta del directorio `images/` para que las rutas relativas se resuelvan correctamente.

### Procesamiento de Datos (Opcional)

Para regenerar los conjuntos de datos con el esquema unificado a partir de los datos fuente originales:

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

## Entrenamiento

El entrenamiento utiliza [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) para SFT multitarea.

```bash
# Multi-task training (G-Substrate)
llamafactory-cli train training/configs/sft_multi_task.yaml

# Single-task training (baseline)
llamafactory-cli train training/configs/sft_single.yaml

# SLURM submission
sbatch training/train.sbatch
```

## Inferencia y Evaluación

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

## Modelo Preentrenado

Descarga el modelo desde HuggingFace:

```bash
huggingface-cli download zmli/G-Substrate-Qwen3-VL-2B --local-dir ./model
```

## Cita

```bibtex
@inproceedings{li2026gsubstrate,
  title={Graph is a Substrate Across Data Modalities},
  author={Li, Ziming and Wu, Xiaoming and Wang, Zehong and Li, Jiazheng and Tian, Yijun and Bi, Jinhe and Ma, Yunpu and Ye, Yanfang and Zhang, Chuxu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
