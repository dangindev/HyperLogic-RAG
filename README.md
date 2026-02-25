# HyperLogic-RAG: Logic-Consistency Enforced Radiology Report Generation via Hypergraph Constraints

<p align="center">
  <img src="figures/pipeline.png" alt="HyperLogic-RAG Pipeline" width="100%">
</p>

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue.svg)](https://miccai2026.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **HyperLogic-RAG**. This repository contains the code for training, evaluation, and inference of our neuro-symbolic framework for radiology report generation.

## 🌟 Overview

Generative models frequently hallucinate when relying solely on internal parametric knowledge. To explicitly ground the generation process, our framework, **HyperLogic-RAG**, introduces a novel dual-branch Retrieval-Augmented Generation (RAG) system:

1. **Vector Database Retrieval**: Utilizes BiomedCLIP embeddings to retrieve visually and semantically similar historical reports.
2. **Hypergraph Knowledge Base Retrieval**: Employs HyperGCN over a clinical hypergraph to model higher-order logical relationships and co-occurring abnormalities.

By conditioning generation on verifiable medical logic, HyperLogic-RAG helps to consistently reduce hallucination issues, yielding a notable improvement in clinical efficacy and structural correctness compared to prior standard baselines.

## 🚀 Main Results

On the large-scale **MIMIC-CXR** dataset, HyperLogic-RAG achieves competitive and robust performance across both Natural Language Generation (NLG) fluency and Clinical Efficacy (CE).

| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CE-F1 |
| --- | :---: | :---: | :---: | :---: | :---: |
| R2Gen (Baseline) | 0.353 | 0.103 | 0.142 | 0.277 | 0.276 |
| **HyperLogic-RAG (Ours)** | **0.386** | **0.112** | **0.172** | **0.287** | **0.618** |

*(Our model demonstrates a substantial gain in clinical accuracy (CE-F1) relative to the R2Gen baseline).*

## 📁 Repository Structure

```
HyperLogicRAG/
├── configs/          # YAML configuration files for models and training
├── data/             # Datasets (MIMIC-CXR, IU X-Ray) & Graph structures
├── figures/          # Images including the architecture pipeline
├── logs/             # Training logs and TensorBoard outputs
├── results/          # Model checkpoints and generated reports
├── scripts/          # Shell scripts for data preprocessing
├── src/              # Source code (models, dataloaders, retrieval, utils)
├── README.md         # This document
└── requirements.txt  # Python package dependencies
```

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/anonymous-researcher/HyperLogic-RAG.git
   cd HyperLogic-RAG
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation

We evaluate our model on two standard benchmarks: **MIMIC-CXR** and **IU X-Ray**. 

1. Download the MIMIC-CXR and IU X-Ray datasets.
2. Extract the images and reports into the `data/` directory.
3. Use the scripts provided in `scripts/` to preprocess the text and build the initial hypergraph structures.

## 🏃‍♂️ Training & Evaluation

We provide SLURM `.sbatch` scripts for running experiments on HPC clusters.

**To train the HyperLogic-RAG model:**
```bash
sbatch run_hyperlogic_biomedclip.sbatch
```

**To evaluate the trained model on the test set:**
```bash
sbatch run_hyperlogic_test.sbatch
# Or manually run:
python eval_hyperlogic_v2_standalone.py --cfg configs/hyperlogic_config_biomedclip_frozen.yaml
```

*(For the frozen BiomedCLIP visual encoder ablations, use `run_hyperlogic_biomedclip_frozen.sbatch`)*.


**License**
This project is licensed under the MIT License - see the LICENSE file for details.
