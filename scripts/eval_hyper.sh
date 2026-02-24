#!/bin/bash
#SBATCH --job-name=eval_hyper
#SBATCH --output=/home/dangnh3/MICCAI2026/logs/eval_hyper-%j.out
#SBATCH --error=/home/dangnh3/MICCAI2026/logs/eval_hyper-%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/miniconda3/bin/activate

cd /home/dangnh3/MICCAI2026/HyperLogicRAG

python scripts/evaluate_hyperlogic.py \
    --config configs/hyperlogic_config_v2_full.yaml \
    --checkpoint results/hyperlogic_rag_v2/best_model.pth

