#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --output=/home/dangnh3/MICCAI2026/logs/eval_baseline-%j.out
#SBATCH --error=/home/dangnh3/MICCAI2026/logs/eval_baseline-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/miniconda3/bin/activate

cd /home/dangnh3/MICCAI2026/HyperLogicRAG

python eval_baseline_standalone.py --checkpoint_path results/baseline/checkpoint_epoch_30.pth
