#!/bin/bash
#SBATCH --job-name=eval_densenet
#SBATCH --output=/home/dangnh3/MICCAI2026/logs/eval_densenet-%j.out
#SBATCH --error=/home/dangnh3/MICCAI2026/logs/eval_densenet-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

echo "=========================================="
echo "DenseNet-121 Baseline Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

cd /home/dangnh3/MICCAI2026

# Activate environment
source ~/miniconda3/bin/activate

# Run evaluation
python /home/dangnh3/MICCAI2026/HyperLogicRAG/scripts/evaluate_densenet.py \
    --config /home/dangnh3/MICCAI2026/HyperLogicRAG/configs/densenet_baseline.yaml \
    --checkpoint /home/dangnh3/MICCAI2026/HyperLogicRAG/results/densenet_baseline/checkpoint_epoch_20.pth

echo
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
