#!/bin/bash
#SBATCH --job-name=densenet_v2
#SBATCH --output=/home/dangnh3/MICCAI2026/logs/densenet_v2-%j.out
#SBATCH --error=/home/dangnh3/MICCAI2026/logs/densenet_v2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

echo "=========================================="
echo "DenseNet-121 Baseline V2 Training"
echo "with Anti-Collapse Fixes"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

cd /home/dangnh3/MICCAI2026/HyperLogicRAG

# Activate environment
source ~/miniconda3/bin/activate

# Print config
echo "Configuration:"
cat configs/densenet_baseline_v2.yaml
echo

# Run training
python src/train_baseline_v2.py \
    --config configs/densenet_baseline_v2.yaml

echo
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
