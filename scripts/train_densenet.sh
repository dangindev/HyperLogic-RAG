#!/bin/bash
#SBATCH --job-name=densenet_baseline
#SBATCH --output=/home/dangnh3/MICCAI2026/logs/densenet-%j.out
#SBATCH --error=/home/dangnh3/MICCAI2026/logs/densenet-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-0
#SBATCH --time=24:00:00
#SBATCH --mem=32G

echo "=========================================="
echo "DenseNet-121 Baseline Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

cd /home/dangnh3/MICCAI2026/HyperLogicRAG

# Activate environment
source ~/miniconda3/bin/activate

# Set PYTHONPATH to include current directory
export PYTHONPATH=/home/dangnh3/MICCAI2026/HyperLogicRAG:$PYTHONPATH

# Run training
python -m src.train_baseline --config configs/densenet_baseline.yaml

echo
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
