"""
Load checkpoint epoch 5 and analyze ACTUAL entity predictions
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import pickle
import numpy as np
from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.datasets import MIMICCXRDataset
from torch.utils.data import DataLoader

print("="*60)
print("ANALYZING ACTUAL MODEL PREDICTIONS (Epoch 5)")
print("="*60)

# Load hypergraph
print("\n[1] Loading hypergraph...")
with open('data/hypergraph.pkl', 'rb') as f:
    hypergraph_data = pickle.load(f)

negative_hyperedges = hypergraph_data['negative_hyperedges']
print(f"✓ Negative hyperedges: {len(negative_hyperedges)}")

# Load checkpoint
print("\n[2] Loading checkpoint epoch 5...")
checkpoint_path = 'results/hyperlogic_rag/checkpoint_epoch_5.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"✓ Checkpoint loaded")
print(f"  Keys: {list(checkpoint.keys())}")

# Create model
print("\n[3] Creating model...")
model = HyperLogicRAGModel(
    num_entities=15814,
    vocab_size=30522,  # BERT vocab
    embed_dim=768,
    num_heads=8,
    num_layers=6,
    use_hypergraph=True,
    hyperedge_index=hypergraph_data['hyperedge_index'],
    hyperedge_weight=hypergraph_data['hyperedge_weight'],
    negative_hyperedges=negative_hyperedges
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Model loaded and set to eval mode")

# Load sample data
print("\n[4] Loading sample data...")
dataset = MIMICCXRDataset(
    split='train',
    max_length=128
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
batch = next(iter(dataloader))
print(f"✓ Batch loaded: {batch['image'].shape}")

# Forward pass to get entity_probs
print("\n[5] Running forward pass...")
with torch.no_grad():
    images = batch['image']
    reports = batch['input_ids']
    logits, entity_probs = model(images, reports[:, :-1])
    
print(f"✓ Entity probs shape: {entity_probs.shape}")
print(f"  Stats: min={entity_probs.min():.6f}, max={entity_probs.max():.6f}, mean={entity_probs.mean():.6f}")

# Analyze distribution
print("\n[6] Entity probability distribution:")
prob_bins = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
entity_probs_flat = entity_probs.flatten().numpy()
hist, _ = np.histogram(entity_probs_flat, bins=prob_bins)
for i in range(len(prob_bins)-1):
    pct = 100 * hist[i] / len(entity_probs_flat)
    print(f"  [{prob_bins[i]:.2f}, {prob_bins[i+1]:.2f}): {hist[i]:8d} ({pct:5.1f}%)")

# Calculate logic loss
print("\n[7] Calculating logic loss...")
from src.utils.hypergraph_utils import logic_constrained_loss
logic_loss = logic_constrained_loss(entity_probs, negative_hyperedges)
print(f"✓ Logic loss: {logic_loss.item():.6f}")

# Deep dive: which conflicts fire?
print("\n[8] Conflict analysis:")
violations = []
for hedge in negative_hyperedges:
    nodes = hedge['nodes']
    if len(nodes) == 2:
        node_a, node_b = nodes
        prob_a = entity_probs[:, node_a]
        prob_b = entity_probs[:, node_b]
        violation = (prob_a * prob_b).mean().item()
        violations.append((violation, node_a, node_b, prob_a.mean().item(), prob_b.mean().item()))

violations.sort(reverse=True, key=lambda x: x[0])

print(f"\n  Top 10 conflicts (highest violations):")
for i, (viol, na, nb, pa, pb) in enumerate(violations[:10]):
    print(f"    {i+1}. Nodes ({na}, {nb}): violation={viol:.6f}, prob_a={pa:.4f}, prob_b={pb:.4f}")

print(f"\n  Bottom 10 conflicts (lowest violations):")
for i, (viol, na, nb, pa, pb) in enumerate(violations[-10:]):
    print(f"    {i+1}. Nodes ({na}, {nb}): violation={viol:.6f}, prob_a={pa:.4f}, prob_b={pb:.4f}")

# Check how many conflicts have BOTH probs > 0.5
high_both = sum(1 for v, na, nb, pa, pb in violations if pa > 0.5 and pb > 0.5)
print(f"\n  Conflicts where BOTH probs > 0.5: {high_both} / {len(violations)} ({100*high_both/len(violations):.1f}%)")

# Check how many have at least ONE prob < 0.1
low_one = sum(1 for v, na, nb, pa, pb in violations if pa < 0.1 or pb < 0.1)
print(f"  Conflicts where AT LEAST ONE prob < 0.1: {low_one} / {len(violations)} ({100*low_one/len(violations):.1f}%)")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
if logic_loss.item() < 0.01:
    print("\n⚠️  CONFIRMED: Logic loss is near 0")
    if low_one > len(violations) * 0.5:
        print("\n✓ ROOT CAUSE: Model learned to suppress ONE entity in each conflict")
        print("  This is a VALID learning strategy to minimize loss,")
        print("  but it's NOT what we want (we want factual accuracy).")
        print("\n→ SOLUTION: Redesign loss to encourage FACTUAL predictions")
        print("  rather than conflict-avoidance tricks.")
    else:
        print("\n✗ UNEXPECTED: Conflicts don't explain zero loss")
        print("  Need to investigate loss calculation further")
else:
    print(f"\n✓ Logic loss is healthy: {logic_loss.item():.6f}")
    print("  The training issue might be elsewhere")
