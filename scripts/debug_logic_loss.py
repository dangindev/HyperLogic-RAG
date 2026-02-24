"""
Deep Debugging Script for Logic Loss = 0 Issue

This script analyzes:
1. Actual entity_probs distribution
2. Negative hyperedges coverage
3. Why logic loss is 0 despite high entity probs
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import pickle
import numpy as np

print("="*60)
print("LOGIC LOSS DEBUGGING ANALYSIS")
print("="*60)

# Load hypergraph
print("\n[1] Loading hypergraph data...")
with open('data/hypergraph.pkl', 'rb') as f:
    hypergraph_data = pickle.load(f)

negative_hyperedges = hypergraph_data.get('negative_hyperedges', [])
print(f"✓ Loaded {len(negative_hyperedges)} negative hyperedges")

# Sample some to inspect
print("\n[2] Sample negative hyperedges:")
for i, hedge in enumerate(negative_hyperedges[:5]):
    print(f"  {i+1}. Nodes: {hedge['nodes']}")

# Simulate entity_probs like in real training
print("\n[3] Simulating entity_probs scenarios...")

num_entities = 15814
batch_size = 4

# Scenario 1: Random high probs (like early training)
print("\n  Scenario 1: Random high probs (mean=0.5)")
entity_probs_random = torch.rand(batch_size, num_entities) * 0.8 + 0.1  # [0.1, 0.9]
print(f"    Stats: min={entity_probs_random.min():.4f}, max={entity_probs_random.max():.4f}, mean={entity_probs_random.mean():.4f}")

# Calculate loss
from src.utils.hypergraph_utils import logic_constrained_loss
loss_random = logic_constrained_loss(entity_probs_random, negative_hyperedges)
print(f"    Logic Loss: {loss_random.item():.6f}")

# Scenario 2: Simulate "learned avoidance" - model learns to set conflicting entities to different probs
print("\n  Scenario 2: Learned avoidance (high mean but low conflicts)")
entity_probs_smart = torch.rand(batch_size, num_entities) * 0.9 + 0.1  # High overall

# For each negative hyperedge, set ONE node to low prob
for hedge in negative_hyperedges[:100]:  # First 100 conflicts
    nodes = hedge['nodes']
    if len(nodes) == 2:
        node_a, node_b = nodes
        # Set one of them to low prob
        entity_probs_smart[:, node_a] = torch.rand(batch_size) * 0.1  # [0, 0.1]

print(f"    Stats: min={entity_probs_smart.min():.4f}, max={entity_probs_smart.max():.4f}, mean={entity_probs_smart.mean():.4f}")
loss_smart = logic_constrained_loss(entity_probs_smart, negative_hyperedges)
print(f"    Logic Loss: {loss_smart.item():.6f}")

# Scenario 3: Analyze ACTUAL distribution from model
print("\n  Scenario 3: Distribution matching training (mean=0.72, min=0.0)")
# Create a distribution like we saw: mean=0.72, but some entities at 0
entity_probs_real = torch.rand(batch_size, num_entities) * 0.5 + 0.5  # [0.5, 1.0] mostly high

# Set some entities to near 0 (like min=0.0 we observed)
num_zero = int(num_entities * 0.1)  # 10% at zero
zero_indices = torch.randperm(num_entities)[:num_zero]
entity_probs_real[:, zero_indices] = torch.rand(batch_size, num_zero) * 0.01  # Near 0

print(f"    Stats: min={entity_probs_real.min():.4f}, max={entity_probs_real.max():.4f}, mean={entity_probs_real.mean():.4f}")
loss_real = logic_constrained_loss(entity_probs_real, negative_hyperedges)
print(f"    Logic Loss: {loss_real.item():.6f}")

# Deep dive: Check which conflicts actually fire
print("\n[4] Detailed conflict analysis (Scenario 3):")
violations = []
for hedge in negative_hyperedges:
    nodes = hedge['nodes']
    if len(nodes) == 2:
        node_a, node_b = nodes
        prob_a = entity_probs_real[:, node_a]
        prob_b = entity_probs_real[:, node_b]
        violation = (prob_a * prob_b).mean().item()
        violations.append(violation)

violations = sorted(violations, reverse=True)
print(f"  Top 10 violations: {violations[:10]}")
print(f"  Bottom 10 violations: {violations[-10:]}")
print(f"  Mean violation: {np.mean(violations):.6f}")
print(f"  Median violation: {np.median(violations):.6f}")
print(f"  % violations > 0.1: {100 * sum(1 for v in violations if v > 0.1) / len(violations):.1f}%")
print(f"  % violations == 0: {100 * sum(1 for v in violations if v == 0) / len(violations):.1f}%")

print("\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)
print("\n**Hypothesis:** Model learned to set ONE entity in each conflict pair to ~0")
print("This makes prob_a * prob_b ≈ 0, thus logic_loss ≈ 0")
print("\nTesting...")

# Count how many negative hyperedge nodes are at ~0
threshold_low = 0.05
conflict_nodes_all = set()
for hedge in negative_hyperedges:
    if len(hedge['nodes']) == 2:
        conflict_nodes_all.update(hedge['nodes'])

print(f"\nTotal unique nodes in conflicts: {len(conflict_nodes_all)}")

# Check how many are near zero
entity_probs_mean = entity_probs_real.mean(dim=0)  # [N]
near_zero_mask = entity_probs_mean < threshold_low
num_near_zero_in_conflicts = sum(1 for node in conflict_nodes_all if near_zero_mask[node])

print(f"Nodes in conflicts with prob < {threshold_low}: {num_near_zero_in_conflicts} / {len(conflict_nodes_all)} ({100*num_near_zero_in_conflicts/len(conflict_nodes_all):.1f}%)")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
If >50% of conflict nodes have prob<0.05, then:
✓ Model IS learning to avoid conflicts (SMART but WRONG for our purpose)
✗ Logic loss correctly computes = 0 (not a bug in loss function)
→ SOLUTION: Need different loss formulation that penalizes LOW probs on conflict nodes
""")
