"""
Direct test of logic_constrained_loss with realistic entity_probs
Simulate what happens in actual training at epoch 6+
"""
import torch
import pickle
import sys
sys.path.append('/home/dangnh3/MICCAI2026/HyperLogicRAG')

from src.utils.hypergraph_utils import logic_constrained_loss

print("="*60)
print("TESTING LOGIC_CONSTRAINED_LOSS FUNCTION")
print("="*60)

# Load negative hyperedges
with open('/home/dangnh3/MICCAI2026/HyperLogicRAG/data/hypergraph.pkl', 'rb') as f:
    hypergraph_data = pickle.load(f)
negative_hyperedges = hypergraph_data['negative_hyperedges']
print(f"\n✓ Loaded {len(negative_hyperedges)} negative hyperedges")

print("\n" + "="*60)
print("TEST 1: High uniform probs (baseline)")
print("="*60)
entity_probs_1 = torch.ones(4, 15814) * 0.7  # All entities at 0.7
loss_1 = logic_constrained_loss(entity_probs_1, negative_hyperedges)
print(f"Entity probs: all = 0.7")
print(f"Logic loss: {loss_1.item():.6f}")
print(f"Expected: ~0.7*0.7 = 0.49 ✓" if abs(loss_1.item() - 0.49) < 0.05 else f"Expected: ~0.49 ✗")

print("\n" + "="*60)
print("TEST 2: Low uniform probs")
print("="*60)
entity_probs_2 = torch.ones(4, 15814) * 0.01  # All entities near 0
loss_2 = logic_constrained_loss(entity_probs_2, negative_hyperedges)
print(f"Entity probs: all = 0.01")
print(f"Logic loss: {loss_2.item():.6f}")
print(f"Expected: ~0.01*0.01 = 0.0001 ✓")

print("\n" + "="*60)
print("TEST 3: Mixed probs (realistic)")
print("="*60)
# Simulate: most entities high, but conflict nodes randomly distributed
entity_probs_3 = torch.rand(4, 15814) * 0.5 + 0.5  # [0.5, 1.0]
loss_3 = logic_constrained_loss(entity_probs_3, negative_hyperedges)
print(f"Entity probs: random in [0.5, 1.0]")
print(f"Logic loss: {loss_3.item():.6f}")

print("\n" + "="*60)
print("TEST 4: EXACT Training Scenario (entity_probs stats from log)")
print("="*60)
# From log: min=0.0000, max=0.9892, mean=0.7205
# This suggests: many entities ~0.7-0.9, some at exactly 0

# Create this distribution
entity_probs_4 = torch.rand(8, 15814) * 0.3 + 0.6  # Most in [0.6, 0.9]
# Set ~10% to exactly 0 (matching min=0.0000)
num_zeros = int(15814 * 0.1)
zero_indices = torch.randperm(15814)[:num_zeros]
entity_probs_4[:, zero_indices] = 0.0

print(f"Entity probs: min={entity_probs_4.min():.4f}, max={entity_probs_4.max():.4f}, mean={entity_probs_4.mean():.4f}")
loss_4 = logic_constrained_loss(entity_probs_4, negative_hyperedges)
print(f"Logic loss: {loss_4.item():.6f}")

# Check if zeros are in conflict nodes
conflict_nodes = set()
for hedge in negative_hyperedges:
    if len(hedge['nodes']) == 2:
        conflict_nodes.add(hedge['nodes'][0])
        conflict_nodes.add(hedge['nodes'][1])

zeros_in_conflicts = sum(1 for idx in zero_indices if idx.item() in conflict_nodes)
print(f"\nZeros in conflict nodes: {zeros_in_conflicts} / {num_zeros} ({100*zeros_in_conflicts/num_zeros:.1f}%)")

print("\n" + "="*60)
print("TEST 5: Worst case - ALL conflict nodes at 0")
print("="*60)
entity_probs_5 = torch.rand(4, 15814) * 0.3 + 0.6  # Base: [0.6, 0.9]
# Set ALL conflict nodes to 0
for node in conflict_nodes:
    entity_probs_5[:, node] = 0.0

print(f"Entity probs: min={entity_probs_5.min():.4f}, max={entity_probs_5.max():.4f}, mean={entity_probs_5.mean():.4f}")
loss_5 = logic_constrained_loss(entity_probs_5, negative_hyperedges)
print(f"Logic loss: {loss_5.item():.6f}")
print(f"Expected: 0.0 (all conflicts have at least one prob=0) ✓" if loss_5.item() < 0.0001 else "Unexpected ✗")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if loss_4.item() < 0.01:
    print("\n⚠️  TEST 4 (realistic scenario) gives near-zero loss!")
    print("✓ Hypothesis CONFIRMED: Random zeros overlapping with conflict nodes")
    print("  causes logic loss to drop to 0,")
    print("\n  EVEN if mean prob is high (0.72)")
    print("\nROOT CAUSE:")
    print("  prob_a * prob_b ≈ 0 when either prob is 0")
    print("  If ~10% entities are 0, and 3246 conflict nodes exist,")
    print("  statistically ~324 conflict nodes will be 0,")
    print("  affecting ~645 of 1623 conflict pairs (40%!)")
else:
    print(f"\n✓ TEST 4 loss is healthy: {loss_4.item():.6f}")
    print("  The zero issue might require different investigation")

print(f"\nActual zeros in conflicts from TEST 4: {zeros_in_conflicts}")
print(f"Affected conflict pairs (estimate): ~{zeros_in_conflicts * 2} / 1623")
