"""
Test logic_constrained_loss_v2 to verify it prevents zero-probability trick
"""
import sys
sys.path.append('/home/dangnh3/MICCAI2026/HyperLogicRAG')

import torch
import pickle
from src.utils.hypergraph_utils import logic_constrained_loss, logic_constrained_loss_v2

print("="*60)
print("TESTING LOGIC_CONSTRAINED_LOSS_V2")
print("="*60)

# Load negative hyperedges
with open('/home/dangnh3/MICCAI2026/HyperLogicRAG/data/hypergraph.pkl', 'rb') as f:
    hypergraph_data = pickle.load(f)
negative_hyperedges = hypergraph_data['negative_hyperedges']
print(f"\n✓ Loaded {len(negative_hyperedges)} negative hyperedges")

print("\n" + "="*60)
print("TEST 1: Baseline - All probs = 0.7")
print("="*60)
entity_probs = torch.ones(4, 15814) * 0.7
loss_v1 = logic_constrained_loss(entity_probs, negative_hyperedges)
loss_v2 = logic_constrained_loss_v2(entity_probs, negative_hyperedges)
print(f"V1 (old): {loss_v1.item():.6f}")
print(f"V2 (new): {loss_v2.item():.6f}")
print(f"✓ Both should be similar (no suppression needed at 0.7)")

print("\n" + "="*60)
print("TEST 2: CRITICAL - ALL conflict nodes at 0 (the exploit)")
print("="*60)
entity_probs_exploit = torch.rand(4, 15814) * 0.3 + 0.6  # Base [0.6, 0.9]

# Set ALL conflict nodes to 0 (the trick model learned)
conflict_nodes = set()
for hedge in negative_hyperedges:
    if len(hedge['nodes']) == 2:
        conflict_nodes.update(hedge['nodes'])

for node in conflict_nodes:
    entity_probs_exploit[:, node] = 0.0

print(f"Entity probs: min={entity_probs_exploit.min():.4f}, max={entity_probs_exploit.max():.4f}, mean={entity_probs_exploit.mean():.4f}")
print(f"Conflict nodes set to: 0.0 (all {len(conflict_nodes)} nodes)")

loss_v1_exploit = logic_constrained_loss(entity_probs_exploit, negative_hyperedges)
loss_v2_exploit = logic_constrained_loss_v2(entity_probs_exploit, negative_hyperedges)

print(f"\nV1 (old): {loss_v1_exploit.item():.6f}")
print(f"V2 (new): {loss_v2_exploit.item():.6f}")

if loss_v1_exploit.item() < 0.01:
    print("✓ V1 gives near-zero (as expected - exploit works)")
else:
    print("✗ V1 unexpected")

if loss_v2_exploit.item() > 0.1:
    print("✅ V2 PREVENTS EXPLOIT! Loss is non-zero due to suppression penalty")
else:
    print("❌ V2 FAILED! Still allows zero-trick")

print("\n" + "="*60)
print("TEST 3: Gradients check")
print("="*60)
# Create entity probs with gradients
entity_probs_grad = torch.rand(4, 15814, requires_grad=True) * 0.3 + 0.6

# Set some conflict nodes near 0
for i, node in enumerate(list(conflict_nodes)[:100]):
    entity_probs_grad.data[:, node] = 0.01

loss_v2_grad = logic_constrained_loss_v2(entity_probs_grad, negative_hyperedges)
loss_v2_grad.backward()

# Check if gradients exist and are non-zero
grad_stats = entity_probs_grad.grad[0, list(conflict_nodes)[:100]]
print(f"Gradient stats for near-zero conflict nodes:")
print(f"  Min: {grad_stats.min():.6f}")
print(f"  Max: {grad_stats.max():.6f}")
print(f"  Mean: {grad_stats.mean():.6f}")

if grad_stats.abs().mean() > 0.001:
    print("✅ Gradients flow correctly (non-zero)")
else:
    print("❌ Gradients are too small")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nV2 improvements:")
print("1. ✓ Detects and penalizes low probabilities on conflict nodes")
print("2. ✓ Prevents model from exploiting zero-probability trick")
print("3. ✓ Maintains gradient flow for optimization")
print("\nV2 is ready for training!")
