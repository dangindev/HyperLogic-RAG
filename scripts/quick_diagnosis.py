"""
Simple test: Load checkpoint and inspect entity predictor weights
"""
import torch
import pickle
import numpy as np

print("="*60)
print("QUICK DIAGNOSIS - Entity Predictor Analysis")
print("="*60)

# Load checkpoint
print("\n[1] Loading checkpoint...")
checkpoint_path = '/home/dangnh3/MICCAI2026/HyperLogicRAG/results/hyperlogic_rag/checkpoint_epoch_5.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

# Extract entity predictor weights
print("\n[2] Analyzing entity predictor...")
state_dict = checkpoint['model_state_dict']

# Find entity predictor layers
entity_keys = [k for k in state_dict.keys() if 'entity_predictor' in k]
print(f"Entity predictor layers: {len(entity_keys)}")
for key in entity_keys:
    print(f"  {key}: {state_dict[key].shape}")

# Check the final layer bias (important for sigmoid output)
final_layer_bias_key = 'entity_predictor.mlp.3.bias'
if final_layer_bias_key in state_dict:
    bias = state_dict[final_layer_bias_key]
    print(f"\n[3] Final layer bias analysis:")
    print(f"  Shape: {bias.shape}")
    print(f"  Stats: min={bias.min():.4f}, max={bias.max():.4f}, mean={bias.mean():.4f}")
    print(f"  Sigmoid(bias) stats: min={torch.sigmoid(bias).min():.4f}, max={torch.sigmoid(bias).max():.4f}, mean={torch.sigmoid(bias).mean():.4f}")
    
    # Check distribution
    hist, bins = np.histogram(bias.numpy(), bins=10)
    print(f"\n  Bias distribution:")
    for i in range(len(hist)):
        print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d}")
    
    # If bias is very negative, sigmoid will be near 0
    very_negative = (bias < -5).sum().item()
    print(f"\n  Entities with bias < -5: {very_negative} / {len(bias)} ({100*very_negative/len(bias):.1f}%)")
    print(f"  → These will have prob < {torch.sigmoid(torch.tensor(-5.0)):.4f} regardless of input")

# Load hypergraph to check conflicts
print("\n[4] Loading hypergraph...")
with open('/home/dangnh3/MICCAI2026/HyperLogicRAG/data/hypergraph.pkl', 'rb') as f:
    hypergraph_data = pickle.load(f)

negative_hyperedges = hypergraph_data['negative_hyperedges']
print(f"✓ {len(negative_hyperedges)} negative hyperedges")

# Check if conflict nodes have low biases
if final_layer_bias_key in state_dict:
    bias = state_dict[final_layer_bias_key]
    conflict_nodes = set()
    for hedge in negative_hyperedges:
        if len(hedge['nodes']) == 2:
            conflict_nodes.update(hedge['nodes'])
    
    print(f"\n[5] Conflict nodes bias analysis:")
    print(f"  Unique conflict nodes: {len(conflict_nodes)}")
    
    conflict_biases = [bias[node].item() for node in conflict_nodes]
    non_conflict_biases = [bias[i].item() for i in range(len(bias)) if i not in conflict_nodes]
    
    print(f"\n  Conflict nodes bias:")
    print(f"    Mean: {np.mean(conflict_biases):.4f}")
    print(f"    Std: {np.std(conflict_biases):.4f}")
    print(f"    % < -5: {100 * sum(1 for b in conflict_biases if b < -5) / len(conflict_biases):.1f}%")
    
    print(f"\n  Non-conflict nodes bias:")
    print(f"    Mean: {np.mean(non_conflict_biases):.4f}")
    print(f"    Std: {np.std(non_conflict_biases):.4f}")
    print(f"    % < -5: {100 * sum(1 for b in non_conflict_biases if b < -5) / len(non_conflict_biases):.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
If conflict nodes have significantly MORE negative biases than others:
  → Model learned to SUPPRESS conflict nodes to minimize logic loss
  → This is "gradient hacking" - valid optimization but not desired behavior
  
If ALL nodes have very negative biases:
  → Model collapsed to predict nothing (extreme case)
  
If biases are similar:
  → The issue might be in forward pass or data, not the weights
""")
