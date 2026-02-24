"""
Test script to verify gradient flow in Hyper-Logic RAG model
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.utils.hypergraph_utils import load_and_convert_hypergraph

print("="*60)
print("TESTING GRADIENT FLOW")
print("="*60)

# Load hypergraph
hypergraph_data = load_and_convert_hypergraph("data/hypergraph.pkl")

# Mock visual encoder
class MockVisualEncoder(nn.Module):
    def forward(self, x):
        B = x.size(0)
        return torch.randn(B, 2048, 7, 7)

visual_encoder = MockVisualEncoder()

# Initialize model
model = HyperLogicRAGModel(
    visual_encoder=visual_encoder,
    vocab_size=10000,
    num_entities=15814,
    embed_dim=512,
    num_heads=8,
    num_decoder_layers=3,
    hypergraph_data=hypergraph_data
)

print(f"\n✓ Model initialized")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check all modules registered
print("\n✓ Registered modules:")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

# Mock data
batch_size = 2
images = torch.randn(batch_size, 3, 224, 224)
target_tokens = torch.randint(1, 1000, (batch_size, 20))

print(f"\nForward pass...")
logits, entity_probs = model(images, target_tokens)

print(f"  Logits: {logits.shape}")
print(f"  Entity probs: {entity_probs.shape}")

# Compute loss
total_loss, gen_loss, logic_loss = model.compute_loss(
    logits, target_tokens, entity_probs, lambda_logic=0.1
)

print(f"\nLosses:")
print(f"  Total: {total_loss.item():.4f}")
print(f"  Generation: {gen_loss.item():.4f}")
print(f"  Logic: {logic_loss.item():.4f}")

# Backward pass
print(f"\nBackward pass...")
total_loss.backward()

# Check gradient norms for key modules
print(f"\n✓ Gradient norms:")
modules_to_check = [
    ('entity_predictor', model.entity_predictor),
    ('hypergcn', model.hypergcn),
    ('visual_proj', model.visual_proj),
    ('text_embedding', model.text_embedding),
    ('decoder', model.decoder),
    ('output_proj', model.output_proj)
]

all_grads_ok = True
for name, module in modules_to_check:
    total_norm = 0.0
    param_count = 0
    for p in module.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
            param_count += 1
    
    total_norm = total_norm ** 0.5
    
    if param_count > 0:
        print(f"  {name}: {total_norm:.4f} ({param_count} params with grad)")
        if total_norm < 1e-8:
            print(f"    ⚠️  WARNING: Very small gradient!")
            all_grads_ok = False
    else:
        print(f"  {name}: No gradients (frozen or no params)")

print("\n" + "="*60)
if all_grads_ok:
    print("✅ GRADIENT FLOW TEST PASSED!")
    print("All modules have reasonable gradients.")
else:
    print("⚠️  Some modules have very small gradients")
    print("   This might be OK depending on the module")
print("="*60)
