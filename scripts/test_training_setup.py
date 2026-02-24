"""
Quick test: Load 1 batch and run forward/backward pass
"""
import sys
sys.path.append('/home/dangnh3/MICCAI2026/HyperLogicRAG')

import torch
import yaml

print("Testing training setup...")

# Load config
with open('configs/hyperlogic_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("✓ Config loaded")

# Test dataloader
from src.datasets import get_dataloader

print("\nTesting dataloader...")
try:
    train_loader = get_dataloader(config, 'train')
    print(f"✓ Train loader created: {len(train_loader)} batches")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"✓ Batch loaded:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Tokens shape: {batch['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ Dataloader error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model creation
print("\nTesting model...")
try:
    from src.models.hyperlogic_rag import HyperLogicRAGModel
    from src.utils.hypergraph_utils import load_and_convert_hypergraph
    from src.models.simple_cnn import SimpleCNNEncoder
    
    # Load hypergraph
    hypergraph_data = load_and_convert_hypergraph('data/hypergraph.pkl')
    print(f"✓ Hypergraph loaded: {hypergraph_data['num_nodes']} nodes")
    
    # Visual encoder (SimpleCNN instead of ResNet)
    visual_encoder = SimpleCNNEncoder()
    print("✓ Visual encoder created (SimpleCNN)")
    
    # Model
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=config['model']['vocab_size'],
        num_entities=hypergraph_data['num_nodes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        hypergraph_data=hypergraph_data
    )
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
print("\nTesting forward pass...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    images = batch['image'].to(device)
    reports = batch['input_ids'].to(device)
    
    logits, entity_probs = model(images, reports[:, :-1])
    print(f"✓ Forward pass OK:")
    print(f"  Logits: {logits.shape}")
    print(f"  Entity probs: {entity_probs.shape}")
    
except Exception as e:
    print(f"❌ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loss computation
print("\nTesting loss...")
try:
    targets = reports[:, 1:]
    total_loss, gen_loss, logic_loss = model.compute_loss(
        logits, targets, entity_probs, lambda_logic=0.1
    )
    print(f"✓ Loss computation OK:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Generation: {gen_loss.item():.4f}")
    print(f"  Logic: {logic_loss.item():.4f}")
    
except Exception as e:
    print(f"❌ Loss error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test backward pass
print("\nTesting backward pass...")
try:
    total_loss.backward()
    print("✓ Backward pass OK")
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"✓ Gradients computed for {len(grad_norms)} parameters")
    
    # Show a few
    print("\nSample gradient norms:")
    for i, (name, norm) in enumerate(list(grad_norms.items())[:5]):
        print(f"  {name}: {norm:.4f}")
    
except Exception as e:
    print(f"❌ Backward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n🚀 Ready to submit SLURM job!")
