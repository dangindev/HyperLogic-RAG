"""
Training script for Hyper-Logic RAG Model with PRETRAINED components
Upgraded version with:
- ResNet-50 pretrained on ImageNet (transformers)
- BERT pretrained embeddings
"""
import sys
import os
# Add the directory containing 'src' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# If running from src, go up one level
if current_dir.endswith('src'):
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
from tqdm import tqdm
from transformers import AutoModel

from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.utils.hypergraph_utils import load_and_convert_hypergraph
from src.datasets import get_dataloader, get_shared_tokenizer
from src.utils.metrics import compute_scores

def create_visual_encoder(config):
    """
    Create visual encoder based on config
    Supports: 'densenet121' (CheXpert/ImageNet) or 'resnet50' (ImageNet)
    """
    model_config = config['model']
    encoder_config = model_config.get('visual_encoder', {})
    encoder_type = encoder_config.get('type', 'resnet50')
    output_dim = model_config.get('embed_dim', 512) # HyperLogicRAG expects this dim
    
    print(f"Initializing Visual Encoder: {encoder_type} (output_dim={output_dim})...")
    
    if encoder_type == 'densenet121':
        try:
            from src.models.densenet_encoder import DenseNet121Encoder
            # DenseNet121Encoder handles projection internally if output_dim != 1024
            encoder = DenseNet121Encoder(
                pretrained=encoder_config.get('pretrained_source', 'chexpert'),
                output_dim=output_dim
            )
            return encoder
        except ImportError as e:
            print(f"⚠️  Import Error: {e}")
            # Fallback
            pass
            
    # Fallback / Default to ResNet-50 (ImageNet)
    print("Loading pretrained ResNet-50 from AutoModel...")
    
    try:
        from transformers import AutoModel
        resnet = AutoModel.from_pretrained("microsoft/resnet-50")
        
        # Extract features (like Baseline model does)
        class ResNetFeatureExtractor(nn.Module):
            def __init__(self, resnet_model, project_dim=None):
                super().__init__()
                self.resnet = resnet_model
                self.project_dim = project_dim
                if project_dim and project_dim != 2048:
                    self.proj = nn.Linear(2048, project_dim)
                else:
                    self.proj = None
                
            def forward(self, x):
                outputs = self.resnet(x)
                # Get spatial features [B, 2048, 7, 7]
                features = outputs.last_hidden_state
                
                # If we need to project, we must flatten first? 
                # NO. HyperLogicRAGModel expects spatial [B, C, H, W] OR flat [B, C]
                # If we return [B, 2048, 7, 7], the model does:
                # view(B, 2048, 49).permute -> [B, 49, 2048] -> proj(2048->512)
                # SO: We should rely on HyperLogicRAGModel's visual_proj
                # BUT: HyperLogicRAGModel's proj is Linear(512, embed_dim).
                # ResNet outputs 2048. 2048 != 512.
                # FIX: We MUST project here if using ResNet
                if self.proj:
                    # [B, 2048, 7, 7] -> [B, 7, 7, 2048] -> proj -> [B, 7, 7, 512] -> [B, 512, 7, 7]
                    features = features.permute(0, 2, 3, 1)
                    features = self.proj(features)
                    features = features.permute(0, 3, 1, 2)
                    
                return features
        
        # Fix ResNet output dim to match what model expects (512 usually, or embed_dim)
        visual_encoder = ResNetFeatureExtractor(resnet, project_dim=output_dim)
        
        freeze_early = encoder_config.get('freeze_backbone', True)
        if freeze_early:
            # Freeze early layers
            params = list(resnet.parameters())
            for param in params[:len(params)//2]:
                param.requires_grad = False
            print("✓ Froze early layers (first 50% of parameters)")
        
        print(f"✓ Loaded pretrained ResNet-50 successfully")
        return visual_encoder
        
    except Exception as e:
        print(f"⚠️  Failed to load pretrained ResNet: {e}")
        # Fallback to SimpleCNN...
        from src.models.simple_cnn import SimpleCNNEncoder
        return SimpleCNNEncoder()

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def train_epoch(model, dataloader, optimizer, device, lambda_logic=0.1, gradient_clip=1.0, current_epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    gen_loss_sum = 0
    logic_loss_sum = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        reports = batch['input_ids'].to(device)  # Tokenized reports [B, T]
        
        # Forward pass
        # Check for retrieval context
        context_ids = batch.get('context_ids')
        if context_ids is not None:
            context_ids = context_ids.to(device)
            
        logits, entity_probs = model(images, reports[:, :-1], retrieval_ids=context_ids)  # Teacher forcing
        
        # Compute loss
        targets = reports[:, 1:]  # Shift for next-token prediction
        # Pass current epoch for logic loss warm-up (detach first 5 epochs)
        batch_loss, gen_loss, logic_loss = model.compute_loss(
            logits, targets, entity_probs, lambda_logic=lambda_logic, current_epoch=current_epoch
        )
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track losses
        total_loss += batch_loss.item()
        gen_loss_sum += gen_loss.item()
        logic_loss_sum += logic_loss.item()
        
        # Update progress bar (use .4f for logic to avoid rounding small values to 0.000)
        pbar.set_postfix({
            'loss': f'{batch_loss.item():.3f}',
            'gen': f'{gen_loss.item():.3f}',
            'logic': f'{logic_loss.item():.4f}'
        })
    
    avg_total = total_loss / len(dataloader)
    avg_gen = gen_loss_sum / len(dataloader)
    avg_logic = logic_loss_sum / len(dataloader)
    
    return avg_total, avg_gen, avg_logic

def validate(model, dataloader, device, lambda_logic=0.1, tokenizer=None):
    """Validate model with Loss and optionally Full BLEU Metrics"""
    model.eval()
    total_loss = 0
    gen_loss_sum = 0
    logic_loss_sum = 0
    
    # Store candidates and references for NLP metrics
    res = {}
    gts = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            images = batch['image'].to(device)
            reports = batch['input_ids'].to(device)
            
            # Check for retrieval context
            context_ids = batch.get('context_ids')
            if context_ids is not None:
                context_ids = context_ids.to(device)
            
            # 1. Compute Loss (Teacher Forcing)
            logits, entity_probs = model(images, reports[:, :-1], retrieval_ids=context_ids)
            targets = reports[:, 1:]
            batch_loss, gen_loss, logic_loss = model.compute_loss(
                logits, targets, entity_probs, lambda_logic=lambda_logic
            )
            
            total_loss += batch_loss.item()
            gen_loss_sum += gen_loss.item()
            logic_loss_sum += logic_loss.item()
            
            # 2. Generate Reports for BLEU Metrics
            if tokenizer is not None:
                # Pass retrieval_ids to generate if available
                # Use tokenizer to decode, skip special tokens
                generated_ids = model.generate(images, max_length=100, num_beams=3, retrieval_ids=context_ids)
                
                # Decode generated and target texts
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)
                
                for i in range(len(generated_texts)):
                    key = str(len(res))
                    res[key] = [generated_texts[i]]
                    gts[key] = [target_texts[i]]
    
    avg_total = total_loss / len(dataloader)
    avg_gen = gen_loss_sum / len(dataloader)
    avg_logic = logic_loss_sum / len(dataloader)
    
    metrics = {
        'loss': avg_total,
        'gen_loss': avg_gen,
        'logic_loss': avg_logic
    }
    
    # Compute NLP metrics if tokenizer provided
    if tokenizer is not None and len(res) > 0:
        print(f"Computing NLP metrics for {len(res)} samples...")
        scores = compute_scores(gts, res)
        metrics.update(scores)
    else:
        metrics.update({'BLEU_1': 0.0, 'BLEU_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0})
    
    return metrics

def main(config_path, resume_from=None):
    """Main training loop"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load hypergraph
    print("\nLoading hypergraph...")
    hypergraph_path = config['dataset']['hypergraph_path']
    hypergraph_data = load_and_convert_hypergraph(hypergraph_path)
    
    # Create dataloaders (this initializes the tokenizer)
    print("\nCreating dataloaders...")
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'validate')  # Manifest uses 'validate' not 'val'
    
    # Get vocab_size from tokenizer (R2Gen word-level tokenizer)
    vocab_size = train_loader.dataset.tokenizer.get_vocab_size()
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Tokenizer vocab size: {vocab_size}")
    
    # Create visual encoder based on config
    print("\nCreating visual encoder...")
    # Fix: imports are now inside function to avoid circles, but pass config
    visual_encoder = create_visual_encoder(config)
    
    # Create model with dynamic vocab_size from tokenizer
    print("\nInitializing model...")
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=vocab_size,  # From tokenizer
        num_entities=hypergraph_data['num_nodes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        hypergraph_data=hypergraph_data,
        dropout=config['model'].get('dropout', 0.1) # Pass dropout from config
    )
    
    model = model.to(device)
    
    # Initialize node embeddings from entity names (Grounding Logic)
    if 'id_to_node' in hypergraph_data:
        model.init_node_embeddings_with_names(
            hypergraph_data['id_to_node'], 
            train_loader.dataset.tokenizer, 
            device
        )
    
    # Load Pretrained Model (e.g., from MIMIC) if specified in config
    pretrained_model = config['model'].get('pretrained_model')
    if pretrained_model and not resume_from:
        if os.path.exists(pretrained_model):
            print(f"\nLoading pretrained weights from {pretrained_model}...")
            try:
                # Load with strict=False to allow for mismatches (e.g., new node embeddings)
                pretrained_dict = torch.load(pretrained_model, map_location=device)
                
                # If saved via train_hyperlogic_rag.py, state_dict is inside 'model_state_dict'
                if 'model_state_dict' in pretrained_dict:
                    pretrained_state = pretrained_dict['model_state_dict']
                elif 'state_dict' in pretrained_dict:
                    pretrained_state = pretrained_dict['state_dict']
                else:
                    pretrained_state = pretrained_dict
                    
                msg = model.load_state_dict(pretrained_state, strict=False)
                print(f"✓ Loaded pretrained model weights successfully")
                print(f"  Missing keys: {len(msg.missing_keys)}")
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained model: {e}")
        else:
            print(f"\n⚠️ Pretrained model path not found: {pretrained_model}")
    
    trainable, total = count_parameters(model)
    print(f"✓ Model created")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Hypergraph: {hypergraph_data['num_nodes']} nodes, "
          f"{len(hypergraph_data['negative_hyperedges'])} conflicts")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}...")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING (PRETRAINED VERSION)")
    print("="*60)
    
    lambda_logic_target = config['training'].get('lambda_logic', 0.1)
    gradient_clip = config['training'].get('gradient_clip', 1.0)
    
    # Logic Loss Warm-up Schedule
    # Epoch 1-20: lambda_logic = 0.0 (decoder learns fluency first)
    # Epoch 21-30: gradual increase
    # Epoch 31+: full lambda_logic
    def get_lambda_logic(epoch, target_lambda=0.1, warmup_epochs=20, ramp_epochs=10):
        """
        Warm-up schedule for logic loss.
        - epoch < warmup_epochs: return 0.0 (pure generation training)
        - epoch < warmup_epochs + ramp_epochs: linear ramp
        - otherwise: target_lambda
        """
        if epoch < warmup_epochs:
            return 0.0
        elif epoch < warmup_epochs + ramp_epochs:
            progress = (epoch - warmup_epochs) / ramp_epochs
            return target_lambda * progress
        else:
            return target_lambda
    
    print(f"Logic Loss Config: target={lambda_logic_target}, warmup=20 epochs, ramp=10 epochs")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Get lambda_logic for this epoch (warm-up schedule)
        lambda_logic = get_lambda_logic(epoch, target_lambda=lambda_logic_target)
        
        # Train
        train_loss, train_gen, train_logic = train_epoch(
            model, train_loader, optimizer, device,
            lambda_logic=lambda_logic,
            gradient_clip=gradient_clip,
            current_epoch=epoch
        )
        
        print(f"\nTrain - Loss: {train_loss:.4f} | Gen: {train_gen:.4f} | Logic: {train_logic:.4f} | λ_logic: {lambda_logic:.4f}")
        
        # Validate (skip if no validation data or not at validation period)
        # Validate if:
        # 1. It's the last epoch
        # 2. OR (epoch + 1) modulo val_every is 0
        val_every = config['training'].get('val_every', 1)
        should_validate = ((epoch + 1) == config['training']['epochs']) or ((epoch + 1) % val_every == 0)

        if len(val_loader) > 0 and should_validate:
            # Get tokenizer reference for BLEU calculation
            tokenizer = train_loader.dataset.tokenizer
            metrics = validate(
                model, val_loader, device, lambda_logic=lambda_logic, tokenizer=tokenizer
            )
            val_loss = metrics['loss']
            print(f"Val   - Loss: {val_loss:.4f} | Gen: {metrics['gen_loss']:.4f} | Logic: {metrics['logic_loss']:.4f}")
            print(f"        BLEU-1: {metrics.get('BLEU_1', 0):.4f} | BLEU-4: {metrics.get('BLEU_4', 0):.4f} | METEOR: {metrics.get('METEOR', 0):.4f} | ROUGE-L: {metrics.get('ROUGE_L', 0):.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"✓ New best model! Val loss: {val_loss:.4f}")
        else:
            if not should_validate:
                 print(f"Val   - Skipped (next validation at epoch {(epoch + 1) + (val_every - (epoch + 1) % val_every)})")
            else:
                 print("Val   - Skipped (no validation data)")
            val_loss = train_loss  # Use train loss for checkpoint selection
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        

    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hyperlogic_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    main(args.config, args.resume)
