"""
Training script for Hyper-Logic RAG Model with BIOMEDCLIP Backbone
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import argparse
from tqdm import tqdm
from transformers import AutoModel

from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.utils.hypergraph_utils import load_and_convert_hypergraph
from src.datasets import get_dataloader, get_shared_tokenizer
from src.models.biomed_clip_encoder import BiomedCLIPEncoder
from src.utils.metrics import compute_scores

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def train_epoch(model, dataloader, optimizer, device, lambda_logic=0.1, gradient_clip=1.0, current_epoch=0, label_smoothing=0.0):
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
        batch_loss, gen_loss, logic_loss = model.compute_loss(
            logits, targets, entity_probs, lambda_logic=lambda_logic, current_epoch=current_epoch,
            label_smoothing=label_smoothing
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
        
        # Update progress bar
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
    """Validate model with Loss and optionally Full Metrics"""
    model.eval()
    total_loss = 0
    gen_loss_sum = 0
    logic_loss_sum = 0
    
    # Store candidates and references for METEOR/BLEU
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
            
            # Forward pass for loss calculation (teacher forcing)
            logits, entity_probs = model(images, reports[:, :-1], retrieval_ids=context_ids)
            targets = reports[:, 1:]
            batch_loss, gen_loss, logic_loss = model.compute_loss(
                logits, targets, entity_probs, lambda_logic=lambda_logic
            )
            
            total_loss += batch_loss.item()
            gen_loss_sum += gen_loss.item()
            logic_loss_sum += logic_loss.item()

            # 2. Generate Reports (Greedy/Beam) for Metrics
            # Only generate if tokenizer is provided (Full Evaluation)
            if tokenizer is not None:
                # Generate
                # Pass retrieval_ids to generate if available
                generated_ids = model.generate(images, max_length=100, num_beams=3, retrieval_ids=context_ids)
                
                # Decode
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)
                
                for i in range(len(generated_texts)):
                    key = str(len(res)) # Unique ID based on count
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
    
    if tokenizer is not None and len(res) > 0:
        print(f"Computing NLP metrics for {len(res)} samples...")
        scores = compute_scores(gts, res)
        # scores is dict e.g. {'BLEU_1': 0.5, ...}
        metrics.update(scores)
    else:
        # Fill with 0
        metrics.update({'BLEU_1': 0.0, 'BLEU_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0})
    
    return metrics

def load_transfer_weights(model, checkpoint_path):
    """
    Load weights from a checkpoint, ignoring layers with shape mismatches.
    Useful for transfer learning between datasets with different vocab sizes.
    """
    print(f"\nLoading transfer weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    model_state = model.state_dict()
    transfer_state = {}
    ignored_keys = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                transfer_state[k] = v
            else:
                ignored_keys.append(f"{k} (shape {v.shape} vs {model_state[k].shape})")
        else:
            ignored_keys.append(f"{k} (not in model)")
            
    # Load compatible weights
    model.load_state_dict(transfer_state, strict=False)
    
    print(f"✓ Transferred {len(transfer_state)}/{len(model_state)} layers")
    if ignored_keys:
        print(f"⚠️  Ignored {len(ignored_keys)} layers due to mismatch (e.g., vocab size):")
        for k in ignored_keys[:5]:
            print(f"   - {k}")
        if len(ignored_keys) > 5:
            print(f"   ... and {len(ignored_keys)-5} more")

def main(config_path, resume_from=None, transfer_from=None):
    """Main training loop"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = config.get('output', {}).get('output_dir', "results/hyperlogic_biomedclip")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load hypergraph
    print("\nLoading hypergraph...")
    hypergraph_path = config['dataset']['hypergraph_path'] # Ensure this is absolute in yaml or use default
    if not os.path.isabs(hypergraph_path):
        hypergraph_path = os.path.join(os.getcwd(), hypergraph_path) # Fallback
    
    hypergraph_data = load_and_convert_hypergraph(hypergraph_path)
    
    # Create dataloaders (this initializes R2Gen tokenizer)
    print("\nCreating dataloaders...")
    # Resolve data_root relative to the project root
    config['dataset']['data_root'] = os.path.join(project_root, 'data')
    train_loader = get_dataloader(config, 'train')
    val_split = config['dataset'].get('val_split', 'validate')
    val_loader = get_dataloader(config, val_split)  # Use configured split (e.g., 'test' for IU X-Ray)
    
    # Get vocab_size from R2Gen tokenizer
    vocab_size = train_loader.dataset.tokenizer.get_vocab_size()
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Tokenizer vocab size: {vocab_size}")
    
    # Create visual encoder (BiomedCLIP)
    print("\nCreating visual encoder (BiomedCLIP)...")
    freeze_backbone = config.get('model', {}).get('visual_encoder', {}).get('freeze_backbone', True)
    print(f"Backbone Freeze Status: {freeze_backbone}")
    visual_encoder = BiomedCLIPEncoder(pretrained=True, freeze=freeze_backbone)
    
    # Create model with dynamic vocab_size
    print("\nInitializing model...")
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=vocab_size,  # From R2Gen tokenizer
        num_entities=hypergraph_data['num_nodes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        hypergraph_data=hypergraph_data,
        use_relational_memory=config['model'].get('use_relational_memory', True),
        use_mcln=config['model'].get('use_mcln', False),
        rm_num_slots=config['model'].get('rm_num_slots', 3),
        rm_d_model=config['model'].get('rm_d_model', 512),
        dropout=config['model'].get('dropout', 0.1)
    )
    
    model = model.to(device)
    
    # Initialize node embeddings from entity names (Grounding Logic)
    if 'id_to_node' in hypergraph_data:
        model.init_node_embeddings_with_names(
            hypergraph_data['id_to_node'], 
            train_loader.dataset.tokenizer, 
            device
        )
            
    # Optimizer
    if config['training'].get('backbone_lr'):
        backbone_lr = float(config['training']['backbone_lr'])
        base_lr = float(config['training']['learning_rate'])
        print(f"✓ Using differential learning rates: Backbone={backbone_lr}, Decoder/Heads={base_lr}")
        
        # Identify backbone parameters (visual_encoder)
        backbone_params = list(model.visual_encoder.parameters())
        backbone_ids = set(map(id, backbone_params))
        
        # Filter rest of parameters
        rest_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        
        params = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': rest_params, 'lr': base_lr}
        ]
        
        optimizer = AdamW(
            params,
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # LR Schedule: Linear Warmup + Cosine Decay
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    total_epochs = config['training']['epochs']
    
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        print(f"✓ LR Schedule: Warmup {warmup_epochs} epochs → Cosine decay")
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6
        )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}...")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    elif transfer_from and os.path.exists(transfer_from):
        # Transfer Learning (load weights but start fresh training)
        load_transfer_weights(model, transfer_from)
    trainable, total = count_parameters(model)
    print(f"✓ Model created")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    
    # Get tokenizer reference for validation
    tokenizer = train_loader.dataset.tokenizer

    # Training loop
    print("\nSTARTING TRAINING (BIOMEDCLIP + HYPERLOGIC + R2GEN TOKENIZER)")
    
    lambda_logic_target = config['training'].get('lambda_logic', 0.1)
    gradient_clip = config['training'].get('gradient_clip', 1.0)
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    early_stopping_patience = config['training'].get('early_stopping_patience', 0)
    no_improve_count = 0
    
    # Logic Loss Warm-up Schedule (same as HL v2)
    def get_lambda_logic(epoch, target_lambda=0.1, warmup_epochs=20, ramp_epochs=10):
        if epoch < warmup_epochs:
            return 0.0
        elif epoch < warmup_epochs + ramp_epochs:
            progress = (epoch - warmup_epochs) / ramp_epochs
            return target_lambda * progress
        else:
            return target_lambda
    
    print(f"Logic Loss Config: target={lambda_logic_target}, warmup=20 epochs, ramp=10 epochs")
    if label_smoothing > 0:
        print(f"Label Smoothing: {label_smoothing}")
    if early_stopping_patience > 0:
        print(f"Early Stopping: patience={early_stopping_patience}")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Get lambda_logic for this epoch (warm-up schedule)
        lambda_logic = get_lambda_logic(epoch, target_lambda=lambda_logic_target)
        
        train_loss, train_gen, train_logic = train_epoch(
            model, train_loader, optimizer, device,
            lambda_logic=lambda_logic,
            gradient_clip=gradient_clip,
            current_epoch=epoch,
            label_smoothing=label_smoothing
        )
        print(f"Train - Loss: {train_loss:.4f} | Gen: {train_gen:.4f} | Logic: {train_logic:.4f} | λ_logic: {lambda_logic:.4f}")
        
        if len(val_loader) > 0:
            # Validate (skip if no validation data or not at validation period)
            # Validate if:
            # 1. It's the last epoch
            # 2. OR (epoch + 1) modulo val_every is 0
            val_every = config['training'].get('val_every', 1)
            should_validate = ((epoch + 1) == config['training']['epochs']) or ((epoch + 1) % val_every == 0)

            if should_validate:
                metrics = validate(
                    model, val_loader, device, lambda_logic=lambda_logic, tokenizer=tokenizer
                )
                print(f"Val   - Loss: {metrics['loss']:.4f} | Gen: {metrics['gen_loss']:.4f} | Logic: {metrics['logic_loss']:.4f}")
                print(f"        BLEU-1: {metrics.get('BLEU_1', 0):.4f} | BLEU-2: {metrics.get('BLEU_2', 0):.4f} | BLEU-3: {metrics.get('BLEU_3', 0):.4f} | BLEU-4: {metrics.get('BLEU_4', 0):.4f} | METEOR: {metrics.get('METEOR', 0):.4f} | ROUGE-L: {metrics.get('ROUGE_L', 0):.4f}")
                val_loss = metrics['loss']
            else:
                print(f"Val   - Skipped (next validation at epoch {(epoch + 1) + (val_every - (epoch + 1) % val_every)})")
                val_loss = train_loss # Use train loss
        else:
            val_loss = train_loss
            
        scheduler.step()
        
        # Save
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        # Save latst
        torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))

        # Save regular checkpoint
        if (epoch + 1) % config['output']['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

        
        if len(val_loader) > 0 and should_validate:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
                print(f"✓ New best model! Val loss: {val_loss:.4f}")
            else:
                no_improve_count += 1
                if early_stopping_patience > 0:
                    print(f"⚠️  No improvement for {no_improve_count}/{early_stopping_patience} validations")
                    if no_improve_count >= early_stopping_patience:
                        print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hyperlogic_config_v2_full.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--transfer_from', type=str, default=None, help="Path to checkpoint to transfer weights from")
    args = parser.parse_args()
    main(args.config, args.resume, args.transfer_from)
