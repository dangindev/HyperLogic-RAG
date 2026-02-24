import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import random
from tqdm import tqdm
from src.datasets import get_dataloader
from src.models.baseline_model import BaselineModel


def get_warmup_lr(step, warmup_steps, base_lr):
    """Linear warmup learning rate"""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr


def get_scheduled_sampling_ratio(epoch, config):
    """Get teacher forcing ratio for current epoch"""
    if not config['training'].get('scheduled_sampling', {}).get('enabled', False):
        return 1.0  # Always teacher forcing
    
    start_epoch = config['training']['scheduled_sampling'].get('start_epoch', 6)
    end_epoch = config['training']['scheduled_sampling'].get('end_epoch', 11)
    final_ratio = config['training']['scheduled_sampling'].get('final_ratio', 0.5)
    
    if epoch < start_epoch:
        return 1.0  # 100% teacher forcing
    elif epoch >= end_epoch:
        return final_ratio  # Final ratio
    else:
        # Linear decay
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return 1.0 - (1.0 - final_ratio) * progress


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: {config['experiment']['name']}\n")
    
    # Create output directory
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Dataloaders
    train_loader = get_dataloader(config, split='train')
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # Model
    model = BaselineModel(config).to(device)
    
    start_epoch = 0
    global_step = 0
    
    if 'resume_from' in config['training'] and config['training']['resume_from']:
        print(f"Resuming from checkpoint: {config['training']['resume_from']}")
        checkpoint = torch.load(config['training']['resume_from'], map_location=device)
        model.load_state_dict(checkpoint)
        try:
            start_epoch = int(config['training']['resume_from'].split('_')[-1].split('.')[0])
            print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not parse epoch, starting from 0")
    
    # Optimizer with warmup
    base_lr = float(config['training']['lr'])
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    warmup_steps = config['training'].get('warmup_steps', 0)
    
    # Loss with label smoothing
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,  # Pad token
        label_smoothing=label_smoothing
    )
    
    print(f"Training config:")
    print(f"  Base LR: {base_lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Gradient clip: {config['training'].get('gradient_clip', 1.0)}")
    print(f"  Scheduled sampling: {config['training'].get('scheduled_sampling', {}).get('enabled', False)}\n")
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['fp16'])
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0
        teacher_forcing_ratio = get_scheduled_sampling_ratio(epoch, config)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Teacher forcing ratio: {teacher_forcing_ratio:.2f}")
        print(f"{'='*60}")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            captions = batch['input_ids'].to(device)
            
            # Warmup learning rate
            if global_step < warmup_steps:
                current_lr = get_warmup_lr(global_step, warmup_steps, base_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config['training']['fp16']):
                # Shift for autoregressive training
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                # Scheduled sampling: randomly use model predictions instead of ground truth
                if teacher_forcing_ratio < 1.0 and random.random() > teacher_forcing_ratio:
                    # Use model's own predictions
                    with torch.no_grad():
                        outputs = model(images, input_captions)
                        sampled_tokens = outputs.argmax(dim=-1)
                        # Use sampled tokens as input (with teacher forcing for first token)
                        input_captions = torch.cat([
                            input_captions[:, :1],
                            sampled_tokens[:, :-1]
                        ], dim=1)
                
                outputs = model(images, input_captions)  # [B, SeqLen, Vocab]
                
                # Compute loss
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    target_captions.reshape(-1)
                )
            
            # Backward with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training'].get('gradient_clip', 1.0)
            )
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Global step: {global_step}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                config['experiment']['output_dir'],
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train(config)
