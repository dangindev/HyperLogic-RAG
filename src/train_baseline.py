import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from src.datasets import get_dataloader
from src.models.baseline_model import BaselineModel

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Dataloaders
    train_loader = get_dataloader(config, split='train')
    # val_loader = get_dataloader(config, split='validate') # Uncomment when real splits available
    
    # Model
    model = BaselineModel(config).to(device)
    
    start_epoch = 0
    if 'resume_from' in config['training'] and config['training']['resume_from']:
        print(f"Resuming from checkpoint: {config['training']['resume_from']}")
        checkpoint = torch.load(config['training']['resume_from'], map_location=device)
        model.load_state_dict(checkpoint)
        # Assuming checkpoint filename format "checkpoint_epoch_N.pth"
        try:
            start_epoch = int(config['training']['resume_from'].split('_')[-1].split('.')[0])
            print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not parse epoch from filename, starting loop but model weights loaded.")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['lr']))
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is pad
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['fp16'])
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            captions = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config['training']['fp16']):
                # Standard causal training: Input is caption[:, :-1], Target is caption[:, 1:]
                # Note: Model forward needs to handle shifting or we do it here
                # Here we pass full captions, let decoder handle masking or teacher forcing?
                # For simplicity in this baseline script, let's assume decoder takes `captions` 
                # and internal implementation does the right shifting or we shift here.
                
                # Let's shift here for clarity
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                outputs = model(images, input_captions) # [B, SeqLen, Vocab]
                
                # Reshape for loss
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_captions.reshape(-1))
            
            # Scaler logic with gradient clipping
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step and update
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
