import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from src.datasets import get_dataloader
# Import RAG Model
from src.models.rag_model import RAGModel

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Dataloaders - get_dataloader now handles 'rag' type
    train_loader = get_dataloader(config, split='train')
    
    # Model - Switch to RAGModel
    model = RAGModel(config).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['lr']))
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    # FP16 (Disabled for stability for now as per Baseline 1 lessons)
    use_fp16 = config['training'].get('fp16', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            # Targets
            captions = batch['input_ids'].to(device)
            
            # RAG Context
            # Note: The dataset returns 'context_ids' and 'context_mask' for RAG
            context_ids = batch.get('context_ids')
            context_mask = batch.get('context_mask')
            
            if context_ids is not None:
                context_ids = context_ids.to(device)
                context_mask = context_mask.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                # Standard causal training: Input is caption[:, :-1], Target is caption[:, 1:]
                input_captions = captions[:, :-1] # Not used directly in forward for RAG model as implemented, 
                                                  # but RAGModel.forward currently takes 'target_ids' for training logic
                                                  # Let's adjust inputs to match RAGModel declaration
                
                # RAGModel forward: (images, context_ids, context_mask, target_ids)
                # It does teacher forcing internally with target_ids input. 
                # Ideally we should pass (images, context, decoder_input_ids) and alignment, but simplified:
                
                target_labels = captions[:, 1:]
                decoder_input = captions[:, :-1] # Pass this if model expects it for teacher forcing
                
                # Our RAGModel forward with 'target_ids' implies it handles embedding and shifting?
                # Looking at rag_model.py:
                # if target_ids is not None:
                #    tgt_embeds = self.embedding(target_ids) ...
                #    ... output = self.decoder(...)
                
                # So we should pass decoder_input as target_ids effectively for teacher forcing context?
                # No, standard transformer decoder takes "tgt" (input to decoder) and predicts next token.
                # So we pass `decoder_input` (shifted right) to model.
                
                outputs = model(images, context_ids, context_mask, target_ids=decoder_input) # [B, SeqLen, Vocab]
                
                # Loss
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_labels.reshape(-1))
            
            scaler.scale(loss).backward()
            
            # Clip Grads
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    parser.add_argument('--config', type=str, default='configs/rag_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
