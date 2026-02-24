import torch
import argparse
import numpy as np
import os
from R2Gen.modules.trainer import Trainer
from R2Gen.modules.loss import compute_loss
from R2Gen.modules.metrics import compute_scores
from R2Gen.modules.optimizers import build_optimizer, build_lr_scheduler
from HyperLogicRAG.src.models.hyper_r2gen import HyperR2GenModel
from HyperLogicRAG.src.datasets import MIMICCXR_RAG_Dataset, get_shared_tokenizer, get_dataloader
from torch.utils.data import DataLoader

# Argument parsing (reuse standard + new args)
# ... (Will implement full arg parser in file)

class HyperTrainer(Trainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(HyperTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        # Override loop to handle dictionary batch
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Unpack RAG Dataset batch
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                targets = batch['input_ids'].to(self.device)
                hypergraph_indices = batch.get('hypergraph_indices', None)
                if hypergraph_indices is not None:
                     hypergraph_indices = hypergraph_indices.to(self.device)
            else:
                # Fallback
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                hypergraph_indices = None
            
            # Forward
            optimizer.zero_grad()
            outputs = self.model(images, targets, mode='forward', hypergraph_indices=hypergraph_indices)
            
            # Loss
            loss = self.criterion(outputs, targets[:, 1:], targets[:, :-1]) # Check shifting
            loss.backward()
            
            # Clip Config
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Logging (print every 50 batches)
            if batch_idx % 50 == 0:
                 print(f"Epoch {epoch} [{batch_idx}/{len(self.train_dataloader)}] Loss: {loss.item():.4f}")

        return total_loss / len(self.train_dataloader)

    # Need to override validate/test to pass hypergraph_indices too
    # ...

def parse_args():
    parser = argparse.ArgumentParser()
    # Copy R2Gen args + HyperLogic args
    # ...
    return parser.parse_args()

def main():
    # ...
    # Initialize tokenizer with FULL clean dataset
    # Initialize RAG Dataset
    pass

if __name__ == '__main__':
    main()
