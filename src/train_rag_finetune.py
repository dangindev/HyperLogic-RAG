
import torch
import argparse
import numpy as np
import os
import sys
import random

# Add Repo Root to Path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, 'R2Gen'))

from R2Gen.modules.trainer import Trainer
from R2Gen.modules.loss import compute_loss
from R2Gen.modules.metrics import compute_scores
from R2Gen.modules.optimizers import build_optimizer, build_lr_scheduler
from R2Gen.modules.rag_model import RAGModel # Use our new RAG Model
from HyperLogicRAG.src.datasets import MIMICCXR_RAG_Dataset, get_shared_tokenizer
from torch.utils.data import DataLoader

# Re-use HyperTrainer from train_hyper_r2gen logic but defined here for simplicity
class HyperTrainer(Trainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(HyperTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
        # IMPORTANT: When finetuning, we start from epoch 1, even if we loaded weights from a later epoch checkpont
        self.start_epoch = 1


    def train(self):
        """Override train() to handle dict-based batches from RAG dataset during validation."""
        best_bleu4 = 0.0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            # Validate EVERY epoch to capture the best checkpoint
            print(f"\n--- Validation (Epoch {epoch}) ---", flush=True)
            self.model.eval()
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, batch in enumerate(self.val_dataloader):
                    images = batch['images'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    output = self.model(images, mode='sample')
                    reports = self.model.tokenizer.batch_decode(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.batch_decode(input_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                log.update(**{'test_' + k: v for k, v in val_met.items()})
                print(f"Val metrics: { {k: f'{v:.4f}' for k, v in val_met.items()} }", flush=True)

            self._record_best(log)
            # Print logs
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))
            
            # Save current checkpoint
            self._save_checkpoint(epoch, save_best=False)
            
            # Save BEST model if BLEU-4 improved
            current_bleu4 = log.get('val_BLEU_4', 0.0)
            if current_bleu4 > best_bleu4:
                best_bleu4 = current_bleu4
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch, 'best_bleu4': best_bleu4}, best_path)
                print(f"★ New Best! BLEU-4: {best_bleu4:.4f} → Saved model_best.pth (Epoch {epoch})", flush=True)
            else:
                print(f"  No improvement (Best: {best_bleu4:.4f}, Current: {current_bleu4:.4f})", flush=True)

    def _train_epoch(self, epoch):
        print(f"\n{'='*60}")
        print(f"Starting RAG Finetune Epoch {epoch}/{self.epochs}")
        train_loss = 0
        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            hypergraph_indices = batch.get('hypergraph_indices', None)
            if hypergraph_indices is not None: hypergraph_indices = hypergraph_indices.to(self.device)

            self.optimizer.zero_grad()
            # RAG Retrieval happens inside model(..., mode='forward')
            output = self.model(images, input_ids, mode='forward', hypergraph_indices=hypergraph_indices)
            loss = self.criterion(output, input_ids, attention_mask)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_dataloader)}, Loss: {loss.item():.4f}")
                sys.stdout.flush()

        avg_loss = train_loss / len(self.train_dataloader)
        print(f"Training complete. Avg loss: {avg_loss:.4f}")
        self.lr_scheduler.step()
        return {'train_loss': avg_loss}

def parse_args():
    parser = argparse.ArgumentParser()
    # R2Gen Standard Args
    parser.add_argument('--image_dir', type=str, default='HyperLogicRAG/data/mimic_cxr/images')
    parser.add_argument('--ann_path', type=str, default='HyperLogicRAG/data/mimic_cxr_clean.jsonl')
    parser.add_argument('--hypergraph_path', type=str, default='HyperLogicRAG/data/hypergraph.pkl')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5) # Short finetuning
    parser.add_argument('--save_dir', type=str, default='results/rag_finetune')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=1e-5) # Very small LR for visual
    parser.add_argument('--lr_ed', type=float, default=5e-5) # Small LR for Decoder
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, required=True) # Must resume from Hyper-R2Gen
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    print(f"Loading tokenizer from {args.ann_path}")
    tokenizer = get_shared_tokenizer(args.ann_path, args.max_seq_length)
    args.bos_idx = tokenizer.bos_token_id
    args.eos_idx = tokenizer.eos_token_id
    args.pad_idx = tokenizer.pad_token_id
    
    print("Creating datasets...")
    dataset_config = {
        'dataset': {
            'data_root': os.path.dirname(args.ann_path),
            'manifest_file': os.path.basename(args.ann_path),
            'image_size': 224,
            'max_length': args.max_seq_length,
            'hypergraph_path': args.hypergraph_path,
            'retrieval_file': 'rag_index.json'
        },
        'model': {'visual_encoder': {'type': 'resnet'}}
    }
    
    train_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='train')
    val_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='validate')
    test_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='test') # Keep test for verification
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)
    
    # Build RAG Model
    print("Building RAG Model (Hyper-R2Gen + Retrieval)...")
    model = RAGModel(args, tokenizer)
    model = model.cuda()
    
    # Load pretrained weights
    resume_path = args.resume
    args.resume = None # FORCE CLEAR for Trainer so it starts fresh finetune (None is required, not "")
    
    if os.path.isfile(resume_path):
        print(f"Loading weights from {resume_path}...")
        checkpoint = torch.load(resume_path)
        state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights with msg: {msg}")
    
    # Build RAG Index
    # Only if index file doesn't exist
    index_path = 'HyperLogicRAG/data/rag_index.pt'
    model.retriever.index_path = index_path
    if not os.path.exists(index_path):
        print("RAG Index not found. Building from Training Dataset...")
        # Note: This takes time. Maybe 20-30 mins for 200k samples on GPU.
        model.retriever.build_index_from_dataset(train_dataset, batch_size=128)
    else:
        print("Loading RAG Index...")
        model.retriever.load_index()

    # Trainer
    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    trainer = HyperTrainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
