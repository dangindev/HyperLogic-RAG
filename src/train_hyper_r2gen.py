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
from HyperLogicRAG.src.models.hyper_r2gen import HyperR2GenModel
from HyperLogicRAG.src.datasets import MIMICCXR_RAG_Dataset, get_shared_tokenizer, get_dataloader
from torch.utils.data import DataLoader

class HyperTrainer(Trainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(HyperTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)

    def train(self):
        """Override train() to handle dict-based batches from RAG dataset during validation."""
        from numpy import inf
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            should_validate = (epoch == self.epochs) or (epoch % self.save_period == 0)

            if should_validate and self.mnt_mode != 'off':
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
                        if batch_idx == 0:
                            print(f"  [DEBUG] Val batch 0: output shape={output.shape}, first report='{reports[0][:100]}', first gt='{ground_truths[0][:100]}'", flush=True)
                        val_res.extend(reports)
                        val_gts.extend(ground_truths)
                    print(f"  [DEBUG] Val: {len(val_gts)} gts, {len(val_res)} res, non-empty res: {sum(1 for r in val_res if r.strip())}", flush=True)
                    val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                               {i: [re] for i, re in enumerate(val_res)})
                    log.update(**{'val_' + k: v for k, v in val_met.items()})
                    print(f"Val metrics: { {k: f'{v:.4f}' for k, v in val_met.items()} }", flush=True)

                self.model.eval()
                with torch.no_grad():
                    test_gts, test_res = [], []
                    for batch_idx, batch in enumerate(self.test_dataloader):
                        images = batch['images'].to(self.device)
                        input_ids = batch['input_ids'].to(self.device)
                        output = self.model(images, mode='sample')
                        reports = self.model.tokenizer.batch_decode(output.cpu().numpy())
                        ground_truths = self.model.tokenizer.batch_decode(input_ids[:, 1:].cpu().numpy())
                        test_res.extend(reports)
                        test_gts.extend(ground_truths)
                    test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                                {i: [re] for i, re in enumerate(test_res)})
                    log.update(**{'test_' + k: v for k, v in test_met.items()})
                    print(f"Test metrics: { {k: f'{v:.4f}' for k, v in test_met.items()} }", flush=True)

            self._record_best(log)

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off' and should_validate:
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. ".format(self.mnt_metric))
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn't improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _train_epoch(self, epoch):
        print(f"\n{'='*60}")
        print(f"Starting Hyper-R2Gen Epoch {epoch}/{self.epochs}")
        print(f"{'='*60}")
        sys.stdout.flush()

        train_loss = 0
        self.model.train()
        print(f"Training on {len(self.train_dataloader)} batches...")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Unpack RAG Dataset batch (Dictionary)
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            hypergraph_indices = batch.get('hypergraph_indices', None)
            
            if hypergraph_indices is not None:
                hypergraph_indices = hypergraph_indices.to(self.device)

            # R2Gen Criterion expects: output, reports_ids, reports_masks
            # reports_ids should be the target labels
            # In R2Gen Dataset, reports_ids includes <bos> and <eos>
            # Model forward inputs match R2Gen logic?
            # R2GenModel forward takes 'targets' and shifts internally?
            # R2GenModel:
            #   att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, None, targets)
            #   _prepare_feature_forward cuts the last token of seq for input (seq[:, :-1])
            # So 'targets' passed to model should be the full sequence.
            
            self.optimizer.zero_grad()
            output = self.model(images, input_ids, mode='forward', hypergraph_indices=hypergraph_indices)
            
            # Loss Calculation
            # compute_loss(output, reports_ids, reports_masks)
            # output is [B, S-1, V] (because input was S-1 length)
            # reports_ids is [B, S]
            # output aligns with reports_ids[:, 1:] (next token prediction)
            loss = self.criterion(output, input_ids, attention_mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # Logging
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='results/hyper_r2gen')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str, help='Path to pretrained weights (partial load, strict=False for transfer learning)')
    parser.add_argument('--rag_index_path', type=str, default=None, help='Path to RAG retrieval index JSON (defaults to rag_index.json in ann_path dir)')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Fix seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # Create Tokenizer using R2GenTokenizer wrapper (reused or reimplemented)
    # HyperLogicRAG has 'get_shared_tokenizer'
    print(f"Loading tokenizer from {args.ann_path}")
    tokenizer = get_shared_tokenizer(args.ann_path, args.max_seq_length)
    
    # Update args with tokenizer info if needed
    args.bos_idx = tokenizer.bos_token_id
    args.eos_idx = tokenizer.eos_token_id
    args.pad_idx = tokenizer.pad_token_id
    
    # Create Datasets using MIMICCXR_RAG_Dataset
    # Note: args.image_dir must be correct
    print("Creating datasets...")
    # Config wrapper to satisfy get_dataloader/MIMICCXR_RAG_Dataset expected config format
    dataset_config = {
        'dataset': {
            'data_root': os.path.dirname(args.ann_path), # Parent of data/mimic_cxr_clean.jsonl -> HyperLogicRAG/data
            'manifest_file': os.path.basename(args.ann_path),
            'image_size': 224,
            'max_length': args.max_seq_length,
            'hypergraph_path': args.hypergraph_path,
            'rag_index_path': args.rag_index_path if args.rag_index_path else os.path.join(os.path.dirname(args.ann_path), 'rag_index.json')
        },
        'model': {
            'visual_encoder': {'type': 'resnet'} # Trigger standard transform
        }
    }
    
    # We manually create instances because get_dataloader might do things differently
    train_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='train')
    val_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='validate')
    test_dataset = MIMICCXR_RAG_Dataset(dataset_config, split='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)
    
    # Build Model
    print("Building Hyper-R2Gen Model...")
    model = HyperR2GenModel(args, tokenizer)
    
    # Load pretrained weights (partial, for transfer learning)
    if hasattr(args, 'pretrained') and args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Filter out mismatched keys (vocab size differs between datasets)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        skipped = [k for k in state_dict if k not in filtered]
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(filtered)}/{len(state_dict)} params. Skipped {len(skipped)}: {skipped}")
        # Clear --resume so Trainer doesn't try strict loading
        args.resume = None
    
    # Setup Trainer
    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    trainer = HyperTrainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
