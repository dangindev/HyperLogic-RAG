#!/usr/bin/env python3
"""
Self-Critical Sequence Training (SCST) for HyperLogic-RAG.
Optimizes CIDEr score directly using REINFORCE.

Uses the SAME data pipeline as train_hyperlogic_biomedclip.py
(src.datasets.get_dataloader + R2GenTokenizer) to ensure vocab consistency.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # HyperLogicRAG/
repo_root = os.path.dirname(project_root)     # MICCAI2026/
sys.path.insert(0, project_root)
sys.path.insert(0, repo_root)
# For pycocoevalcap
sys.path.insert(0, os.path.join(repo_root, 'R2Gen_original'))

# --- IMPORTS ---
from pycocoevalcap.cider.cider import Cider

from src.datasets import get_dataloader, get_shared_tokenizer
from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.models.biomed_clip_encoder import BiomedCLIPEncoder
from src.utils.hypergraph_utils import load_and_convert_hypergraph


def compute_cider_scores(gen_texts, gt_texts):
    """Compute per-sample CIDEr scores."""
    cider = Cider()
    gts = {i: [gt_texts[i]] for i in range(len(gt_texts))}
    res = {i: [gen_texts[i]] for i in range(len(gen_texts))}
    _, scores = cider.compute_score(gts, res)
    return np.array(scores)


def train_scst(config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output dir
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. DATA (same pipeline as train_hyperlogic_biomedclip.py) ----
    print("\nCreating dataloaders...")
    config['dataset']['data_root'] = os.path.join(repo_root, 'HyperLogicRAG', 'data')
    
    # CRITICAL FIX: Initialize tokenizer with FULL dataset to match checkpoint vocab (21648)
    # The subset (50k) has smaller vocab (12020) causing partial load failure
    from src.datasets import get_shared_tokenizer
    full_manifest = os.path.join(config['dataset']['data_root'], 'mimic_cxr_clean.jsonl')
    print(f"Force-initializing tokenizer from full dataset: {full_manifest}")
    get_shared_tokenizer(full_manifest, max_length=config['dataset']['max_length'])
    
    train_loader = get_dataloader(config, 'train')
    val_split = config['dataset'].get('val_split', 'validate')
    val_loader = get_dataloader(config, val_split)

    # Get tokenizer from dataset
    tokenizer = train_loader.dataset.tokenizer
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Tokenizer vocab size: {vocab_size}")

    # ---- 2. MODEL (same init as train_hyperlogic_biomedclip.py) ----
    print("\nLoading hypergraph...")
    hypergraph_path = config['dataset']['hypergraph_path']
    if not os.path.isabs(hypergraph_path):
        hypergraph_path = os.path.join(repo_root, hypergraph_path)
    hypergraph_data = load_and_convert_hypergraph(hypergraph_path)

    print("\nCreating visual encoder (BiomedCLIP)...")
    freeze_backbone = config['model']['visual_encoder'].get('freeze_backbone', True)
    visual_encoder = BiomedCLIPEncoder(pretrained=True, freeze=freeze_backbone)

    print("\nInitializing model...")
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=vocab_size,
        num_entities=hypergraph_data['num_nodes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        hypergraph_data=hypergraph_data,
        use_relational_memory=config['model'].get('use_relational_memory', True),
        use_mcln=config['model'].get('use_mcln', False),
        dropout=config['model'].get('dropout', 0.1)
    )

    # Load pretrained CE checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    print("✓ Model loaded successfully")

    # ---- 2b. CHEXBERT (Clinical Reward) ----
    print("\nLoading CheXbert for Clinical Reward...")
    try:
        import eval_ce_metrics
        # Reload module in case of updates
        import importlib
        importlib.reload(eval_ce_metrics)
        
        chexbert_model, chexbert_tokenizer = eval_ce_metrics.load_chexbert(device)
        chexbert_model.eval()
        for p in chexbert_model.parameters():
            p.requires_grad = False
        print("✓ CheXbert loaded for reward calculation")
    except ImportError:
        print("❌ Could not import eval_ce_metrics. Make sure it is in PYTHONPATH.")
        sys.exit(1)

    def compute_ce_rewards_batch(gen_texts, gt_texts):
        # Compute F1 per sample
        # Returns np.array of shape (B,)
        
        # 1. Get labels
        # label_reports returns (N_samples, 14) binary
        # We need to process in one go
        # Note: label_reports expects List[str]
        
        with torch.no_grad():
            gen_labels = eval_ce_metrics.label_reports(chexbert_model, chexbert_tokenizer, gen_texts, device, batch_size=len(gen_texts))
            gt_labels = eval_ce_metrics.label_reports(chexbert_model, chexbert_tokenizer, gt_texts, device, batch_size=len(gt_texts))
            
            # Apply threshold 0.5 (label_reports returns probs? Check implementation)
            # label_reports in Step 1006 returns "probs" (softmax/sigmoid output)
            gen_binary = (gen_labels > 0.5).astype(int)
            gt_binary = (gt_labels > 0.5).astype(int)
            
            # 2. Compute F1 per sample
            # F1 = 2 * (P * R) / (P + R)
            # TP = sum(gen & gt)
            # FP = sum(gen & !gt)
            # FN = sum(!gen & gt)
            
            scores = []
            for i in range(len(gen_texts)):
                tp = np.sum(gen_binary[i] & gt_binary[i])
                fp = np.sum(gen_binary[i] & ~gt_binary[i])
                fn = np.sum(~gen_binary[i] & gt_binary[i])
                
                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2 * (precision * recall) / (precision + recall + epsilon)
                
                # Special case: If both empty (No Finding), F1=1?
                if np.sum(gt_binary[i]) == 0:
                    if np.sum(gen_binary[i]) == 0:
                        f1 = 1.0
                    else:
                        f1 = 0.0 # GT empty but Gen has hallucination
                
                scores.append(f1)
                
            return np.array(scores)

    # ---- 3. OPTIMIZER ----
    lr = config['training'].get('learning_rate', 5e-6)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config['training'].get('weight_decay', 0.01))
    
    # Reward weights
    w_cider = config['training'].get('reward_mix', {}).get('cider', 0.5)
    w_ce = config['training'].get('reward_mix', {}).get('ce_f1', 0.5)
    print(f"Reward Mixing: CIDEr={w_cider} | Clinical F1={w_ce}")

    # ---- 4. SCST TRAINING LOOP ----
    num_epochs = config['training']['epochs']
    print(f"\n{'='*60}")
    print(f"STARTING SCST TRAINING ({num_epochs} epochs, lr={lr})")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_ce_f1 = 0.0
        num_batches = 0

        # Use tqdm variable to update postfix
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            images = batch['image'].to(device)       # [B, 3, 224, 224]
            input_ids = batch['input_ids'].to(device) # [B, seq_len]
            captions = batch['caption']                # list of strings

            optimizer.zero_grad()

            # --- Greedy baseline (no grad) ---
            model.eval()
            with torch.no_grad():
                greedy_seq, _ = model.sample_scst(images, sample_method='greedy', retrieval_ids=None) # Correct signature
            model.train()

            # --- Sampled sequence (with grad) ---
            sample_seq, sample_log_probs = model.sample_scst(images, sample_method='sample', retrieval_ids=None)

            # --- Decode to text ---
            greedy_txt = [tokenizer.decode(seq, skip_special_tokens=True) for seq in greedy_seq.cpu().numpy()]
            sample_txt = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sample_seq.cpu().numpy()]
            gt_txt = captions  # Already strings from dataset

            # --- Compute Rewards ---
            try:
                # 1. CIDEr
                r_cider_g = compute_cider_scores(greedy_txt, gt_txt)
                r_cider_s = compute_cider_scores(sample_txt, gt_txt)
                
                # 2. Clinical F1
                r_ce_g = compute_ce_rewards_batch(greedy_txt, gt_txt)
                r_ce_s = compute_ce_rewards_batch(sample_txt, gt_txt)
                
                # Mix
                reward_greedy = w_cider * r_cider_g + w_ce * r_ce_g
                reward_sample = w_cider * r_cider_s + w_ce * r_ce_s
                
            except Exception as e:
                print(f"  [WARN] Reward calc failed: {e}, skipping batch")
                continue

            # Self-critical advantage
            advantage = torch.from_numpy(reward_sample - reward_greedy).float().to(device)

            # --- REINFORCE loss ---
            mask = (sample_seq != 0).float().to(device) # Assuming 0 is pad
            loss = -(advantage.unsqueeze(1) * sample_log_probs * mask)
            loss = loss.sum() / mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update stats
            epoch_loss += loss.item()
            epoch_reward += reward_sample.mean()
            epoch_ce_f1 += r_ce_s.mean()
            num_batches += 1
            
            # --- PROGRESS UPDATE ---
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'rwd': f"{reward_sample.mean():.4f}",
                'f1': f"{r_ce_s.mean():.4f}"
            })
            
            # --- INTERMEDIATE LOGGING (Every 500 batches ~ 1h 20m) ---
            if num_batches % 500 == 0:
                avg_l = epoch_loss / num_batches
                avg_r = epoch_reward / num_batches
                avg_f = epoch_ce_f1 / num_batches
                print(f"\n[Batch {num_batches}] Loss={avg_l:.4f} | Rwd={avg_r:.4f} | CE_F1={avg_f:.4f}")


        # --- Epoch summary ---
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_reward = epoch_reward / max(num_batches, 1)
        avg_f1 = epoch_ce_f1 / max(num_batches, 1)
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f} | Avg Reward={avg_reward:.4f} | Avg CE F1={avg_f1:.4f}")

        # --- Save ---
        save_path = os.path.join(output_dir, f'checkpoint_scst_epoch_{epoch+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'avg_reward': avg_reward
        }, save_path)
        print(f"  Saved: {save_path}")

    print("\n✓ SCST Training Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCST Training for HyperLogic-RAG')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to CE pre-trained model")
    args = parser.parse_args()

    train_scst(args.config, args.checkpoint)
