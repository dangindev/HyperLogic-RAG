#!/usr/bin/env python3
"""
Evaluate DenseNet Baseline Model
"""
import sys
sys.path.insert(0, '/home/dangnh3/MICCAI2026/HyperLogicRAG')

import torch
import yaml
import argparse
from tqdm import tqdm
from src.datasets import get_dataloader
from src.models.baseline_model import BaselineModel
from transformers import AutoTokenizer

# Metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_scores(gts, res):
    """Compute BLEU, METEOR, ROUGE-L"""
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]
    
    scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score
        except Exception as e:
            print(f"Warning: {method} computation failed: {e}")
            if isinstance(method, list):
                for m in method:
                    scores[m] = 0.0
            else:
                scores[method] = 0.0
    
    return scores


def evaluate(config, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Load model
    model = BaselineModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✓ Model loaded\n")
    
    # Tokenizer - MUST MATCH TRAINING!
    tokenizer_name = config.get('dataset', {}).get('tokenizer', 'bert-base-uncased')
    print(f"Using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Test dataloader (use num_workers=0 to avoid shm errors)
    eval_config = config.copy()
    eval_config['dataset']['num_workers'] = 0
    test_loader = get_dataloader(eval_config, split='test')
    print(f"Test set: {len(test_loader.dataset)} samples\n")
    
    predictions = []
    ground_truths = []
    
    print("Generating reports...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            captions = batch['input_ids']
            
            # Generate with greedy decoding
            batch_size = images.size(0)
            max_len = config['dataset']['max_length']
            
            # Start with CLS token (BERT) or BOS token (GPT-2)
            start_token = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None else tokenizer.bos_token_id
            if start_token is None:
                start_token = tokenizer.pad_token_id  # Fallback
                
            end_token = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None else tokenizer.eos_token_id
            if end_token is None:
                end_token = tokenizer.pad_token_id  # Fallback
                
            generated = torch.full((batch_size, 1), start_token, 
                                   dtype=torch.long, device=device)
            
            # Greedy decoding
            for _ in range(max_len - 1):
                outputs = model(images, generated)
                next_tokens = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Stop if all sequences have EOS
                if end_token is not None and (next_tokens == end_token).all():
                    break
            
            # Decode predictions
            for i in range(batch_size):
                pred_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                gt_text = tokenizer.decode(captions[i], skip_special_tokens=True)
                
                predictions.append(pred_text)
                ground_truths.append(gt_text)
    
    # Compute metrics
    print("\nComputing metrics...")
    gts = {i: [gt] for i, gt in enumerate(ground_truths)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    
    scores = compute_scores(gts, res)
    
    # Print results
    print(f"\n{'='*60}")
    print("DENSENET-121 BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    for metric, value in scores.items():
        print(f"{metric:15s}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Save sample predictions
    print("Sample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"GT:   {ground_truths[i][:100]}...")
        print(f"Pred: {predictions[i][:100]}...")
    
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.checkpoint)
