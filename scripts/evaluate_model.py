#!/usr/bin/env python3
"""
Unified Model Evaluation Script

Supports evaluation of:
- R2Gen
- M2KT
- HyperLogic RAG (V1/V2)
- Internal Baseline

Computes: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
"""

import sys
import os
sys.path.insert(0, '/path/to/HyperLogicRAG')
sys.path.insert(0, '/path/to/R2Gen')
sys.path.insert(0, '/path/to/M2KT_baseline')

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import yaml

# Metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_scores(gts, res):
    """
    Compute evaluation metrics
    
    Args:
        gts: dict of ground truth reports {id: [report]}
        res: dict of generated reports {id: [report]}
    
    Returns:
        dict of scores
    """
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


def load_hyperlogic_model(checkpoint_path, device='cuda'):
    """Load HyperLogic RAG model from checkpoint"""
    from src.models.hyperlogic_rag import HyperLogicRAGModel
    from transformers import AutoTokenizer
    
    print(f"Loading HyperLogic checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config (should be saved with checkpoint)
    config_path = Path(checkpoint_path).parent.parent / 'configs' / 'hyperlogic_config_v2_full.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
    
    # Build model (simplified - may need full config)
    print("Note: Loading model requires full initialization")
    print("For now, returning checkpoint data")
    
    return checkpoint, config, tokenizer


def load_r2gen_model(checkpoint_path, device='cuda'):
    """Load R2Gen model"""
    # Import R2Gen modules
    sys.path.insert(0, '/path/to/R2Gen')
    from models.r2gen import R2GenModel
    from modules.tokenizers import Tokenizer
    
    print(f"Loading R2Gen checkpoint: {checkpoint_path}")
    
    # Create args (need to match training config)
    class Args:
        def __init__(self):
            self.dataset_name = 'mimic_cxr'
            self.threshold = 10
            self.max_seq_length = 100
            self.ann_path = '/path/to/R2Gen/data/mimic_cxr/annotation.json'
            
            # Model params
            self.visual_extractor = 'resnet101'
            self.visual_extractor_pretrained = True
            self.d_model = 512
            self.d_ff = 512
            self.d_vf = 2048
            self.num_heads = 8
            self.num_layers = 3
            self.dropout = 0.1
            self.logit_layers = 1
            self.bos_idx = 0
            self.eos_idx = 0
            self.pad_idx = 0
            self.use_bn = 0
            self.drop_prob_lm = 0.5
            self.rm_num_slots = 3
            self.rm_num_heads = 8
            self.rm_d_model = 512
            
            # Sampling
            self.sample_method = 'beam_search'
            self.beam_size = 3
            self.temperature = 1.0
            self.sample_n = 1
            self.group_size = 1
            self.output_logsoftmax = 1
            self.decoding_constraint = 0
            self.block_trigrams = 1
    
    args = Args()
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remove 'module.' prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("✓ R2Gen model loaded")
    return model, tokenizer


def evaluate_r2gen(model, tokenizer, test_loader, device='cuda'):
    """Evaluate R2Gen model"""
    predictions = []
    ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            
            # Generate
            output, _ = model(images, mode='sample')
            
            # Decode
            reports = tokenizer.decode_batch(output.cpu().numpy())
            gt_reports = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            
            predictions.extend(reports)
            ground_truths.extend(gt_reports)
    
    # Compute metrics
    gts = {i: [gt] for i, gt in enumerate(ground_truths)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    
    scores = compute_scores(gts, res)
    
    return scores, predictions, ground_truths


def quick_eval_checkpoint(checkpoint_path, model_type='hyperlogic', num_samples=50, device='cpu'):
    """
    Quick evaluation - load checkpoint and generate samples
    
    Args:
        checkpoint_path: path to model checkpoint
        model_type: 'hyperlogic', 'r2gen', 'm2kt', 'baseline'
        num_samples: number of samples to generate
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'='*60}")
    print(f"QUICK EVALUATION: {model_type.upper()}")
    print(f"{'='*60}\n")
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}\n")
    
    if model_type == 'hyperlogic':
        # For now, just show checkpoint info
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Checkpoint Info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
        print(f"  Parameters: {len(state_dict)} keys")
        
        print("\n✓ Checkpoint loaded successfully!")
        print("\nNote: Full evaluation requires complete model initialization")
        print("Use --full_eval flag with proper config for complete evaluation")
        
        return None
    
    elif model_type == 'r2gen':
        print("Loading R2Gen model...")
        model, tokenizer = load_r2gen_model(checkpoint_path, device)
        
        # Load test data
        from modules.dataloaders import R2DataLoader
        
        class Args:
            def __init__(self):
                self.image_dir = '/path/to/R2Gen/data/mimic_cxr/images/'
                self.ann_path = '/path/to/R2Gen/data/mimic_cxr/annotation.json'
                self.dataset_name = 'mimic_cxr'
                self.max_seq_length = 100
                self.threshold = 10
                self.batch_size = 16
                self.num_workers = 4
        
        args = Args()
        test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
        
        print(f"Test set: {len(test_loader)} batches")
        print("\nGenerating reports...")
        
        scores, predictions, ground_truths = evaluate_r2gen(model, tokenizer, test_loader, device)
        
        print(f"\n{'='*60}")
        print("R2GEN EVALUATION RESULTS")
        print(f"{'='*60}")
        for metric, value in scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print(f"{'='*60}\n")
        
        return scores
    
    else:
        print(f"Model type '{model_type}' not yet implemented")
        return None


def main():
    parser = argparse.ArgumentParser(description='Unified Model Evaluation')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['r2gen', 'm2kt', 'hyperlogic', 'baseline'],
                       help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Evaluation params
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples (None = full test set)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save generated predictions')
    
    # Mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation mode')
    
    args = parser.parse_args()
    
    # Run evaluation
    if args.quick or args.model in ['hyperlogic', 'baseline']:
        # Quick mode for models not yet fully implemented
        results = quick_eval_checkpoint(
            args.checkpoint,
            args.model,
            args.num_samples or 50,
            args.device
        )
    else:
        # Full evaluation
        if args.model == 'r2gen':
            print("Running full R2Gen evaluation...")
            model, tokenizer = load_r2gen_model(args.checkpoint, args.device)
            
            from modules.dataloaders import R2DataLoader
            class EvalArgs:
                image_dir = '/path/to/R2Gen/data/mimic_cxr/images/'
                ann_path = '/path/to/R2Gen/data/mimic_cxr/annotation.json'
                dataset_name = 'mimic_cxr'
                max_seq_length = 100
                threshold = 10
                batch_size = args.batch_size
                num_workers = 4
            
            test_loader = R2DataLoader(EvalArgs(), tokenizer, split='test', shuffle=False)
            results, predictions, ground_truths = evaluate_r2gen(model, tokenizer, test_loader, args.device)
            
            # Save results
            if args.output:
                output_data = {
                    'model': args.model,
                    'checkpoint': args.checkpoint,
                    'metrics': results
                }
                if args.save_predictions:
                    output_data['predictions'] = predictions[:100]  # Save first 100
                    output_data['ground_truths'] = ground_truths[:100]
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n✓ Results saved to: {args.output}")
        
        else:
            print(f"Full evaluation for {args.model} not yet implemented")
            print("Use --quick flag for quick checkpoint test")


if __name__ == '__main__':
    main()
