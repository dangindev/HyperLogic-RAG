#!/usr/bin/env python3
"""
Evaluate HyperLogicRAG Model
"""
import sys
import os
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.utils.hypergraph_utils import load_and_convert_hypergraph
from src.datasets import get_dataloader

# Metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def create_visual_encoder():
    """Replicate visual encoder creation from training script"""
    print("Loading pretrained ResNet-50 from AutoModel...")
    resnet = AutoModel.from_pretrained("microsoft/resnet-50")
    
    class ResNetFeatureExtractor(nn.Module):
        def __init__(self, resnet_model):
            super().__init__()
            self.resnet = resnet_model
            
        def forward(self, x):
            outputs = self.resnet(x)
            return outputs.last_hidden_state
    
    return ResNetFeatureExtractor(resnet)


def compute_scores(gts, res):
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
    return scores


def evaluate(config_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load Hypergraph
    hypergraph_path = os.path.join(os.path.dirname(config_path), '..', config['dataset']['hypergraph_path'])
    if not os.path.exists(hypergraph_path):
        # Fallback to local path if relative path fails
        hypergraph_path = config['dataset']['hypergraph_path']
        
    print(f"Loading hypergraph from {hypergraph_path}...")
    hypergraph_data = load_and_convert_hypergraph(hypergraph_path)
    
    # Create Model
    print("Creating model...")
    visual_encoder = create_visual_encoder()
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=config['model']['vocab_size'],
        num_entities=len(hypergraph_data['node_to_id']),
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        hypergraph_data=hypergraph_data
    ).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Remove module. prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Dataloader
    # Update config for dataloader to find data
    if not os.path.isabs(config['dataset']['data_root']):
        config['dataset']['data_root'] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            config['dataset']['data_root']
        )
            
    # Force test split and num_workers=0 (to avoid shm errors)
    orig_workers = config['dataset']['num_workers']
    config['dataset']['num_workers'] = 0
    test_loader = get_dataloader(config, split='test')
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    predictions = []
    ground_truths = []
    
    print("Generating reports...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            captions = batch['input_ids'].to(device)
            # Retrieval context logic here if needed, but datasets.py should handle it
            # The model forward takes retrieval_context if present
            # Let's check if dataloader provides it.
            # RAG dataset should provide 'context_ids'
            
            retrieval_context = None
            if 'context_ids' in batch:
                # Need to encode context if model expects embeddings, 
                # or pass IDs if model has context encoder.
                # HyperLogicRAGModel forward expects `retrieval_context` as embeddings [B, C, D]?
                # Let's check forward signature in hyperlogic_rag.py... 
                # It takes `retrieval_context` as optional.
                # If the model integrates retrieval, it might need embeddings.
                # Inspecting the model code would clarify, but for now passing None is safe baseline.
                pass

            # Generation Loop
            # Start with CLS token (BERT)
            batch_size = images.size(0)
            generated = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
            
            for _ in range(config['dataset']['max_length']):
                # Forward pass
                # Note: Testing logic might need `target_tokens` for teacher forcing training
                # But for generation we pass generated so far.
                # HyperLogicRAGModel forward signature: (images, target_tokens, retrieval_context=None)
                # It returns logits.
                
                logits, _ = model(images, generated, retrieval_context)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == tokenizer.sep_token_id).all():
                    break
            
            # Decode
            for i in range(batch_size):
                pred = tokenizer.decode(generated[i], skip_special_tokens=True)
                gt = tokenizer.decode(captions[i], skip_special_tokens=True)
                predictions.append(pred)
                ground_truths.append(gt)

    # Compute Metrics
    print("\nComputing metrics...")
    gts = {i: [gt] for i, gt in enumerate(ground_truths)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    scores = compute_scores(gts, res)
    
    print(f"\n{'='*60}")
    print("HYPERLOGIC RAG EVALUATION RESULTS")
    print(f"{'='*60}")
    for k, v in scores.items():
        print(f"{k:15s}: {v:.4f}")
    print(f"{'='*60}\n")
    
    # Samples
    for i in range(5):
        print(f"[{i}] GT:   {ground_truths[i][:100]}...")
        print(f"    Pred: {predictions[i][:100]}...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
