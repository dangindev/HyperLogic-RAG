"""
Debug script to examine generated reports from HL v2 checkpoint
"""
import torch
import torch.nn as nn
import os
import sys
import yaml
import json
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from src.models.hyperlogic_rag import HyperLogicRAGModel
from src.utils.hypergraph_utils import load_and_convert_hypergraph
import torchvision.models as models

def get_visual_encoder(config):
    base_model = models.densenet121(weights=None)
    base_model.classifier = nn.Identity()
    output_dim = config['model'].get('embed_dim', 512)
    
    class VisualEncoder(nn.Module):
        def __init__(self, base, output_dim):
            super().__init__()
            self.model = base
            self.feature_dim = 1024
            if output_dim != self.feature_dim:
                self.projection = nn.Linear(self.feature_dim, output_dim)
            else:
                self.projection = nn.Identity()
        
        def forward(self, x):
            features = self.model.features(x)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            features = self.projection(features)
            return features
    
    return VisualEncoder(base_model, output_dim)

def main():
    device = torch.device('cpu')
    
    # Load config
    config_path = "configs/hyperlogic_config_v2_resume.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load manifest
    manifest_path = "/home/dangnh3/MICCAI2026/HyperLogicRAG/data/mimic_cxr_clean.jsonl"
    samples = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry['split'] == 'test':
                samples.append(entry)
    
    print(f"Found {len(samples)} test samples")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Model
    visual_encoder = get_visual_encoder(config)
    hypergraph_data = load_and_convert_hypergraph("/home/dangnh3/MICCAI2026/HyperLogicRAG/data/hypergraph.pkl")
    
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=30522,
        num_entities=15814,
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model'].get('num_decoder_layers', 6),
        hypergraph_data=hypergraph_data
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load("results/hyperlogic_rag_v2/checkpoint_epoch_20.pth", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("\n" + "="*80)
    print("SAMPLE COMPARISON: HL v2 Generated vs Ground Truth")
    print("="*80)
    
    # Generate a few samples
    for i in range(5):
        sample = samples[i]
        image_path = sample['image_path']
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join("/home/dangnh3/MICCAI2026/HyperLogicRAG", image_path)
        gt_report = sample['report']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            img_array = np.array(image).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            image_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(image_tensor.to(device), max_length=100, num_beams=3)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Ground Truth ({len(gt_report.split())} words):")
        print(f"  {gt_report[:300]}...")
        print(f"Generated ({len(generated_text.split())} words):")
        print(f"  {generated_text[:300]}...")
        print()

if __name__ == "__main__":
    main()
