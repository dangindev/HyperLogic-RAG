import torch
import torch.nn as nn
import argparse
import os
import sys
import yaml
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

# Add path to src
sys.path.append(os.getcwd())

from src.models.hyperlogic_rag import HyperLogicRAGModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# Mock Visual Encoder for loading if needed (though we load checkpoints)
# But HyperLogicRAGModel takes 'visual_encoder' as init arg.
# We need to reconstruct the visual encoder to pass it to __init__.
from transformers import AutoModel

class MockVisualEncoder(nn.Module):
    def __init__(self, encoder_type='densenet121'):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'densenet121':
            import torchvision.models as models
            self.base = models.densenet121(pretrained=False) # Weights loaded from checkpoint
            self.encoder = nn.Sequential(*list(self.base.children())[:-1])
        else:
            self.encoder = AutoModel.from_pretrained("microsoft/resnet-50") # Fallback

    def forward(self, x):
        features = self.encoder(x)
        if self.encoder_type == 'densenet121':
             # [B, 1024, 7, 7]
             return features
        return features

def get_visual_encoder(config):
    """
    Create visual encoder matching training (ImageNet DenseNet121 fallback)
    The training used torchvision DenseNet121 (3-channel input) due to missing torchxrayvision
    """
    import torchvision.models as models
    
    # Use DenseNet121 with ImageNet weights (same as training fallback)
    base_model = models.densenet121(weights=None)  # We'll load from checkpoint anyway
    base_model.classifier = nn.Identity()
    
    # Get output dimension from config
    output_dim = config['model'].get('embed_dim', 512)
    
    class VisualEncoder(nn.Module):
        def __init__(self, base, output_dim):
            super().__init__()
            self.model = base
            self.feature_dim = 1024  # DenseNet121 feature dimension
            if output_dim != self.feature_dim:
                self.projection = nn.Linear(self.feature_dim, output_dim)
            else:
                self.projection = nn.Identity()
        
        def forward(self, x):
            features = self.model.features(x)  # [B, 1024, 7, 7]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [B, 1024, 1, 1]
            features = features.view(features.size(0), -1)  # [B, 1024]
            features = self.projection(features)  # [B, output_dim]
            return features
    
    return VisualEncoder(base_model, output_dim)

def compute_scores(gts, res):
    """
    Compute evaluation metrics
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

def generate_greedy(model, images, tokenizer, max_length=100, device='cuda'):
    """
    Greedy decoding generation
    """
    model.eval()
    
    # We use model.generate() if available (Beam Search), or implement manual greedy
    # HyperLogicRAGModel has 'generate' method using beam search.
    # Let's use that if possible, but it returns IDs.
    
    with torch.no_grad():
        # model.generate returns [B, seq_len]
        # We need to handle potential beam search or greedy manually if we want speed
        # For now, let's use the model's generate method which uses beam search (better for metrics)
        
        # It needs 'max_length' and 'num_beams'
        generated_ids = model.generate(images, max_length=max_length, num_beams=3)
        
    return generated_ids

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from file or use manual
    config_path = args.config
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dataset
    print("Loading test dataset...")
    # Manual dataset loading to ensure control
    class MIMICTestDataset(torch.utils.data.Dataset):
        def __init__(self, config):
            self.data_root = config['dataset']['data_root']
            # self.image_dir = config['dataset']['image_dir'] # Config might differ
            self.image_dir = "/home/dangnh3/MICCAI2026/HyperLogicRAG/data/mimic_cxr/images"
            self.max_length = config['dataset']['max_length']
            self.image_size = config['dataset']['image_size']
            
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            self.samples = []
            manifest = "/home/dangnh3/MICCAI2026/HyperLogicRAG/data/mimic_cxr_clean.jsonl"
            if not os.path.exists(manifest):
                 # Try alternative path if first one fails
                 manifest = "/home/dangnh3/MICCAI2026/data/mimic_cxr_clean.jsonl"
            
            print(f"Loading manifest from {manifest}")
            with open(manifest, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry['split'] == 'test':
                        self.samples.append(entry)
            print(f"Found {len(self.samples)} test samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            image_path = sample['image_path']
            
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.image_size, self.image_size))
                img_array = np.array(image).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_array = (img_array - mean) / std
                image = torch.from_numpy(img_array.transpose(2, 0, 1))
            except:
                image = torch.zeros((3, self.image_size, self.image_size))

            return {
                'image': image,
                'caption': sample['report']
            }

    dataset = MIMICTestDataset(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("Loading model...")
    # 1. Visual Encoder
    visual_encoder = get_visual_encoder(config)
    
    # 2. Hypergraph (Mock or Load)
    # The model fails if we don't pass hypergraph_data structurally compatible if it was trained with it.
    # Typically logic loss is training only, but architecture (HyperGCN) requires graph structure (edge_index).
    # We must load it.
    from src.utils.hypergraph_utils import load_and_convert_hypergraph
    hypergraph_path = "/home/dangnh3/MICCAI2026/HyperLogicRAG/data/hypergraph.pkl"
    if os.path.exists(hypergraph_path):
        print(f"Loading hypergraph from {hypergraph_path}")
        hypergraph_data = load_and_convert_hypergraph(hypergraph_path)
    else:
        print("⚠️ Hypergraph not found! Initializing with None (may fail if checkpoint has HyperGCN weights)")
        hypergraph_data = None

    # 3. Initialize
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=30522, # BERT vocab
        num_entities=15814, # Fixed for this dataset
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_decoder_layers=config['model'].get('num_decoder_layers', 6),
        hypergraph_data=hypergraph_data
    ).to(device)
    
    # 4. Load Checkpoint
    checkpoint_path = args.checkpoint_path
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Handle state_dict keys (remove 'module.' if DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Strict=False to allow missing logic loss buffers/auxiliary stuff if needed
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Generation
    print("Generating reports...")
    predictions = []
    ground_truths = []
    
    tokenizer = dataset.tokenizer
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Generate (Beam Search by default in model.generate)
            # Returns [B, seq_len] tensor of IDs
            generated_ids = model.generate(images, max_length=100, num_beams=3)
            
            # Decode
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            predictions.extend(generated_texts)
            ground_truths.extend(captions)
            
    # Compute Metrics
    print("Computing metrics...")
    gts = {i: [gt] for i, gt in enumerate(ground_truths)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    
    scores = compute_scores(gts, res)
    
    print("\n==================================================")
    print(f"HyperLogic v2 Results (Checkpoint: {os.path.basename(checkpoint_path)})")
    print("==================================================")
    for k, v in scores.items():
        print(f"{k:10s}: {v:.4f}")
    print("==================================================")
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'metrics': scores,
        # 'predictions': predictions[:10]
    }
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/hyperlogic_config_v2_full.yaml')
    parser.add_argument('--output', type=str, default='hyperlogic_v2_results.json')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
