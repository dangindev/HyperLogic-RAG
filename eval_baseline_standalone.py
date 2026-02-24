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

from src.models.baseline_model import BaselineModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

class MockArgs:
    def __init__(self):
        pass

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
    batch_size = images.size(0)
    
    # Encoder
    with torch.no_grad():
        features = model.encoder(images) # [B, embed_dim]
        
        # Start tokens
        # BertTokenizer uses [CLS]=101 as start? No, typically it doesn't have BOS for generation in BERT.
        # But this model was trained with 'caption[:, :-1]' and 'caption[:, 1:]'.
        # Let's check dataset used. It uses 'bert-base-uncased'. 
        # BERT is not autoregressive, but here it's used as decoder? 
        # Wait, the DecoderRNN uses TransformerDecoder.
        # It expects embeddings.
        # We need a BOS token. BERT doesn't strictly have one for Decoder.
        # Usually [CLS] (101) is used as BOS if not specified.
        
        bos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
        eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
        
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        
        for i in range(max_length):
            outputs = model.decoder(features, generated)
            # outputs: [B, Seq, Vocab]
            next_token_logits = outputs[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)
            
            generated = torch.cat((generated, next_token), dim=1)
            
            # Check for EOS
            is_eos = (next_token.squeeze(1) == eos_token_id)
            finished = finished | is_eos
            
            if finished.all():
                break
                
    return generated

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from file or use manual
    config_path = 'configs/baseline_config.yaml'
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dataset
    print("Loading test dataset...")
    # Manual dataset loading to ensure control
    class MIMICTestDataset(torch.utils.data.Dataset):
        def __init__(self, config):
            self.data_root = config['dataset']['data_root']
            self.image_dir = config['dataset']['image_dir']
            self.max_length = config['dataset']['max_length']
            self.image_size = config['dataset']['image_size']
            
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            self.samples = []
            manifest = os.path.join(self.data_root, 'mimic_cxr_clean.jsonl')
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    print("Loading model...")
    model = BaselineModel(config).to(device)
    
    checkpoint_path = args.checkpoint_path
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = None
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Patch state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.resnet.'):
            new_k = k.replace('encoder.resnet.', 'encoder.encoder.')
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
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
            
            # Generate
            generated_ids = generate_greedy(model, images, tokenizer, max_length=100, device=device)
            
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
    print(f"Baseline Results (Checkpoint: {os.path.basename(checkpoint_path)})")
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
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
