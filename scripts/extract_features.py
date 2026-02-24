import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
# from torchvision import transforms # REMOVED
import os
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
from PIL import Image

# Re-use the dataset class logic properly or import it?
# Let's import the dataset class to ensure consistency in preprocessing (Resize, Normalize)
from src.datasets import get_dataloader, MIMICCXRDataset

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use same backbone as baseline: Microsoft ResNet-50
        self.resnet = AutoModel.from_pretrained("microsoft/resnet-50")
        self.resnet.eval() 

    def forward(self, images):
        with torch.no_grad():
            outputs = self.resnet(images)
            # pooler_output: [B, 2048, 1, 1] -> [B, 2048]
            features = outputs.pooler_output.flatten(1)
            # L2 Normalize for Cosine Similarity later
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features

def extract_features(config_path, output_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force batch size and workers for extraction
    config['dataset']['batch_size'] = 64
    config['dataset']['num_workers'] = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # We only need the TRAIN set for the index (retrieval database)
    # But usually for RAG we retrieve from TRAIN to generate for VAL/TEST.
    # So we need to build Index from TRAIN.
    # We also need to extract query features for VAL/TEST later? 
    # Actually, for training the RAG model, we need to retrieve for each training sample too (excluding itself).
    
    # Let's extract features for ALL splits.
    
    model = FeatureExtractor().to(device)
    
    splits = ['train', 'validate', 'test']
    
    all_features = {}
    all_ids = {} # map index to sample ID (dicom_id or study_id)
    
    manifest_file = config['dataset'].get('manifest_file', 'mimic_cxr_clean.jsonl')
    data_root = config['dataset']['data_root']
    
    # We will instantiate dataset manually to iterate all splits
    # Or just use get_dataloader
    
    for split in splits:
        print(f"Processing split: {split}")
        # Note: We need a way to get IDs. verify dataset returns IDs
        # Our current dataset __getitem__ returns: {'image': ..., 'image_path': ..., 'report': ..., 'input_ids': ...}
        # It has `image_path` which can serve as ID.
        
        try:
            dataset = MIMICCXRDataset(config, split=split)
            # Create a simple loader
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=config['dataset']['batch_size'], 
                shuffle=False, 
                num_workers=config['dataset']['num_workers'],
                pin_memory=True
            )
        except Exception as e:
            print(f"Skipping split {split} due to error or empty: {e}")
            continue
            
        feature_list = []
        path_list = []
        
        for batch in tqdm(loader, desc=f"Extracting {split}"):
            images = batch['image'].to(device)
            paths = batch['image_path'] # list of strings
            
            feats = model(images)
            feature_list.append(feats.cpu().numpy())
            path_list.extend(paths)
            
        if feature_list:
            all_features[split] = np.concatenate(feature_list, axis=0) # [N, 2048]
            all_ids[split] = path_list
            print(f"Saved {all_features[split].shape[0]} features for {split}")
            
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, 
                        train_feats=all_features.get('train'), 
                        train_ids=all_ids.get('train'),
                        val_feats=all_features.get('validate'),
                        val_ids=all_ids.get('validate'),
                        test_feats=all_features.get('test'),
                        test_ids=all_ids.get('test'))
    print(f"All features saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline_config.yaml')
    parser.add_argument('--output', type=str, default='results/features.npz')
    args = parser.parse_args()
    
    extract_features(args.config, args.output)
