import argparse
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

import yaml
import json
import numpy as np
from tqdm import tqdm
import torch

def build_index(features_path, manifest_path, output_path, k=1):
    print(f"Loading features from {features_path}...")
    data = np.load(features_path)
    
    train_feats = data['train_feats'] # [N_train, 2048]
    train_ids = data['train_ids'] # [N_train] (paths)
    
    val_feats = data['val_feats']
    val_ids = data['val_ids']
    
    test_feats = data['test_feats']
    test_ids = data['test_ids']
    
    # Load manifest to get reports
    print(f"Loading manifest from {manifest_path}...")
    id_to_report = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            id_to_report[entry['image_path']] = entry['report']
            
    # Build Index Dictionary
    retrieval_index = {}
    
    # Helper for retrieval
    # Since N_train is small (~200k), we can do batched matrix mult using Torch or just Numpy
    # Moving to GPU for speed
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device} for index search")
    
    train_feats_t = torch.from_numpy(train_feats).to(device) # [N, D]
    
    # 1. Processing Train Set (Leave-One-Out)
    print("Building index for Training set (Leave-One-Out)...")
    batch_size = 1000
    num_train = train_feats.shape[0]
    
    for i in tqdm(range(0, num_train, batch_size)):
        end = min(i + batch_size, num_train)
        queries = train_feats_t[i:end] # [B, D]
        
        # Cosine Sim: Queries are normalized, Train is normalized -> Dot product
        # scores: [B, N_train]
        scores = torch.mm(queries, train_feats_t.T)
        
        # Mask self (diagonal for this block if we were doing full matrix)
        # But here queries are a slice.
        # We need to mask the identity.
        # Global indices for this batch: range(i, end)
        
        # For each query k in batch, valid indices are ALL except k + i
        # Simple way: get top 2, if top 1 is self, take top 2.
        
        topk = torch.topk(scores, k=k+1, dim=1)
        indices = topk.indices.cpu().numpy() # [B, k+1]
        
        for idx_in_batch, global_indices in enumerate(indices):
            query_global_idx = i + idx_in_batch
            
            # Find best match that is NOT self
            best_idx = -1
            for candidate_idx in global_indices:
                if candidate_idx != query_global_idx:
                    best_idx = candidate_idx
                    break
            
            if best_idx != -1:
                 # Map IDs
                query_id = train_ids[query_global_idx]
                retrieved_id = train_ids[best_idx]
                retrieved_report = id_to_report.get(retrieved_id, "")
                retrieval_index[query_id] = retrieved_report
    
    # 2. Processing Val/Test Set (Search in full Train)
    for split_name, feats, ids in [('validate', val_feats, val_ids), ('test', test_feats, test_ids)]:
        if feats is None: continue
        print(f"Building index for {split_name}...")
        
        feats_t = torch.from_numpy(feats).to(device)
        num_samples = feats.shape[0]
        
        for i in tqdm(range(0, num_samples, batch_size)):
            end = min(i + batch_size, num_samples)
            queries = feats_t[i:end]
            scores = torch.mm(queries, train_feats_t.T)
            
            topk = torch.topk(scores, k=k, dim=1)
            indices = topk.indices.cpu().numpy()
            
            for idx_in_batch, global_indices in enumerate(indices):
                best_idx = global_indices[0] # Just top 1
                
                query_id = ids[i + idx_in_batch]
                retrieved_id = train_ids[best_idx]
                retrieved_report = id_to_report.get(retrieved_id, "")
                retrieval_index[query_id] = retrieved_report

    # Save
    print(f"Saving index with {len(retrieval_index)} entries to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(retrieval_index, f)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='results/features.npz')
    parser.add_argument('--manifest', type=str, default='data/mimic_cxr_clean.jsonl')
    parser.add_argument('--output', type=str, default='data/mimic_cxr/rag_index.json')
    args = parser.parse_args()
    
    build_index(args.features, args.manifest, args.output)
