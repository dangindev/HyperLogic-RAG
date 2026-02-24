import os
import json
import argparse
from tqdm import tqdm
import re

def preprocess(args):
    # IU-CXR new structure: annotation.json with "train": [...]
    
    print(f"Reading {args.annotation_json}...")
    with open(args.annotation_json, 'r') as f:
        data = json.load(f)
        
    # data is {"train": [...], "val": [...], "test": [...]} or just "train"?
    # The snippet showed '{"train": [{'
    # I will assume it might have splits.
    
    records = []
    
    for split_name, samples in data.items():
        print(f"Processing split: {split_name} with {len(samples)} samples")
        
        for sample in tqdm(samples):
            # sample keys: id, report, image_path (list), split
            report = sample.get('report', '')
            image_paths = sample.get('image_path', [])
            sample_id = sample.get('id')
            
            # Clean report
            report = re.sub(r'\s+', ' ', report).strip()
            
            for img_rel_path in image_paths:
                # image_path is list of strings like "CXR2384_IM-0942/0.png"
                # absolute path = images_dir / img_rel_path
                abs_path = os.path.join(args.images_dir, img_rel_path)
                
                records.append({
                    "image_path": abs_path,
                    "report": report,
                    "split": split_name if split_name in ['train', 'val', 'test'] else 'train',
                    "study_id": sample_id,
                    "dataset_source": "iu_cxr"
                })

    print(f"Created {len(records)} records for IU-CXR.")
    
    output_path = os.path.join(args.output_dir, "iu_cxr_clean.jsonl")
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    preprocess(args)
