import pandas as pd
import os
import json
import argparse
from tqdm import tqdm

def preprocess(args):
    print("Reading PadChest CSV...")
    df = pd.read_csv(args.csv_path)
    
    # Structure: ImageID, labels...
    # Label columns start from index 3? 'Normal', ...
    # Let's dynamically find label columns.
    # We assume 'ImageID', 'StudyDate_DICOM', 'PatientID' are metadata.
    # Rest are labels.
    
    metadata_cols = ['ImageID', 'StudyDate_DICOM', 'PatientID']
    label_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Found {len(label_cols)} label columns.")
    
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['ImageID']
        
        # Get active labels (value == 1)
        active_labels = []
        for label in label_cols:
            if row[label] == 1:
                active_labels.append(label)
        
        if 'Normal' in active_labels and len(active_labels) > 1:
            active_labels.remove('Normal') # Remove 'Normal' if other pathologies exist
            
        if not active_labels:
            report = "No finding."
        else:
            # Construct pseudo-sentence
            report = "Findings: " + ", ".join(active_labels) + "."
            
        # Image path
        # images in images_dir / image_id
        image_path = os.path.join(args.images_dir, image_id)
        
        records.append({
            "image_path": image_path,
            "report": report,
            "split": "train", # PadChest CXRLT might be all train?
            "study_id": image_id,
            "dataset_source": "padchest"
        })
        
    print(f"Created {len(records)} records for PadChest.")
    
    output_path = os.path.join(args.output_dir, "padchest_clean.jsonl")
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    preprocess(args)
