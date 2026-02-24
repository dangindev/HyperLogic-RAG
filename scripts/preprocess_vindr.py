import pandas as pd
import os
import json
import argparse
from tqdm import tqdm

def preprocess(args):
    # VinDr-CXR structure:
    # train.csv
    # train/ (images - dicom or jpg? User listed separate folder, likely jpgs or dicoms)
    # The list_dir showed `train ` and `train.csv`.
    
    print("Reading VinDr-CXR train CSV...")
    df = pd.read_csv(args.train_csv)
    
    # VinDr is detection mainly. 'class_name', 'rad_id', ...
    # It does NOT have full reports. It has 'findings' sometimes or just labels?
    # VinDr-CXR typically has bounding boxes.
    # Re-reading prompt: "VinDr-CXR (có thể có bbox/labels)".
    # If no report, we can construct a pseudo-report from labels or skip?
    # Or maybe we use it for 'Auxiliary Task' (Detection/Classification).
    
    # Checking dataset columns usually: image_id, class_name, x_min, ...
    # If 'class_name' is the only text, we can make a dummy report: "Findings: [class_name]."
    
    # Let's aggregate labels per image.
    
    # Group by image_id
    grouped = df.groupby('image_id')['class_name'].apply(lambda x: list(set(x))).reset_index()
    
    records = []
    
    for idx, row in tqdm(grouped.iterrows(), total=len(grouped)):
        image_id = row['image_id']
        labels = row['class_name']
        
        # Filter "No finding"
        labels = [l for l in labels if l != 'No finding']
        if not labels:
            report = "No finding."
        else:
            report = "Findings: " + ", ".join(labels) + "."
            
        # Image path: assuming jpg in train/
        # Check if dicom or jpg
        image_path = os.path.join(args.images_dir, f"{image_id}.jpg") 
        # Modify extension if needed based on actual files
        
        records.append({
            "image_path": image_path,
            "report": report, # Pseudo-report
            "split": "train",
            "study_id": image_id,
            "dataset_source": "vindr_cxr"
        })
        
    print(f"Created {len(records)} records for VinDr-CXR.")
    
    output_path = os.path.join(args.output_dir, "vindr_cxr_clean.jsonl")
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    preprocess(args)
