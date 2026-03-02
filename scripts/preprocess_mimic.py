import pandas as pd
import os
import json
import zipfile
import argparse
from tqdm import tqdm
import re

def preprocess(args):
    print("Loading splits and metadata...")
    split_df = pd.read_csv(args.split_csv)
    meta_df = pd.read_csv(args.metadata_csv)
    
    # Merge split and metadata
    # Both have dicom_id, subject_id, study_id
    df = pd.merge(split_df, meta_df, on=['subject_id', 'study_id', 'dicom_id'])
    
    # Filter for PA/AP views only if desired? For now keep all.
    # df = df[df['ViewPosition'].isin(['PA', 'AP'])]
    
    print(f"Total images: {len(df)}")
    
    # Reports are typically in pXX/pXXXXXX/sXXXXXX.txt within the zip
    # We will read them on the fly or extract.
    # Let's assume we extract them to a temporary location or read from zip if possible.
    # Reading 200GB zip randomly is slow. Better to unzip once manually or assume unzipped.
    # The user directory showed 'mimic-cxr-reports.zip'.
    # Let's try to read from zip to avoid massive inode usage if possible, 
    # OR better, create a dictionary of reports first.
    
    reports = {}
    print("Reading reports from zip...")
    try:
        with zipfile.ZipFile(args.reports_zip, 'r') as z:
            for filename in tqdm(z.namelist()):
                if filename.endswith('.txt'):
                    # filename like: mimic-cxr-reports/files/p10/p10000032/s50414267.txt
                    # Extract study_id from filename
                    try:
                        study_id = int(os.path.basename(filename).replace('s', '').replace('.txt', ''))
                        text = z.read(filename).decode('utf-8')
                        # Simple extraction of Findings/Impression
                        # This is a naive extraction.
                        reports[study_id] = text
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading zip: {e}")
        return

    print(f"Loaded {len(reports)} reports.")
    
    # Create final records
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        subject_id = row['subject_id']
        split = row['split']
        
        # Image path construction (MIMIC structure)
        # files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        # The user has `images/` folder. It usually mimics the structure?
        # Let's verify structure later. Assuming standard MIMIC-JPG structure.
        p_group = f"p{str(subject_id)[:2]}"
        p_subject = f"p{subject_id}"
        image_rel_path = os.path.join(p_group, p_subject, f"s{study_id}", f"{dicom_id}.jpg")
        image_abs_path = os.path.join(args.images_root, image_rel_path)
        
        if study_id in reports:
            report_text = reports[study_id]
            # Robust Cleaning
            # 1. Remove de-identification placeholders regex [** ... **]
            report_text = re.sub(r'\[\*\*.*?\*\*\]', '', report_text)
            # 2. Remove multiple spaces/newlines
            report_text = re.sub(r'\s+', ' ', report_text).strip()
            
            records.append({
                "image_path": image_abs_path,
                "report": report_text,
                "split": split,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "dataset_source": "mimic_cxr"
            })
    
    print(f"Created {len(records)} records.")
    
    # Save to JSONL
    output_path = os.path.join(args.output_dir, "mimic_cxr_clean.jsonl")
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--metadata_csv', type=str, required=True)
    parser.add_argument('--reports_zip', type=str, required=True)
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    preprocess(args)
