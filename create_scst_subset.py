import json
import os
import heapq

# Paths
DATA_ROOT = 'HyperLogicRAG/data'
INPUT_FILE = os.path.join(DATA_ROOT, 'mimic_cxr_clean.jsonl')
OUTPUT_FILE = os.path.join(DATA_ROOT, 'mimic_scst_50k.jsonl')

# Keywords
DISEASE_KEYWORDS = [
    'pneumonia', 'consolidation', 'effusion', 'edema', 'atelectasis', 
    'pneumothorax', 'nodule', 'mass', 'cardiomegaly', 'fracture', 
    'lesion', 'opacity', 'congestion', 'infection', 'tubes', 'lines', 
    'device', 'support', 'severe', 'mild', 'moderate'
]

NEGATIVE_KEYWORDS = [
    'no acute', 'unremarkable', 'normal', 'clear', 'stable', 'unchanged'
]

def score_report(report):
    """
    Score report complexity.
    Higher score = More likely to contain pathology/details.
    """
    report_lower = report.lower()
    words = report_lower.split()
    
    # Base score: Length (longer reports usually have more findings)
    score = len(words)
    
    # Bonus for disease keywords
    for kw in DISEASE_KEYWORDS:
        if kw in report_lower:
            score += 20  # Significant boost for pathology
            
    # Penalize for normal/negative keywords
    for kw in NEGATIVE_KEYWORDS:
        if kw in report_lower:
            score -= 10
            
    return score

def main():
    print(f"Reading from {INPUT_FILE}...")
    
    candidates = []
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    count_all = 0
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry['split'] == 'train':
                    count_all += 1
                    report = entry['report']
                    score = score_report(report)
                    
                    # Store tuple (score, entry) - negate score for min-heap if we wanted top-k with heap
                    # But simpler: just collect all and sort. 220k is small enough for memory.
                    candidates.append((score, entry))
                elif entry['split'] in ['validate', 'test']:
                    # Keep ALL validation/test data (do not filter them!)
                    # We need consistent valuation sets.
                    pass 
            except Exception as e:
                continue

    print(f"Found {count_all} training samples.")
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50,000
    top_50k = candidates[:50000]
    
    print(f"Selected top {len(top_50k)} samples.")
    print(f"  Max score: {top_50k[0][0]}")
    print(f"  Min score: {top_50k[-1][0]}")
    
    # Write to output
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        # 1. Write top 50k TRAIN samples
        for score, entry in top_50k:
            f.write(json.dumps(entry) + '\n')
            
        # 2. Append ALL VALIDATE/TEST samples (read again to find them)
        with open(INPUT_FILE, 'r') as fin:
            for line in fin:
                entry = json.loads(line)
                if entry['split'] != 'train':
                    f.write(line)
                    
    print("Done!")

if __name__ == '__main__':
    main()
