"""
RadGraph Extraction Script for MIMIC-CXR Reports
Extracts entities and relations from reports to build knowledge graph
"""
import sys
import os
sys.path.append(os.getcwd())

import json
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter

def extract_with_radgraph(manifest_path, output_dir):
    """
    Extract RadGraph entities and relations from MIMIC-CXR reports
    
    Args:
        manifest_path: Path to mimic_cxr_clean.jsonl
        output_dir: Directory to save extracted knowledge
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to import radgraph
    try:
        from radgraph import RadGraph
        extractor = RadGraph(cuda=0)  # Use GPU 0
        print("✓ RadGraph loaded successfully")
    except ImportError:
        print("ERROR: RadGraph not installed. Installing from source...")
        print("Run: pip install git+https://github.com/Stanford-AIMI/radgraph.git")
        return
    
    # Load manifest
    print(f"Loading reports from {manifest_path}...")
    reports = []
    with open(manifest_path, 'r') as f:
        for line in tqdm(f, desc="Loading"):
            entry = json.loads(line)
            reports.append({
                'image_path': entry['image_path'],
                'report': entry['report'],
                'split': entry['split']
            })
    
    print(f"Total reports: {len(reports)}")
    
    # Extract entities and relations
    all_graphs = []
    entity_vocab = Counter()
    relation_vocab = Counter()
    
    print("Extracting RadGraph annotations...")
    for i, item in enumerate(tqdm(reports, desc="RadGraph Extraction")):
        report_text = item['report']
        
        # RadGraph inference
        try:
            results_dict = extractor([report_text])  # Returns dict: {0: result}
            result = list(results_dict.values())[0]  # Get first (and only) result
            
            # Parse RadGraph output
            entities = []
            relations = []
            
            if 'entities' in result:
                for ent_id, ent_data in result['entities'].items():
                    entity = {
                        'id': ent_id,
                        'label': ent_data.get('label', ''),
                        'tokens': ent_data.get('tokens', ''),
                        'start_ix': ent_data.get('start_ix', -1),
                        'end_ix': ent_data.get('end_ix', -1)
                    }
                    entities.append(entity)
                    entity_vocab[ent_data.get('label', 'UNKNOWN')] += 1
            
            if 'relations' in result:
                for rel in result['relations']:
                    relation = {
                        'src': rel.get('src', ''),
                        'tgt': rel.get('tgt', ''),
                        'relation': rel.get('relation', '')
                    }
                    relations.append(relation)
                    relation_vocab[rel.get('relation', 'UNKNOWN')] += 1
            
            all_graphs.append({
                'image_path': item['image_path'],
                'split': item['split'],
                'entities': entities,
                'relations': relations
            })
            
        except Exception as e:
            print(f"Error processing report {i}: {e}")
            all_graphs.append({
                'image_path': item['image_path'],
                'split': item['split'],
                'entities': [],
                'relations': []
            })
        
        # Save intermediate results every 10k
        if (i + 1) % 10000 == 0:
            temp_path = os.path.join(output_dir, f'radgraph_temp_{i+1}.jsonl')
            with open(temp_path, 'w') as f:
                for g in all_graphs:
                    f.write(json.dumps(g) + '\n')
            print(f"Saved checkpoint at {i+1} reports")
    
    # Save final results
    output_path = os.path.join(output_dir, 'radgraph_annotations.jsonl')
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for graph in all_graphs:
            f.write(json.dumps(graph) + '\n')
    
    # Save vocabularies
    entity_vocab_path = os.path.join(output_dir, 'entity_vocabulary.json')
    relation_vocab_path = os.path.join(output_dir, 'relation_vocabulary.json')
    
    with open(entity_vocab_path, 'w') as f:
        json.dump(dict(entity_vocab.most_common()), f, indent=2)
    
    with open(relation_vocab_path, 'w') as f:
        json.dump(dict(relation_vocab.most_common()), f, indent=2)
    
    # Print statistics
    print("\n" + "="*50)
    print("RADGRAPH EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total reports processed: {len(all_graphs)}")
    print(f"Unique entity types: {len(entity_vocab)}")
    print(f"Unique relation types: {len(relation_vocab)}")
    print(f"\nTop 10 Entity Types:")
    for ent, count in entity_vocab.most_common(10):
        print(f"  {ent}: {count}")
    print(f"\nTop 10 Relation Types:")
    for rel, count in relation_vocab.most_common(10):
        print(f"  {rel}: {count}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='data/mimic_cxr_clean.jsonl')
    parser.add_argument('--output', type=str, default='data/radgraph')
    args = parser.parse_args()
    
    extract_with_radgraph(args.manifest, args.output)
