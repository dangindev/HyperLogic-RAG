"""
Hypergraph Construction via Association Rule Mining
Builds hypergraph from RadGraph entity co-occurrences
"""
import sys
import os
sys.path.append(os.getcwd())

import json
import pickle
from collections import defaultdict, Counter
from itertools import combinations
import argparse
from tqdm import tqdm
import numpy as np

class HypergraphBuilder:
    def __init__(self, min_support=0.01, min_confidence=0.5):
        """
        Args:
            min_support: Minimum support threshold (fraction of reports)
            min_confidence: Minimum confidence for association rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # Storage
        self.entity_vocab = set()
        self.entity_counts = Counter()
        self.cooccurrence_counts = defaultdict(int)
        self.total_reports = 0
        
        # Hypergraph components
        self.positive_hyperedges = []  # Co-occurrence patterns
        self.negative_hyperedges = []  # Conflict patterns
        self.node_to_id = {}
        self.id_to_node = {}
    
    def load_radgraph_data(self, radgraph_file):
        """Load RadGraph annotations"""
        print(f"Loading RadGraph data from {radgraph_file}...")
        
        entity_sets = []
        
        with open(radgraph_file, 'r') as f:
            for line in tqdm(f, desc="Reading"):
                data = json.loads(line)
                
                # Extract entity labels (focus on observations)
                entities = []
                for ent in data['entities']:
                    label = ent['label']
                    tokens = ent['tokens'].lower().strip()
                    
                    # Only keep meaningful observations (not anatomy alone)
                    if 'Observation' in label:
                        # Standardize entity representation
                        entity_key = f"{tokens}::{label.split('::')[1]}"  # e.g., "cardiomegaly::definitely present"
                        entities.append(entity_key)
                        self.entity_vocab.add(entity_key)
                        self.entity_counts[entity_key] += 1
                
                if entities:
                    entity_sets.append(set(entities))
        
        self.total_reports = len(entity_sets)
        print(f"✓ Loaded {self.total_reports} reports")
        print(f"✓ Found {len(self.entity_vocab)} unique entities")
        
        return entity_sets
    
    def mine_frequent_itemsets(self, entity_sets, max_size=3):
        """
        Mine frequent itemsets using Apriori-like algorithm
        """
        print(f"\nMining frequent itemsets (min_support={self.min_support})...")
        
        min_count = int(self.min_support * self.total_reports)
        print(f"Minimum occurrence count: {min_count} reports")
        
        # Count co-occurrences of all pairs, triples, etc.
        frequent_itemsets = defaultdict(list)
        
        # Size 2 (pairs)
        print("  Mining pairs...")
        for entities in tqdm(entity_sets, desc="  Pairs"):
            if len(entities) < 2:
                continue
            for pair in combinations(sorted(entities), 2):
                self.cooccurrence_counts[pair] += 1
        
        # Filter by support
        for itemset, count in self.cooccurrence_counts.items():
            if count >= min_count:
                frequent_itemsets[2].append((itemset, count))
        
        print(f"    Found {len(frequent_itemsets[2])} frequent pairs")
        
        # Size 3 (triples)
        if max_size >= 3:
            print("  Mining triples...")
            triple_counts = defaultdict(int)
            for entities in tqdm(entity_sets, desc="  Triples"):
                if len(entities) < 3:
                    continue
                for triple in combinations(sorted(entities), 3):
                    # Check if all pairs in this triple are frequent
                    pairs = [tuple(sorted(p)) for p in combinations(triple, 2)]
                    if all(self.cooccurrence_counts.get(p, 0) >= min_count for p in pairs):
                        triple_counts[triple] += 1
            
            for itemset, count in triple_counts.items():
                if count >= min_count:
                    frequent_itemsets[3].append((itemset, count))
            
            print(f"    Found {len(frequent_itemsets[3])} frequent triples")
        
        # Size 4 (4-way)
        if max_size >= 4:
            print("  Mining 4-itemsets...")
            quad_counts = defaultdict(int)
            for entities in tqdm(entity_sets, desc="  4-itemsets"):
                if len(entities) < 4:
                    continue
                for quad in combinations(sorted(entities), 4):
                    # Check if represented in smaller itemsets
                    triples = [tuple(sorted(t)) for t in combinations(quad, 3)]
                    if any(t in [x[0] for x in frequent_itemsets[3]] for t in triples):
                        quad_counts[quad] += 1
            
            for itemset, count in quad_counts.items():
                if count >= min_count:
                    frequent_itemsets[4].append((itemset, count))
            
            print(f"    Found {len(frequent_itemsets[4])} frequent 4-itemsets")
        
        return frequent_itemsets
    
    def build_hypergraph(self, frequent_itemsets):
        """
        Convert frequent itemsets to hypergraph structure
        """
        print("\nBuilding hypergraph...")
        
        # Create node vocabulary
        for entity in self.entity_vocab:
            node_id = len(self.node_to_id)
            self.node_to_id[entity] = node_id
            self.id_to_node[node_id] = entity
        
        print(f"  Nodes: {len(self.node_to_id)}")
        
        # Create positive hyperedges (co-occurrence patterns)
        for size in sorted(frequent_itemsets.keys()):
            for itemset, count in frequent_itemsets[size]:
                support = count / self.total_reports
                
                # Convert to node IDs
                nodes = [self.node_to_id[entity] for entity in itemset]
                
                self.positive_hyperedges.append({
                    'nodes': nodes,
                    'entities': list(itemset),
                    'support': support,
                    'count': count,
                    'type': 'co-occurrence'
                })
        
        print(f"  Positive hyperedges: {len(self.positive_hyperedges)}")
        
        # Create negative hyperedges (conflict patterns)
        # Example: "normal" vs "abnormal" findings
        print("  Mining conflict patterns...")
        conflicts_found = 0
        
        # Find present/absent pairs of same entity
        entity_base_names = defaultdict(list)
        for entity in self.entity_vocab:
            base_name = entity.split('::')[0]
            entity_base_names[base_name].append(entity)
        
        for base_name, variants in entity_base_names.items():
            if len(variants) > 1:
                # Check if present and absent variants exist
                present_var = [v for v in variants if 'present' in v and 'absent' not in v]
                absent_var = [v for v in variants if 'absent' in v]
                
                if present_var and absent_var:
                    # These conflict
                    for p in present_var:
                        for a in absent_var:
                            nodes = [self.node_to_id[p], self.node_to_id[a]]
                            self.negative_hyperedges.append({
                                'nodes': nodes,
                                'entities': [p, a],
                                'type': 'conflict',
                                'reason': f'{base_name} cannot be both present and absent'
                            })
                            conflicts_found += 1
        
        print(f"  Negative hyperedges (conflicts): {len(self.negative_hyperedges)}")
        
        return {
            'nodes': self.id_to_node,
            'node_to_id': self.node_to_id,
            'positive_hyperedges': self.positive_hyperedges,
            'negative_hyperedges': self.negative_hyperedges,
            'entity_counts': dict(self.entity_counts),
            'total_reports': self.total_reports,
            'params': {
                'min_support': self.min_support,
                'min_confidence': self.min_confidence
            }
        }

def main(radgraph_file, output_file, min_support=0.01):
    builder = HypergraphBuilder(min_support=min_support)
    
    # Load data
    entity_sets = builder.load_radgraph_data(radgraph_file)
    
    # Mine patterns
    frequent_itemsets = builder.mine_frequent_itemsets(entity_sets, max_size=3)
    
    # Build hypergraph
    hypergraph = builder.build_hypergraph(frequent_itemsets)
    
    # Save
    print(f"\nSaving hypergraph to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(hypergraph, f)
    
    print("\n" + "="*60)
    print("HYPERGRAPH CONSTRUCTION COMPLETE")
    print("="*60)
    print(f"Total nodes: {len(hypergraph['nodes'])}")
    print(f"Positive hyperedges: {len(hypergraph['positive_hyperedges'])}")
    print(f"Negative hyperedges: {len(hypergraph['negative_hyperedges'])}")
    print(f"Total reports analyzed: {hypergraph['total_reports']}")
    print("="*60)
    
    # Show examples
    print("\nExample Positive Hyperedges (Co-occurrence patterns):")
    for i, hedge in enumerate(hypergraph['positive_hyperedges'][:5]):
        print(f"\n{i+1}. {hedge['entities']}")
        print(f"   Support: {hedge['support']:.3f} ({hedge['count']} reports)")
    
    print("\nExample Negative Hyperedges (Conflict patterns):")
    for i, hedge in enumerate(hypergraph['negative_hyperedges'][:5]):
        print(f"\n{i+1}. {hedge['entities']}")
        print(f"   Reason: {hedge['reason']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--radgraph', type=str, default='data/radgraph/radgraph_annotations.jsonl')
    parser.add_argument('--output', type=str, default='data/hypergraph.pkl')
    parser.add_argument('--min_support', type=float, default=0.01, help='Minimum support (0.01 = 1%)')
    args = parser.parse_args()
    
    main(args.radgraph, args.output, args.min_support)
