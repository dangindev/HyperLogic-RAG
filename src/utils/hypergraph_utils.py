"""
Hypergraph Preprocessing Utilities
Convert hypergraph.pkl to PyTorch Geometric format
"""
import pickle
import torch
import numpy as np

def load_and_convert_hypergraph(hypergraph_path):
    """
    Load hypergraph and convert to PyG format
    
    Args:
        hypergraph_path: Path to hypergraph.pkl
    
    Returns:
        dict with:
            - hyperedge_index: [2, E] tensor
            - hyperedge_weight: [E] tensor  
            - num_nodes: int
            - node_to_id: dict mapping entity name → ID
            - id_to_node: dict mapping ID → entity name
            - negative_hyperedges: list of conflict pairs
    """
    print(f"Loading hypergraph from {hypergraph_path}...")
    
    with open(hypergraph_path, 'rb') as f:
        hypergraph = pickle.load(f)
    
    num_nodes = len(hypergraph['nodes'])
    node_to_id = hypergraph['node_to_id']
    id_to_node = hypergraph['nodes']
    
    print(f"✓ Loaded {num_nodes} nodes")
    print(f"✓ {len(hypergraph['positive_hyperedges'])} positive hyperedges")
    print(f"✓ {len(hypergraph['negative_hyperedges'])} negative hyperedges")
    
    # Convert positive hyperedges to PyG format
    edges = []
    weights = []
    
    for hedge_idx, hedge in enumerate(hypergraph['positive_hyperedges']):
        nodes = hedge['nodes']
        support = hedge['support']
        
        # Each node in the hyperedge connects to this hyperedge ID
        for node_id in nodes:
            edges.append([node_id, hedge_idx])
        
        weights.append(support)
    
    hyperedge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
    hyperedge_weight = torch.tensor(weights, dtype=torch.float)   # [num_hyperedges]
    
    print(f"✓ Converted to PyG format:")
    print(f"  hyperedge_index: {hyperedge_index.shape}")
    print(f"  hyperedge_weight: {hyperedge_weight.shape}")
    
    # Extract negative hyperedges (conflict patterns)
    negative_hyperedges = hypergraph['negative_hyperedges']
    
    return {
        'hyperedge_index': hyperedge_index,
        'hyperedge_weight': hyperedge_weight,
        'num_nodes': num_nodes,
        'node_to_id': node_to_id,
        'id_to_node': id_to_node,
        'negative_hyperedges': negative_hyperedges,
        'total_reports': hypergraph['total_reports'],
        'entity_counts': hypergraph['entity_counts']
    }


def logic_constrained_loss(entity_predictions, negative_hyperedges, threshold=0.5):
    """
    Compute logic-constrained loss using CONTINUOUS probabilities
    Penalizes violations of negative hyperedges (conflicts)
    
    DEPRECATED: This version allows model to set entities to 0 to avoid penalty.
    Use logic_constrained_loss_v2 instead.
    
    Args:
        entity_predictions: [B, N] tensor of entity probabilities
        negative_hyperedges: list of negative hyperedge dicts
        threshold: DEPRECATED - kept for compatibility but not used
    
    Returns:
        loss: scalar penalty for conflicts
    """
    if len(negative_hyperedges) == 0:
        return torch.tensor(0.0, device=entity_predictions.device)
    
    total_violation = 0.0
    num_conflicts = 0
    
    for hedge in negative_hyperedges:
        nodes = hedge['nodes']
        if len(nodes) != 2:
            continue  # Only handle pairwise conflicts for now
        
        node_a, node_b = nodes
        
        # Use CONTINUOUS probabilities (differentiable!)
        # This enables gradient flow unlike binary threshold
        prob_a = entity_predictions[:, node_a]  # [B]
        prob_b = entity_predictions[:, node_b]  # [B]
        
        # Penalty if BOTH conflicting entities are activated
        # Using product of probabilities (smooth, differentiable)
        violation = prob_a * prob_b
        total_violation += violation.mean()  # Average over batch
        num_conflicts += 1
    
    if num_conflicts == 0:
        return torch.tensor(0.0, device=entity_predictions.device)
    
    # Normalize by number of conflict pairs
    return total_violation / num_conflicts


def logic_constrained_loss_v2(entity_predictions, negative_hyperedges, suppression_weight=1.0, min_prob=0.1):
    """
    IMPROVED logic loss with suppression penalty
    
    Prevents model from setting conflict entities to 0 as a trick to minimize loss.
    
    Loss = conflict_penalty + suppression_weight * suppression_penalty
    
    Where:
    - conflict_penalty: prob_a * prob_b (penalize if BOTH high)
    - suppression_penalty: relu(min_prob - prob_a) + relu(min_prob - prob_b)
                          (penalize if EITHER too low)
    
    Args:
        entity_predictions: [B, N] tensor of entity probabilities
        negative_hyperedges: list of negative hyperedge dicts
        suppression_weight: Weight for suppression penalty (default: 1.0)
        min_prob: Minimum acceptable probability (default: 0.1)
    
    Returns:
        loss: scalar penalty encouraging factual predictions without conflicts
    """
    if len(negative_hyperedges) == 0:
        return torch.tensor(0.0, device=entity_predictions.device)
    
    import torch.nn.functional as F
    
    total_conflict = 0.0
    total_suppression = 0.0
    num_conflicts = 0
    
    for hedge in negative_hyperedges:
        nodes = hedge['nodes']
        if len(nodes) != 2:
            continue
        
        node_a, node_b = nodes
        prob_a = entity_predictions[:, node_a]  # [B]
        prob_b = entity_predictions[:, node_b]  # [B]
        
        # Component 1: Conflict penalty (if BOTH high)
        conflict = prob_a * prob_b
        total_conflict += conflict.mean()
        
        # Component 2: Suppression penalty (if EITHER too low)
        # This prevents model from setting entities to 0 to avoid conflicts
        suppress_a = F.relu(min_prob - prob_a)  # Penalty if prob_a < min_prob
        suppress_b = F.relu(min_prob - prob_b)  # Penalty if prob_b < min_prob
        total_suppression += (suppress_a.mean() + suppress_b.mean())
        
        num_conflicts += 1
    
    if num_conflicts == 0:
        return torch.tensor(0.0, device=entity_predictions.device)
    
    # Normalize and combine
    conflict_loss = total_conflict / num_conflicts
    suppression_loss = total_suppression / num_conflicts
    
    total_loss = conflict_loss + suppression_weight * suppression_loss
    
    return total_loss


if __name__ == "__main__":
    # Test conversion
    import sys
    import os
    
    hypergraph_path = "data/hypergraph.pkl"
    
    if os.path.exists(hypergraph_path):
        print("="*60)
        print("TESTING HYPERGRAPH CONVERSION")
        print("="*60)
        
        result = load_and_convert_hypergraph(hypergraph_path)
        
        print("\n" + "="*60)
        print("CONVERSION SUCCESS!")
        print("="*60)
        print(f"Nodes: {result['num_nodes']}")
        print(f"Total reports: {result['total_reports']}")
        print(f"Hyperedge index shape: {result['hyperedge_index'].shape}")
        print(f"Hyperedge weight shape: {result['hyperedge_weight'].shape}")
        print(f"Negative hyperedges: {len(result['negative_hyperedges'])}")
        
        # Test logic loss
        print("\n" + "="*60)
        print("TESTING LOGIC LOSS")
        print("="*60)
        
        # Simulate entity predictions
        batch_size = 4
        entity_preds = torch.rand(batch_size, result['num_nodes'])
        
        loss = logic_constrained_loss(
            entity_preds, 
            result['negative_hyperedges'],
            threshold=0.5
        )
        
        print(f"Entity predictions: {entity_preds.shape}")
        print(f"Logic loss: {loss.item():.4f}")
        print("✓ Logic loss computation successful!")
        
        # Show some example entity names
        print("\n" + "="*60)
        print("EXAMPLE ENTITIES (first 10)")
        print("="*60)
        for i in range(min(10, result['num_nodes'])):
            entity_name = result['id_to_node'][i]
            count = result['entity_counts'].get(entity_name, 0)
            print(f"{i}: {entity_name} (count: {count})")
        
    else:
        print(f"❌ Hypergraph file not found: {hypergraph_path}")
        print("Please run build_hypergraph.py first!")
