"""
HyperGCN Encoder Module
Using PyTorch Geometric HypergraphConv for encoding clinical knowledge
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv

class HyperGCNEncoder(nn.Module):
    """
    Hypergraph Convolutional Encoder
    
    Encodes hypergraph structure (clinical co-occurrence rules) 
    into entity embeddings that guide report generation
    
    Based on:
    - PyTorch Geometric HypergraphConv (official implementation)
    - DAMPER (AAAI 2025) validation for medical reports
    """
    def __init__(self, num_nodes, embed_dim=512, num_layers=2, dropout=0.1, use_attention=False):
        """
        Args:
            num_nodes: Number of entity nodes (15,814 in our hypergraph)
            embed_dim: Embedding dimension
            num_layers: Number of hypergraph conv layers
            dropout: Dropout rate
            use_attention: Whether to use attention in HypergraphConv
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # Node embedding initialization
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        
        # Hypergraph convolution layers (using PyG official implementation)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HypergraphConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    use_attention=use_attention,
                    heads=1,  # Single attention head
                    concat=False,
                    dropout=dropout,
                    bias=True,
                    # hyperedge_attr_channels will be 1 (just weight scalar)
                )
            )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, entity_predictions, hyperedge_index, hyperedge_weight=None):
        """
        Args:
            entity_predictions: [B, N] tensor, probability of each entity
            hyperedge_index: [2, E] tensor, hypergraph structure
                             [node_idx, hyperedge_idx] pairs
            hyperedge_weight: [E] tensor, optional edge weights
        
        Returns:
            node_embeddings: [B, N, D] refined entity embeddings
        """
        batch_size = entity_predictions.size(0)
        
        # Initialize node features with embeddings
        # Weighted by entity predictions
        node_features = self.node_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]
        
        # Weight by entity predictions (use as attention scores)
        entity_weights = entity_predictions.unsqueeze(-1)  # [B, N, 1]
        node_features = node_features * entity_weights  # [B, N, D]
        
        # Process each sample in batch
        # Note: PyG HypergraphConv expects [N, D] input, not batched
        # So we need to process batch sequentially
        batch_outputs = []
        
        for b in range(batch_size):
            x = node_features[b]  # [N, D]
            
            # Apply hypergraph convolutions
            for conv in self.convs:
                x_residual = x
                # Don't use hyperedge_weight with attention for now (simpler)
                x = conv(x, hyperedge_index)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + x_residual  # Residual connection
                x = self.layer_norm(x)
            
            batch_outputs.append(x)
        
        # Stack batch
        node_embeddings = torch.stack(batch_outputs, dim=0)  # [B, N, D]
        
        return node_embeddings


class EntityPredictor(nn.Module):
    """
    Predicts entity probabilities from visual features
    Used to initialize HyperGCN with image-based entity predictions
    """
    def __init__(self, visual_dim=512, num_entities=15814, hidden_dim=1024):  # Changed from 2048 to 512 (DenseNet)
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_entities),
            nn.Sigmoid()  # Output probabilities
        )
        
        # Initialize last layer bias to 0 (sigmoid(0)=0.5) to prevent initial collapse
        # And use small weights
        nn.init.zeros_(self.mlp[3].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.01)
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: [B, C, H, W] or [B, C] image features
        
        Returns:
            entity_probs: [B, num_entities] entity probabilities
        """
        if len(visual_features.shape) == 4:
            # Global average pooling if 4D
            visual_features = visual_features.mean(dim=[2, 3])  # [B, C]
        
        entity_probs = self.mlp(visual_features)  # [B, num_entities]
        return entity_probs


def test_hypergcn():
    """Test HyperGCN encoder"""
    print("Testing HyperGCN Encoder...")
    
    # Mock data
    batch_size = 2
    num_nodes = 100
    embed_dim = 128
    num_hyperedges = 50
    
    # Create mock hypergraph
    # Each hyperedge connects 3-4 nodes
    hyperedge_index = []
    for e in range(num_hyperedges):
        num_nodes_in_edge = torch.randint(3, 5, (1,)).item()
        nodes = torch.randperm(num_nodes)[:num_nodes_in_edge]
        for node in nodes:
            hyperedge_index.append([node.item(), e])
    
    hyperedge_index = torch.tensor(hyperedge_index).t()  # [2, E]
    hyperedge_weight = torch.rand(num_hyperedges)  # Random weights
    
    # Mock entity predictions
    entity_predictions = torch.rand(batch_size, num_nodes)  # Random probabilities
    
    # Initialize encoder
    encoder = HyperGCNEncoder(
        num_nodes=num_nodes,
        embed_dim=embed_dim,
        num_layers=2,
        use_attention=False  # Keep simple for now
    )
    
    # Forward pass
    print(f"Input entity predictions: {entity_predictions.shape}")
    print(f"Hypergraph structure: {hyperedge_index.shape}")
    
    output = encoder(entity_predictions, hyperedge_index, hyperedge_weight)
    
    print(f"Output embeddings: {output.shape}")
    print("✓ HyperGCN test passed!")
    
    return encoder, output


if __name__ == "__main__":
    test_hypergcn()
