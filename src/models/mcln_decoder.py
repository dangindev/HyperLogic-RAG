import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.models.relational_memory import ConditionalLayerNorm, MultiHeadedAttention

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ConditionalSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as in the paper.
    """
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        "Apply residual connection to any sublayer with the same size."
        # R2Gen Implementation: x + dropout(sublayer(norm(x, memory)))
        # Note: norm takes (x, memory)
        return x + self.dropout(sublayer(self.norm(x, memory)))

class MCLNDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rm_num_slots, rm_d_model):
        super(MCLNDecoderLayer, self).__init__()
        self.d_model = d_model
        # Use MultiHeadedAttention from relational_memory.py
        self.self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.src_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        
        # 3 sublayers: Self-Attn, Cross-Attn, FF
        # All conditioned on memory via MCLN
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        """
        x: [B, T, D] (target sequence)
        hidden_states: [B, S, D] (encoder output - visual features)
        src_mask: [B, 1, S] (mask for encoder output)
        tgt_mask: [B, 1, T, T] (mask for target sequence)
        memory: [B, T, Slots*D] (Relational Memory state)
        """
        m = hidden_states
        
        # 1. Self Attention (Query=x, Key=x, Value=x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0], memory)
        
        # 2. Cross Attention (Query=x, Key=m, Value=m)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)[0], memory)
        
        # 3. Feed Forward
        x = self.sublayer[2](x, self.feed_forward, memory)
        
        return x

class MCLNDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, rm_num_slots, rm_d_model):
        super(MCLNDecoder, self).__init__()
        layer = MCLNDecoderLayer(d_model, nhead, dim_feedforward, dropout, rm_num_slots, rm_d_model)
        self.layers = clones(layer, num_layers)
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        
    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        """
        x: [B, T, D] - Target Embeddings
        hidden_states: [B, S, D] - Encoder Output
        src_mask: Mask for Encoder Output
        tgt_mask: Mask for Target
        memory: Relational Memory
        """
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x, memory)
