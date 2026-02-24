"""
Relational Memory module adapted from R2Gen
for conditioning decoder with learned memory patterns.

Reference: R2Gen (Chen et al.)
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class RelationalMemory(nn.Module):
    """
    Relational Memory module from R2Gen.
    
    Uses LSTM-style gating with self-attention over memory slots
    to learn common patterns in medical reports.
    
    Args:
        num_slots: Number of memory slots
        d_model: Dimension of each memory slot
        num_heads: Number of attention heads
    """
    def __init__(self, num_slots=3, d_model=512, num_heads=8):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU()
        )

        # LSTM-style gates
        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        """Initialize memory slots as identity-like patterns."""
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward_step(self, input, memory):
        """Single step of memory update."""
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        
        # Self-attention over memory + input
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        # LSTM-style gating
        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        """
        Process sequence through memory.
        
        Args:
            inputs: [B, T, D] input sequence
            memory: [B, num_slots * d_model] initial memory
        
        Returns:
            outputs: [B, T, num_slots * d_model] memory states at each step
        """
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class ConditionalLayerNorm(nn.Module):
    """
    Layer Normalization conditioned on Relational Memory.
    
    The memory modulates gamma and beta of LayerNorm,
    allowing the decoder to adapt based on learned patterns.
    """
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        # MLP to compute gamma/beta adjustments from memory
        self.mlp_gamma = nn.Sequential(
            nn.Linear(rm_num_slots * rm_d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.mlp_beta = nn.Sequential(
            nn.Linear(rm_num_slots * rm_d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        """
        Args:
            x: [B, T, D] input
            memory: [B, T, num_slots * rm_d_model] memory context
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Memory-conditioned adjustments
        delta_gamma = self.mlp_gamma(memory)  # [B, T, D]
        delta_beta = self.mlp_beta(memory)    # [B, T, D]
        
        # Expand base gamma/beta to match batch
        gamma_hat = self.gamma.unsqueeze(0).unsqueeze(0).expand_as(x)
        beta_hat = self.beta.unsqueeze(0).unsqueeze(0).expand_as(x)
        
        # Add memory adjustments
        gamma_hat = gamma_hat + delta_gamma
        beta_hat = beta_hat + delta_beta
        
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


if __name__ == "__main__":
    # Test
    print("Testing RelationalMemory...")
    rm = RelationalMemory(num_slots=3, d_model=512, num_heads=8)
    
    batch_size = 4
    seq_len = 10
    
    inputs = torch.randn(batch_size, seq_len, 512)
    memory = rm.init_memory(batch_size).to(inputs.device)
    
    outputs = rm(inputs, memory)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected: [4, 10, {3 * 512}]")
    assert outputs.shape == (batch_size, seq_len, 3 * 512)
    print("✓ Test passed!")
