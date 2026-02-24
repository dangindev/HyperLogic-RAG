import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

# Import original modules to reuse standard parts
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.att_model import AttModel, pack_wrapper
from R2Gen.models.r2gen import R2GenModel

# --- Modified Transformer Components ---

class HyperRelationalMemory(nn.Module):
    def __init__(self, num_slots, d_model, num_heads=1):
        super(HyperRelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size, knowledge_emb=None):
        """
        Initialize memory. If knowledge_emb provided, inject it.
        knowledge_emb: [B, K, D]
        """
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        
        # Inject Knowledge
        if knowledge_emb is not None:
            device = knowledge_emb.device
            memory = memory.to(device)
            
            # Crop RAG/Graph embeddings to fit slots or overwrite slots
            # Strategy: Overwrite first K slots with Knowledge
            K = min(knowledge_emb.size(1), self.num_slots)
            # Use data from knowledge, keep identity for the rest
            memory[:, :K, :] = knowledge_emb[:, :K, :]
        
        # Pad if needed (standard logic)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff)).to(memory.device)
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model) # Flatten for CLN

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)
        return outputs

# --- Standard Transformer Parts (re-implemented to avoid partial imports) ---

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None: mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model), nn.ReLU(inplace=True), nn.Linear(rm_d_model, rm_d_model))
        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Determine if memory is static [B, F] or dynamic [B, L, F]
        if memory.dim() == 2:
             # Static memory: [B, F]
             delta_gamma = self.mlp_gamma(memory) # [B, D]
             delta_beta = self.mlp_beta(memory)   # [B, D]
             delta_gamma = delta_gamma.unsqueeze(1) # [B, 1, D]
             delta_beta = delta_beta.unsqueeze(1)   # [B, 1, D]
        else:
             # Dynamic memory: [B, L, F]
             delta_gamma = self.mlp_gamma(memory) # [B, L, D]
             delta_beta = self.mlp_beta(memory)   # [B, L, D]
        
        gamma_hat = self.gamma.view(1, 1, -1) + delta_gamma
        beta_hat = self.beta.view(1, 1, -1) + delta_beta
        
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat

class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)
    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)

class HyperTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(HyperTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask, knowledge_emb=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, knowledge_emb)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, knowledge_emb=None):
        memory = self.rm.init_memory(hidden_states.size(0), knowledge_emb).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)

class HyperR2GenModel(R2GenModel):
    def __init__(self, args, tokenizer):
        super(HyperR2GenModel, self).__init__(args, tokenizer)
        # Override encoder_decoder
        self.encoder_decoder = HyperEncoderDecoder(args, tokenizer)
        
        # HyperGraph Projector
        # If HG dim != d_model, project it
        self.hg_projector = nn.Linear(512, args.d_model) # Assuming HG_dim=512

    def forward_mimic_cxr(self, images, targets=None, mode='train', knowledge_emb=None, hypergraph_indices=None):
        att_feats, fc_feats = self.visual_extractor(images)
        
        # Handle Indices -> Embeddings via EncoderDecoder's embedding definition
        if hypergraph_indices is not None and knowledge_emb is None:
            knowledge_emb = self.encoder_decoder.node_embedding(hypergraph_indices)
        
        # Project Knowledge
        if knowledge_emb is not None:
            knowledge_emb = self.hg_projector(knowledge_emb)
        
        if mode == 'train' or mode == 'forward':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', knowledge_emb=knowledge_emb)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample', knowledge_emb=knowledge_emb)
        else:
            raise ValueError
        return output

class HyperEncoderDecoder(AttModel):
    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = HyperRelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = HyperTransformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        # We need to manually init because we don't want to call super().__init__ which creates the wrong model
        nn.Module.__init__(self) # Skip AttModel init
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.seq_length = args.max_seq_length
        self.att_embed = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.1)) # Assume resnet

        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        
        # HyperGraph Node Embeddings
        # Default 15814 nodes if not specified
        self.num_nodes = getattr(args, 'num_nodes', 15814) 
        self.node_embedding = nn.Embedding(self.num_nodes, 512) # Fixed 512 dim for HG nodes

    def forward(self, fc_feats, att_feats, targets=None, mode='forward', hypergraph_indices=None, knowledge_emb=None):
        # Get Knowledge Embeddings if not provided
        if knowledge_emb is None and hypergraph_indices is not None:
             knowledge_emb = self.node_embedding(hypergraph_indices) # [B, K, 512]
        
        if mode == 'forward':
            att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, None, targets)
            out = self.model(att_feats, seq, att_masks, seq_mask, knowledge_emb)
            outputs = F.log_softmax(self.logit(out), dim=-1)
            return outputs
        elif mode == 'sample':
            return self.sample(fc_feats, att_feats, knowledge_emb)

    # ... (Keep _prepare_feature_forward) ...

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        # Project features if needed (2048 -> 512)
        if att_feats.size(-1) != self.d_model:
            att_feats = self.att_embed(att_feats)
            
        if att_masks is not None:
            att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask.device)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask

    def sample(self, fc_feats, att_feats, knowledge_emb=None):
        """
        Greedy decode for validation/testing.
        Uses HyperTransformer.decode() at each step for correct RM handling.
        """
        batch_size = att_feats.size(0)
        device = att_feats.device
        
        # 1. Prepare visual features (project 2048 -> 512)
        att_feats = self.att_embed(att_feats)
        att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long).unsqueeze(-2)
        
        # 2. Encode visual features
        memory = self.model.encode(att_feats, att_masks)
        
        # 3. Greedy decode loop
        bos_idx = getattr(self.tokenizer, 'bos_token_id', 0)
        eos_idx = getattr(self.tokenizer, 'eos_token_id', 0)
        
        seq = torch.zeros(batch_size, self.seq_length, dtype=torch.long, device=device)
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        
        for t in range(self.seq_length):
            # Use HyperTransformer.decode which handles RM init + update correctly
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = self.model.decode(memory, att_masks, ys, tgt_mask, knowledge_emb)
            
            # Get logits for last position only
            logprobs = F.log_softmax(self.logit(out[:, -1]), dim=-1)
            
            # Greedy selection
            _, next_word = logprobs.max(1)  # [B]
            seq[:, t] = next_word
            
            # Append to sequence for next step
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        
        return seq, None


