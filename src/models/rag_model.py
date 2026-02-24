import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RAGModel(nn.Module):
    def __init__(self, config):
        super(RAGModel, self).__init__()
        self.config = config
        
        # 1. Image Encoder (ResNet-50)
        self.resnet = AutoModel.from_pretrained("microsoft/resnet-50")
        
        # Project ResNet features to embed_dim
        # ResNet pooler output is [B, 2048, 1, 1] -> flatten -> [B, 2048]
        self.visual_projection = nn.Linear(2048, config['model']['embed_dim'])
        
        # 2. Context Encoder (Simple Embedding or Small Transformer)
        # Re-using a small encoder (e.g. BERT-tiny) or just embeddings would be lighter.
        # But to be robust, let's use the SAME embeddings as the decoder but independent encoder layers?
        # For "Simple RAG", we can just assume the Context is a sequence of tokens.
        # We can just embed them and pass them as "Memory" to the decoder.
        
        self.vocab_size = config['model']['vocab_size']
        self.embed_dim = config['model']['embed_dim']
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=512)
        
        # We process context with a few Transformer Encoder layers to mix info
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=config['model']['num_heads'], batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['model']['num_layers'])
        
        # 4. Head
        self.fc_out = nn.Linear(self.embed_dim, self.vocab_size)
        
    def forward(self, images, context_ids, context_mask, target_ids=None):
        # --- Encode Images ---
        # [B, 3, H, W] -> [B, 2048]
        visual_feats = self.resnet(images).pooler_output.flatten(1) 
        visual_embeds = self.visual_projection(visual_feats).unsqueeze(1) # [B, 1, Dim]
        
        # --- Encode Context (Report from similar image) ---
        # [B, SeqLen] -> [B, SeqLen, Dim]
        context_embeds = self.embedding(context_ids)
        context_embeds = self.pos_encoder(context_embeds)
        
        # Pass through Context Encoder
        # We need padding mask for context: [B, SeqLen] (True for ignore)
        # PyTorch Transformer mask: True for NOT attending.
        # Our mask usually 1 for attend, 0 for pad. So we flip.
        src_key_padding_mask = (context_mask == 0)
        
        context_encoded = self.context_encoder(context_embeds, src_key_padding_mask=src_key_padding_mask) # [B, Seq, Dim]
        
        # --- Fusion (Concatenation) ---
        # Memory = [Visual, Context] -> [B, 1+Seq, Dim]
        memory = torch.cat([visual_embeds, context_encoded], dim=1)
        
        # Create memory mask (Visual is valid, Context depends on padding)
        # Visual mask: [B, 1] -> False (attend)
        visual_mask = torch.zeros((images.size(0), 1), dtype=torch.bool, device=images.device)
        memory_key_padding_mask = torch.cat([visual_mask, src_key_padding_mask], dim=1) # [B, 1+Seq]
        
        # --- Decode ---
        if target_ids is not None:
            # Training Mode
            tgt_embeds = self.embedding(target_ids)
            tgt_embeds = self.pos_encoder(tgt_embeds)
            
            # Causal Mask
            seq_len = tgt_embeds.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=images.device) * float('-inf'), diagonal=1)
            
            output = self.decoder(
                tgt=tgt_embeds,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            return self.fc_out(output)
            
        else:
            # Inference Mode (Greedy/Beam - not implemented here fully, just returning None or loop)
            # Placeholder for training script compatibility
            return None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)
