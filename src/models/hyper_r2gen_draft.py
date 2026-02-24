import torch
import torch.nn as nn
import copy
from R2Gen.models.r2gen import R2GenModel
from R2Gen.modules.encoder_decoder import EncoderDecoder, Transformer, RelationalMemory, Decoder, DecoderLayer, ConditionalSublayerConnection, ConditionalLayerNorm, Embeddings, PositionalEncoding, Encoder, EncoderLayer, SublayerConnection, MultiHeadedAttention, PositionwiseFeedForward

class HyperRelationalMemory(RelationalMemory):
    """
    Enhanced Relational Memory that initializes from External Knowledge (HyperGraph + RAG).
    """
    def init_memory(self, batch_size, knowledge_emb=None):
        """
        knowledge_emb: [B, num_slots, d_model] - Pre-computed knowledge embeddings
        """
        if knowledge_emb is not None:
            # If knowledge provided, use it as initial memory
            # Ensure shape matches
            if knowledge_emb.size(1) != self.num_slots:
                # Resize or project if needed (simple crop/pad for now)
                if knowledge_emb.size(1) > self.num_slots:
                    knowledge_emb = knowledge_emb[:, :self.num_slots, :]
                else:
                    pad = torch.zeros(batch_size, self.num_slots - knowledge_emb.size(1), self.d_model).to(knowledge_emb)
                    knowledge_emb = torch.cat([knowledge_emb, pad], dim=1)
            return knowledge_emb
        
        return super().init_memory(batch_size)

class HyperTransformer(Transformer):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__(encoder, decoder, src_embed, tgt_embed, rm)
        self.rm = rm # HyperRelationalMemory

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, knowledge_emb=None):
        # Pass knowledge_emb to init_memory
        memory = self.rm.init_memory(hidden_states.size(0), knowledge_emb).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)

class HyperEncoderDecoder(EncoderDecoder):
    def make_model(self, tgt_vocab):
        # Copy-paste from R2Gen but use Hyper classes
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        
        # USE HYPER RM
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

    def forward(self, fc_feats, att_feats, targets, mode='forward', knowledge_emb=None):
        # Override to accept knowledge_emb
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats) # Simplified for brevity, need full check
        
        # Re-implement _prepare_feature logic inside forward/decode
        # Actually R2Gen calls self.model(...) which is Transformer.forward
        # But we need to call Transformer.decode with knowledge
        
        if mode == 'forward':
            # Training
            # Need to replicate _prepare_feature_forward logic perfectly
            # ...
            # Shortcuts: R2Gen forward calls self.model(att_feats, seq, att_masks, seq_mask)
            # We intercept and call self.model.decode(...) directly?
            pass 
        return super().forward(fc_feats, att_feats, targets, mode)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # Hard to override cleanly without copying massive code.
        # But we can update the 'memory' initialization logic in the Transformer instance.
        pass

# SIMPLER APPROACH:
# No new classes. Just Monkey Patching at runtime in train_hyper_r2gen.py?
# Or just copy the whole EncoderDecoder file? It's cleaner.

