"""
Hyper-Logic RAG Model
Combines:
- Visual encoder
- Entity prediction
- HyperGCN (hypergraph knowledge encoding)
- RAG retrieval
- Report decoder with logic constraints
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.models.hypergcn_encoder import HyperGCNEncoder, EntityPredictor
from src.utils.hypergraph_utils import logic_constrained_loss
from src.models.relational_memory import RelationalMemory, ConditionalLayerNorm
from src.models.mcln_decoder import MCLNDecoder

class HyperLogicRAGModel(nn.Module):
    """
    Full Hyper-Logic RAG Model
    
    Architecture:
        1. Visual Encoder (ResNet/ViT)
        2. Entity Predictor (image → entity probabilities)
        3. HyperGCN Encoder (refine entities with hypergraph)
        4. Report Decoder (Transformer with retrieval context)
        5. [Optional] RelationalMemory for learning report patterns
    """
    def __init__(
        self,
        visual_encoder,
        vocab_size,
        num_entities=15814,
        embed_dim=512,
        num_heads=8,
        num_decoder_layers=6,
        hypergraph_data=None,  # Output from hypergraph_utils.load_and_convert_hypergraph
        use_relational_memory=True,  # Enable RelationalMemory from R2Gen
        use_mcln=False, # Enable Memory-Driven Layer Norm (R2Gen Style)
        rm_num_slots=3,
        rm_d_model=512,
        dropout=0.1  # Added dropout parameter
    ):
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        
        # Entity prediction from visual features
        visual_dim = 512  # DenseNet-121 output dim (changed from 2048 ResNet-50)
        self.entity_predictor = EntityPredictor(
            visual_dim=visual_dim,
            num_entities=num_entities,
            hidden_dim=1024
        )
        
        # HyperGCN for knowledge reasoning
        self.use_hypergraph = hypergraph_data is not None
        if self.use_hypergraph:
            self.hypergcn = HyperGCNEncoder(
                num_nodes=num_entities,
                embed_dim=embed_dim,
                num_layers=2,
                dropout=dropout, # Use configurable dropout
                use_attention=False
            )
            self.register_buffer('hyperedge_index', hypergraph_data['hyperedge_index'])
            self.register_buffer('hyperedge_weight', hypergraph_data['hyperedge_weight'])
            
            # Store negative_hyperedges as a persistent attribute
            # Python lists are not moved by .to(device), but that's OK
            # They contain dict structure that logic_constrained_loss needs
            self.negative_hyperedges = hypergraph_data['negative_hyperedges']
            
            # CRITICAL: Log to verify loading
            print(f"✓ Loaded {len(self.negative_hyperedges)} negative hyperedges (conflict rules)")
        else:
            self.hypergcn = None
            self.negative_hyperedges = []
            print("⚠️  No hypergraph data provided, logic loss will be 0")
        
        self.entity_proj = nn.Linear(num_entities, embed_dim)
        
        # Text embeddings (standard, like Baseline 1/2 - proven approach)
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.positional_encoding = PositionalEncoding(embed_dim)
        
        # Transformer decoder
        self.use_mcln = use_mcln
        if use_mcln:
            print("✓ Using MCLN Decoder (R2Gen Style)")
            self.decoder = MCLNDecoder(
                d_model=embed_dim,
                nhead=num_heads,
                num_layers=num_decoder_layers,
                dim_feedforward=2048,
                dropout=dropout,
                rm_num_slots=rm_num_slots,
                rm_d_model=rm_d_model
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=2048,
                dropout=dropout, # Use configurable dropout
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Cross-attention for visual features (if needed)
        self.visual_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Visual feature projection (DenseNet outputs 512-d with config projection)
        self.visual_proj = nn.Linear(512, embed_dim)
        
        # Relational Memory (from R2Gen)
        self.use_relational_memory = use_relational_memory
        if use_relational_memory:
            self.rm_num_slots = rm_num_slots
            self.rm_d_model = rm_d_model
            self.relational_memory = RelationalMemory(
                num_slots=rm_num_slots,
                d_model=rm_d_model,
                num_heads=num_heads
            )
            # Project memory output to embed_dim (for residual connection)
            self.rm_proj = nn.Linear(rm_num_slots * rm_d_model, embed_dim)
            print(f"✓ RelationalMemory enabled: {rm_num_slots} slots × {rm_d_model}d")
        else:
            self.relational_memory = None
            self.rm_proj = None
            print("⚠️  RelationalMemory disabled")
        
    def init_node_embeddings_with_names(self, id_to_node, tokenizer, device):
        """
        Initialize HyperGCN node embeddings using the model's text embeddings of entity names.
        
        Args:
            id_to_node: Dict mapping node_id -> entity_name
            tokenizer: Tokenizer to encode entity names
            device: torch device
        """
        if not self.use_hypergraph or self.hypergcn is None:
            return
            
        print(f"Initializing {len(id_to_node)} node embeddings from names...")
        
        with torch.no_grad():
            # Create a tensor for all node embeddings
            new_embeddings = torch.zeros_like(self.hypergcn.node_embedding.weight)
            
            count = 0
            # Use list logic instead of tqdm to avoid import if not available in model file
            for idx in range(len(id_to_node)):
                if idx not in id_to_node:
                    continue
                    
                name = id_to_node[idx]
                
                # Tokenize name
                # R2Gen tokenizer returns dict. We need generic handling.
                if hasattr(tokenizer, 'encode'):
                    enc = tokenizer.encode(name, add_special_tokens=False)
                    if isinstance(enc, dict):
                        token_ids = enc['input_ids']
                    else:
                        token_ids = enc
                else:
                    count += 0 # Skip if no encode
                    continue
                
                if len(token_ids) == 0:
                    continue
                    
                # Convert to tensor
                token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
                
                # Get embeddings from model's text embedding layer
                # [L, D]
                word_embeddings = self.text_embedding(token_tensor)
                
                # Average to get entity embedding [D]
                entity_embedding = word_embeddings.mean(dim=0)
                
                new_embeddings[idx] = entity_embedding
                count += 1
            
            # Update HyperGCN weights
            self.hypergcn.node_embedding.weight.data.copy_(new_embeddings)
            print(f"✓ Initialized {count} node embeddings from text")

    def unused_method_placeholder(self):
        pass
    
    def forward(self, images, target_tokens, retrieval_context=None, retrieval_ids=None):
        """
        Args:
            images: [B, 3, H, W] input images
            target_tokens: [B, T] target token IDs
            retrieval_context: [B, C, D] optional retrieved report embeddings (deprecated)
            retrieval_ids: [B, L_r] optional retrieved report token IDs (preferred)
        
        Returns:
            logits: [B, T, vocab_size]
            entity_probs: [B, N] entity predictions (for logic loss)
        """
        batch_size = images.size(0)
        
        # 1. Visual encoding
        if images.dim() == 5:
            # Multi-image: [B, 2, C, H, W]
            vf_0 = self.visual_encoder(images[:, 0]) # [B, C', H', W'] or [B, C']
            vf_1 = self.visual_encoder(images[:, 1])
            
            # 2. Entity prediction (average probabilities from both views)
            ep_0 = self.entity_predictor(vf_0)
            ep_1 = self.entity_predictor(vf_1)
            entity_probs = (ep_0 + ep_1) / 2.0  # [B, N]
            
            # Combine visual features
            if len(vf_0.shape) == 4:
                # Concatenate along channel dimension (like R2Gen fc_feats)
                # Wait, if we concat channel, visual_proj needs to handle 2x input size.
                # Instead, let's concat along spatial sequence length later to keep channel size same!
                # vf_0: [B, C, H, W] -> sequence [B, C, H*W]
                visual_features = [vf_0, vf_1]
            else:
                visual_features = [vf_0, vf_1]
        else:
            visual_features = self.visual_encoder(images)  # [B, 2048, 7, 7] or [B, 2048]
            entity_probs = self.entity_predictor(visual_features)  # [B, N]
        
        # 3. HyperGCN refinement (if enabled)
        if self.use_hypergraph:
            # Refine entity embeddings using hypergraph
            entity_embeddings = self.hypergcn(
                entity_probs,
                self.hyperedge_index,
                self.hyperedge_weight
            )  # [B, N, D]
            
            # Pool entity embeddings to single vector
            # FIXED: Use MAX pooling instead of MEAN to preserve strong signals from sparse entities
            entity_context = entity_embeddings.max(dim=1)[0]  # [B, D]
        else:
            # Just project entity probabilities
            entity_context = self.entity_proj(entity_probs)  # [B, D]
        
        # 4. Prepare encoder memory (visual + entity + retrieval)
        encoder_memory = []
        
        # Visual features (flatten spatial dims)
        if isinstance(visual_features, list):
            # Multi-view processing
            vf_seqs = []
            for vf in visual_features:
                if len(vf.shape) == 4:
                    B, C, H, W = vf.shape
                    v_flat = vf.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
                    v_flat = self.visual_proj(v_flat)
                    vf_seqs.append(v_flat)
                else:
                    vf_seqs.append(vf.unsqueeze(1)) # [B, 1, C]
            
            # Concat spatial/sequence dims: [B, H*W + H*W, D] or [B, 1 + 1, C]
            multi_v_flat = torch.cat(vf_seqs, dim=1)
            encoder_memory.append(multi_v_flat)
        else:
            # Single view
            if len(visual_features.shape) == 4:
                B, C, H, W = visual_features.shape
                visual_flat = visual_features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
                # Project to embed_dim using registered module
                visual_flat = self.visual_proj(visual_flat)
                encoder_memory.append(visual_flat)
            else:
                encoder_memory.append(visual_features.unsqueeze(1))  # [B, 1, C]
        
        # Entity context
        encoder_memory.append(entity_context.unsqueeze(1))  # [B, 1, D]
        
        # Retrieval context (if provided)
        retrieval_mask = None
        if retrieval_ids is not None:
            # Embed retrieved tokens
            retrieval_emb = self.text_embedding(retrieval_ids) # [B, L_r, D]
            retrieval_emb = self.positional_encoding(retrieval_emb)
            encoder_memory.append(retrieval_emb)
            
            # Create padding mask (True for PAD/0)
            # retrieval_ids: [B, L_r]
            retrieval_mask = (retrieval_ids == 0).to(images.device)
            
        elif retrieval_context is not None:
            encoder_memory.append(retrieval_context)  # [B, C, D]
            # No mask for embeddings (assume valid)
        
        encoder_memory = torch.cat(encoder_memory, dim=1)  # [B, L, D]
        
        # Create full memory mask for decoder
        if retrieval_mask is not None:
            batch_size = images.size(0)
            # Create masks for Visual and Entity parts (all False/Valid)
            # Visual: [B, V]
            if isinstance(visual_features, list):
                if len(visual_features[0].shape) == 4:
                    V = visual_features[0].size(2) * visual_features[0].size(3) * len(visual_features)
                else:
                    V = len(visual_features)
            elif len(visual_features.shape) == 4:
                 V = visual_features.size(2) * visual_features.size(3)
            else:
                 V = 1
            visual_mask = torch.zeros((batch_size, V), dtype=torch.bool, device=images.device)
            
            # Entity: [B, 1]
            entity_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=images.device)
            
            # Concat: [Visual, Entity, Retrieval]
            memory_key_padding_mask = torch.cat([visual_mask, entity_mask, retrieval_mask], dim=1)
        else:
            memory_key_padding_mask = None

        # 5. Decode report
        tgt_embeddings = self.text_embedding(target_tokens)  # [B, T, D]
        
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # Create causal mask
        T = target_tokens.size(1)
        
        # Decode
        if self.use_mcln and self.use_relational_memory:
             # MCLN uses R2Gen-style MultiHeadedAttention which expects boolean masks
             # Causal mask: [1, T, T] boolean (True = attend, False = mask)
             tgt_mask = torch.tril(torch.ones(T, T, device=images.device)).unsqueeze(0)  # [1, T, T]
             
             # MCLN requires Relational Memory State
             memory = self.relational_memory.init_memory(batch_size).to(images.device)
             # Evolve memory: [B, T, Slots*D]
             memory = self.relational_memory(tgt_embeddings, memory)
             
             decoder_output = self.decoder(
                tgt_embeddings,
                encoder_memory,
                src_mask=None,  # No masking on encoder memory for now
                tgt_mask=tgt_mask,
                memory=memory
             )
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(images.device)
            decoder_output = self.decoder(
                tgt_embeddings,
                encoder_memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )  # [B, T, D]
            
            # 6. Apply RelationalMemory conditioning (if enabled via Residual)
            if self.use_relational_memory and self.relational_memory is not None:
                # Initialize memory and process through target embeddings
                memory = self.relational_memory.init_memory(batch_size).to(images.device)
                memory_output = self.relational_memory(tgt_embeddings, memory)  # [B, T, num_slots * rm_d_model]
                # Project and add as residual
                memory_context = self.rm_proj(memory_output)  # [B, T, D]
                decoder_output = decoder_output + 0.5 * memory_context  # Weighted residual
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)  # [B, T, vocab_size]
        
        return logits, entity_probs
    
    def compute_loss(self, logits, targets, entity_probs, lambda_logic=0.1, current_epoch=0, label_smoothing=0.0):
        """
        Combined loss: generation + logic constraints
        Args:
            current_epoch: Current training epoch (0-indexed). Used for warm-up strategy.
            label_smoothing: Label smoothing factor (0.0 = hard labels, 0.1 = recommended).
        """
        # Generation loss (cross-entropy with optional label smoothing)
        gen_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,  # Padding token
            label_smoothing=label_smoothing
        )
        
        # Logic constraint loss
        if self.use_hypergraph and len(self.negative_hyperedges) > 0:
            # Import v2 loss function with suppression penalty
            from src.utils.hypergraph_utils import logic_constrained_loss_v2
            
            if current_epoch < 5:
                # WARM-UP PHASE (Epoch 0-4): Detach gradients to prevent entity collapse
                # Model learns basic accuracy first without heavy penalties
                l_logic = logic_constrained_loss_v2(entity_probs.detach(), self.negative_hyperedges)
            else:
                # STRICT PHASE (Epoch 5+): Enforce logic constraints with gradients
                # V2 prevents zero-probability trick with suppression penalty
                l_logic = logic_constrained_loss_v2(entity_probs, self.negative_hyperedges)
        else:
            # DEBUG: This should NOT happen if hypergraph loaded correctly
            if self.use_hypergraph:
                print(f"⚠️  WARNING: use_hypergraph=True but negative_hyperedges is empty!")
            l_logic = torch.tensor(0.0, device=logits.device)
        
        # Combined loss
        total_loss = gen_loss + lambda_logic * l_logic
        
        # DEBUG: Print periodically to track logic loss across batches
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1
        
        if self._batch_counter % 500 == 1:  # Print at batch 1, 501, 1001, ...
            print(f"\n🔍 BATCH {self._batch_counter} DEBUG:")
            print(f"  gen_loss: {gen_loss.item():.4f}")
            print(f"  l_logic (raw): {l_logic.item():.4f}")
            print(f"  lambda_logic: {lambda_logic}")
            print(f"  entity_probs stats: min={entity_probs.min().item():.4f}, max={entity_probs.max().item():.4f}, mean={entity_probs.mean().item():.4f}")
            print(f"  negative_hyperedges count: {len(self.negative_hyperedges)}")
        
        return total_loss, gen_loss, l_logic

    def generate(self, images, max_length=128, num_beams=3, retrieval_ids=None):
        """
        Generate reports using beam search.
        Args:
            images: [B, 3, H, W]
            max_length: max generation length
            num_beams: beam size
            retrieval_ids: [B, L_r] optional retrieved tokens
        Returns:
            generated_ids: [B, seq_len] (Best beam only)
        """
        outputs = self._beam_search(images, beam_size=num_beams, max_len=max_length, retrieval_ids=retrieval_ids)
        # Return only the best beam for each sample (stride by num_beams)
        return outputs[::num_beams]

    def _beam_search(self, images, beam_size=3, max_len=100, retrieval_ids=None):
        """
        Beam Search generation implementation.
        UPDATED: Now includes RelationalMemory conditioning and RAG support.
        """
        B = images.size(0)
        device = images.device
        
        # 1. Encode images & entities
        if images.dim() == 5:
            vf_0 = self.visual_encoder(images[:, 0])
            vf_1 = self.visual_encoder(images[:, 1])
            ep_0 = self.entity_predictor(vf_0)
            ep_1 = self.entity_predictor(vf_1)
            entity_probs = (ep_0 + ep_1) / 2.0
            
            if len(vf_0.shape) == 4:
                visual_features = [vf_0, vf_1]
            else:
                visual_features = [vf_0, vf_1]
        else:
            visual_features = self.visual_encoder(images)
            entity_probs = self.entity_predictor(visual_features)
        
        if self.use_hypergraph:
            entity_embeddings = self.hypergcn(entity_probs, self.hyperedge_index, self.hyperedge_weight)
            # FIXED: MAX pooling
            entity_context = entity_embeddings.max(dim=1)[0]
        else:
            entity_context = self.entity_proj(entity_probs)
            
        # Prepare Encoder Memory
        encoder_memory = []
        if isinstance(visual_features, list):
            vf_seqs = []
            for vf in visual_features:
                if len(vf.shape) == 4:
                    B_v, C, H, W = vf.shape
                    v_flat = vf.view(B_v, C, H*W).permute(0, 2, 1)
                    v_flat = self.visual_proj(v_flat)
                    vf_seqs.append(v_flat)
                else:
                    vf_seqs.append(vf.unsqueeze(1))
            multi_v_flat = torch.cat(vf_seqs, dim=1)
            encoder_memory.append(multi_v_flat)
        else:
            if len(visual_features.shape) == 4:
                B_v, C, H, W = visual_features.shape
                visual_flat = visual_features.view(B_v, C, H*W).permute(0, 2, 1)
                visual_flat = self.visual_proj(visual_flat)
                encoder_memory.append(visual_flat)
            else:
                encoder_memory.append(visual_features.unsqueeze(1))
        encoder_memory.append(entity_context.unsqueeze(1))
        
        # Add RAG context if provided
        retrieval_mask = None
        if retrieval_ids is not None:
            # Embed retrieval IDs
            retrieval_emb = self.text_embedding(retrieval_ids)
            retrieval_emb = self.positional_encoding(retrieval_emb)
            encoder_memory.append(retrieval_emb)
            
            # Create padding mask
            retrieval_mask = (retrieval_ids == 0).to(device)
            
        encoder_memory = torch.cat(encoder_memory, dim=1)  # [B, L, D]
        
        # Create full memory mask for decoder
        if retrieval_mask is not None:
            # Visual: [B, V]
            if isinstance(visual_features, list):
                if len(visual_features[0].shape) == 4:
                    V = visual_features[0].size(2) * visual_features[0].size(3) * len(visual_features)
                else:
                    V = len(visual_features)
            elif len(visual_features.shape) == 4:
                 V = visual_features.size(2) * visual_features.size(3)
            else:
                 V = 1
            visual_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
            
            # Entity: [B, 1]
            entity_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            
            # Concat: [Visual, Entity, Retrieval]
            memory_key_padding_mask = torch.cat([visual_mask, entity_mask, retrieval_mask], dim=1)
        else:
            memory_key_padding_mask = None
        
        # 2. Initialize RelationalMemory (same as forward())
        if self.use_relational_memory and self.relational_memory is not None:
            rm_memory = self.relational_memory.init_memory(B).to(device)
        else:
            rm_memory = None
        
        # 3. Beam Search Loop
        # R2Gen tokenizer: bos=2, eos=3, pad=0, unk=1
        start_token = 2  # R2Gen <bos> token (NOT BERT [CLS] = 101!)
        eos_token = 3    # R2Gen <eos> token
        
        # Expand memory for beam size
        encoder_memory = encoder_memory.repeat_interleave(beam_size, dim=0)
        if rm_memory is not None:
            rm_memory = rm_memory.repeat_interleave(beam_size, dim=0)
        
        # Expand mask for beam size
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(beam_size, dim=0)
        
        # Initial input: [B*beam, 1]
        ys = torch.full((B * beam_size, 1), start_token, dtype=torch.long, device=device)
        
        # Initial scores: [B*beam]
        scores = torch.full((B * beam_size,), -float('inf'), device=device)
        for i in range(B):
            scores[i * beam_size] = 0.0
            
        cur_len = 1
        
        for t in range(max_len - 1):
            # Embeddings
            tgt_emb = self.text_embedding(ys)
            tgt_emb = self.positional_encoding(tgt_emb)
            
            if self.use_mcln and self.use_relational_memory:
                 # MCLN: boolean causal mask [1, T, T]
                 tgt_mask = torch.tril(torch.ones(cur_len, cur_len, device=device)).unsqueeze(0)
                 # Process through RelationalMemory first
                 memory_output = self.relational_memory(tgt_emb, rm_memory)
                 
                 decoder_output = self.decoder(
                    tgt_emb, 
                    encoder_memory, 
                    src_mask=None,
                    tgt_mask=tgt_mask,
                    memory=memory_output
                 )
            else:
                # Standard decoder: float causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)
                decoder_output = self.decoder(
                    tgt_emb, 
                    encoder_memory, 
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            
            # Apply RelationalMemory conditioning (SAME AS FORWARD!)
            if self.use_relational_memory and self.relational_memory is not None:
                # Process through RelationalMemory
                memory_output = self.relational_memory(tgt_emb, rm_memory)  # [B*beam, T, num_slots*d]
                memory_context = self.rm_proj(memory_output)  # [B*beam, T, D]
                decoder_output = decoder_output + 0.5 * memory_context  # Weighted residual
            
            # Logits for last token
            last_token_logits = self.output_proj(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(last_token_logits, dim=-1)
            
            # Add to cumulative scores
            cand_scores = scores.unsqueeze(1) + log_probs
            cand_scores_flat = cand_scores.view(B, -1)
            
            # Select top-k per sample
            best_scores, best_indices = cand_scores_flat.topk(beam_size, dim=1)
            
            vocab_size = log_probs.size(1)
            cand_beam_idx = best_indices // vocab_size
            cand_token_idx = best_indices % vocab_size
            
            # Update global batch indices
            base_indices = torch.arange(0, B * beam_size, beam_size, device=device).unsqueeze(1)
            selected_beam_absolute = (base_indices + cand_beam_idx).view(-1)
            
            # Update scores & sequences
            scores = best_scores.view(-1)
            ys = torch.cat([ys[selected_beam_absolute], cand_token_idx.view(-1).unsqueeze(1)], dim=1)
            
            cur_len += 1
            
        return ys[:, 1:] # Exclude start token

    def sample_scst(self, images, max_len=100, sample_method='greedy', retrieval_ids=None):
        """
        Sampling for Self-Critical Sequence Training (SCST).
        Returns:
            seq: [B, T] (generated tokens)
            seq_log_probs: [B, T] (log probs of generated tokens)
        """
        B = images.size(0)
        device = images.device
        
        # 1. Encode images & entities (Same as _beam_search)
        if images.dim() == 5:
            vf_0 = self.visual_encoder(images[:, 0])
            vf_1 = self.visual_encoder(images[:, 1])
            ep_0 = self.entity_predictor(vf_0)
            ep_1 = self.entity_predictor(vf_1)
            entity_probs = (ep_0 + ep_1) / 2.0
            
            if len(vf_0.shape) == 4:
                visual_features = [vf_0, vf_1]
            else:
                visual_features = [vf_0, vf_1]
        else:
            visual_features = self.visual_encoder(images)
            entity_probs = self.entity_predictor(visual_features)
        
        if self.use_hypergraph:
            entity_embeddings = self.hypergcn(entity_probs, self.hyperedge_index, self.hyperedge_weight)
            entity_context = entity_embeddings.max(dim=1)[0]
        else:
            entity_context = self.entity_proj(entity_probs)
            
        # Prepare Encoder Memory
        encoder_memory = []
        if isinstance(visual_features, list):
            vf_seqs = []
            for vf in visual_features:
                if len(vf.shape) == 4:
                    B_v, C, H, W = vf.shape
                    v_flat = vf.view(B_v, C, H*W).permute(0, 2, 1)
                    v_flat = self.visual_proj(v_flat)
                    vf_seqs.append(v_flat)
                else:
                    vf_seqs.append(vf.unsqueeze(1))
            multi_v_flat = torch.cat(vf_seqs, dim=1)
            encoder_memory.append(multi_v_flat)
        else:
            if len(visual_features.shape) == 4:
                B_v, C, H, W = visual_features.shape
                visual_flat = visual_features.view(B_v, C, H*W).permute(0, 2, 1)
                visual_flat = self.visual_proj(visual_flat)
                encoder_memory.append(visual_flat)
            else:
                encoder_memory.append(visual_features.unsqueeze(1))
        encoder_memory.append(entity_context.unsqueeze(1))
        
        # Add RAG context
        retrieval_mask = None
        if retrieval_ids is not None:
            retrieval_emb = self.text_embedding(retrieval_ids)
            retrieval_emb = self.positional_encoding(retrieval_emb)
            encoder_memory.append(retrieval_emb)
            retrieval_mask = (retrieval_ids == 0).to(device)
            
        encoder_memory = torch.cat(encoder_memory, dim=1)
        
        # Memory Mask
        if retrieval_mask is not None:
            if isinstance(visual_features, list):
                if len(visual_features[0].shape) == 4:
                    V = visual_features[0].size(2) * visual_features[0].size(3) * len(visual_features)
                else:
                    V = len(visual_features)
            elif len(visual_features.shape) == 4:
                 V = visual_features.size(2) * visual_features.size(3)
            else:
                 V = 1
            visual_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
            entity_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([visual_mask, entity_mask, retrieval_mask], dim=1)
        else:
            memory_key_padding_mask = None
        
        # 2. Initialize RelationalMemory
        if self.use_relational_memory and self.relational_memory is not None:
            rm_memory = self.relational_memory.init_memory(B).to(device)
        else:
            rm_memory = None
            
        # 3. Autoregressive Loop
        start_token = 2
        eos_token = 3
        
        inputs = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        
        seqs = []
        log_probs = []
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            # Embeddings
            cur_len = inputs.size(1)
            tgt_emb = self.text_embedding(inputs)
            tgt_emb = self.positional_encoding(tgt_emb)
            
            # Decoder
            if self.use_mcln and self.use_relational_memory:
                 # MCLN: boolean causal mask [1, T, T]
                 tgt_mask = torch.tril(torch.ones(cur_len, cur_len, device=device)).unsqueeze(0)
                 memory_output = self.relational_memory(tgt_emb, rm_memory)
                 decoder_output = self.decoder(
                    tgt_emb, encoder_memory, src_mask=None, tgt_mask=tgt_mask, memory=memory_output
                 )
            else:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)
                decoder_output = self.decoder(
                    tgt_emb, encoder_memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
                )
                if self.use_relational_memory and self.relational_memory is not None:
                   # Only need memory update for the LAST step, but implementation updates all.
                   # For efficiency in loop, we re-compute full sequence. 
                   # Ideal: Cache memory states. For now: re-compute is safer.
                    memory_output = self.relational_memory(tgt_emb, rm_memory)
                    memory_context = self.rm_proj(memory_output)
                    decoder_output = decoder_output + 0.5 * memory_context

            # Logits for the LAST token
            logits = self.output_proj(decoder_output[:, -1, :])
            
            if sample_method == 'greedy':
                log_p = torch.log_softmax(logits, dim=-1)
                curr_log_prob, next_word = torch.max(log_p, dim=1)
            elif sample_method == 'sample':
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_word = dist.sample()
                curr_log_prob = dist.log_prob(next_word)
                
            # Store
            seqs.append(next_word)
            log_probs.append(curr_log_prob)
            
            # Update inputs
            inputs = torch.cat([inputs, next_word.unsqueeze(1)], dim=1)
            
            # Check eos
            finished |= (next_word == eos_token)
            if finished.all():
                break
                
        return torch.stack(seqs, dim=1), torch.stack(log_probs, dim=1)


class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def test_model():
    """Test Hyper-Logic RAG model"""
    print("="*60)
    print("TESTING HYPER-LOGIC RAG MODEL")
    print("="*60)
    
    # Mock hypergraph data
    from src.utils.hypergraph_utils import load_and_convert_hypergraph
    
    hypergraph_path = "data/hypergraph.pkl"
    if os.path.exists(hypergraph_path):
        hypergraph_data = load_and_convert_hypergraph(hypergraph_path)
    else:
        print("⚠️  Hypergraph not found, testing without hypergraph...")
        hypergraph_data = None
    
    # Mock visual encoder (ResNet-50 style)
    class MockVisualEncoder(nn.Module):
        def forward(self, x):
            B = x.size(0)
            return torch.randn(B, 2048, 7, 7)
    
    visual_encoder = MockVisualEncoder()
    
    # Initialize model
    model = HyperLogicRAGModel(
        visual_encoder=visual_encoder,
        vocab_size=10000,
        num_entities=15814 if hypergraph_data else 100,
        embed_dim=512,
        num_heads=8,
        num_decoder_layers=3,  # Smaller for testing
        hypergraph_data=hypergraph_data
    )
    
    print(f"\n✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hypergraph enabled: {model.use_hypergraph}")
    
    # Mock inputs
    batch_size = 2
    seq_len = 20
    
    images = torch.randn(batch_size, 3, 224, 224)
    target_tokens = torch.randint(1, 1000, (batch_size, seq_len))
    
    print(f"\nMock inputs:")
    print(f"  Images: {images.shape}")
    print(f"  Target tokens: {target_tokens.shape}")
    
    # Forward pass
    logits, entity_probs = model(images, target_tokens)
    
    print(f"\nOutputs:")
    print(f"  Logits: {logits.shape}")
    print(f"  Entity probs: {entity_probs.shape}")
    
    # Compute loss
    total_loss, gen_loss, logic_loss = model.compute_loss(
        logits, target_tokens, entity_probs, lambda_logic=0.1
    )
    
    print(f"\nLosses:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Generation: {gen_loss.item():.4f}")
    print(f"  Logic: {logic_loss.item():.4f}")
    
    print("\n" + "="*60)
    print("✓ HYPER-LOGIC RAG MODEL TEST PASSED!")
    print("="*60)


    def beam_search(self, images, beam_size=3, max_len=100):
        """
        Beam Search generation for better quality reports.
        Args:
            images: [B, 3, H, W]
            beam_size: Number of beams
            max_len: Max sequence length
        Returns:
            start_tokens: [B, max_len] (or stopped early)
        """
        B = images.size(0)
        device = images.device
        
        # 1. Encode images & entities
        visual_features = self.visual_encoder(images)
        entity_probs = self.entity_predictor(visual_features)
        
        if self.use_hypergraph:
            entity_embeddings = self.hypergcn(entity_probs, self.hyperedge_index, self.hyperedge_weight)
            entity_context = entity_embeddings.mean(dim=1)
        else:
            entity_context = self.entity_proj(entity_probs)
            
        # Prepare Encoder Memory
        encoder_memory = []
        if len(visual_features.shape) == 4:
            B_v, C, H, W = visual_features.shape
            visual_flat = visual_features.view(B_v, C, H*W).permute(0, 2, 1)
            visual_flat = self.visual_proj(visual_flat)
            encoder_memory.append(visual_flat)
        else:
            encoder_memory.append(visual_features.unsqueeze(1))
        encoder_memory.append(entity_context.unsqueeze(1))
        encoder_memory = torch.cat(encoder_memory, dim=1)  # [B, L, D]
        
        # 2. Beam Search Loop
        # Initialize beams: [B, beam_size, seq_len]
        # We start with [CLS] token (assuming ID 101 for BERT)
        start_token = 101 
        end_token = 102
        
        # [B*beam_size, 1]
        completed_sentences = [[] for _ in range(B)]
        
        # Current active beams: (score, sequence)
        # We flatten batch for loop efficiency or keep structured.
        # Let's use simple per-sample loop for clarity since B is small in eval,
        # or vectorized for speed. Vectorized is better for GPU.
        
        # Expand memory for beam size
        # encoder_memory: [B, L, D] -> [B*beam, L, D]
        encoder_memory = encoder_memory.repeat_interleave(beam_size, dim=0)
        
        # Initial input: [B*beam, 1]
        ys = torch.full((B * beam_size, 1), start_token, dtype=torch.long, device=device)
        
        # Initial scores: [B*beam]
        # For the first step, only the first beam (index 0) has score 0, others -inf
        scores = torch.full((B * beam_size,), -float('inf'), device=device)
        for i in range(B):
            scores[i * beam_size] = 0.0
            
        batch_indices = torch.arange(B, device=device).repeat_interleave(beam_size)
        
        tokens_generated = torch.zeros((B*beam_size, max_len), dtype=torch.long, device=device)
        tokens_generated[:, 0] = start_token
        
        cur_len = 1
        active_mask = torch.ones(B * beam_size, dtype=torch.bool, device=device)
        
        for t in range(max_len - 1):
            if not active_mask.any():
                break
                
            # Forward pass
            # TransformerDecoder needs [S, N, E] or [N, S, E] if batch_first=True
            # Our decoder is batch_first=True
            
            # Create mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)
            
            # Embeddings
            tgt_emb = self.text_embedding(ys) # [B*beam, cur_len, D]
            tgt_emb = self.positional_encoding(tgt_emb)
            
            decoder_output = self.decoder(tgt_emb, encoder_memory, tgt_mask=tgt_mask)
            
            # Logits for last token
            last_token_logits = self.output_proj(decoder_output[:, -1, :]) # [B*beam, vocab]
            log_probs = torch.log_softmax(last_token_logits, dim=-1) # [B*beam, vocab]
            
            # Add to cumulative scores
            # scores: [B*beam] -> [B*beam, 1] + [B*beam, vocab] -> [B*beam, vocab]
            cand_scores = scores.unsqueeze(1) + log_probs
            
            # Flatten to [B, beam*vocab] to select topk per sample
            cand_scores_flat = cand_scores.view(B, -1)
            
            # Select top-k per sample
            # best_scores: [B, beam], best_indices: [B, beam]
            best_scores, best_indices = cand_scores_flat.topk(beam_size, dim=1)
            
            # Recover beam index and token ID
            # best_indices are in range [0, beam*vocab - 1]
            # beam_idx = best_indices // vocab
            # token_idx = best_indices % vocab
            vocab_size = log_probs.size(1)
            cand_beam_idx = best_indices // vocab_size
            cand_token_idx = best_indices % vocab_size
            
            # Update global batch indices (adjust for creating new B*beam structure)
            # We need to construct the next step inputs
            
            # Base indices for each sample: [0, beam, 2*beam, ...]
            base_indices = torch.arange(0, B * beam_size, beam_size, device=device).unsqueeze(1)
            
            # Selected global beam indices to pick from 'ys'
            # [B, beam] -> [B*beam]
            selected_beam_absolute = (base_indices + cand_beam_idx).view(-1)
            
            # Next tokens to append
            next_tokens = cand_token_idx.view(-1)
            
            # Update scores
            scores = best_scores.view(-1)
            
            # Update ys
            # Pick history from selected beams and append new token
            ys_history = ys[selected_beam_absolute]
            ys = torch.cat([ys_history, next_tokens.unsqueeze(1)], dim=1)
            
            cur_len += 1
            
            # Handle finished beams (token == 102)
            # In a real rigorous implementation, we'd move finished sequences aside.
            # Here, strict length constraints or PAD masking is simpler for 'fallback' script.
            # But the 'fallback' script expects [B, seq_len] tensor.
            # We'll rely on post-processing to truncate at 102.
            
            # Optimization: if all top-1 beams for all samples are finished, stop.
            # But simpler to just run max_len in this script.
            
        return ys[:, 1:] # Return generating part excluding start token

if __name__ == "__main__":
    test_model()
