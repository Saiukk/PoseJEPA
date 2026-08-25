import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class CrossAttnPosePredictor(nn.Module):
    """Predictor using Cross-Attention: 2D target tokens fused with pose act as the query, 
    and the 3D context acts as the memory."""

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        depth: int,
        mlp_dim: int,
        target_seq_len: int,
        pose_dim: int,
        dropout: float,
    ):
        super().__init__()
        mlp_dim = mlp_dim if mlp_dim is not None else embed_dim * 4
        self.target_seq_len = target_seq_len
        self.depth = depth

        # Embeddings
        self.context_proj = nn.Linear(context_dim, embed_dim) if context_dim != embed_dim else nn.Identity()
        self.pose_embed = nn.Linear(pose_dim, embed_dim)
        self.mask_cond_embed = nn.Linear(1, embed_dim)
        
        # Target sequence initialization
        self.void_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, target_seq_len, embed_dim))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim)
        )

        self.output_proj = nn.Linear(embed_dim, output_dim)
        self._init_weights()
        
        logging.info("CrossAttnPosePredictor initialized (depth=%d)", depth)

    def _init_weights(self):
        nn.init.normal_(self.void_token, std=0.02)
        nn.init.normal_(self.pos_embed_2d, std=0.02)
        
        def _init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        self.apply(_init_module)

    def forward(
            self, context_3d: torch.Tensor, pose: torch.Tensor, mask: torch.Tensor,
        ) -> torch.Tensor:
            B = context_3d.shape[0]

            # 1. Memory: 3D context tokens
            memory_tokens = self.context_proj(context_3d)

            # 2. Query di base (void + mask)
            target_tokens = self.void_token.expand(B, self.target_seq_len, -1)
            target_tokens = target_tokens + self.mask_cond_embed(mask.unsqueeze(-1))
            
            # 3. FUSIONE SPAZIALE: La posa modula direttamente le coordinate 2D
            pose_feat = self.pose_embed(pose).unsqueeze(1) # (B, 1, embed_dim)
            spatial_query_pos = self.pos_embed_2d + pose_feat  # (B, target_seq_len, embed_dim)
            
            query_tokens = target_tokens + spatial_query_pos

            # 4. Cross-Attention
            out = self.transformer(tgt=query_tokens, memory=memory_tokens)

            # Normalize output to match DINOv2 targets
            return F.normalize(self.output_proj(out), dim=-1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class AdaLNPosePredictor(nn.Module):
    """
    Predictor with Cross-Attention and AdaLN-Zero pose modulation.
    - Query: 2D Target tokens modulated by Pose via AdaLN
    - Memory: 3D Context tokens
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        depth: int,
        mlp_dim: int,
        target_seq_len: int,
        pose_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        mlp_dim = mlp_dim if mlp_dim is not None else embed_dim * 4
        self.target_seq_len = target_seq_len

        # --- Embeddings Base ---
        self.context_proj = nn.Linear(context_dim, embed_dim) if context_dim != embed_dim else nn.Identity()
        self.mask_cond_embed = nn.Linear(1, embed_dim)
        
        self.void_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, target_seq_len, embed_dim))

        # --- AdaLN Modulation ---
        # Proiettiamo la posa al doppio della dimensione per generare (Scale, Shift)
        self.adaLN_modulation = nn.Linear(pose_dim, embed_dim * 2)
        # LayerNorm pura, senza pesi imparati (elementwise_affine=False) perché useremo la posa
        self.ln_query = nn.LayerNorm(embed_dim, elementwise_affine=False)
        
        # --- Transformer Decoder (Cross-Attention) ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim,
            dropout=dropout, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=depth, 
            norm=nn.LayerNorm(embed_dim)
        )

        self.output_proj = nn.Linear(embed_dim, output_dim)
        self._init_weights()
        
        logging.info("AdaLNPosePredictor initialized (depth=%d, AdaLN-Zero enabled)", depth)

    def _init_weights(self):
        # Inizializzazione standard
        nn.init.normal_(self.void_token, std=0.02)
        nn.init.normal_(self.pos_embed_2d, std=0.02)
        
        def _init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                    
        self.apply(_init_module)

        # TRUCCO ADALN-ZERO: Inizializza i pesi della modulazione a 0
        # Così scale = 0 e shift = 0 all'inizio del training.
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self, context_3d: torch.Tensor, pose: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        B = context_3d.shape[0]

        # 1. MEMORY: Proietta il contesto 3D
        memory_tokens = self.context_proj(context_3d)

        # 2. BASE TARGET TOKENS: Token vuoti + posizione 2D + maschera
        target_tokens = self.void_token.expand(B, self.target_seq_len, -1)
        target_tokens = target_tokens + self.pos_embed_2d
        target_tokens = target_tokens + self.mask_cond_embed(mask.unsqueeze(-1))
        
        # 3. ADALN MODULATION: Usa la posa per trasformare geometricamente le Query
        # Calcoliamo Shift e Scale. Chunk divide il tensore a metà sull'ultima dimensione.
        shift, scale = self.adaLN_modulation(pose).unsqueeze(1).chunk(2, dim=-1)
        
        # Modulazione: LN(x) * (1 + scale) + shift
        query_tokens = self.ln_query(target_tokens) * (1 + scale) + shift

        # 4. CROSS-ATTENTION: La camera orientata cerca nel modello 3D
        out = self.transformer(tgt=query_tokens, memory=memory_tokens)

        # 5. Output Projection e Normalizzazione sferica (per matchare DINOv2)
        return F.normalize(self.output_proj(out), dim=-1)