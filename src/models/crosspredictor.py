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



################################
#  * * * *  * * * *  *     *  #      
#  *     *  *     *    * *    # 
#  *     *  *     *     *     #
#  * * *.   * * * *     *     #
#  *    *.  *     *     *     #
#  *     *. *     *     *     #
###############################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class FourierEmbedding(nn.Module):
    def __init__(self, num_bands=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands) * math.pi)

    def forward(self, x):
        x_expanded = x.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)         # (..., 3, 2*num_bands)
        return emb.flatten(-2, -1)


class RayCrossAttnPredictor(nn.Module):
    """
    CROSS ATTENTION PREDICTOR
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        depth: int,
        mlp_dim: int = None,
        dropout: float = 0.1,
        fourier_bands: int = 6,
    ):
        super().__init__()
        mlp_dim = mlp_dim if mlp_dim is not None else embed_dim * 4

        self.fourier = FourierEmbedding(num_bands=fourier_bands)
        fourier_dim = 3 * 2 * fourier_bands                         # (xyz), (sin,cos) <--- Ricorda

        self.ray_dir_embed = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ray_moment_embed = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.center_embed = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.context_proj = nn.Linear(context_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.occupancy_proj = nn.Linear(embed_dim, 1)

        self._init_weights()

        logging.info(
            "RayCrossAttnPredictor initialized (depth=%d, FourierBands=%d, plucker=True)",
            depth, fourier_bands,
        )

    def _init_weights(self):
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

    def forward(
        self,
        ray_dirs: torch.Tensor,       # (B, N, 3) direzione unitaria, gia' ruotata nel frame oggetto
        ray_moments: torch.Tensor,    # (B, N, 3) momento Plucker O x d, stesso frame
        context_features: torch.Tensor,
        context_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        center_fourier = self.fourier(context_centers)
        k_v_pos = self.center_embed(center_fourier)
        k_v_feat = self.context_proj(context_features)
        memory_tokens = k_v_feat + k_v_pos

        dir_fourier = self.fourier(ray_dirs)
        moment_fourier = self.fourier(ray_moments)
        query_tokens = self.ray_dir_embed(dir_fourier) + self.ray_moment_embed(moment_fourier)

        out = self.transformer(tgt=query_tokens, memory=memory_tokens)

        occupancy_logits = self.occupancy_proj(out).squeeze(-1)   # (B, N), logit pre-sigmoid

        content = self.output_proj(out)
        content = F.normalize(content, dim=-1)

        return content, occupancy_logits