import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(nn.Module):
    def __init__(self, num_bands=10):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands) * math.pi)

    def forward(self, x):
        x_expanded = x.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
        return emb.flatten(-2, -1)


class PoseEncoder(nn.Module):
    """
    """

    def __init__(self, embed_dim: int, fourier_bands: int = 10, use_rot6d: bool = True):
        super().__init__()
        self.use_rot6d = use_rot6d
        self.fourier = FourierEmbedding(num_bands=fourier_bands)
        n_scalars = (6 if use_rot6d else 9) + 3
        fourier_dim = n_scalars * 2 * fourier_bands
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, R, t, centroid, scale):
        """
        R: (..., 3, 3)   t: (..., 3)   centroid: (..., 3)   scale: (...,)
        centroid/scale devono fare broadcast con R/t (vedi Predictor.forward per come
        vengono espansi sulla dimensione N delle context view).
        """
        centroid_col = centroid.unsqueeze(-1)                                      # (..., 3, 1)
        t_norm = (t.unsqueeze(-1) + torch.matmul(R, centroid_col)).squeeze(-1)     # (..., 3)
        t_norm = t_norm / scale.unsqueeze(-1)

        if self.use_rot6d:
            rot_part = R[..., :, :2].reshape(*R.shape[:-2], 6)
        else:
            rot_part = R.flatten(-2, -1)

        pose_flat = torch.cat([rot_part, t_norm], dim=-1)
        return self.mlp(self.fourier(pose_flat))


class RenderPredictor(nn.Module):
    

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        depth: int,
        expand_depth: int = 2,
        grid_size: int = 16,
        mlp_dim: int = None,
        dropout: float = 0.1,
        pose_fourier_bands: int = 10,
        use_rot6d: bool = True,
    ):
        super().__init__()
        mlp_dim = mlp_dim or embed_dim * 4
        self.grid_size = grid_size
        n_patches = grid_size * grid_size

        self.pose_encoder = PoseEncoder(embed_dim, fourier_bands=pose_fourier_bands, use_rot6d=use_rot6d)
        self.context_proj = nn.Linear(context_dim, embed_dim)

        stage1_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.stage1 = nn.TransformerDecoder(stage1_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim))

        self.spatial_queries = nn.Parameter(torch.randn(n_patches, embed_dim) * 0.02)
        stage2_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.stage2 = nn.TransformerDecoder(stage2_layer, num_layers=expand_depth, norm=nn.LayerNorm(embed_dim))

        self.output_proj = nn.Linear(embed_dim, output_dim)
        self._init_weights()

        logging.info(
            "RenderPredictor initialized (depth=%d, expand_depth=%d, grid=%dx%d, rot6d=%s, no rays)",
            depth, expand_depth, grid_size, grid_size, use_rot6d,
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
        R_query: torch.Tensor,        # (B, 3, 3)  posa VERA (training) o variabile latente (inference)
        t_query: torch.Tensor,        # (B, 3)
        context_dino: torch.Tensor,   # (B, N, P, context_dim)
        context_R: torch.Tensor,      # (B, N, 3, 3)
        context_t: torch.Tensor,      # (B, N, 3)
        centroid: torch.Tensor,       # (B, 3)
        scale: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        B, N, P, _ = context_dino.shape

        ctx_pose_tok = self.pose_encoder(
            context_R, context_t, centroid.unsqueeze(1), scale.unsqueeze(1)
        )                                                                       # (B, N, embed_dim)
        ctx_feat = self.context_proj(context_dino)                              # (B, N, P, embed_dim)
        memory = (ctx_feat + ctx_pose_tok.unsqueeze(2)).reshape(B, N * P, -1)    # (B, N*P, embed_dim)

        query = self.pose_encoder(R_query, t_query, centroid, scale).unsqueeze(1)   # (B, 1, embed_dim)

        pose_latent = self.stage1(tgt=query, memory=memory)                     # (B, 1, embed_dim)

        spatial_q = self.spatial_queries.unsqueeze(0).expand(B, -1, -1)         # (B, n_patches, embed_dim)
        grid_out = self.stage2(tgt=spatial_q, memory=pose_latent)               # (B, n_patches, embed_dim)

        content = self.output_proj(grid_out)
        return F.normalize(content, dim=-1)