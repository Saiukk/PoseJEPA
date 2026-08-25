import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class TransformerPosePredictor(nn.Module):
    """Predictor that takes as input a 3D context, a pose and a mask, and predicts a sequence of 2D tokens."""

    def __init__(
        self,
        context_dim: int,
        embed_dim: int ,
        output_dim: int,
        num_heads: int,
        depth: int ,
        mlp_dim: int ,
        target_seq_len: int ,
        pose_dim: int ,
        dropout: float ,
    ):
        
        super().__init__()

        mlp_dim = mlp_dim if mlp_dim is not None else embed_dim * 4
        self.target_seq_len = target_seq_len

        # project the 3D context tokens to the embedding dimension
        self.context_proj = (
            nn.Linear(context_dim, embed_dim) if context_dim != embed_dim else nn.Identity()
        )

        # pose into a single token (global context)
        self.pose_embed = nn.Linear(pose_dim, embed_dim)

        # project the mask condition (1D) into the embedding dimension
        self.mask_cond_embed = nn.Linear(1, embed_dim)

        # Void Token to be predicted, initialized to zeros
        self.void_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding for the target sequence
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, target_seq_len, embed_dim))


        # IMPLEMENTATION OF THE TRANSFORMER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
       
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

        # Output projection to the desired output dimension (not used if output_dim == embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        self._init_weights()

        logging.info(f"TransformerPosePredictor initialized ")

    def _init_weights(self):
        nn.init.normal_(self.void_token, std=0.02)
        nn.init.normal_(self.pos_embed_2d, std=0.02)

        # Initialize weights of Transformer  (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
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
        self,
        context_3d: torch.Tensor,
        pose: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        
        B = context_3d.shape[0]

        # Embedding of the 3D context tokens
        context_tokens = self.context_proj(context_3d)

        # Global Pose Token
        pose_token = self.pose_embed(pose).unsqueeze(1)  # (B, 1, embed_dim)

        # --- Token target da completare: vuoto + posizione + maschera ---
        target_tokens = self.void_token.expand(B, self.target_seq_len, -1)
        target_tokens = target_tokens + self.pos_embed_2d
        target_tokens = target_tokens + self.mask_cond_embed(mask.unsqueeze(-1))

        # Concatenation
        transformer_input = torch.cat([pose_token, context_tokens, target_tokens], dim=1)
        out_sequence = self.transformer(transformer_input)

        # only the predicted
        predicted_2d_tokens = out_sequence[:, -self.target_seq_len:, :]
        return F.normalize(self.output_proj(predicted_2d_tokens), dim=-1)