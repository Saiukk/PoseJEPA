import torch
import torch.nn as nn
from .base import BaseEncoder3D


def fourier_encode(x: torch.Tensor, num_freqs: int = 6, include_input: bool = True) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    x_scaled = x.unsqueeze(-1) * freqs * torch.pi
    enc = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1).flatten(-2)
    return torch.cat([x, enc], dim=-1) if include_input else enc


class ViewCrossAttentionPool(nn.Module):
    def __init__(self, in_dim, embed_dim, n_heads=8, dropout=0.0, pos_num_freqs=6):
        super().__init__()
        self.pos_num_freqs = pos_num_freqs
        pos_dim = 3 * (1 + 2 * pos_num_freqs)

        self.feat_proj = nn.Linear(in_dim, embed_dim)
        self.view_dir_mlp = nn.Sequential(
            nn.Linear(3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.point_pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, view_feats, view_dirs, mask, point_xyz):
        """
        view_feats: (B, P, N, in_dim)
        view_dirs:  (B, P, N, 3)
        mask:       (B, P, N) bool, True = vista valida
        point_xyz:  (B, P, 3)
        """
        B, P, N, _ = view_feats.shape

        kv = self.feat_proj(view_feats) + self.view_dir_mlp(view_dirs)
        kv = self.norm_kv(kv.reshape(B * P, N, -1))

        pos_enc = fourier_encode(point_xyz, self.pos_num_freqs)
        q = self.point_pos_mlp(pos_enc).reshape(B * P, 1, -1)
        q = self.norm_q(q)

        key_padding_mask = ~mask.reshape(B * P, N)
        all_masked = key_padding_mask.all(dim=-1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        out = self.out_norm(out.squeeze(1)).reshape(B, P, -1)

        if all_masked.any():
            out = out.masked_fill(all_masked.reshape(B, P, 1), 0.0)
        return out


class Aggregator(BaseEncoder3D):
    def __init__(self, in_dim, embed_dim, pool_heads=8, pool_dropout=0.0, pos_num_freqs=6):
        super().__init__()
        self._output_dim = embed_dim
        self.pool = ViewCrossAttentionPool(in_dim, embed_dim, n_heads=pool_heads,
                                            dropout=pool_dropout, pos_num_freqs=pos_num_freqs)

    # Return context features and center coordinates
    def extractFeatures(self, view_feats, view_dirs, mask, point_xyz):
        context_3d = self.pool(view_feats, view_dirs, mask, point_xyz)
        center_3d = point_xyz
        return context_3d, center_3d

    @property
    def output_dim(self):
        return ((1, None, self._output_dim), (1, None, 3))