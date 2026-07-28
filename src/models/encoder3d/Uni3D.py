import torch
import torch.nn as nn
import timm
from .base import BaseEncoder3D
from .uni3d_backbone import Group, Encoder, PointcloudEncoder, fps, knn_point, square_distance
import logging

class Uni3DEncoder(BaseEncoder3D):
    def __init__(self, ckpt_path,
                 pc_model, trans_dim, group_size, num_group,
                 embed_dim, pc_encoder_dim, patch_dropout, freeze):
        
        super().__init__()
        self._output_dim = trans_dim
        self.freeze = freeze
        self.num_group = num_group
       # create the point transformer model
        point_transformer = timm.create_model(pc_model, pretrained=False, drop_path_rate=0.0)

        # args minimale per PointcloudEncoder
        from types import SimpleNamespace
        args = SimpleNamespace(pc_feat_dim=trans_dim, embed_dim=embed_dim, 
                               group_size=group_size, num_group=num_group,
                               pc_encoder_dim=pc_encoder_dim, patch_dropout=patch_dropout)
        
        self.point_encoder = PointcloudEncoder(point_transformer, args)

        logging.info(f"Uni3D constructed, loading Uni3D weights from {ckpt_path} ...")
        self._load_weights(ckpt_path)

        if self.freeze:
            self.point_encoder.eval()
            for p in self.point_encoder.parameters():
                p.requires_grad_(False)
        
        self._output_shape = None  # Will be inferred later

    def _load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["module"]

        # load only the point_encoder weights 
        clean = {k[len("point_encoder."):]: v for k, v in sd.items() if k.startswith("point_encoder.")}
        missing, unexpected = self.point_encoder.load_state_dict(clean, strict=False)

        # Print missing and unexpected keys for debugging
        if missing or unexpected:
            logging.info(f"-- MISSING: {missing}")
            logging.info(f"-- UNEXPECTED: {unexpected}")

    def extractFeatures(self, pc):
        # pc: (B, N, 6) → xyz + rgb
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        return self._forward_dense(xyz, color)   # (B, num_group, 768)

    def _forward_dense(self, pts, colors):
        ''' In this case we do not use the last layers as UNI3D does, we do not need the allignement with the CLIP space'''

        # this part is the same as in the original Uni3D code
        pe = self.point_encoder
        _, center, features = pe.group_divider(pts, colors)
        tok = pe.encoder(features)
        tok = pe.encoder2trans(tok)
        cls = pe.cls_token.expand(tok.size(0), -1, -1)
        cls_pos = pe.cls_pos.expand(tok.size(0), -1, -1)
        pos = pe.pos_embed(center)
        x = torch.cat((cls, tok), dim=1) + torch.cat((cls_pos, pos), dim=1)
        x = pe.patch_dropout(x)
        x = pe.visual.pos_drop(x)
        for blk in pe.visual.blocks:
            x = blk(x)
        x = pe.visual.norm(x)          
        return x[:, 1:, :]             # Do not return the cls token, only the patch tokens (B, num_group, C)

    
    def _infer_output_shape(self):
        device = next(self.point_encoder.parameters()).device
        was_training = self.point_encoder.training
        self.point_encoder.eval()
        with torch.no_grad():
            # num_group punti bastano: FPS campiona num_group da num_group (identità),
            # kNN con group_size <= num_group è sempre soddisfatto
            dummy = torch.zeros(1, self.num_group, 6, device=device)
            out = self._forward_dense(dummy[:, :, :3].contiguous(),
                                       dummy[:, :, 3:].contiguous())
        self.point_encoder.train(was_training)
        return tuple(out.shape)   # (1, num_group, C)

    @property
    def output_dim(self) -> tuple:
        if self._output_shape is None:
            self._output_shape = self._infer_output_shape()
        return self._output_shape
