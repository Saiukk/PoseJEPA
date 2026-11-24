import argparse
import csv
import json
import math
import os
import sys
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from PIL import Image
from pycocotools import mask as mask_utils
from pytorch3d.transforms import so3_exp_map
from torch.utils.data import DataLoader, Dataset
from bop_toolkit_lib import inout
# --------------------------------------------------------------------------
# SO(3): sampling e mappe (INVARIATO)
# --------------------------------------------------------------------------
def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)
def super_fibonacci(n, device="cpu"):
    """Uniform sampling in SO(3)."""
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = torch.arange(n, device=device, dtype=torch.float64) + 0.5
    r, R = torch.sqrt(s / n), torch.sqrt(1.0 - s / n)
    a, b = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    q = torch.stack([r * torch.sin(a), r * torch.cos(a),
                     R * torch.sin(b), R * torch.cos(b)], dim=-1)
    return quat_to_mat(q).float()
def random_rotations(shape, device):
    return quat_to_mat(torch.randn(*shape, 4, device=device))
def so3_exp(w):
    """Mappa esponenziale di SO(3) (pytorch3d)."""
    return so3_exp_map(w.reshape(-1, 3)).reshape(*w.shape[:-1], 3, 3)
# --------------------------------------------------------------------------
# REFINER: implementazione ESATTA fornita (nessun import da moduli esterni)
# --------------------------------------------------------------------------
GRID, PATCH, TOKEN_COUNT = 16, 16, 256
class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, kernel=3, stride=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel, stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(y)), inplace=True)
def positional_encoding(length: int, channels: int, device, dtype):
    # Classic sinusoidal positional encoding, same conceptual role as FoundationPose.
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, channels, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / channels))
    pe = torch.zeros(length, channels, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.to(dtype=dtype)
class LatentRefineNet(nn.Module):
    """Same block/channel counts as official FoundationPose RefineNet.
    Its RGB-D encoder starts with 7x7/3x3 stride-2 convolutions; here we instead
    project DINO tokens using 1x1, preserving the 16x16 spatial token grid.
    """
    def __init__(self, token_dim: int = 1024):
        super().__init__()
        self.encodeAB = nn.Sequential(
            ConvBNReLU(token_dim * 2, 256, kernel=1),
            BasicResidualBlock(256),
            BasicResidualBlock(256),
            ConvBNReLU(256, 512, kernel=3, stride=2),
            BasicResidualBlock(512),
            BasicResidualBlock(512),
        )
        self.rot_transformer = nn.TransformerEncoderLayer(
        d_model=512, nhead=4, dim_feedforward=512, batch_first=True)
        self.trans_transformer = nn.TransformerEncoderLayer(
        d_model=512, nhead=4, dim_feedforward=512, batch_first=True)
        self.rot_linear = nn.Linear(512, 3)
        self.trans_linear = nn.Linear(512, 3)
        nn.init.zeros_(self.rot_linear.weight)
        nn.init.zeros_(self.rot_linear.bias)
        nn.init.zeros_(self.trans_linear.weight)
        nn.init.zeros_(self.trans_linear.bias)
    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor):
        if z_pred.shape != z_target.shape or z_pred.shape[1] != TOKEN_COUNT:
            raise ValueError(f'Expected matching [N,{TOKEN_COUNT},C], got '
            f'{tuple(z_pred.shape)} vs {tuple(z_target.shape)}')
        H = W = GRID
        a = z_pred.transpose(1, 2).reshape(-1, z_pred.shape[-1], H, W)
        b = z_target.transpose(1, 2).reshape(-1, z_target.shape[-1], H, W)
        ab = self.encodeAB(torch.cat((a, b), dim=1))
        seq = ab.flatten(2).transpose(1, 2)
        seq = seq + positional_encoding(seq.shape[1], 512, seq.device, seq.dtype)[None]
        out_rot = self.rot_linear(self.rot_transformer(seq).mean(dim=1))
        out_trans = self.trans_linear(self.trans_transformer(seq).mean(dim=1))
        return out_rot, out_trans
# --------------------------------------------------------------------------
# Dataset: detection CNOS (INVARIATO)
# --------------------------------------------------------------------------
class CnosDetDataset(Dataset):
    """Crop coerente di RGB, maschera e depth, come nel training."""
    def __init__(self, root, cnos_json, targets_json, split="test", crop_size=256,
        margin=1.2, min_score=0.0, min_mask_px=50, det_per_target=3):
        with open(cnos_json) as f:
            dets = json.load(f)
        n_all = len(dets)
        with open(targets_json) as f:
            inst = {(t["scene_id"], t["im_id"], t["obj_id"]): t["inst_count"]
                    for t in json.load(f)}
        groups = defaultdict(list)
        for d in dets:
            if d["score"] < min_score:
                continue
            k = (d["scene_id"], d["image_id"], d["category_id"])
            if k in inst:
                groups[k].append(d)
        sel = []
        for k, ds_ in groups.items():
            if det_per_target > 0:
                ds_.sort(key=lambda d: d["score"], reverse=True)
                ds_ = ds_[:max(inst[k] * det_per_target, 1)]
            sel += ds_
        sel.sort(key=lambda d: (d["category_id"], d["scene_id"], d["image_id"]))
        self.dets = sel
        print(f"detection CNOS: {n_all} -> {len(self.dets)} after 2 filter applied", flush=True)
        self.root, self.split = root, split
        self.size, self.margin, self.min_px = crop_size, margin, min_mask_px
        self._cams = {}
    def __len__(self):
        return len(self.dets)
    def _cam(self, sid):
        if sid not in self._cams:
            p = f"{self.root}/{self.split}/{sid:06d}/scene_camera.json"
            with open(p) as f:
                self._cams[sid] = json.load(f)
        return self._cams[sid]
    def _read_image(self, base, iid):
        for sub, ext in (("rgb", "png"), ("rgb", "jpg"), ("gray", "tif")):
            p = f"{base}/{sub}/{iid:06d}.{ext}"
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                return img[..., ::-1]
        raise FileNotFoundError(f"no image found for {base} / {iid}")
    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3, np.float32), False
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = (mask > 0) & (depth > 0)
        if valid.sum() < self.min_px:
            return np.zeros(3, np.float32), False
        zc = float(np.median(depth[valid]))
        t0 = (np.linalg.inv(K) @ np.array([uc, vc, 1.0])) * zc
        return t0.astype(np.float32), True
    def _get_crop_box(self, mask_bool, bbox):
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            x, y, w, h = bbox
            x0, x1, y0, y1 = x, x + w, y, y + h
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(max(x1 - x0, y1 - y0) * self.margin / 2.0, 1.0)
        return (int(round(cx - half)), int(round(cy - half)),
                int(round(cx + half)), int(round(cy + half)))
    def __getitem__(self, i):
        d = self.dets[i]
        sid, iid = d["scene_id"], d["image_id"]
        base = f"{self.root}/{self.split}/{sid:06d}"
        img = self._read_image(base, iid)
        cam = self._cam(sid)[str(iid)]
        K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)
        depth = None
        for ext in (".png", ".tif", ".tiff"):
            depth_path = f"{base}/depth/{iid:06d}{ext}"
            if os.path.isfile(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    break
        depth = depth.astype(np.float32) * cam.get("depth_scale", 1.0)
        rle = dict(d["segmentation"])
        h, w = rle["size"]
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, h, w)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_utils.decode(rle).astype(np.uint8)
        t0, ok = self.guess_translation(depth, mask, K)
        box = self._get_crop_box(mask > 0, d["bbox"])
        xmin, ymin, xmax, ymax = box
        dsz = (self.size, self.size)
        img_c = np.asarray(Image.fromarray(np.ascontiguousarray(img)).crop(box).resize(dsz, Image.BICUBIC))
        msk_c = np.asarray(Image.fromarray(mask * 255).crop(box).resize(dsz, Image.NEAREST))
        dep_c = np.asarray(Image.fromarray(depth).crop(box).resize(dsz, Image.NEAREST))
        scale_x = self.size / float(xmax - xmin)
        scale_y = self.size / float(ymax - ymin)
        K_c = K.copy()
        K_c[0, 0] = K[0, 0] * scale_x
        K_c[1, 1] = K[1, 1] * scale_y
        K_c[0, 2] = (K[0, 2] - xmin) * scale_x
        K_c[1, 2] = (K[1, 2] - ymin) * scale_y
        return {
            "img": torch.from_numpy(img_c.copy()).permute(2, 0, 1).float() / 255,
            "mask": torch.from_numpy(msk_c.copy()).float()[None] / 255,
            "depth": torch.from_numpy(dep_c.copy())[None],
            "K": torch.from_numpy(K_c),
            "t0": torch.from_numpy(t0),
            "valid": torch.tensor(ok),
            "scene_id": sid, "im_id": iid, "obj_id": d["category_id"],
            "det_score": float(d["score"]),
        }
# --------------------------------------------------------------------------
# Stessa generazione dei raggi del training (INVARIATO)
# --------------------------------------------------------------------------
def generate_rays(K, R, trans, centroid, scale, grid_size=16, patch_size=16):
    B = K.shape[0]
    device = K.device
    y = torch.arange(grid_size, device=device, dtype=torch.float32)
    x = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    u = grid_x * patch_size + (patch_size / 2.0)
    v = grid_y * patch_size + (patch_size / 2.0)
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3)
    uv1 = uv1.unsqueeze(0).expand(B, -1, -1)
    K_inv = torch.linalg.inv(K)
    d_c = torch.bmm(uv1, K_inv.transpose(1, 2))
    R_inv = R.transpose(1, 2)
    d_o = F.normalize(torch.bmm(d_c, R_inv.transpose(1, 2)), p=2, dim=-1)
    centroid_col = centroid.unsqueeze(-1)
    trans_norm = (trans.unsqueeze(-1) + torch.bmm(R, centroid_col)).squeeze(-1)
    trans_norm = trans_norm / scale.unsqueeze(-1)
    O = -torch.bmm(R_inv, trans_norm.unsqueeze(-1)).squeeze(-1)
    O_exp = O.unsqueeze(1).expand(-1, d_o.shape[1], -1)
    moment = torch.cross(O_exp, d_o, dim=-1)
    return d_o, moment
class EnergyNet(nn.Module):
    GRID, PATCH = 16, 16
    def __init__(self, cfg, checkpoint, assets_dir, device="cuda", amp=False):
        super().__init__()
        assert checkpoint is not None and os.path.isfile(checkpoint), f"No valid checkpoint at path {checkpoint}"
        assert assets_dir is not None and os.path.isdir(assets_dir), f"assets_dir non valida: {assets_dir}"
        self.assets_dir, self.amp = assets_dir, amp
        self.img_size = self.GRID * self.PATCH
        self.dev = torch.device(device)
        enc2d = hydra.utils.instantiate(cfg.encoder2d)
        enc3d = hydra.utils.instantiate(cfg.encoder3d, in_dim=enc2d.output_dim[-1])
        pred = hydra.utils.instantiate(
        cfg.Predictor,
        context_dim=enc3d.output_dim[0][-1],
        output_dim=enc2d.output_dim[-1],
        )
        self.encoder2d, self.encoder3d, self.predictor = enc2d, enc3d, pred
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"weights_only=True fallito ({type(e).__name__}), uso weights_only=False", flush=True)
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name in ("encoder2d", "encoder3d", "predictor"):
            getattr(self, name).load_state_dict(ckpt["models"][name], strict=True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(self.dev).eval()
        for p_ in self.parameters():
            p_.requires_grad_(False)
        self._obj_cache = {}
        self._target = self._tok = None
        self._groups, self._inv = None, None
    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.amp)
    def _load_assets(self, obj_id):
        d = os.path.join(self.assets_dir, f"obj_{obj_id:06d}", "3DEmbedd_v3")
        ld = lambda n: torch.from_numpy(np.load(os.path.join(d, n + ".npy"))).to(self.dev)
        return {"view_feats": ld("view_feats"), "view_dirs": ld("view_dirs"),
                "view_mask": ld("mask"), "point_xyz": ld("points"),
                "centroid": ld("centroid"), "scale": ld("scale")}
    @torch.no_grad()
    def _get_obj(self, obj_id):
        if obj_id not in self._obj_cache:
            a = self._load_assets(obj_id)
            ctx_f, ctx_c = self.encoder3d.extractFeatures(
                a["view_feats"].float().unsqueeze(0),
                a["view_dirs"].float().unsqueeze(0),
                a["view_mask"].bool().unsqueeze(0),
                a["point_xyz"].float().unsqueeze(0),
            )
            self._obj_cache[obj_id] = {
                "ctx_f": ctx_f, "ctx_c": ctx_c,
                "centroid": a["centroid"].reshape(1, 3).float(),
                "scale": a["scale"].reshape(1).float(),
            }
        return self._obj_cache[obj_id]
    @torch.no_grad()
    def prepare(self, b):
        img = (b["img"] - self.mean) / self.std
        with self._autocast():
            self._target = self.encoder2d(img).float()
        m = F.avg_pool2d(b["mask"], self.PATCH)
        self._tok = m.flatten(1)
        groups = defaultdict(list)
        for i, o in enumerate(b["obj_id"].tolist()):
            groups[int(o)].append(i)
        self._groups, order = [], []
        for o, ids in groups.items():
            self._groups.append((self._get_obj(o), torch.tensor(ids, device=self.dev)))
            order += ids
        if len(self._groups) > 1:
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            self._inv = inv.to(self.dev)
        else:
            self._inv = None
    # Nuovo helper: ritorna le features del predictor per ogni ipotesi.
    # Usa esattamente lo stesso generate_rays / predictor del vecchio forward.
    def predict_tokens(self, K, R, t):
        N = R.shape[1]
        predictions = []
        for o, idx in self._groups:
            G = idx.numel()
            GN = G * N
            with torch.autocast("cuda", enabled=False):
                d, mo = generate_rays(
                    K[idx].float().repeat_interleave(N, dim=0),
                    R[idx].reshape(GN, 3, 3).float(),
                    t[idx].reshape(GN, 3).float(),
                    o["centroid"].expand(GN, -1), o["scale"].expand(GN),
                    grid_size=self.GRID, patch_size=self.PATCH,
                )
            with self._autocast():
                pred = self.predictor(
                ray_dirs=d, ray_moments=mo,
                context_features=o["ctx_f"].expand(GN, -1, -1),
                context_centers=o["ctx_c"].expand(GN, -1, -1),
                )
            predictions.append(pred.float().reshape(G, N, *pred.shape[1:]))
        if len(predictions) == 1:
            return predictions[0]
        return torch.cat(predictions, dim=0)[self._inv]
    def forward(self, img, mask, depth, K, obj_id, R, t):
        assert self._target is not None, "chiama net.prepare(b) prima di forward"
        pred = self.predict_tokens(K, R, t)
        err = (pred - self._target[:, None]).abs().mean(-1)
        w = self._tok[:, None, :]
        return (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)
def _energy(net, b, R, t):
    return net(b["img"], b["mask"], b["depth"], b["K"], b["obj_id"], R, t)
# --------------------------------------------------------------------------
# NUOVO: sostituisce ESCLUSIVAMENTE il refinement con gradient descent.
# --------------------------------------------------------------------------
def load_refiner(checkpoint_path, token_dim, device):
    assert os.path.isfile(checkpoint_path), f"Refiner checkpoint non trovato: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    refiner = LatentRefineNet(token_dim=token_dim).to(device)
    refiner.load_state_dict(checkpoint["refiner"], strict=True)
    refiner.eval().requires_grad_(False)
    rot_normalizer = checkpoint["arch"]["rot_normalizer"]
    trans_normalizer = checkpoint["arch"]["trans_normalizer"]
    print(f"Refiner loaded: {checkpoint_path}", flush=True)
    print(f"Training step: {checkpoint['step']}", flush=True)
    print(f"Rotation normalizer: {rot_normalizer}", flush=True)
    print(f"Translation normalizer: {trans_normalizer}", flush=True)
    rot_normalizer = torch.as_tensor(rot_normalizer, device=device, dtype=torch.float32)
    trans_normalizer = torch.as_tensor(trans_normalizer, device=device, dtype=torch.float32)
    return refiner, rot_normalizer, trans_normalizer
@torch.no_grad()
def refine(net, refiner, b, R, t, steps, rot_normalizer, trans_normalizer):
    """Refinement originale, senza calcolare la L1 e senza top-K intermedi.
    La chiamata avviene su chunk di al massimo `args.chunk` pose.
    """
    R = R.detach().clone()
    t = t.detach().clone()
    B, N = R.shape[:2]
    for _ in range(steps):
        z_pred = net.predict_tokens(b["K"], R, t)
        z_target = net._target[:, None].expand_as(z_pred)
        r_norm, t_norm = refiner(
            z_pred.reshape(B * N, *z_pred.shape[2:]),
            z_target.reshape(B * N, *z_target.shape[2:]),
        )
        delta_r = (r_norm.float() * rot_normalizer).reshape(B, N, 3)
        delta_t = (t_norm.float() * trans_normalizer).reshape(B, N, 3)
        R = so3_exp(delta_r) @ R  # aggiornamento LEFT come nel tuo codice
        t = t + delta_t
    return R, t
class FoundationPoseScorer:
    """Usa esclusivamente il pretrained ScoreNetMultiPair di FoundationPose.
    Per ridurre la VRAM: render + preprocessing + extract_feat in chunk,
    una sola cross-attention finale sulle N feature di tutte le pose.
    Niente scorers retrained e niente refinement FoundationPose.
    """
    def __init__(self, fp_root, dataset_root, amp=True, fp_chunk=16, device='cuda'):
        if torch.device(device).type != 'cuda':
            raise ValueError('FoundationPose scorer richiede CUDA')
        root = os.path.abspath(fp_root)
        needed = [
            'Utils.py',
            'learning/models/network_modules.py',
            'learning/models/score_network.py',
            'learning/datasets/pose_dataset.py',
            'learning/datasets/h5_dataset.py',
            'weights/2024-01-11-20-02-45/config.yml',
            'weights/2024-01-11-20-02-45/model_best.pth',
        ]
        missing = [p for p in needed if not os.path.isfile(os.path.join(root, p))]
        if missing:
            raise FileNotFoundError(
                'File FoundationPose mancanti in ' + root + ':\n  ' +
                '\n  '.join(missing) +
                '\nEsegui prima setup_fp_scorer.sh'
            )
        sys.path.insert(0, root)
        try:
            import kornia
            import trimesh
            import nvdiffrast.torch as dr
            from learning.models.score_network import ScoreNetMultiPair
            from learning.datasets.h5_dataset import ScoreMultiPairH5Dataset
            from learning.datasets.pose_dataset import BatchPoseData
            from Utils import (make_mesh_tensors, compute_crop_window_tf_batch,
                               transform_pts, nvdiffrast_render)
            from omegaconf import OmegaConf
        except Exception as exc:
            raise ImportError(
                'Impossibile importare lo scorer FoundationPose. Verifica le dipendenze '
                '(nvdiffrast, kornia, trimesh, open3d, transformations, h5py, ecc.).'
            ) from exc
        self.kornia = kornia
        self.trimesh = trimesh
        self.BatchPoseData = BatchPoseData
        self.make_mesh_tensors = make_mesh_tensors
        self.compute_crop_window_tf_batch = compute_crop_window_tf_batch
        self.transform_pts = transform_pts
        self.nvdiffrast_render = nvdiffrast_render
        self.fp_chunk = fp_chunk
        self.amp = amp
        self.dataset_root = dataset_root
        self.glctx = dr.RasterizeCudaContext()
        weights_dir = os.path.join(root, 'weights', '2024-01-11-20-02-45')
        self.cfg = OmegaConf.load(os.path.join(weights_dir, 'config.yml'))
        for key, value in {
            'use_normal': False, 'use_BN': False, 'zfar': float('inf'),
            'c_in': 4, 'normalize_xyz': False, 'crop_ratio': 1.2,
        }.items():
            if key not in self.cfg or self.cfg[key] is None:
                self.cfg[key] = value
        self.dataset = ScoreMultiPairH5Dataset(
        cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
        self.model = ScoreNetMultiPair(
        cfg=self.cfg, c_in=int(self.cfg['c_in'])).cuda().eval()
        ckpt_path = os.path.join(weights_dir, 'model_best.pth')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(
            checkpoint['model'] if 'model' in checkpoint else checkpoint,
        strict=True)
        self.model.requires_grad_(False)
        with open(os.path.join(dataset_root, 'models', 'models_info.json')) as f:
            info = json.load(f)
        self.diameters = {int(k): float(v['diameter']) / 1000.0 for k, v in info.items()}
        self.current_obj = None
        self.mesh = None
        self.mesh_tensors = None
        print('FoundationPose scorer pretrained caricato: ' + ckpt_path, flush=True)
    def _load_obj(self, obj_id):
        if self.current_obj == obj_id:
            return
        # Un solo modello BOP su GPU alla volta, non tutti gli oggetti.
        self.mesh_tensors = None
        self.mesh = None
        mesh_file = os.path.join(self.dataset_root, 'models', f'obj_{obj_id:06d}.ply')
        if not os.path.isfile(mesh_file):
            raise FileNotFoundError(mesh_file)
        mesh = self.trimesh.load(mesh_file, force='mesh', process=False)
        # BOP mesh in mm -> metri; NON centrare la mesh, la posa BOP e' rispetto
        # al frame originale della mesh, e va mantenuto per il renderer.
        mesh.apply_scale(0.001)
        self.mesh = mesh
        self.mesh_tensors = self.make_mesh_tensors(mesh)
        self.current_obj = obj_id
    @torch.inference_mode()
    def _make_crop(self, rgb, depth, K, pose_chunk, diameter):
        """Preprocessing fedele a make_crop_data_batch di predict_score.py FP.
        Input originali HxW: RGB uint8 [0,255], depth in metri, K originale.
        Tutti i crop qui sono quelli pose-conditioned di FP, non i crop CNOS.
        """
        render_size = tuple(int(v) for v in self.cfg['input_resize'])
        H, W = depth.shape
        B = len(pose_chunk)
        poseAs = torch.as_tensor(pose_chunk, dtype=torch.float32, device='cuda')
        # Utils.compute_crop_window_tf_batch imposta globalmente il tensor type CUDA
        # (implementazione FP originale). Lo ripristiniamo subito dopo.
        try:
            tf_to_crops = self.compute_crop_window_tf_batch(
            pts=self.mesh.vertices, H=H, W=W, poses=poseAs, K=K,
            crop_ratio=float(self.cfg['crop_ratio']),
            out_size=(render_size[1], render_size[0]),
            method='box_3d', mesh_diameter=diameter)
        finally:
            torch.set_default_tensor_type(torch.FloatTensor)
        bbox2d_crop = torch.as_tensor(
            np.array([0, 0, render_size[0]-1, render_size[1]-1]).reshape(2, 2),
            dtype=torch.float32, device='cuda')
        bbox2d_ori = self.transform_pts(
            bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(-1, 4)
        extra = {}
        rgb_r, depth_r, _ = self.nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=poseAs, context='cuda',
        get_normal=bool(self.cfg['use_normal']), glctx=self.glctx,
        mesh_tensors=self.mesh_tensors, output_size=self.cfg['input_resize'],
        bbox2d=bbox2d_ori, use_light=True, extra=extra)
        rgbAs = rgb_r.permute(0, 3, 1, 2) * 255.0
        depthAs = depth_r[..., None].permute(0, 3, 1, 2)
        xyz_mapAs = extra['xyz_map'].permute(0, 3, 1, 2)
        rgb_ori = torch.as_tensor(rgb, device='cuda', dtype=torch.float32)
        dep_ori = torch.as_tensor(depth, device='cuda', dtype=torch.float32)
        rgbBs = self.kornia.geometry.transform.warp_perspective(
            rgb_ori.permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
        depthBs = self.kornia.geometry.transform.warp_perspective(
            dep_ori[None, None].expand(B, -1, -1, -1),
            tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
        if rgbAs.shape[-2:] != render_size:
            rgbAs = self.kornia.geometry.transform.warp_perspective(
                rgbAs, tf_to_crops, dsize=render_size,
                mode='bilinear', align_corners=False)
            depthAs = self.kornia.geometry.transform.warp_perspective(
                depthAs, tf_to_crops, dsize=render_size,
                mode='nearest', align_corners=False)
            xyz_mapAs = self.kornia.geometry.transform.warp_perspective(
                xyz_mapAs, tf_to_crops, dsize=render_size,
                mode='nearest', align_corners=False)
        Ks = torch.as_tensor(K, dtype=torch.float32).reshape(1, 3, 3).expand(B, 3, 3)
        mesh_diameters = torch.full((B,), float(diameter), device='cuda')
        pose_data = self.BatchPoseData(
        rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs,
        normalAs=None, normalBs=None, poseA=poseAs, xyz_mapAs=xyz_mapAs,
        tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
        return self.dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)
    @torch.inference_mode()
    def score(self, rgb, depth_mm, K, R, t_mm, obj_id):
        """Returns [N] logits FP per la selezione finale; maggiore e' meglio."""
        self._load_obj(obj_id)
        N = R.shape[0]
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], N, axis=0)
        poses[:, :3, :3] = np.asarray(R, dtype=np.float32)
        poses[:, :3, 3] = np.asarray(t_mm, dtype=np.float32) / 1000.0
        # Depth BOP in mm -> metri. depth_mm e' gia' depth_scale-corrected.
        depth_m = np.asarray(depth_mm, dtype=np.float32) / 1000.0
        K = np.asarray(K, dtype=np.float32)
        feats = []
        for s in range(0, N, self.fp_chunk):
            pose_data = self._make_crop(
            rgb, depth_m, K, poses[s:s+self.fp_chunk],
        self.diameters[obj_id])
            A = torch.cat((pose_data.rgbAs, pose_data.xyz_mapAs), dim=1).float()
            B = torch.cat((pose_data.rgbBs, pose_data.xyz_mapBs), dim=1).float()
            if pose_data.normalAs is not None:
                A = torch.cat((A, pose_data.normalAs.float()), dim=1)
                B = torch.cat((B, pose_data.normalBs.float()), dim=1)
            # extract_feat opera su ciascuna ipotesi indipendentemente.
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                f = self.model.extract_feat(A, B)
            feats.append(f.float())
            del pose_data, A, B, f
        # UNICA cross-attention su tutte le N pose: nessun ranking per chunk!
        all_feats = torch.cat(feats, dim=0).unsqueeze(0)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            x, _ = self.model.att_cross(
                all_feats, all_feats, all_feats, need_weights=False)
            logits = self.model.linear(x).reshape(-1).float()
        return logits.cpu()
def _load_full_bop_frame(ds, scene_id, im_id):
    """RGB/depth/K full-frame, SOLO per scorer FP (non cambia input JEPA)."""
    base = f'{ds.root}/{ds.split}/{scene_id:06d}'
    rgb = np.ascontiguousarray(ds._read_image(base, im_id))  # no negative strides RGB
    cam = ds._cam(scene_id)[str(im_id)]
    K = np.asarray(cam['cam_K'], dtype=np.float32).reshape(3, 3)
    depth = None
    for ext in ('.png', '.tif', '.tiff'):
        p = f'{base}/depth/{im_id:06d}{ext}'
        if os.path.isfile(p):
            depth = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                break
    if depth is None:
        raise FileNotFoundError(f'Depth mancante: {base}/depth/{im_id:06d}')
    depth_mm = depth.astype(np.float32) * float(cam.get('depth_scale', 1.0))
    return rgb, depth_mm, K
def run(net, refiner, rot_normalizer, trans_normalizer, ds, args, fp_scorer=None):
    dev = args.device
    grid = super_fibonacci(args.N, dev)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
    shuffle=False, pin_memory=True,
    persistent_workers=args.workers > 0)
    out = []
    image_times = defaultdict(float)
    total_inference_time = 0.0
    total_processed_detections = 0
    detection_counter = 0
    for batch in tqdm(dl, desc=f'inference [{args.selection}]'):
        # Processiamo una detection per volta: anche con batch_size=64 e N=128
        # nessuna operazione espande B*N in VRAM.
        for j in range(len(batch['obj_id'])):
            b = {k: v[j:j+1].to(dev, non_blocking=True) for k, v in batch.items()}
            scene_id = int(b['scene_id'][0].item())
            im_id = int(b['im_id'][0].item())
            obj_id = int(b['obj_id'][0].item())
            detection_counter += 1
            if not bool(b['valid'][0]):
                continue
            if torch.device(dev).type == 'cuda':
                torch.cuda.synchronize(device=dev)
            start_time = time.perf_counter()
            with torch.inference_mode():
                net.prepare(b)
                # Identico sampling SO(3) / t0; nessuna energy intermedia.
                # Stesse ipotesi per --selection energy e FPScorer:
                # la randomizzazione non dipende dal forward dello scorer.
                rng = torch.Generator(device='cpu')
                rng.manual_seed(args.seed + detection_counter)
                q = torch.randn(1, 1, 4, generator=rng).to(dev)
                Rg = quat_to_mat(q)
                R0 = Rg @ grid[None]
                t0 = b['t0'][:, None, :].expand(1, args.N, 3).contiguous()
                R_parts, t_parts = [], []
                for s in range(0, args.N, args.chunk):
                    R_c, t_c = refine(
                        net, refiner, b, R0[:, s:s+args.chunk],
                        t0[:, s:s+args.chunk], args.steps,
                        rot_normalizer, trans_normalizer)
                    R_parts.append(R_c)
                    t_parts.append(t_c)
                R = torch.cat(R_parts, dim=1)  # [1,N,3,3] piccolo
                t = torch.cat(t_parts, dim=1)  # [1,N,3]
                if args.selection == 'energy':
                    # L1 SOLO dopo il refinement di tutte le ipotesi.
                    E = torch.cat([
                        _energy(net, b, R[:, s:s+args.chunk], t[:, s:s+args.chunk])
                        for s in range(0, args.N, args.chunk)
                    ], dim=1)
                    best = int(E[0].argmin().item())
                    pose_score = -float(E[0, best].item())
                else:
                    rgb, depth_mm, K = _load_full_bop_frame(ds, scene_id, im_id)
                    logits = fp_scorer.score(
                        rgb, depth_mm, K,
                        R[0].float().cpu().numpy(),
                        t[0].float().cpu().numpy(), obj_id)
                    best = int(logits.argmax().item())
                    # FP score_logits sono relativi all'insieme della detection.
                    # Per scegliere tra detection diverse dello stesso oggetto,
                    # usa la confidence CNOS; nessuna energy entra nella scelta.
                    pose_score = float(b['det_score'][0].item())
                chosen_R = R[0, best].float().cpu().numpy()
                chosen_t = t[0, best].float().cpu().numpy().reshape(3, 1)
            if torch.device(dev).type == 'cuda':
                torch.cuda.synchronize(device=dev)
            detection_time = time.perf_counter() - start_time
            total_inference_time += detection_time
            total_processed_detections += 1
            image_times[(scene_id, im_id)] += detection_time
            out.append({
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': obj_id,
                'score': pose_score,
                'R': chosen_R,
                't': chosen_t,  # mm per BOP; metri SOLO internamente a FP
                'time': -1,
                'detection_time': detection_time,
            })
    # BOP: tempo complessivo di tutte le detection processate per immagine.
    # Il valore e' identico per tutte le pose della stessa immagine.
    for r in out:
        r["time"] = image_times[(r["scene_id"], r["im_id"])]
    if total_processed_detections:
        print(f'Tempo inferenza totale: {total_inference_time:.3f} s', flush=True)
        print(f'Tempo medio per detection: '
             f'{total_inference_time / total_processed_detections:.6f} s', flush=True)
        print(f'Detection processate: {total_processed_detections}', flush=True)
        print(f'Immagini distinte: {len(image_times)}', flush=True)
        print(f'Tempo medio per immagine (BOP): '
             f'{total_inference_time / len(image_times):.6f} s', flush=True)
    return out
def select_and_save(results, targets_json, out_csv):
    with open(targets_json) as f:
        cnt = {(t["scene_id"], t["im_id"], t["obj_id"]): t["inst_count"]
               for t in json.load(f)}
    g = defaultdict(list)
    for r in results:
        g[(r["scene_id"], r["im_id"], r["obj_id"])].append(r)
    final = []
    for k, hs in g.items():
        if k in cnt:
            hs.sort(key=lambda r: r["score"], reverse=True)
            final += hs[:cnt[k]]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # Il parser BOP si aspetta soltanto 7 colonne standard.
    bop_results = [{k: v for k, v in r.items() if k != "detection_time"} for r in final]
    inout.save_bop_results(out_csv, bop_results)
    print(f"BOP CSV: {out_csv}", flush=True)
def load_cfg(config_path, config_name, overrides):
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=list(overrides))
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/riccardop02/datasets/tless")
    p.add_argument("--cnos", default="/data/riccardop02/detection/cnos/cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json")
    p.add_argument("--out", default="/experiments/riccardop02/Inference")
    p.add_argument("--split", default="test_primesense")
    p.add_argument("--targets", default=None)
    p.add_argument("--ckpt", default="/experiments/riccardop02/outputs/2026-11-04/10-40-44/checkpoints/best.pt")
    p.add_argument("--refiner_ckpt", default="/experiments/riccardop02/outputs/refinement_near_gt/long_ref_best.pt")
    p.add_argument("--assets_dir", default="/data/riccardop02/datasets/tless/assets3d")
    p.add_argument("--config_path", default="configs")
    p.add_argument("--config_name", default="config")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_score", type=float, default=0.3)
    p.add_argument("--det_per_target", type=int, default=0)
    p.add_argument("--crop_ratio", type=float, default=1.2)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--seed", type=int, default=42,
    help="Rotazioni identiche tra esecuzioni energy/FPScorer")
    #p.add_argument("--topk", type=int, default=0, help="Legacy: ignorato, ora si raffinano tutte le N pose")
    p.add_argument("--chunk", type=int, default=40,
    help="Numero massimo di pose per chiamata al refiner/energy")
    p.add_argument("--selection", choices=["energy", "FPScorer"], default="energy")
    p.add_argument("--fp_root", default="/data/riccardop02/SCORER")
    p.add_argument("--fp_chunk", type=int, default=16,
    help="Batch rendering+extract_feat FoundationPose; attention globale su N pose")
    p.add_argument("--steps", type=int, default=2)  # neural refinement; reference REFINE_STEPS=2
    # p.add_argument("--lr_t", type=float, default=2.0)  # legacy, non usato dal neural refiner
    # p.add_argument("--lr_R", type=float, default=0.03)  # legacy, non usato dal neural refiner
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval", type=bool, default=True)
    args = p.parse_args()
    if args.N < 1 or args.chunk < 1 or args.fp_chunk < 1:
        p.error('--N, --chunk e --fp_chunk devono essere >= 1')
    targets = args.targets or f"{args.root}/test_targets_bop19.json"
    cfg = load_cfg(args.config_path, args.config_name, args.overrides)
    net = EnergyNet(cfg, args.ckpt, args.assets_dir, device=args.device, amp=args.amp)
    refiner, rot_normalizer, trans_normalizer = load_refiner(
        args.refiner_ckpt, net.encoder2d.output_dim[-1], args.device)
    ds = CnosDetDataset(args.root, args.cnos, targets, args.split, net.img_size,
    margin=args.crop_ratio, min_score=args.min_score,
    det_per_target=args.det_per_target)
    fp_scorer = None
    if args.selection == 'FPScorer':
        fp_scorer = FoundationPoseScorer(
            args.fp_root, args.root, amp=args.amp,
            fp_chunk=args.fp_chunk, device=args.device)
    res = run(net, refiner, rot_normalizer, trans_normalizer,
              ds, args, fp_scorer=fp_scorer)
    out = args.out
    if not out.endswith(".csv"):
        run_dir = os.path.join(out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_dir = run_dir + "_" + args.root.split('/')[-1] + f"_{args.selection}_refSTEP2_CNOS"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f)
        out = os.path.join(run_dir, f"{args.selection.lower()}_{os.path.basename(args.root.rstrip('/'))}-{args.split}.csv")
    print(f"Results saved to: {out}", flush=True)
    select_and_save(res, targets, out)
    if args.eval:
        out_dir = os.path.dirname(os.path.abspath(out))
        env = os.environ.copy()
        env["BOP_PATH"] = os.path.dirname(args.root.rstrip("/"))
        env["BOP_RESULTS_PATH"] = out_dir
        env["BOP_EVAL_PATH"] = out_dir
        env["BOP_OUTPUT_PATH"] = out_dir
        subprocess.run(
            [sys.executable, "scripts/eval_bop19_pose.py",
             "--renderer_type", "vispy",
             "--result_filenames", os.path.basename(out)],
             cwd="/home/riccardop02/JEPA/bop_toolkit",
             env=env,
             check=True,
        )
if __name__ == "__main__":
    main()
