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


@torch.no_grad()
def sample_and_score(net, b, grid, chunk):
    B, N, dev = b["t0"].shape[0], grid.shape[0], grid.device
    Rg = random_rotations((B, 1), dev)
    R = Rg @ grid[None]
    t = b["t0"][:, None, :].expand(B, N, 3).contiguous()
    E = torch.cat([_energy(net, b, R[:, i:i + chunk], t[:, i:i + chunk])
                   for i in range(0, N, chunk)], dim=1)
    return R, t, E


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
    # [B, topk, 3, 3], [B, topk, 3]
    # Uguale al test neural refiner di riferimento: nessun gradiente,
    # update LEFT, nuova prediction a ogni step, L1 finale per la scelta.
    R = R.detach().clone()
    t = t.detach().clone()
    B, N = R.shape[:2]
    for _ in range(steps):
        z_pred = net.predict_tokens(b["K"], R, t)             # [B, N, 256, C]
        z_target = net._target[:, None].expand_as(z_pred)     # [B, N, 256, C]
        r_norm, t_norm = refiner(
            z_pred.reshape(B * N, *z_pred.shape[2:]),
            z_target.reshape(B * N, *z_target.shape[2:]),
        )
        delta_r = (r_norm.float() * rot_normalizer).reshape(B, N, 3)
        delta_t = (t_norm.float() * trans_normalizer).reshape(B, N, 3)
        R = so3_exp(delta_r) @ R
        t = t + delta_t
    E = _energy(net, b, R, t)
    return R, t, E


class GTOracle:
    """GT-only diagnostic. Error is geodesic_deg/180 + translation_error/diameter.

    For repeated instances, each hypothesis is matched to the closest GT instance.
    This is NOT a symmetry-aware error or an official BOP metric.
    """
    def __init__(self, dataset_root, split):
        self.dataset_root = dataset_root
        self.split = split
        self._scene_cache = {}
        model_info_path = os.path.join(dataset_root, "models", "models_info.json")
        if not os.path.isfile(model_info_path):
            raise FileNotFoundError(f"Missing model diameters: {model_info_path}")
        with open(model_info_path) as f:
            info = json.load(f)
        self.diameters = {int(k): float(v["diameter"]) for k, v in info.items()}

    def ground_truth(self, scene_id, im_id, obj_id):
        scene_id, im_id, obj_id = int(scene_id), int(im_id), int(obj_id)
        if scene_id not in self._scene_cache:
            path = os.path.join(self.dataset_root, self.split,
                                f"{scene_id:06d}", "scene_gt.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"GT diagnostic requires: {path}")
            with open(path) as f:
                self._scene_cache[scene_id] = json.load(f)
        entries = self._scene_cache[scene_id].get(str(im_id), [])
        poses = []
        for item in entries:
            if int(item["obj_id"]) == obj_id:
                poses.append((
                    np.asarray(item["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                    np.asarray(item["cam_t_m2c"], dtype=np.float32).reshape(3),
                ))
        if not poses:
            raise ValueError(f"No GT scene={scene_id} im={im_id} obj={obj_id}")
        return poses

    def errors(self, R, t, scene_id, im_id, obj_id):
        """Return composite cost and its rotation/translation components [N]."""
        poses = self.ground_truth(scene_id, im_id, obj_id)
        diameter = max(self.diameters[int(obj_id)], 1e-6)
        best = np.full(len(R), np.inf, dtype=np.float64)
        best_rot = np.full(len(R), np.inf, dtype=np.float64)
        best_trans = np.full(len(R), np.inf, dtype=np.float64)
        for Rgt, tgt in poses:
            traces = np.trace(np.matmul(R, Rgt.T), axis1=-2, axis2=-1)
            angles = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
            translations = np.linalg.norm(t - tgt[None], axis=-1)
            cost = angles / 180.0 + translations / diameter
            mask = cost < best
            best[mask] = cost[mask]
            best_rot[mask] = angles[mask]
            best_trans[mask] = translations[mask]
        return best, best_rot, best_trans


def _slice_detection(batch, index):
    """Slice original CPU dataloader batch to limit working GPU memory."""
    B = len(batch["obj_id"])
    return {k: (v[index:index + 1] if torch.is_tensor(v) and v.ndim and v.shape[0] == B else v)
            for k, v in batch.items()}


@torch.no_grad()
def refine_chunked(net, refiner, b, R, t, steps, rot_normalizer,
                   trans_normalizer, chunk):
    """Refine all supplied poses, never materializing all latent predictions."""
    Rs, ts, energies = [], [], []
    for i in range(0, R.shape[1], chunk):
        rc, tc, ec = refine(net, refiner, b, R[:, i:i + chunk],
                            t[:, i:i + chunk], steps,
                            rot_normalizer, trans_normalizer)
        Rs.append(rc)
        ts.append(tc)
        energies.append(ec)
    return torch.cat(Rs, 1), torch.cat(ts, 1), torch.cat(energies, 1)


def run(net, refiner, rot_normalizer, trans_normalizer, ds, args, diag_path):
    """One row per valid detection: GT-oracle errors before/after and final L1 rank."""
    dev = args.device
    grid = super_fibonacci(args.N, dev)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, pin_memory=torch.device(dev).type == "cuda",
                    persistent_workers=args.workers > 0)
    oracle = GTOracle(args.root, args.split)
    out_energy, out_oracle = [], []
    rows = []
    fields = ["scene_id", "im_id", "obj_id",
              "rot_prima_deg", "rot_dopo_deg",
              "trasl_prima_mm", "trasl_dopo_mm", "gt_rank_l1_finale"]

    os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
    with open(diag_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for raw_batch in tqdm(dl, desc="inference", unit="batch"):
            # One detection on GPU at a time; hypotheses are refined in chunks.
            for item_index in range(len(raw_batch["obj_id"])):
                batch = _slice_detection(raw_batch, item_index)
                b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
                if not bool(b["valid"][0]):
                    continue
                scene_id = int(b["scene_id"][0])
                im_id = int(b["im_id"][0])
                obj_id = int(b["obj_id"][0])

                with torch.no_grad():
                    net.prepare(b)
                    R0, t0, E0 = sample_and_score(net, b, grid, args.chunk)
                    if args.all_poses:
                        indices = torch.arange(args.N, device=R0.device)
                    else:
                        indices = E0[0].topk(min(args.topk, args.N), largest=False).indices
                    R1, t1, E1 = refine_chunked(
                        net, refiner, b, R0[:, indices], t0[:, indices],
                        args.steps, rot_normalizer, trans_normalizer,
                        args.refine_chunk)

                # Pre-refinement best among the SAME candidates used post-refinement.
                # This makes the before/after comparison directly interpretable.
                Rpre = R0[0, indices].detach().float().cpu().numpy()
                tpre = t0[0, indices].detach().float().cpu().numpy()
                Rpost = R1[0].detach().float().cpu().numpy()
                tpost = t1[0].detach().float().cpu().numpy()
                epost = E1[0].detach().float().cpu().numpy()

                pre_cost, pre_rot, pre_trans = oracle.errors(
                    Rpre, tpre, scene_id, im_id, obj_id)
                post_cost, post_rot, post_trans = oracle.errors(
                    Rpost, tpost, scene_id, im_id, obj_id)
                best_pre = int(np.argmin(pre_cost))
                best_post = int(np.argmin(post_cost))
                best_l1 = int(np.argmin(epost))
                rank = int(np.where(np.argsort(epost) == best_post)[0][0]) + 1

                row = {
                    "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id,
                    "rot_prima_deg": round(float(pre_rot[best_pre]), 3),
                    "rot_dopo_deg": round(float(post_rot[best_post]), 3),
                    "trasl_prima_mm": round(float(pre_trans[best_pre]), 3),
                    "trasl_dopo_mm": round(float(post_trans[best_post]), 3),
                    "gt_rank_l1_finale": rank,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)

                base = dict(scene_id=scene_id, im_id=im_id, obj_id=obj_id,
                            time=-1, detection_time=0.0)
                score = -float(epost[best_l1])
                out_energy.append(dict(**base, score=score,
                                       R=Rpost[best_l1], t=tpost[best_l1].reshape(3, 1)))
                out_oracle.append(dict(**base, score=score,
                                       R=Rpost[best_post], t=tpost[best_post].reshape(3, 1)))

    if rows:
        ranks = np.array([r["gt_rank_l1_finale"] for r in rows])
        print(f"\nDetection analizzate: {len(rows)}", flush=True)
        print(f"Migliore GT selezionata anche dalla L1 (rank=1): "
              f"{int((ranks == 1).sum())}/{len(rows)} "
              f"({100*(ranks == 1).mean():.1f}%)", flush=True)
        print(f"Posizione media della migliore GT nella L1 finale: {ranks.mean():.2f}", flush=True)
        print(f"Errore rotazione prima/dopo (migliore GT): "
              f"{np.mean([r['rot_prima_deg'] for r in rows]):.2f}° / "
              f"{np.mean([r['rot_dopo_deg'] for r in rows]):.2f}°", flush=True)
        print(f"Errore traslazione prima/dopo (migliore GT): "
              f"{np.mean([r['trasl_prima_mm'] for r in rows]):.2f} / "
              f"{np.mean([r['trasl_dopo_mm'] for r in rows]):.2f} (unità BOP)", flush=True)
    print(f"CSV semplice: {diag_path}", flush=True)
    return out_energy, out_oracle


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

    personal_csv = os.path.splitext(out_csv)[0] + "_timings.csv"
    with open(personal_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "detection_time"])
        for r in final:
            writer.writerow([r["scene_id"], r["im_id"], r["obj_id"],
                             r["score"], f"{r['detection_time']:.8f}"])
    print(f"BOP CSV: {out_csv}", flush=True)
    print(f"Timing CSV: {personal_csv}", flush=True)


def load_cfg(config_path, config_name, overrides):
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=list(overrides))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/riccardop02/datasets/ycbv")
    p.add_argument("--cnos", default="/data/riccardop02/detection/gt/gt_ycbv-test.json")
    p.add_argument("--out", default="/experiments/riccardop02/Inference")
    p.add_argument("--split", default="test")
    p.add_argument("--targets", default=None)
    p.add_argument("--ckpt", default="/experiments/riccardop02/outputs/2026-11-04/10-40-44/checkpoints/best.pt")
    p.add_argument("--refiner_ckpt", default="/experiments/riccardop02/outputs/refinement_perturbed/ref_best.pt")
    p.add_argument("--assets_dir", default="/data/riccardop02/datasets/ycbv/assets3d")
    p.add_argument("--config_path", default="configs")
    p.add_argument("--config_name", default="config")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_score", type=float, default=0.3)
    p.add_argument("--det_per_target", type=int, default=0)
    p.add_argument("--crop_ratio", type=float, default=1.2)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--chunk", type=int, default=40, help="Coarse-search L1 chunk")
    p.add_argument("--refine_chunk", type=int, default=40, help="Refinement chunk in pose hypotheses")
    p.add_argument("--all_poses", action=argparse.BooleanOptionalAction, default=True,
                   help="Refine all N; --no-all_poses refines only initial L1 topk")
    p.add_argument("--steps", type=int, default=3)  # neural refinement; reference REFINE_STEPS=2
    p.add_argument("--lr_t", type=float, default=2.0)  # legacy, non usato dal neural refiner
    p.add_argument("--lr_R", type=float, default=0.03)  # legacy, non usato dal neural refiner
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--save_bop", action="store_true",
                   help="Save and optionally evaluate the two original BOP pose CSVs")
    p.add_argument("--eval", action="store_true", default='True',
                   help="Evaluate BOP CSVs (requires --save_bop)")
    args = p.parse_args()
    if args.N < 1 or args.topk < 1 or args.chunk < 1 or args.refine_chunk < 1:
        p.error("N, topk, chunk and refine_chunk must be positive")

    targets = args.targets or f"{args.root}/test_targets_bop19.json"
    cfg = load_cfg(args.config_path, args.config_name, args.overrides)
    net = EnergyNet(cfg, args.ckpt, args.assets_dir, device=args.device, amp=args.amp)
    refiner, rot_normalizer, trans_normalizer = load_refiner(
        args.refiner_ckpt, net.encoder2d.output_dim[-1], args.device)
    ds = CnosDetDataset(args.root, args.cnos, targets, args.split, net.img_size,
                        margin=args.crop_ratio, min_score=args.min_score,
                        det_per_target=args.det_per_target)
    out = args.out
    if not out.endswith(".csv"):
        run_dir = os.path.join(out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_dir = run_dir + "_" + args.root.split('/')[-1] + "_allpose_gt_simple"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f)
        out = os.path.join(run_dir, f"energy_{os.path.basename(args.root.rstrip('/'))}-{args.split}.csv")

    diag_path = os.path.splitext(out)[0] + "_gt_diagnostics.csv"
    res_energy, res_oracle = run(
        net, refiner, rot_normalizer, trans_normalizer, ds, args, diag_path)
    if args.save_bop or args.eval:
        oracle_out = os.path.splitext(out)[0] + "_oracle_gt.csv"
        select_and_save(res_energy, targets, out)
        select_and_save(res_oracle, targets, oracle_out)

        if args.eval:
            out_dir = os.path.dirname(os.path.abspath(out))
            env = os.environ.copy()
            env["BOP_PATH"] = os.path.dirname(args.root.rstrip("/"))
            env["BOP_RESULTS_PATH"] = out_dir
            env["BOP_EVAL_PATH"] = out_dir
            env["BOP_OUTPUT_PATH"] = out_dir
            for csv_path in (out, oracle_out):
                subprocess.run(
                    [sys.executable, "scripts/eval_bop19_pose.py",
                     "--renderer_type", "vispy",
                     "--result_filenames", os.path.basename(csv_path)],
                    cwd="/home/riccardop02/JEPA/bop_toolkit",
                    env=env, check=True)



if __name__ == "__main__":
    main()
