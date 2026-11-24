from __future__ import annotations
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from tqdm.auto import tqdm
GPU = "0"
os.environ["EGL_DEVICE_ID"] = GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from pytorch3d.transforms import matrix_to_axis_angle, so3_exp_map
from data.dataset_HF_REF import GSOMegaposeDataset
try:
    import wandb
except ImportError:
    wandb = None
LOG = logging.getLogger('posejepa.refiner')
GRID, PATCH, TOKEN_COUNT = 16, 16, 256
# ---------- EXACT sampling convention from the supplied inference ----------
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)
def super_fibonacci(n: int, device: torch.device) -> torch.Tensor:
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = torch.arange(n, device=device, dtype=torch.float64) + 0.5
    r, big_r = torch.sqrt(s / n), torch.sqrt(1.0 - s / n)
    a, b = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    q = torch.stack([r * torch.sin(a), r * torch.cos(a),
                     big_r * torch.sin(b), big_r * torch.cos(b)], dim=-1)
    return quat_to_mat(q).float()
def global_fibonacci(grid: torch.Tensor, batch_size: int,
generator: torch.Generator | None = None) -> torch.Tensor:
    # Inference convention: Rg = random_rotations((B, 1)); R = Rg @ grid[None]
    if generator is None:
        q = torch.randn(batch_size, 1, 4, device=grid.device)
    else:
        q = torch.randn(batch_size, 1, 4, generator=generator).to(grid.device)
    return quat_to_mat(q) @ grid.unsqueeze(0)
    # ---------- EXACT Pluecker formula from the supplied inference ------------
def generate_rays(K, R, trans, centroid, scale, grid_size=GRID, patch_size=PATCH):
    B = K.shape[0]
    y = torch.arange(grid_size, device=K.device, dtype=torch.float32)
    x = torch.arange(grid_size, device=K.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    u = gx * patch_size + patch_size / 2.0
    v = gy * patch_size + patch_size / 2.0
    uv1 = torch.stack((u, v, torch.ones_like(u)), -1).reshape(-1, 3)
    uv1 = uv1.unsqueeze(0).expand(B, -1, -1)
    K_inv = torch.linalg.inv(K)
    d_c = torch.bmm(uv1, K_inv.transpose(1, 2))
    R_inv = R.transpose(1, 2)
    d_o = F.normalize(torch.bmm(d_c, R_inv.transpose(1, 2)), p=2, dim=-1)
    trans_norm = (trans.unsqueeze(-1) + torch.bmm(R, centroid.unsqueeze(-1))).squeeze(-1)
    trans_norm = trans_norm / scale.unsqueeze(-1)
    O = -torch.bmm(R_inv, trans_norm.unsqueeze(-1)).squeeze(-1)
    moment = torch.cross(O.unsqueeze(1).expand_as(d_o), d_o, dim=-1)
    return d_o, moment
    # ---------- FoundationPose-inspired latent architecture -------------------
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
        # Official FoundationPose uses separate Transformer layers per head.
        self.rot_transformer = nn.TransformerEncoderLayer(
        d_model=512, nhead=4, dim_feedforward=512, batch_first=True)
        self.trans_transformer = nn.TransformerEncoderLayer(
        d_model=512, nhead=4, dim_feedforward=512, batch_first=True)
        self.rot_linear = nn.Linear(512, 3)
        self.trans_linear = nn.Linear(512, 3)
        # Identity update at initialization: lets before/after checks be meaningful.
        nn.init.zeros_(self.rot_linear.weight)
        nn.init.zeros_(self.rot_linear.bias)
        nn.init.zeros_(self.trans_linear.weight)
        nn.init.zeros_(self.trans_linear.bias)
    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor):
        # [batch_of_hypotheses, 256, token_dim]
        if z_pred.shape != z_target.shape or z_pred.shape[1] != TOKEN_COUNT:
            raise ValueError(f'Expected matching [N,{TOKEN_COUNT},C], got '
f'{tuple(z_pred.shape)} vs {tuple(z_target.shape)}')
        H = W = GRID
        a = z_pred.transpose(1, 2).reshape(-1, z_pred.shape[-1], H, W)
        b = z_target.transpose(1, 2).reshape(-1, z_target.shape[-1], H, W)
        # Shared weights, like encodeA(torch.cat([A,B],dim=0)) in FoundationPose.
        ab = self.encodeAB(torch.cat((a, b), dim=1))
        seq = ab.flatten(2).transpose(1, 2)  # [N,64,512]
        seq = seq + positional_encoding(seq.shape[1], 512, seq.device, seq.dtype)[None]
        out_rot = self.rot_linear(self.rot_transformer(seq).mean(dim=1))
        out_trans = self.trans_linear(self.trans_transformer(seq).mean(dim=1))
        # Output is a normalized delta; no tanh clamp (global rotations up to pi).
        return out_rot, out_trans
        # ---------- Data validation + inference-aligned translation ----------------
def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}
def first_present(batch: dict, names: tuple[str, ...]):
    for name in names:
        if name in batch and torch.is_tensor(batch[name]):
            return batch[name], name
    return None, None
def ensure_bhw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f'{name} must be [B,H,W] or [B,1,H,W], got {tuple(x.shape)}')
def guess_t0_from_observation(batch, args, image_hw):
    """Equivalent to CnosDetDataset.guess_translation in the provided inference.
    Finds bbox center (u,v) of the object mask; z=median(depth[mask & depth>0]);
    computes t0=inv(K_crop)@[u,v,1]*z. The depth and mask here MUST both be
    aligned to the same cropped image and the cropped intrinsics K.
    """
    # Preferred: already stored by an inference-equivalent dataset preprocess.
    t0, t0_name = first_present(batch, ('t0', 'initial_translation', 'trans_init'))
    if t0 is not None:
        t0 = t0.reshape(-1, 3).float()
        valid, _ = first_present(batch, ('valid', 't0_valid'))
        valid = valid.reshape(-1).bool() if valid is not None else torch.isfinite(t0).all(-1)
        return t0, valid
    depth, depth_name = first_present(batch, ('depth', 'depth_image', 'depth_crop', 'depth_img'))
    mask, mask_name = first_present(batch, ('mask', 'mask_visib', 'object_mask', 'mask_img', 'mask_crop'))
    if depth is None:
        raise KeyError(
            "Missing observed depth: the existing Hydra dataset must return 'depth' "
            "(in the SAME crop as 'image') or precomputed inference-equivalent 't0'. "
            "The old training Engine never uses depth; do NOT substitute GT 'trans'.")
    depth = ensure_bhw(depth, depth_name).float()
    B, H, W = depth.shape
    if (H, W) != tuple(image_hw):
        raise ValueError(f'{depth_name} is {H}x{W} but image crop is {image_hw}. '
                         'Need depth aligned to crop and K_c (as in inference).')
    if mask is None:
        if not args.allow_token_mask or 'mask_tokens' not in batch:
            raise KeyError(
                "Missing pixel-level object mask for inference t0. Add 'mask' to "
                "your Hydra dataset or explicitly pass --allow_token_mask for "
                "an approximate bbox estimated from 16x16 mask_tokens.")
        tok = batch['mask_tokens'].float().reshape(B, 1, GRID, GRID)
        mask = F.interpolate(tok, size=(H, W), mode='nearest')[:, 0]
        mask_name = 'mask_tokens (upsampled approximation)'
        if not getattr(guess_t0_from_observation, '_warned', False):
            LOG.warning('t0 from UPSAMPLED 16x16 mask_tokens: not pixel-exact CNOS bbox')
            guess_t0_from_observation._warned = True
    else:
        mask = ensure_bhw(mask, mask_name).float()
        if mask.shape[-2:] != (H, W):
            raise ValueError(f'{mask_name} {tuple(mask.shape)} != depth {tuple(depth.shape)}')
    K = batch['K'].float()
    if K.shape != (B, 3, 3):
        raise ValueError(f'Expected K crop [B,3,3], got {tuple(K.shape)}')
    depth_scale, _ = first_present(batch, ('depth_scale',))
    if depth_scale is not None:
        depth = depth * depth_scale.reshape(B, 1, 1).float()
    depth = depth * float(args.depth_to_pose_units)
    # The original CnosDetDataset min_mask_px=50 (valid observed depth pixels).
    t0 = torch.zeros(B, 3, device=depth.device, dtype=torch.float32)
    valid = torch.zeros(B, device=depth.device, dtype=torch.bool)
    for b in range(B):
        ys, xs = torch.where(mask[b] > 0)
        if xs.numel() == 0:
            continue
        uc = (xs.min().float() + xs.max().float()) * 0.5
        vc = (ys.min().float() + ys.max().float()) * 0.5
        good = (mask[b] > 0) & (depth[b] > 0) & torch.isfinite(depth[b])
        if int(good.sum()) < args.min_mask_px:
            continue
        zc = depth[b][good].median()
        t0[b] = torch.linalg.solve(K[b], torch.stack((uc, vc, uc.new_tensor(1.0)))) * zc
        valid[b] = True
    return t0, valid
def rotation_error_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    relative = Ra @ Rb.transpose(-1, -2)
    tr = relative.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)) * (180.0 / math.pi)
    # Angle buckets for diagnosing global SO(3) refinement.
#ANGLE_BINS = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180))
ANGLE_BINS = (
    (0, 5),
    (5, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 45),
)
class MetricAccumulator:
    def __init__(self):
        self.sums = {}
        self.count = 0
        self.images = 0
    @torch.no_grad()
    def add(self, R0, t0, R1, t1, Rgt, tgt, loss_r, loss_t,
    delta_r=None, delta_t=None):
        br = rotation_error_deg(R0.float(), Rgt.float()).flatten()
        ar = rotation_error_deg(R1.float(), Rgt.float()).flatten()
        bt = (t0.float() - tgt.float()).norm(dim=-1).flatten()
        at = (t1.float() - tgt.float()).norm(dim=-1).flatten()
        n = br.numel()
        self.count += n
        values = {
            'rot_before_deg': br.sum(), 'rot_after_deg': ar.sum(),
            'trans_before_units': bt.sum(), 'trans_after_units': at.sum(),
            'rot_improved_frac': (ar < br - 1e-4).float().sum(),
            'trans_improved_frac': (at < bt - 1e-7).float().sum(),
            'joint_improved_frac': ((ar < br - 1e-4) & (at < bt - 1e-7)).float().sum(),
            'loss_rot': loss_r.detach().float().sum(),
            'loss_trans': loss_t.detach().float().sum(),
        }
        if delta_r is not None:
            values['rotation_delta_norm_rad'] = delta_r.detach().float().norm(dim=-1).sum()
            values['rotation_delta_norm_deg'] = delta_r.detach().float().norm(dim=-1).sum() * (180 / math.pi)
        if delta_t is not None:
            values['translation_delta_norm_units'] = delta_t.detach().float().norm(dim=-1).sum()
        for low, high in ANGLE_BINS:
            label = f'{low}_{high}'
            mask = (br >= low) & (br < high if high < 180 else br <= high + 1e-4)
            values[f'rot_bin_{label}_count'] = mask.float().sum()
            values[f'rot_bin_{label}_gain_deg_sum'] = ((br - ar) * mask).sum()
            values[f'rot_bin_{label}_improved_sum'] = ((ar < br - 1e-4) & mask).float().sum()
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.item())
    @torch.no_grad()
    def add_oracle(self, before_rot, after_rot, before_trans, after_trans):
        # Inputs [B,N]; take the min ACROSS ALL 128 candidates, not per chunk.
        B = before_rot.shape[0]
        assert after_rot.shape == before_rot.shape
        self.images += B
        vals = {
            'oracle_rot_before_deg': before_rot.min(dim=1).values.sum(),
            'oracle_rot_after_deg': after_rot.min(dim=1).values.sum(),
            'oracle_trans_before_units': before_trans.min(dim=1).values.sum(),
            'oracle_trans_after_units': after_trans.min(dim=1).values.sum(),
        }
        vals['oracle_rot_improved_frac'] = (after_rot.min(1).values < before_rot.min(1).values - 1e-4).float().sum()
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.item())
    def result(self):
        if not self.count:
            return {}
        n = self.count
        out = {k: v / n for k, v in self.sums.items()
               if not k.endswith(('_count', '_sum')) and not k.startswith('oracle_')}
        for low, high in ANGLE_BINS:
            label = f'{low}_{high}'
            cnt = self.sums.get(f'rot_bin_{label}_count', 0.0)
            gain = self.sums.get(f'rot_bin_{label}_gain_deg_sum', 0.0)
            imp = self.sums.get(f'rot_bin_{label}_improved_sum', 0.0)
            out[f'rot_bin_{label}_count'] = cnt
            out[f'rot_bin_{label}_gain_deg'] = gain / cnt if cnt else 0.0
            out[f'rot_bin_{label}_improved_frac'] = imp / cnt if cnt else 0.0
            out[f'rot_bin_{label}_fraction'] = cnt / n
        if self.images:
            for k, v in self.sums.items():
                if k.startswith('oracle_'):
                    out[k] = v / self.images
            out['oracle_rot_gain_deg'] = out['oracle_rot_before_deg'] - out['oracle_rot_after_deg']
            out['oracle_trans_gain_units'] = out['oracle_trans_before_units'] - out['oracle_trans_after_units']
        out['rot_gain_deg'] = out['rot_before_deg'] - out['rot_after_deg']
        out['trans_gain_units'] = out['trans_before_units'] - out['trans_after_units']
        out['loss'] = out['loss_rot'] + out['loss_trans']
        out['hypotheses'] = n
        out['images'] = self.images
        return out
        # ---------- Trainer ---------------------------------------------------------
class RefinementTrainer:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'{self.device} requested but CUDA not available')
        self.amp = bool(args.amp and self.device.type == 'cuda')
        self.grid = super_fibonacci(args.N, self.device)
        self.step = 0
        self.epoch = 0
        self.out_dir = Path(args.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = None
        self._missing_val_warned = False
        self._fixed_val_batches = None
        self.best_val = float("inf")
        LOG.info('Building JEPA modules from Hydra...')
        self.encoder2d = hydra.utils.instantiate(cfg.encoder2d).to(self.device)
        self.encoder3d = hydra.utils.instantiate(
        cfg.encoder3d, in_dim=self.encoder2d.output_dim[-1]).to(self.device)
        self.predictor = hydra.utils.instantiate(
        cfg.Predictor, context_dim=self.encoder3d.output_dim[0][-1],
output_dim=self.encoder2d.output_dim[-1]).to(self.device)
        self._load_jepa(args.jepa_ckpt)
        for m in (self.encoder2d, self.encoder3d, self.predictor):
            m.eval().requires_grad_(False)
        self.refiner = LatentRefineNet(self.encoder2d.output_dim[-1]).to(self.device)
        self.optimizer = torch.optim.AdamW(
        self.refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.02)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.train_loader = self._make_loader('train')
        self.val_loader = self._make_loader('val') if args.val_every > 0 else None
        if args.resume:
            self._load_resume(args.resume)
        if args.wandb and wandb is None:
            raise ImportError('wandb is not installed: pip install wandb, or use --no-wandb')
        if args.wandb:
            self.wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or None,
            dir=str(self.out_dir),
            config={**vars(args), 'hydra_cfg': OmegaConf.to_container(cfg, resolve=True)},
            id=getattr(self, '_wandb_id', None),
            resume='allow' if getattr(self, '_wandb_id', None) else None,
            )
        LOG.info('Refiner parameters: %d; N=%d; chunk=%d; batch=%d',
                 sum(p.numel() for p in self.refiner.parameters()),
                 args.N, args.chunk, args.batch_size)
    def _load_jepa(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f'JEPA checkpoint not found: {path}')
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            LOG.warning('weights_only=True cannot read trusted local checkpoint (%s); '
                        'retrying weights_only=False', exc)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'models' not in ckpt:
            raise KeyError("Checkpoint must contain {'models': {'encoder2d','encoder3d','predictor'}}")
        for name, model in (('encoder2d', self.encoder2d),
                            ('encoder3d', self.encoder3d),
                            ('predictor', self.predictor)):
            model.load_state_dict(ckpt['models'][name], strict=True)
            LOG.info('Loaded JEPA module %s', name)
    def _make_loader(self, split):
        # Same Hydra data parameters, but force the companion dataset subclass
        # without requiring any YAML edits. All architectures remain Hydra-loaded.
        data_args = OmegaConf.to_container(self.cfg.data, resolve=True)
        data_args.pop('_target_', None)
        data_args.update(dataset_source=str(self.cfg.gso.root), split=split,
        out_size=256, patch_size=16,
        context_dirname=self.args.context_dirname or data_args.get('context_dirname', '3DEmbedd'))
        ds = GSOMegaposeDataset(**data_args)
        n_shards = len(ds.shard_paths)
        workers = self.args.workers if split == 'train' else self.args.val_workers
        workers = min(workers, n_shards)
        is_iter = isinstance(ds, IterableDataset)
        batch_size = self.args.batch_size if split == 'train' else self.args.val_batch_size
        return DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train' and not is_iter),
            num_workers=workers, pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(split == 'train' and workers > 0),
            prefetch_factor=1 if workers > 0 else None,
        )
    @staticmethod
    def _set_epoch(loader, epoch):
        if loader is None:
            return
        ds = loader.dataset
        if hasattr(ds, '_epoch_counter'):
            ds._epoch_counter.value = epoch
        elif hasattr(ds, 'set_epoch'):
            ds.set_epoch(epoch)
    def _load_resume(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.refiner.load_state_dict(state['refiner'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        self.step, self.epoch = state['step'], state.get('epoch', 0)
        self._wandb_id = state.get('wandb_run_id')
        self.best_val = state.get('best_val', float("inf"))
        LOG.info('Resumed refiner at global step %d', self.step)
    def _save(self, filename):
        path = self.out_dir / filename
        payload = {
            'refiner': self.refiner.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val': self.best_val,
            'wandb_run_id': self.wandb_run.id if self.wandb_run else None,
            'jepa_ckpt': self.args.jepa_ckpt,
            'arch': {'token_dim': self.encoder2d.output_dim[-1],
                     'rot_normalizer': self.args.rot_normalizer,
                     'trans_normalizer': self.args.trans_normalizer},
            'args': vars(self.args),
        }
        tmp = path.with_name(path.name + '.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, path)
        LOG.info('Saved checkpoint %s', path)
    def _prepare_batch(self, raw):
        batch = to_device(raw, self.device)
        if 'rot_matrix' not in batch or 'trans' not in batch:
            raise KeyError("Hydra training dataset must supply GT 'rot_matrix' and 'trans'")
        if 'K' not in batch or 'centroid' not in batch or 'scale' not in batch:
            raise KeyError("Hydra dataset must supply 'K', 'centroid', 'scale'")
        image, image_key = first_present(batch, ('image', 'img'))
        if image is None:
            raise KeyError("Hydra dataset must supply 'image' (already normalized) or 'img' (0..1)")
        if image.ndim != 4 or image.shape[-2:] != (256, 256):
            raise ValueError(f'Expected 256x256 target crop, got {tuple(image.shape)}')
        if 't0' not in batch:
            raise KeyError('dataset_HF_REF must provide observed t0; GT trans is forbidden for initialization')
        t0, good = guess_t0_from_observation(batch, self.args, image.shape[-2:])
        if not bool(good.any()):
            LOG.warning('No valid initial translation in this batch: skipping')
            return None
        if not bool(good.all()):
            batch = {k: v[good] if torch.is_tensor(v) and v.ndim > 0
                     and v.shape[0] == good.numel() else v
                     for k, v in batch.items()}
            image = image[good]
            t0 = t0[good]
        batch['_t0'] = t0.detach()
        if image_key == 'img':
            # Identical preprocessing to EnergyNet.prepare() of supplied inference.
            mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            image = (image.float() - mean) / std
        batch['_image'] = image.float()
        B = len(t0)
        gt_t = batch['trans'].reshape(B, 3).float()
        ratio = (t0.norm(dim=-1).median() /
                 gt_t.norm(dim=-1).median().clamp_min(1e-8)).item()
        if ratio > 50.0 or ratio < 0.02:
            raise ValueError(
f'Likely depth/GT translation UNIT mismatch: median |t0|/|GT t|={ratio:.5f}. '
                'Set --depth_to_pose_units, verify depth_scale and the units of GT trans.')
        return batch
    @torch.no_grad()
    def _encode_frozen(self, b):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            z_target = self.encoder2d(b['_image']).float()
            if all(k in b for k in ('view_feats', 'view_dirs', 'view_mask', 'point_xyz')):
                ctx_f, ctx_c = self.encoder3d.extractFeatures(
                b['view_feats'].float(), b['view_dirs'].float(),
                b['view_mask'].bool(), b['point_xyz'].float())
            elif 'context_3d' in b and 'center_3d' in b:
                ctx_f, ctx_c = b['context_3d'], b['center_3d']
            else:
                raise KeyError('No 3D context. Need view_feats/view_dirs/view_mask/point_xyz '
                               'or context_3d/center_3d from Hydra dataset.')
        if z_target.ndim != 3 or z_target.shape[1] != TOKEN_COUNT:
            raise ValueError(f'DINO target shape should be [B,256,C], got {tuple(z_target.shape)}')
        return z_target.detach(), ctx_f.detach(), ctx_c.detach()
    @torch.no_grad()
    def _predict_chunk(self, b, R, t, ctx_f, ctx_c):
        # R [B,M,3,3], t [B,M,3]. Identical rays and predictor as inference.
        B, M = R.shape[:2]
        BN = B * M
        K = b['K'].float().repeat_interleave(M, dim=0)
        centroid = b['centroid'].reshape(B, 3).float().repeat_interleave(M, dim=0)
        scale = b['scale'].reshape(B).float().repeat_interleave(M)
        # Ray calculation must stay FP32, just like supplied inference.
        with torch.autocast(self.device.type, enabled=False):
            dirs, moments = generate_rays(
                K, R.reshape(BN, 3, 3).float(), t.reshape(BN, 3).float(),
                centroid, scale)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            z_pred = self.predictor(
                ray_dirs=dirs, ray_moments=moments,
                context_features=ctx_f.repeat_interleave(M, dim=0),
                context_centers=ctx_c.repeat_interleave(M, dim=0))
        return z_pred.detach()
    def _run_batch(self, raw, training: bool, validate_seed: int | None = None):
        b = self._prepare_batch(raw)
        if b is None:
            return None
        B = b['_image'].shape[0]
        gen = None
        if validate_seed is not None:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(validate_seed)
        gt_R = b['rot_matrix'].reshape(B, 3, 3).float()
        # Rotation sampling biased toward small errors, with exact GT examples.
        # Fractions sum to 1.0. R=GT cases teach delta_R=0.
        # Some of these will also have t=GT, teaching both heads to output zero.
        N = self.args.N
        sample_device = 'cpu' if gen is not None else self.device
        remaining = 1.0 - self.args.rot_gt_fraction
        fractions = [
            self.args.rot_gt_fraction,           # exactly 0 degrees
            remaining * (0.35 / 0.90),          # 0-5 degrees
            remaining * (0.25 / 0.90),          # 5-15 degrees
            remaining * (0.20 / 0.90),          # 15-30 degrees
            remaining * (0.10 / 0.90),          # 30-45 degrees
        ]

        counts = [int(N * f) for f in fractions[:-1]]
        counts.append(N - sum(counts))
        bins = [(0.0, 0.0), (0.0, 5.0), (5.0, 15.0),
                (15.0, 30.0), (30.0, 45.0)]
        pieces = []
        for (lo, hi), count in zip(bins, counts):
            if count == 0:
                continue
            if hi == lo:
                angles_piece = torch.zeros(B, count, 1, device=sample_device)
            else:
                angles_piece = (
                    lo + (hi - lo) * torch.rand(
                        B, count, 1, device=sample_device, generator=gen
                    )
                ) * (math.pi / 180.0)
            pieces.append(angles_piece)
        angles = torch.cat(pieces, dim=1).to(self.device)
        axes = F.normalize(
            torch.randn(B, N, 3, device=sample_device, generator=gen), dim=-1
        ).to(self.device)
        delta_R = so3_exp_map((axes * angles).reshape(-1, 3)).reshape(B, N, 3, 3)
        R_all = delta_R @ gt_R[:, None]

        # The first counts[0] candidate rotations are exactly GT.
        # A subset is also forced to GT translation after t sampling.
        n_exact_R = counts[0]
        n_exact_pose = min(n_exact_R, int(N * self.args.full_gt_fraction))

        # ==========================================================
        # TRANSLATION SAMPLING FOR ITERATIVE REFINEMENT
        # ==========================================================
        gt_t = b['trans'].reshape(B, 3).float()
        t_base = b['_t0'].reshape(B, 3).float()
        N = self.args.N
        # Usa lo stesso generatore della validation
        sample_device = 'cpu' if gen is not None else self.device
        choice = torch.rand(
            B, N, 1,
            device=sample_device,
            generator=gen,
        )
        alpha = torch.rand(
            B, N, 1,
            device=sample_device,
            generator=gen,
        )
        noise_dir = torch.randn(
            B, N, 3,
            device=sample_device,
            generator=gen,
        )
        noise_dir = F.normalize(noise_dir, dim=-1)
        noise_radius = torch.rand(
            B, N, 1,
            device=sample_device,
            generator=gen,
        )
        choice = choice.to(self.device)
        alpha = alpha.to(self.device)
        noise_dir = noise_dir.to(self.device)
        noise_radius = noise_radius.to(self.device)
        # Errore iniziale della traslazione
        initial_error = (t_base - gt_t).norm(
                    dim=-1, keepdim=True
        )[:, None, :]
        alpha = torch.where(
            choice < 0.25,
            torch.ones_like(alpha),
            torch.where(
                choice < 0.75,
                alpha,
                torch.zeros_like(alpha),
            ),
        )
        noise_strength = torch.where(
            choice < 0.25,
            torch.zeros_like(choice),
            torch.where(
                choice < 0.75,
                torch.full_like(choice, 0.25),
                torch.where(
                    choice < 0.95,
                    torch.full_like(choice, 0.10),
                    torch.zeros_like(choice),
                ),
            ),
        )
        noise = (
            noise_dir
            * noise_radius
            * initial_error
            * noise_strength
        )
        t_all = (
            gt_t[:, None, :]
            + alpha * (t_base - gt_t)[:, None, :]
            + noise
        ).contiguous()

        # Exact GT pose examples: zero rotation AND zero translation correction.
        # Avoid in-place changes to tensors needed for autograd; these are inputs.
        if n_exact_pose:
            t_all[:, :n_exact_pose] = gt_t[:, None, :]

        # Randomize candidate order so exact GT and near GT are spread over
        # chunks, instead of collecting the easiest cases in the first chunk.
        perm = torch.randperm(N, device=sample_device, generator=gen).to(self.device)
        R_all = R_all[:, perm]
        t_all = t_all[:, perm]
        #R_all = global_fibonacci(self.grid, B, generator=gen)
        #t_all = b['_t0'][:, None, :].expand(B, self.args.N, 3).contiguous()
        #gt_R = b['rot_matrix'].reshape(B, 3, 3).float()
        #gt_t = b['trans'].reshape(B, 3).float()
        z_target, ctx_f, ctx_c = self._encode_frozen(b)
        meter = MetricAccumulator()
        oracle_br, oracle_ar, oracle_bt, oracle_at = [], [], [], []
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for start in range(0, self.args.N, self.args.chunk):
            end = min(self.args.N, start + self.args.chunk)
            M = end - start
            R0 = R_all[:, start:end]
            t0 = t_all[:, start:end]
            z_pred = self._predict_chunk(b, R0, t0, ctx_f, ctx_c)
            z_tgt = z_target[:, None].expand(-1, M, -1, -1).reshape(B*M, *z_target.shape[1:])
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                r_norm, t_norm = self.refiner(z_pred, z_tgt)
            # Geometric targets: relative rotation and translation, in the same
            # camera/object pose convention used by your inference refinement.
            Rgt = gt_R[:, None].expand(-1, M, -1, -1)
            tgt = gt_t[:, None].expand(-1, M, -1)
            R_delta_gt = Rgt @ R0.transpose(-1, -2)
            rot_gt = matrix_to_axis_angle(R_delta_gt.float().reshape(B*M, 3, 3))
            rot_gt = rot_gt / self.args.rot_normalizer
            trans_gt = (tgt - t0).reshape(B*M, 3) / self.args.trans_normalizer
            # L2 squared per 3-vector, equally weighted on normalized deltas.
            lrot = (r_norm.float() - rot_gt).square().sum(-1)
            ltrans = (t_norm.float() - trans_gt).square().sum(-1)
            loss = (lrot + self.args.trans_weight * ltrans).sum() / (B * self.args.N)
            if training:
                self.scaler.scale(loss).backward()
            with torch.no_grad():
                R_delta_pred = so3_exp_map(
                    (r_norm.float() * self.args.rot_normalizer).reshape(B*M, 3))
                R1 = (R_delta_pred @ R0.reshape(B*M, 3, 3)).reshape(B, M, 3, 3)
                t1 = t0 + t_norm.float().reshape(B, M, 3) * self.args.trans_normalizer
                meter.add(R0, t0, R1, t1, Rgt, tgt, lrot,
                          ltrans * self.args.trans_weight,
                          delta_r=r_norm.float().reshape(B, M, 3) * self.args.rot_normalizer,
                          delta_t=t_norm.float().reshape(B, M, 3) * self.args.trans_normalizer)
                oracle_br.append(rotation_error_deg(R0, Rgt))
                oracle_ar.append(rotation_error_deg(R1, Rgt))
                oracle_bt.append((t0 - tgt).norm(dim=-1))
                oracle_at.append((t1 - tgt).norm(dim=-1))
        # Oracle diagnostics: min over ALL N hypotheses for each image.
        # Ground truth is used only for measurement, never to select an inference pose.
        meter.add_oracle(*(torch.cat(v, dim=1) for v in
                           (oracle_br, oracle_ar, oracle_bt, oracle_at)))
        if training:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.refiner.parameters(),
                self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        else:
            grad_norm = None
        result = meter.result()
        if grad_norm is not None:
            result['grad_norm'] = float(grad_norm.item())
        return result
    def _val(self):
        if self.val_loader is None:
            return {}
        self.refiner.eval()
        output = []
        try:
            if self._fixed_val_batches is None:
                self._set_epoch(self.val_loader, 0)
                self._fixed_val_batches = []
                iterator = iter(self.val_loader)
                for _ in range(self.args.val_batches):
                    try:
                        raw = next(iterator)
                    except StopIteration:
                        break
                    # Cached CPU batches: fixed scene + fixed sampled rotations.
                    self._fixed_val_batches.append(raw)
                LOG.info('Fixed validation set: %d batches', len(self._fixed_val_batches))
            for i, raw in enumerate(self._fixed_val_batches):
                with torch.no_grad():
                    result = self._run_batch(raw, training=False,
                    validate_seed=self.args.seed + i)
                if result:
                    output.append(result)
        finally:
            self.refiner.train()
        if not output:
            return {}
        total = sum(v['hypotheses'] for v in output)
        images = sum(v['images'] for v in output)
        result = {}
        for k in output[0]:
            if k in ('hypotheses', 'images'):
                continue
            if k.endswith('_count'):
                result[k] = sum(v[k] for v in output)
            elif k.startswith('oracle_'):
                result[k] = sum(v[k] * v['images'] for v in output) / max(images, 1)
            elif k.startswith('rot_bin_') and k.endswith(('_gain_deg', '_improved_frac')):
                label = k.removeprefix('rot_bin_').rsplit('_', 2)[0] if k.endswith('_improved_frac') else k.removeprefix('rot_bin_').removesuffix('_gain_deg')
                count_key = f'rot_bin_{label}_count'
                denom = sum(v[count_key] for v in output)
                result[k] = sum(v[k] * v[count_key] for v in output) / denom if denom else 0.0
            else:
                result[k] = sum(v[k] * v['hypotheses'] for v in output) / total
        result['hypotheses'] = total
        result['images'] = images
        return result
    def _log(self, prefix, metrics):
        if not metrics:
            return
        fields = {f'{prefix}/{key}': value for key, value in metrics.items()}
        if self.wandb_run:
            wandb.log(fields, step=self.step)
        LOG.info('%s step=%d: %s', prefix, self.step,
                 ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()
                           if k in ('loss', 'rot_before_deg', 'rot_after_deg',
                                    'trans_before_units', 'trans_after_units',
                                    'rot_improved_frac', 'trans_improved_frac')))
    def fit(self):
        self._set_epoch(self.train_loader, self.epoch)
        self.refiner.train()
        # Validation iniziale
        if self.step == 0 and self.val_loader is not None:
            self._log('val', self._val())
        it = iter(self.train_loader)
        pbar = tqdm(
        total=self.args.steps,
        initial=self.step,
        desc="Training Refiner",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.1,
        )
        try:
            while self.step < self.args.steps:
                try:
                    raw = next(it)
                except StopIteration:
                    self.epoch += 1
                    self._set_epoch(self.train_loader, self.epoch)
                    it = iter(self.train_loader)
                    try:
                        raw = next(it)
                    except StopIteration as exc:
                        raise RuntimeError(
                            'Training dataset is empty'
                        ) from exc
                result = self._run_batch(raw, training=True)
                if not result:
                    continue
                self.step += 1
                # Aggiorna la progress bar
                pbar.update(1)
                pbar.set_postfix(
                loss=f"{result['loss']:.4f}",
                rot=f"{result['rot_after_deg']:.2f}°",
                epoch=self.epoch,
                )
                if (
        self.step == 1
                    or self.step % self.args.log_every == 0
                ):
                    self._log('train', result)
                    if self.wandb_run:
                        wandb.log(
                            {
                                'train/lr':
                                    self.optimizer.param_groups[0]['lr']
                            },
                            step=self.step,
                        )
                if (
        self.val_loader is not None
                    and self.step % self.args.val_every == 0
                ):
                    val = self._val()
                    self._log('val', val)
                    if val and val['loss'] < self.best_val:
                        self.best_val = val['loss']
                        self._save('best.pt')
                if self.step % self.args.save_every == 0:
                    self._save('last.pt')
        finally:
            pbar.close()
            if self.step > 0:
                self._save('last.pt')
            if self.wandb_run:
                wandb.finish()
def parse_args():
    p = argparse.ArgumentParser(description='Train PoseJEPA latent pose refiner')
    p.add_argument('--config_path', default='configs')
    p.add_argument('--config_name', default='config')
    p.add_argument('--overrides', nargs='*', default=[])
    p.add_argument('--jepa_ckpt', default='/experiments/riccardop02/outputs/2026-11-04/10-40-44/checkpoints/best.pt', help='Full checkpoint containing 3 frozen JEPA models')
    p.add_argument('--resume', default='/experiments/riccardop02/outputs/refinement_near_gt/last.pt', help='Refiner last.pt from an earlier run')
    p.add_argument('--out_dir', default='/experiments/riccardop02/outputs/refinement_near_gt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--rot_gt_fraction', type=float, default=0.10,
                   help='Fraction of exact GT rotations (zero delta_R)')
    p.add_argument('--full_gt_fraction', type=float, default=0.05,
                   help='Fraction of exact GT poses (both R and t)')
    p.add_argument('--chunk', type=int, default=16, help='Hypotheses per forward; gradient accumulation over N')
    p.add_argument('--batch_size', type=int, default=12, help='Number of images, not hypotheses')
    p.add_argument('--val_batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--val_workers', type=int, default=0)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--val_every', type=int, default=100)
    p.add_argument('--val_batches', type=int, default=4)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--rot_normalizer', type=float, default=math.pi,
help='Scale of 3D axis-angle normalized correction (global <=180deg)')
    p.add_argument('--trans_normalizer', type=float, default=100.0,
help='Scale of delta translation in SAME units as dataset trans')
    p.add_argument('--trans_weight', type=float, default=0.1)
    p.add_argument('--min_mask_px', type=int, default=50)
    p.add_argument('--context_dirname', default=None,
help='Context folder within each GSO object; default existing dataset setting (usually 3DEmbedd)')
    p.add_argument('--depth_to_pose_units', type=float, default=1.0,
help='Multiply observed depth by this to match GT trans units')
    p.add_argument('--allow_token_mask', action='store_true',
help='Approximate t0 bbox using mask_tokens when no pixel-level mask exists')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wandb_project', default='jepa-pose-refiner')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--wandb_name', default='latent-refiner-near-GT-zero-corrections')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.N < 1 or args.chunk < 1 or args.steps < 1:
        p.error('N, chunk and steps must be > 0')
    if not (0 <= args.full_gt_fraction <= args.rot_gt_fraction <= 1.0):
        p.error('Require 0 <= full_gt_fraction <= rot_gt_fraction <= 1.0')
    if args.rot_normalizer <= 0 or args.trans_normalizer <= 0:
        p.error('rot_normalizer and trans_normalizer must be > 0')
    return args
def main():
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Allow project-local architecture modules to resolve via Hydra _target_.
    sys.path.insert(0, str(Path.cwd()))
    config_dir = str(Path(args.config_path).expanduser().resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    LOG.info('Using Hydra config: %s (%s)', config_dir, args.config_name)
    LOG.info('N=%d, rotation GT fraction=%.2f, full GT fraction=%.2f',
             args.N, args.rot_gt_fraction, args.full_gt_fraction)
    RefinementTrainer(cfg, args).fit()
if __name__ == '__main__':
    main()
