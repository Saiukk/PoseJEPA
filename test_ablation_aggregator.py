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
# SO(3): sampling e mappe
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
# Dataset: detection CNOS
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
# Stessa generazione dei raggi del training
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

            feats = a["view_feats"].float()
            mask = a["view_mask"].bool()
            points = a["point_xyz"].float()

            weights = mask.unsqueeze(-1).to(feats.dtype)
            ctx_f = (feats * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

            ctx_f = ctx_f.unsqueeze(0)
            ctx_c = points.unsqueeze(0)

            self._obj_cache[obj_id] = {
                "ctx_f": ctx_f,
                "ctx_c": ctx_c,
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

    def forward(self, img, mask, depth, K, obj_id, R, t):
        assert self._target is not None, "chiama net.prepare(b) prima di forward"
        N = R.shape[1]
        Es = []
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
            tgt = self._target[idx][:, None]
            err = (pred.float().view(G, N, *tgt.shape[2:]) - tgt).abs().mean(-1)
            w = self._tok[idx][:, None, :]
            Es.append((err * w).sum(-1) / w.sum(-1).clamp(min=1.0))
        if len(Es) == 1:
            return Es[0]
        return torch.cat(Es, 0)[self._inv]


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


def refine(net, b, R, t, steps, lr_t, lr_R, tol=1e-5, patience=20,
           lr_end_frac=0.02, min_frac=0.5):
    R = R.detach().clone()
    t = t.detach().clone().requires_grad_(True)
    w = torch.zeros_like(t, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_R},
                            {"params": [t], "lr": lr_t}])

    def factor(k):
        c = 0.5 * (1.0 + math.cos(math.pi * min(k, steps) / steps))
        return lr_end_frac + (1.0 - lr_end_frac) * c

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=factor)
    best_E = torch.full(R.shape[:2], float("inf"), device=R.device)
    best_R, best_t = R.clone(), t.detach().clone()
    stall = 0
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        E = _energy(net, b, so3_exp(w) @ R, t)
        E.sum().backward()
        with torch.no_grad():
            imp = E.detach() < best_E
            gain = (best_E - E.detach())[imp].max() if imp.any() else 0.0
            best_E = torch.where(imp, E.detach(), best_E)
            # Usa la rotazione corrente invece della base R: e' quella valutata nell'energia.
            evaluated_R = so3_exp(w.detach()) @ R
            best_R[imp] = evaluated_R[imp]
            best_t[imp] = t.detach()[imp]
            stall = 0 if gain > tol else stall + 1

        if it >= int(min_frac * steps) and stall >= patience:
            break
        opt.step()
        sched.step()
        with torch.no_grad():
            R = so3_exp(w) @ R
            w.zero_()
    with torch.no_grad():
        E = _energy(net, b, R, t)
        imp = E < best_E
        best_E = torch.where(imp, E, best_E)
        best_R[imp] = R[imp]
        best_t[imp] = t.detach()[imp]
    return best_R, best_t, best_E


def run(net, ds, args):
    dev = args.device
    grid = super_fibonacci(args.N, dev)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, pin_memory=True,
                    persistent_workers=args.workers > 0)

    out = []
    image_times = defaultdict(float)
    total_inference_time = 0.0
    total_processed_detections = 0

    for b in tqdm(dl, desc="inference"):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}

        # Misura il batch senza cambiare il batching corrente.
        # Synchronize obbligatoria per una misura wall-clock CUDA.
        if torch.cuda.is_available() and torch.device(dev).type == "cuda":
            torch.cuda.synchronize(device=dev)
        start = time.perf_counter()

        net.prepare(b)
        R, t, E = sample_and_score(net, b, grid, args.chunk)
        idx = E.topk(args.topk, dim=1, largest=False).indices
        R = torch.gather(R, 1, idx[..., None, None].expand(-1, -1, 3, 3))
        t = torch.gather(t, 1, idx[..., None].expand(-1, -1, 3))
        with torch.enable_grad():
            R, t, E = refine(net, b, R, t, args.steps, args.lr_t, args.lr_R)

        if torch.cuda.is_available() and torch.device(dev).type == "cuda":
            torch.cuda.synchronize(device=dev)
        batch_time = time.perf_counter() - start
        n_batch = len(b["obj_id"])
        detection_time = batch_time / n_batch  # Media ammortizzata, NON latenza individuale
        total_inference_time += batch_time
        total_processed_detections += n_batch

        best = E.argmin(dim=1)
        ar = torch.arange(len(best), device=dev)
        for j in range(len(best)):
            scene_id = int(b["scene_id"][j])
            im_id = int(b["im_id"][j])
            # Stima per immagine aggregata sui batch (non un runtime BOP ufficiale).
            image_times[(scene_id, im_id)] += detection_time
            if not bool(b["valid"][j]):
                continue
            e = E[ar[j], best[j]]
            out.append({
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": int(b["obj_id"][j]),
                "score": float(-e),
                "R": R[ar[j], best[j]].cpu().numpy(),
                "t": t[ar[j], best[j]].cpu().numpy().reshape(3, 1),
                "time": -1,  # invariato: nessun runtime BOP completo misurato
                "detection_time": detection_time,
            })

    if total_processed_detections:
        print(f"Tempo inferenza batch totale: {total_inference_time:.3f} s", flush=True)
        print(f"Tempo medio ammortizzato per detection: "
              f"{total_inference_time / total_processed_detections:.6f} s", flush=True)
        print(f"Detection processate: {total_processed_detections}", flush=True)
        print(f"Immagini distinte: {len(image_times)}", flush=True)
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

    # File supplementare, riferito esattamente alle detection selezionate per il BOP.
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
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--lr_t", type=float, default=2.0)#2.0
    p.add_argument("--lr_R", type=float, default=0.03) #0.03
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval", type=bool, default=True)
    args = p.parse_args()

    targets = args.targets or f"{args.root}/test_targets_bop19.json"
    cfg = load_cfg(args.config_path, args.config_name, args.overrides)
    net = EnergyNet(cfg, args.ckpt, args.assets_dir, device=args.device, amp=args.amp)
    ds = CnosDetDataset(args.root, args.cnos, targets, args.split, net.img_size,
                        margin=args.crop_ratio, min_score=args.min_score,
                        det_per_target=args.det_per_target)
    res = run(net, ds, args)

    out = args.out
    if not out.endswith(".csv"):
        run_dir = os.path.join(out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_dir = run_dir + "_" + args.root.split('/')[-1] + "_refinement_ablation_aggregator"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f)
        out = os.path.join(run_dir, f"energy_{os.path.basename(args.root.rstrip('/'))}-{args.split}.csv")

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
