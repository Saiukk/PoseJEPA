import argparse
import json
import math
import os
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
    """Uniform sampling in SO3"""
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041            # radice reale di x^4 = x + 4
    s = torch.arange(n, device=device, dtype=torch.float64) + 0.5
    r, R = torch.sqrt(s / n), torch.sqrt(1.0 - s / n)
    a, b = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    q = torch.stack([r * torch.sin(a), r * torch.cos(a),
                     R * torch.sin(b), R * torch.cos(b)], dim=-1)
    return quat_to_mat(q).float()


def random_rotations(shape, device):
    return quat_to_mat(torch.randn(*shape, 4, device=device))


def so3_exp(w):
    """Mappa esponenziale di SO(3) (pytorch3d). w: (..., 3) -> (..., 3, 3)"""
    return so3_exp_map(w.reshape(-1, 3)).reshape(*w.shape[:-1], 3, 3)


# --------------------------------------------------------------------------
# Dataset: detection CNOS -> crop identico al training
# --------------------------------------------------------------------------
class CnosDetDataset(Dataset):
    """
    Prepara i dati per l'inference, con lo stesso crop usato nel training
    (GSOMegaposeDataset): box quadrato dal bbox della MASCHERA, lato =
    max(w, h) * crop_ratio, coordinate arrotondate, PIL crop + resize
    (BICUBIC per l'immagine, NEAREST per maschera/depth).

    Le detection vengono filtrate PRIMA dell'inference: si tengono solo quelle
    che compaiono nei target BOP19 e, per ogni (scene, im, obj), le prime
    inst_count * det_per_target per score CNOS.
    """

    def __init__(self, root, cnos_json, targets_json, split="test", crop_size=256,
                 margin=1.2, min_score=0.0, min_mask_px=50, det_per_target=3):
        dets = json.load(open(cnos_json))
        n_all = len(dets)

        inst = {(t["scene_id"], t["im_id"], t["obj_id"]): t["inst_count"]
                for t in json.load(open(targets_json))}

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

        # ordinato per oggetto: i batch sono (quasi) omogenei -> 1 sola chiamata al predictor
        sel.sort(key=lambda d: (d["category_id"], d["scene_id"], d["image_id"]))
        self.dets = sel
        print(f"detection CNOS: {n_all} -> {len(self.dets)} dopo il filtro sui target", flush=True)

        self.root, self.split = root, split
        self.size, self.margin, self.min_px = crop_size, margin, min_mask_px
        self._cams = {}

    def __len__(self):
        return len(self.dets)

    def _cam(self, sid):
        # carica (se assente) il dict dei parametri camera della scena sid
        if sid not in self._cams:
            p = f"{self.root}/{self.split}/{sid:06d}/scene_camera.json"
            self._cams[sid] = json.load(open(p))
        return self._cams[sid]

    def _read_image(self, base, iid):
        # Legge l'immagine RGB per scena e image id
        for sub, ext in (("rgb", "png"), ("rgb", "jpg"), ("gray", "tif")):
            p = f"{base}/{sub}/{iid:06d}.{ext}"
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                # BGR -> RGB
                return img[..., ::-1]
        raise FileNotFoundError(f"no image found for {base} / {iid}")

    def guess_translation(self, depth, mask, K):
        """Stima iniziale di t dal centro della maschera + depth mediana"""
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3, np.float32), False            # maschera vuota
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = (mask > 0) & (depth > 0)
        if valid.sum() < self.min_px:
            return np.zeros(3, np.float32), False            # niente depth valida
        zc = float(np.median(depth[valid]))
        t0 = (np.linalg.inv(K) @ np.array([uc, vc, 1.0])) * zc
        return t0.astype(np.float32), True

    def _get_crop_box(self, mask_bool, bbox):
        """Identico al training: box quadrato centrato sul bbox della MASCHERA.
        Se la maschera e' vuota ripiega sul bbox di CNOS (il sample verra'
        comunque scartato perche' valid=False)."""
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            x, y, w, h = bbox
            x0, x1, y0, y1 = x, x + w, y, y + h
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(max(x1 - x0, y1 - y0) * self.margin / 2.0, 1.0)   # evita box a lato 0
        return (
            int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)),
        )

    def __getitem__(self, i):
        d = self.dets[i]
        sid, iid = d["scene_id"], d["image_id"]
        base = f"{self.root}/{self.split}/{sid:06d}"

        # Legge i parametri (img, mask, K, depth)
        img = self._read_image(base, iid)
        cam = self._cam(sid)[str(iid)]
        K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)
        depth = cv2.imread(f"{base}/depth/{iid:06d}.png", cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) * cam.get("depth_scale", 1.0)   # mm

        rle = dict(d["segmentation"])
        h, w = rle["size"]
        if isinstance(rle["counts"], list):                  # RLE non compressa
            rle = mask_utils.frPyObjects(rle, h, w)
        elif isinstance(rle["counts"], str):                 # RLE compressed (stringa)
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_utils.decode(rle).astype(np.uint8)

        # init t con il metodo FP
        t0, ok = self.guess_translation(depth, mask, K)

        # ---- crop identico al training ----
        box = self._get_crop_box(mask > 0, d["bbox"])
        xmin, ymin, xmax, ymax = box
        dsz = (self.size, self.size)

        img_c = np.asarray(
            Image.fromarray(np.ascontiguousarray(img)).crop(box).resize(dsz, Image.BICUBIC))
        msk_c = np.asarray(
            Image.fromarray(mask * 255).crop(box).resize(dsz, Image.NEAREST))
        dep_c = np.asarray(
            Image.fromarray(depth).crop(box).resize(dsz, Image.NEAREST))   # mode "F"

        scale_x = self.size / float(xmax - xmin)
        scale_y = self.size / float(ymax - ymin)
        K_c = K.copy()
        K_c[0, 0] = K[0, 0] * scale_x
        K_c[1, 1] = K[1, 1] * scale_y
        K_c[0, 2] = (K[0, 2] - xmin) * scale_x
        K_c[1, 2] = (K[1, 2] - ymin) * scale_y

        return {
            "img": torch.from_numpy(img_c.copy()).permute(2, 0, 1).float() / 255,
            "mask": torch.from_numpy(msk_c.copy()).float()[None] / 255,   # 0/255 -> 0/1 come to_tensor
            "depth": torch.from_numpy(dep_c.copy())[None],
            "K": torch.from_numpy(K_c),
            "t0": torch.from_numpy(t0),
            "valid": torch.tensor(ok),
            "scene_id": sid, "im_id": iid, "obj_id": d["category_id"],
            "det_score": float(d["score"]),
        }


# --------------------------------------------------------------------------
# Rete di energia: encoder2d + aggregator (encoder3d) + predictor
# --------------------------------------------------------------------------
def generate_rays(K, R, trans, centroid, scale, grid_size=16, patch_size=16):
    """ Ray equal to training"""
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
    d_o = F.normalize(torch.bmm(d_c, R_inv.transpose(1, 2)), p=2, dim=-1)   # (B, T, 3)

    centroid_col = centroid.unsqueeze(-1)                                            # (B, 3, 1)
    trans_norm = (trans.unsqueeze(-1) + torch.bmm(R, centroid_col)).squeeze(-1)      # (B, 3)
    trans_norm = trans_norm / scale.unsqueeze(-1)

    O = -torch.bmm(R_inv, trans_norm.unsqueeze(-1)).squeeze(-1)                      # (B, 3)
    O_exp = O.unsqueeze(1).expand(-1, d_o.shape[1], -1)
    moment = torch.cross(O_exp, d_o, dim=-1)                                         # (B, T, 3)
    return d_o, moment


class EnergyNet(nn.Module):
    # for DINOV3
    GRID, PATCH = 16, 16

    def __init__(self, cfg, checkpoint, assets_dir, device="cuda", amp=False):
        super().__init__()
        assert checkpoint is not None and os.path.isfile(checkpoint), \
            f"No valid checkpoint at path {checkpoint}"
        assert assets_dir is not None and os.path.isdir(assets_dir), \
            f"assets_dir non valida: {assets_dir}"
        self.assets_dir, self.amp = assets_dir, amp
        self.img_size = self.GRID * self.PATCH
        self.dev = torch.device(device)

        # costruisce i 3 modelli: aggregator, enc2d e predictor
        enc2d = hydra.utils.instantiate(cfg.encoder2d)
        enc3d = hydra.utils.instantiate(cfg.encoder3d, in_dim=enc2d.output_dim[-1])
        pred = hydra.utils.instantiate(
            cfg.Predictor,
            context_dim=enc3d.output_dim[0][-1],
            output_dim=enc2d.output_dim[-1],
        )
        self.encoder2d, self.encoder3d, self.predictor = enc2d, enc3d, pred

        # carica i pesi (prima in modo sicuro; se il ckpt contiene oggetti custom, ripiega)
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

    # ---- assets per oggetto -------------------------------------------------
    def _load_assets(self, obj_id):
        d = os.path.join(self.assets_dir, f"obj_{obj_id:06d}", "3DEmbedd")
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
                "ctx_f": ctx_f, "ctx_c": ctx_c,                              # (1, P, D), (1, P, 3)
                "centroid": a["centroid"].reshape(1, 3).float(),
                "scale": a["scale"].reshape(1).float(),
            }
        return self._obj_cache[obj_id]

    @torch.no_grad()
    def prepare(self, b):
        img = (b["img"] - self.mean) / self.std
        with self._autocast():
            self._target = self.encoder2d(img).float()                       # (B, T, D)
        m = F.avg_pool2d(b["mask"], self.PATCH)                              # (B, 1, g, g), copertura in [0,1]
        self._tok = m.flatten(1)                                             # (B, T), pesi continui come in training

        # raggruppa le detection del batch per obj_id: una chiamata al predictor per gruppo
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
        # K (B,3,3) del crop; R (B,N,3,3); t (B,N,3) in mm, frame camera.
        # img/mask/depth/obj_id non servono qui: sono gia' stati usati in prepare().
        assert self._target is not None, "chiama net.prepare(b) prima di forward"
        N = R.shape[1]
        Es = []
        for o, idx in self._groups:
            G = idx.numel()
            GN = G * N
            with torch.autocast("cuda", enabled=False):                      # raggi in fp32
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
                )                                                            # (G*N, T, D)
            tgt = self._target[idx][:, None]                                 # (G, 1, T, D)
            err = (pred.float().view(G, N, *tgt.shape[2:]) - tgt).abs().mean(-1)   # (G, N, T)
            w = self._tok[idx][:, None, :]                                   # (G, 1, T)
            Es.append((err * w).sum(-1) / w.sum(-1).clamp(min=1.0))          # (G, N)
        if len(Es) == 1:
            return Es[0]                                                     # (B, N)
        return torch.cat(Es, 0)[self._inv]                                   # (B, N), ordine originale


# --------------------------------------------------------------------------
# Ricerca: sampling rotazioni -> top-K -> refinement
# --------------------------------------------------------------------------
def _energy(net, b, R, t):
    return net(b["img"], b["mask"], b["depth"], b["K"], b["obj_id"], R, t)


@torch.no_grad()
def sample_and_score(net, b, grid, chunk):
    """grid: (N,3,3) Super-Fibonacci. Per ogni detection: rotazione globale
    casuale (cambia l'offset della griglia) + stessa traslazione t0 per tutte
    le ipotesi (come FoundationPose): la traslazione viene corretta solo dal
    refinement."""
    B, N, dev = b["t0"].shape[0], grid.shape[0], grid.device
    Rg = random_rotations((B, 1), dev)
    R = Rg @ grid[None]                                          # (B,N,3,3)
    t = b["t0"][:, None, :].expand(B, N, 3).contiguous()         # (B,N,3), uguale per tutte
    E = torch.cat([_energy(net, b, R[:, i:i + chunk], t[:, i:i + chunk])
                   for i in range(0, N, chunk)], dim=1)
    return R, t, E


def refine(net, b, R, t, steps, lr_t, lr_R):
    """Adam su t + Adam riemanniano su R (left-trivialization, come nei test):
    R <- Exp(w) @ R, con w nello spazio tangente, azzerato dopo ogni step.
    Le ipotesi sono indipendenti (loss = somma), e Adam e' per-elemento,
    quindi ogni ipotesi evolve come se fosse ottimizzata da sola."""
    R = R.detach().clone()
    t = t.detach().clone().requires_grad_(True)
    w = torch.zeros_like(t, requires_grad=True)                 # (B, K, 3)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_R},
                            {"params": [t], "lr": lr_t}])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _energy(net, b, so3_exp(w) @ R, t).sum().backward()
        opt.step()
        with torch.no_grad():
            R = so3_exp(w) @ R                                   # retrazione: resta in SO(3)
            w.zero_()
    with torch.no_grad():
        E = _energy(net, b, R, t)
    return R, t.detach(), E


def run(net, ds, args):
    dev = args.device
    # Sampling su R (la traslazione parte da t0 per tutte le ipotesi)
    grid = super_fibonacci(args.N, dev)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, pin_memory=True,
                    persistent_workers=args.workers > 0)

    # loop su ogni segmentazione
    out = []
    for b in tqdm(dl, desc="inference"):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        net.prepare(b)                                           # target, token mask, contesto 3D

        R, t, E = sample_and_score(net, b, grid, args.chunk)
        idx = E.topk(args.topk, dim=1, largest=False).indices
        R = torch.gather(R, 1, idx[..., None, None].expand(-1, -1, 3, 3))
        t = torch.gather(t, 1, idx[..., None].expand(-1, -1, 3))

        with torch.enable_grad():
            R, t, E = refine(net, b, R, t, args.steps, args.lr_t, args.lr_R)

        best = E.argmin(dim=1)
        ar = torch.arange(len(best), device=dev)
        for j in range(len(best)):
            if not b["valid"][j]:
                continue                                         # nessuna depth valida
            e = E[ar[j], best[j]]
            out.append({
                "scene_id": int(b["scene_id"][j]), "im_id": int(b["im_id"][j]),
                "obj_id": int(b["obj_id"][j]),
                "score": float(b["det_score"][j] * torch.exp(-e)),
                "R": R[ar[j], best[j]].cpu().numpy(),
                "t": t[ar[j], best[j]].cpu().numpy().reshape(3, 1),
                "time": -1,                                      # tempo non misurato (BOP richiede lo stesso valore per immagine)
            })
    return out


def select_and_save(results, targets_json, out_csv):
    """Top-inst_count per (scene, im, obj) come richiede BOP19, poi CSV."""
    cnt = {(t["scene_id"], t["im_id"], t["obj_id"]): t["inst_count"]
           for t in json.load(open(targets_json))}
    g = defaultdict(list)
    for r in results:
        g[(r["scene_id"], r["im_id"], r["obj_id"])].append(r)
    final = []
    for k, hs in g.items():
        if k in cnt:
            hs.sort(key=lambda r: r["score"], reverse=True)
            final += hs[:cnt[k]]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    inout.save_bop_results(out_csv, final)


def load_cfg(config_path, config_name, overrides):
    """Config hydra del training, caricata con compose() (niente @hydra.main:
    altrimenti hydra e argparse si contendono sys.argv)."""
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=list(overrides))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/riccardop02/datasets/lmo", help="cartella del dataset BOP (es. .../lmo)")
    p.add_argument("--cnos", default="/data/riccardop02/detection/cnos/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json", help="JSON delle detection CNOS")
    p.add_argument("--out", default="/experiments/riccardop02/Inference", help="output directory")
    p.add_argument("--split", default="test")
    p.add_argument("--targets", default=None, help="default: <root>/test_targets_bop19.json")
    p.add_argument("--ckpt", default="/experiments/riccardop02/outputs/2026-10-09/07-12-11/checkpoints/best.pt", help="checkpoint con i 3 modelli (default: cfg.checkpoint)")
    p.add_argument("--assets_dir", default="/data/riccardop02/datasets/lmo/assets3d", help="cartella con gli asset 3D per oggetto (vedi EnergyNet._load_assets)")
    p.add_argument("--config_path", default="configs", help="relativo a questo file")
    p.add_argument("--config_name", default="config")
    p.add_argument("--overrides", nargs="*", default=[], help="override hydra, es. device=cuda:0")
    p.add_argument("--amp", action="store_true", help="fp16 autocast")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_score", type=float, default=0.3)
    p.add_argument("--det_per_target", type=int, default=0,
                   help="tiene le prime inst_count*det_per_target detection per (scene,im,obj); 0 = tutte quelle nei target")
    p.add_argument("--crop_ratio", type=float, default=1.2, help="stesso crop_ratio del training")
    p.add_argument("--N", type=int, default=512, help="Initial Hypothesis (rotazioni)")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr_t", type=float, default=1.0)
    p.add_argument("--lr_R", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
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
        os.makedirs(run_dir, exist_ok=True)
        out = os.path.join(run_dir, f"energy_{os.path.basename(args.root.rstrip('/'))}-{args.split}.csv")
    print(f"salvo i risultati in: {out}", flush=True)
    select_and_save(res, targets, out)


if __name__ == "__main__":
    main()