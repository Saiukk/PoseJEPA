import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
GPU = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["EGL_DEVICE_ID"] = GPU
import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _read_image(base, iid):
    for sub, ext in (("rgb", "png"), ("rgb", "jpg"), ("gray", "tif")):
        p = f"{base}/{sub}/{iid:06d}.{ext}"
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1].copy()
    raise FileNotFoundError(f"{base} / {iid}")


class MaskRenderer:
    """Un solo OffscreenRenderer riusato per tutte le maschere di una immagine."""
    def __init__(self, W, H):
        import pyrender
        self.pr = pyrender
        self.r = pyrender.OffscreenRenderer(W, H)

    def __call__(self, mesh, K, R, t):
        pr = self.pr
        scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        scene.add(pr.Mesh.from_trimesh(mesh, smooth=False), pose=T)
        cam = pr.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                  znear=1.0, zfar=20000.0)
        scene.add(cam, pose=np.diag([1.0, -1.0, -1.0, 1.0]))   # OpenCV -> OpenGL
        _, depth = self.r.render(scene)
        return depth > 0

    def close(self):
        self.r.delete()


def _pick_gt(gt_list, obj_id, K, t_ref):
    """Istanza GT con stesso obj_id il cui centro proiettato e' piu' vicino a t_ref."""
    def proj(t):
        p = K @ t
        return p[:2] / p[2]
    ref = proj(t_ref)
    cands = [g for g in gt_list if g["obj_id"] == obj_id]
    if not cands:
        return None
    return min(cands, key=lambda g: np.linalg.norm(
        proj(np.array(g["cam_t_m2c"], dtype=np.float64)) - ref))


def _rot_err_deg(R, Rg):
    cos = (np.trace(np.matmul(R, Rg.T), axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u > 0 else 0.0


def _clamped_window(cx, cy, half, W, H):
    """Finestra quadrata attorno a (cx,cy) che resta dentro l'immagine (niente bande vuote)."""
    half = min(half, W / 2, H / 2)
    x0 = float(np.clip(cx - half, 0, W - 2 * half))
    y0 = float(np.clip(cy - half, 0, H - 2 * half))
    return x0, x0 + 2 * half, y0, y0 + 2 * half


def _contour(ax, mask, color, lw=1.8, ls="solid", halo=True):
    if mask is None or not mask.any():
        return
    m = mask.astype(float)
    if halo:  # bordo scuro sotto la linea per la leggibilita'
        ax.contour(m, levels=[0.5], colors=["black"], linewidths=lw + 1.6, alpha=0.7)
    ax.contour(m, levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def _fill(ax, mask, color, alpha=0.35):
    """Riempimento semitrasparente della maschera."""
    if mask is None or not mask.any():
        return
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask] = matplotlib.colors.to_rgba(color, alpha)
    ax.imshow(rgba)


def _get_det_mask(tr, H, W):
    """Ricostruisce la maschera CNOS a risoluzione piena dal crop salvato + box."""
    if "mask" not in tr or "box" not in tr:
        print(f"maschera non trovata. Chiavi nella trace: {list(tr.keys())}")
        return None
    m = tr["mask"].numpy().astype(np.uint8)
    x0, y0, x1, y1 = [int(v) for v in tr["box"]]
    m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H, W), bool)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)   # il box puo' uscire dall'immagine
    full[sy0:sy1, sx0:sx1] = m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] > 0
    return full


# Draw the result
def draw_detection(fig, tr, idx, root, split="test", models_dir=None, use_iou=False,
                   n_steps=5, zoom=False):
    import trimesh
    sid, iid, oid = tr["scene_id"], tr["im_id"], tr["obj_id"]
    E = tr["E"].numpy()                                         # (S+1, K)
    R = tr["R"].numpy().astype(np.float64)                      # (S+1, K, 3, 3)
    t = tr["t"].numpy().astype(np.float64)                      # (S+1, K, 3)
    S, K_hyp = E.shape[0] - 1, E.shape[1]
    all_steps = np.arange(S + 1)
    show_steps = np.unique(np.linspace(0, S, n_steps).round().astype(int))
    s_best = E.argmin(axis=0)                                   # (K,) step della best per ipotesi

    cmap = plt.get_cmap("coolwarm")
    cols = [cmap(k / max(K_hyp - 1, 1)) for k in range(K_hyp)]
    names = [f"rank {k}" + (" (migliore)" if k == 0 else " (peggiore)" if k == K_hyp - 1 else "")
             for k in range(K_hyp)]

    base = f"{root}/{split}/{sid:06d}"
    img = _read_image(base, iid)
    H, W = img.shape[:2]
    cam = json.load(open(f"{base}/scene_camera.json"))[str(iid)]
    Kc = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    gts = json.load(open(f"{base}/scene_gt.json"))[str(iid)]
    gt = _pick_gt(gts, oid, Kc, t[0, 0])
    mesh = trimesh.load(os.path.join(models_dir or f"{root}/models", f"obj_{oid:06d}.ply"),
                        force="mesh", process=False)

    det_mask = _get_det_mask(tr, H, W)

    render = MaskRenderer(W, H)
    try:
        gt_mask, Rg, tg = None, None, None
        if gt is not None:
            Rg = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            tg = np.array(gt["cam_t_m2c"], dtype=np.float64)
            gt_mask = render(mesh, Kc, Rg, tg)
            c = Kc @ tg
            cx, cy = c[0] / c[2], c[1] / c[2]
            half = 0.9 * float(mesh.extents.max()) * Kc[0, 0] / tg[2]
        else:
            cx, cy, half = W / 2, H / 2, min(W, H) / 2
        if zoom:
            x0, x1, y0, y1 = _clamped_window(cx, cy, half, W, H)
        else:
            x0, x1, y0, y1 = 0, W, 0, H                         # immagine intera, nessuno zoom

        # metriche (S+1, K) per i grafici con tutte le K
        metrics = [("energia E", E)]
        if gt is not None:
            dt = t - tg[None, None, :]                              # (S+1, K, 3) errore con segno
            metrics.append(("errore rotazione [deg]", _rot_err_deg(R, Rg)))
            metrics.append(("errore traslazione |t| [mm]", np.linalg.norm(dt, axis=-1)))
            metrics.append(("errore x [mm] (stima - GT)", dt[..., 0]))
            metrics.append(("errore y [mm] (stima - GT)", dt[..., 1]))
            metrics.append(("errore z [mm] (stima - GT)", dt[..., 2]))
            if use_iou:
                iou = np.array([[_iou(render(mesh, Kc, R[s, k], t[s, k]), gt_mask)
                                 for k in range(K_hyp)] for s in all_steps])
                metrics.append(("IoU maschera con GT", iou))

        n_img_cols = 1 + len(show_steps) + 1                        # originale + passi + BEST
        outer = fig.add_gridspec(K_hyp + 1, 1,
                                 height_ratios=[1.0] * K_hyp + [1.3],
                                 hspace=0.35, top=0.95, bottom=0.04, left=0.04, right=0.98)

        def setup_img(ax):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([]); ax.set_yticks([])

        # --- una riga per ogni K: originale+GT+maschera detection, passi, best ---
        for k in range(K_hyp):
            gs_row = outer[k].subgridspec(1, n_img_cols, wspace=0.04)
            ax = fig.add_subplot(gs_row[0, 0])
            ax.imshow(img)
            _fill(ax, det_mask, "yellow", alpha=0.35)
            _contour(ax, det_mask, "yellow", lw=1.2, halo=False)
            _contour(ax, gt_mask, "lime", lw=1.2, ls="dashed", halo=False)
            setup_img(ax)
            ax.set_ylabel(names[k], fontsize=9, color=cols[k], fontweight="bold")
            ax.set_title("originale (verde = GT, giallo = maschera det.)" if k == 0 else "",
                         fontsize=7)
            for j, s in enumerate(show_steps):
                ax = fig.add_subplot(gs_row[0, j + 1])
                ax.imshow(img)
                m = render(mesh, Kc, R[s, k], t[s, k])
                _fill(ax, m, cols[k], alpha=0.35)
                _contour(ax, m, cols[k], lw=1.8)
                _contour(ax, gt_mask, "lime", lw=1.2, ls="dashed", halo=False)
                setup_img(ax)

                title_step = f"passo {s} | tz={t[s, k, 2]:.0f}\nE={E[s, k]:.4f}"
                if Rg is not None and tg is not None:
                    rot_err = _rot_err_deg(R[s, k], Rg)
                    t_err = np.linalg.norm(t[s, k] - tg)
                    title_step += f"\nrot={rot_err:.1f}° | t_err={t_err:.1f}mm"
                ax.set_title(title_step, fontsize=7)

                

            # ultima colonna: la BEST (step con E minima per questa ipotesi)
            sb = int(s_best[k])
            ax = fig.add_subplot(gs_row[0, n_img_cols - 1])
            ax.imshow(img)
            m = render(mesh, Kc, R[sb, k], t[sb, k])
            _fill(ax, m, cols[k], alpha=0.35)
            _contour(ax, m, "white", lw=2.2)
            _contour(ax, gt_mask, "lime", lw=1.2, ls="dashed", halo=False)
            setup_img(ax)
            title = f"BEST (passo {sb}) | tz={t[sb, k, 2]:.0f}\nE={E[sb, k]:.4f}"
            if Rg is not None:
                title += f"\nrot={_rot_err_deg(R[sb, k], Rg):.1f}°"
            ax.set_title(title, fontsize=7, fontweight="bold")

        # --- grafici con tutte le K insieme ---
        gs_plt = outer[K_hyp].subgridspec(1, len(metrics), wspace=0.35)
        for i, (lab, M) in enumerate(metrics):
            ax = fig.add_subplot(gs_plt[0, i])
            for k in range(K_hyp):
                ax.plot(all_steps, M[:, k], color=cols[k], label=names[k])
                ax.plot(s_best[k], M[s_best[k], k], marker="*", ms=10,
                        color=cols[k], mec="black", zorder=5)       # stella = best
            for s in show_steps:                       # segna i passi mostrati sopra
                ax.axvline(s, color="gray", alpha=0.25, lw=0.8)
            ax.set_ylabel(lab, fontsize=8)
            if "(stima - GT)" in lab:                  # linea dello zero per gli errori con segno
                ax.axhline(0, color="black", ls="--", lw=0.8)
            ax.set_xlabel("passo di refinement", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=6)

        extra = f" | tz GT={tg[2]:.0f} tz0(rank0)={t[0, 0, 2]:.0f}" if tg is not None else ""
        fig.suptitle(f"idx {idx} | scena {sid} img {iid} obj {oid}{extra}", fontsize=10)
    finally:
        render.close()

def plot_all(traces, root, idxs, out_path, split, models_dir, use_iou, n_steps, zoom=False):
    idxs = [i for i in idxs if i in traces]
    if not idxs:
        print("nessun idx valido")
        return
    K_hyp = traces[idxs[0]]["E"].shape[1]
    n = len(idxs)
    h_det = 2.4 * K_hyp + 3.6                        # altezza di un blocco-detection
    fig = plt.figure(figsize=(2.6 * (n_steps + 2), h_det * n))   # +1 colonna per la best
    subfigs = np.atleast_1d(fig.subfigures(n, 1, hspace=0.02))
    for sf, idx in zip(subfigs, idxs):
        try:
            draw_detection(sf, traces[idx], idx, root, split, models_dir, use_iou, n_steps, zoom)
        except Exception as e:
            sf.suptitle(f"idx {idx} fallito: {e!r}", color="red", fontsize=9)
            print(f"idx {idx} fallito: {e!r}")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print("saved in", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default='/experiments/riccardop02/Inference/nvdiffrast_traces.pt')
    ap.add_argument("--out", default="try.png", help="nome del file di output")
    ap.add_argument("--root", default="/data/riccardop02/datasets/ycbv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models_dir", default=None)
    ap.add_argument("--idx", type=int, nargs="+", default=None, help="uno o piu' indici")
    ap.add_argument("--n_obj", type=int, default=30, help="campione casuale di N detection")
    ap.add_argument("--all", action="store_true", help="tutte le detection (immagine enorme)")
    ap.add_argument("--n_steps", type=int, default=5,
                    help="quanti passi di refinement mostrare negli overlay (da 0 a S)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou", action="store_true", help="aggiunge grafico IoU con GT (piu' lento)")
    ap.add_argument("--zoom", action="store_true",
                    help="zoom attorno all'oggetto (default: immagine intera)")
    a = ap.parse_args()

    traces = torch.load(a.traces, weights_only=False)
    idxs = a.idx
    if a.all or a.n_obj:
        keys = sorted(traces.keys())
        if a.all:
            idxs = keys
        else:
            rs = np.random.RandomState(a.seed)
            idxs = [keys[i] for i in sorted(rs.choice(len(keys), size=min(a.n_obj, len(keys)),
                                                      replace=False))]

    out_path = os.path.join(os.path.dirname(a.traces), a.out)
    plot_all(traces, a.root, idxs, out_path, a.split, a.models_dir, a.iou, a.n_steps, a.zoom)