from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--inference-script', default='/home/riccardop02/JEPA/test_ref_enrgy_NVDiff_LASTREF.py', type=Path,
                   help='Your LATEST inference .py with GroupNorm and per-object translation scale')
    p.add_argument('--bop-toolkit-dir', type=Path,
                   default=Path('/home/riccardop02/JEPA/bop_toolkit'))
    p.add_argument('--root', type=Path, default='/data/riccardop02/datasets/ycbv',
                   help='BOP dataset directory, e.g. /data/riccardop02/datasets/ycbv')
    p.add_argument('--split', default='test', help='Use test for ycbv/lmo and test_primesense for tless')
    p.add_argument('--cnos', type=Path, default='/data/riccardop02/detection/gt/gt_ycbv-test.json', help='The detection JSON used in the actual inference')
    p.add_argument('--ckpt', type=Path, default=Path('/experiments/riccardop02/outputs/2026-11-04/10-40-44/checkpoints/best.pt'))
    p.add_argument('--refiner-ckpt', type=Path, default=Path('/experiments/riccardop02/outputs/refinement_GT_GROUPNORM/best.pt'))
    p.add_argument('--assets-dir', type=Path, default=None)
    p.add_argument('--config-path', default='configs')
    p.add_argument('--config-name', default='config')
    p.add_argument('--out', type=Path, default=Path('/data/riccardop02/Qualitative/refiner_bop_qualitative_ycbv'))
    p.add_argument('--cases-csv', type=Path, default=None,
                   help='Optional all_cases.csv / selected_cases.csv: analyze only those scene/im/obj')
    p.add_argument('--max-detections', type=int, default=24)
    p.add_argument('--max-per-object', type=int, default=4)
    p.add_argument('--n', type=int, default=128, help='Number of real sampled hypotheses; match latest inference')
    p.add_argument('--steps', type=int, default=3, help='Number of neural refinement steps')
    p.add_argument('--chunk', type=int, default=16, help='Max hypotheses per refiner forward (reduce if OOM)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-score', type=float, default=0.3)
    p.add_argument('--det-per-target', type=int, default=0)
    p.add_argument('--crop-ratio', type=float, default=1.2)
    p.add_argument('--min-mask-iou', type=float, default=0.05)
    p.add_argument('--n-per-category', type=int, default=6)
    p.add_argument('--sym-disc-step', type=float, default=0.01,
                   help='BOP misc.get_symmetry_transformations step; smaller = more accurate, slower')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-controlled', action='store_true', help='Skip GT-perturbation local refinement test')
    p.add_argument('--no-real', action='store_true', help='Skip actual sampled hypotheses')
    return p.parse_args()


def load_inference(path: Path, toolkit: Path):
    path = path.resolve()
    if not path.is_file() or path.suffix != '.py':
        raise FileNotFoundError(f'--inference-script must point to your latest .py: {path}')
    if toolkit.is_dir():
        sys.path.insert(0, str(toolkit.resolve()))
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('posejepa_inference_for_qualitative', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = ['EnergyNet', 'CnosDetDataset', 'load_refiner', 'refine',
                'quat_to_mat', 'super_fibonacci', 'load_cfg', '_energy']
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f'Latest inference script is missing these functions: {missing}')
    return mod


def read_json(path: Path):
    with path.open('r', encoding='utf8') as f:
        return json.load(f)


def select_indices(ds, cases_csv, max_detections, max_per_object):
    allowed = None
    if cases_csv:
        if not cases_csv.is_file():
            raise FileNotFoundError(cases_csv)
        with cases_csv.open(newline='', encoding='utf8') as f:
            allowed = {(int(r['scene_id']), int(r['im_id']), int(r['obj_id']))
                       for r in csv.DictReader(f)}
        print(f'Case filter: {len(allowed)} distinct scene/im/obj combinations', flush=True)
    chosen, per_obj = [], Counter()
    for idx, det in enumerate(ds.dets):
        key = (int(det['scene_id']), int(det['image_id']), int(det['category_id']))
        if allowed is not None and key not in allowed:
            continue
        obj = key[-1]
        if max_per_object > 0 and per_obj[obj] >= max_per_object:
            continue
        chosen.append(idx)
        per_obj[obj] += 1
        if max_detections > 0 and len(chosen) >= max_detections:
            break
    return chosen


def match_detection_gt(ds, index, scene, im, obj, scene_gt, mask_utils, min_iou):
    """Match CNOS detection to a visible BOP GT instance by *mask* IoU.

    This avoids equating the ordinal est_id in a BOP CSV with the GT index.
    """
    gt_items = scene_gt['gt'].get(str(im), [])
    candidates = [gi for gi, x in enumerate(gt_items) if int(x['obj_id']) == obj]
    if not candidates:
        return None
    rle = dict(ds.dets[index]['segmentation'])
    h, w = rle['size']
    if isinstance(rle['counts'], list):
        rle = mask_utils.frPyObjects(rle, h, w)
    elif isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('ascii')
    det_mask = mask_utils.decode(rle)
    if det_mask.ndim == 3:
        det_mask = np.any(det_mask > 0, axis=-1)
    else:
        det_mask = det_mask > 0
    best_iou, best_gt = -1., None
    for gi in candidates:
        p = ds.root + f'/{ds.split}/{scene:06d}/mask_visib/{im:06d}_{gi:06d}.png'
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f'Cannot match GT: missing visible mask {p}')
        gt_mask = m > 0
        if gt_mask.shape != det_mask.shape:
            raise ValueError(f'CNOS/GT mask shape mismatch: {det_mask.shape} vs {gt_mask.shape}')
        intersection = np.count_nonzero(det_mask & gt_mask)
        union = np.count_nonzero(det_mask | gt_mask)
        iou = intersection / max(union, 1)
        if iou > best_iou:
            best_iou, best_gt = iou, gi
    if best_gt is None or best_iou < min_iou:
        return None
    pose = gt_items[best_gt]
    Rgt = np.asarray(pose['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
    tgt = np.asarray(pose['cam_t_m2c'], dtype=np.float64).reshape(3)
    info = scene_gt['info'].get(str(im), [])
    visibility = float(info[best_gt].get('visib_fract', np.nan)) if best_gt < len(info) else np.nan
    return best_gt, best_iou, Rgt, tgt, visibility


class ModelData:
    def __init__(self, root, sym_disc_step):
        from bop_toolkit_lib import misc
        self.root = root
        self.misc = misc
        self.sym_disc_step = sym_disc_step
        info_path = root / 'models_eval' / 'models_info.json'
        if not info_path.is_file():
            info_path = root / 'models' / 'models_info.json'
        self.model_info = read_json(info_path)
        self.cache = {}

    def get(self, obj):
        if obj not in self.cache:
            folder = self.root / 'models_eval'
            mesh_path = folder / f'obj_{obj:06d}.ply'
            if not mesh_path.is_file():
                mesh_path = self.root / 'models' / f'obj_{obj:06d}.ply'
            mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
            if not hasattr(mesh, 'vertices'):
                raise RuntimeError(f'Not a mesh: {mesh_path}')
            info = self.model_info[str(obj)]
            transforms = self.misc.get_symmetry_transformations(
                info, max_sym_disc_step=self.sym_disc_step)
            if not transforms:
                transforms = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
            syms = [(np.asarray(s['R'], dtype=np.float64).reshape(3, 3),
                     np.asarray(s['t'], dtype=np.float64).reshape(3)) for s in transforms]
            self.cache[obj] = (np.asarray(mesh.vertices, dtype=np.float64),
                               np.asarray(mesh.faces, dtype=np.int32),
                               float(info['diameter']), syms)
            print(f'Obj {obj}: {len(mesh.vertices)} evaluation vertices, {len(syms)} symmetry transforms', flush=True)
        return self.cache[obj]


def metric_bundle(Rs, ts, gt_R, gt_t, points, diameter, syms):
    """Symmetry-aware MSSD in mm, equivalent geometric definition to BOP.

    All vertices of models_eval are used. Translation error is raw Euclidean mm.
    Orientation is minimum geodesic angle modulo annotated BOP rotations.
    """
    Rs = np.asarray(Rs, dtype=np.float64).reshape(-1, 3, 3)
    ts = np.asarray(ts, dtype=np.float64).reshape(-1, 3)
    n = len(Rs)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    gt_sym_R = np.stack([gt_R @ sr for sr, _ in syms])
    gt_sym_t = np.stack([gt_R @ st + gt_t for _, st in syms])
    # Trace(R_est * R_gt_sym^T) equals Frobenius inner product.
    cosine = np.clip((np.einsum('nij,sij->ns', Rs, gt_sym_R) - 1.) / 2., -1., 1.)
    rot_sym = np.rad2deg(np.arccos(cosine)).min(axis=1)
    trans_mm = np.linalg.norm(ts - gt_t[None], axis=-1)
    mssd_mm = np.full(n, np.inf, dtype=np.float64)
    # Chunking prevents N x number_of_vertices arrays from becoming too large.
    gt_pts = [points @ r.T + t for r, t in zip(gt_sym_R, gt_sym_t)]
    for start in range(0, n, 12):
        end = min(n, start + 12)
        est_pts = np.einsum('nij,pj->npi', Rs[start:end], points) + ts[start:end, None, :]
        best = np.full(end - start, np.inf, dtype=np.float64)
        for gpts in gt_pts:
            delta = est_pts - gpts[None, :, :]
            dist2 = np.sum(delta * delta, axis=-1)
            best = np.minimum(best, np.sqrt(np.max(dist2, axis=1)))
        mssd_mm[start:end] = best
    return mssd_mm, rot_sym, trans_mm


def to_batch(item, torch, device):
    out = {}
    for k, v in item.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, (int, float, bool, np.number)):
            out[k] = torch.as_tensor([v], device=device)
        else:
            out[k] = v
    return out


def refine_in_chunks(mod, net, refiner, rot_norm, b, R, t, chunk, steps, torch):
    rr, tt = [], []
    for start in range(0, R.shape[1], chunk):
        r, tr = mod.refine(net, refiner, b, R[:, start:start + chunk],
                           t[:, start:start + chunk], steps, rot_norm)
        rr.append(r)
        tt.append(tr)
    return torch.cat(rr, dim=1), torch.cat(tt, dim=1)


def sample_real(mod, b, grid, seed, dataset_index, torch):
    """Sampling convention from the user's latest inference (CPU random generator)."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed) + int(dataset_index) + 1)
    q = torch.randn(1, 1, 4, generator=gen).to(grid.device)
    Rg = mod.quat_to_mat(q)
    R = Rg @ grid[None]
    t = b['t0'][:, None, :].expand(1, grid.shape[0], 3).contiguous()
    return R, t


def rodrigues(axis, angle_deg):
    angle = math.radians(angle_deg)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3) + math.sin(angle) * a + (1. - math.cos(angle)) * (a @ a)


def sample_controlled(gt_R, gt_t, diameter, key, torch, device):
    """GT-conditioned probes: artificial starts, NEVER mixed with actual inference."""
    seed = int(key[0]) * 1000003 + int(key[1]) * 9176 + int(key[2]) * 101
    rng = np.random.default_rng(seed)
    rotations, translations, descriptions = [], [], []
    for angle in (0., 2., 5., 10., 20., 40.):
        for repeat in range(2):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            R = rodrigues(axis, angle) @ gt_R
            # Include exact translation and small translations up to 3% diameter.
            trans_level = 0. if repeat == 0 else .03
            shift = rng.normal(size=3)
            shift /= np.linalg.norm(shift)
            t = gt_t + shift * (diameter * trans_level)
            rotations.append(R)
            translations.append(t)
            descriptions.append(f'GT+{angle:g}deg/t{trans_level * 100:g}%d')
    return (torch.as_tensor(np.array(rotations), device=device, dtype=torch.float32)[None],
            torch.as_tensor(np.array(translations), device=device, dtype=torch.float32)[None],
            descriptions)


def candidate_rows(scene, im, obj, gt_id, iou, vis, ds_i, mode, pre_R, pre_t,
                   post_R, post_t, e_post, diameter, points, syms, Rgt, tgt, desc):
    a, ra, ta = metric_bundle(pre_R, pre_t, Rgt, tgt, points, diameter, syms)
    b, rb, tb = metric_bundle(post_R, post_t, Rgt, tgt, points, diameter, syms)
    rows = []
    for i in range(len(a)):
        rows.append(dict(scene_id=scene, im_id=im, obj_id=obj, gt_id=gt_id,
                         detection_index=ds_i, mask_iou=round(iou, 4),
                         visib_fract=round(vis, 4), mode=mode, hypothesis_index=i,
                         init_type=desc[i] if desc is not None else 'SuperFibonacci',
                         mssd_before_mm=round(float(a[i]), 5), mssd_after_mm=round(float(b[i]), 5),
                         mssd_before_d=round(float(a[i] / diameter), 6),
                         mssd_after_d=round(float(b[i] / diameter), 6),
                         improvement_d=round(float((a[i] - b[i]) / diameter), 6),
                         rot_before_sym_deg=round(float(ra[i]), 4),
                         rot_after_sym_deg=round(float(rb[i]), 4),
                         trans_before_mm=round(float(ta[i]), 4),
                         trans_after_mm=round(float(tb[i]), 4),
                         post_energy=round(float(e_post[i]), 7) if e_post is not None else '',
                         chosen_energy=0, chosen_gt_best=0))
    if mode == 'real':
        energy_i = int(np.argmin(e_post))
        oracle_i = int(np.argmin(b))
        rows[energy_i]['chosen_energy'] = 1
        rows[oracle_i]['chosen_gt_best'] = 1
    return rows


def load_scene(root, split, scene, cache):
    if scene not in cache:
        base = root / split / f'{scene:06d}'
        cache[scene] = dict(gt=read_json(base / 'scene_gt.json'),
                            info=read_json(base / 'scene_gt_info.json'),
                            camera=read_json(base / 'scene_camera.json'))
    return cache[scene]


def read_rgb(root, split, scene, im):
    base = root / split / f'{scene:06d}' / 'rgb'
    for ext in ('.png', '.jpg', '.jpeg'):
        path = base / f'{im:06d}{ext}'
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    raise FileNotFoundError(f'BOP RGB not found: {base}/{im:06d}.png/jpg')


def project(pts, R, t, K):
    cam = pts @ R.T + t.reshape(1, 3)
    valid = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1.)
    xy = np.zeros((len(pts), 2), dtype=np.float64)
    if valid.any():
        q = cam[valid] @ K.T
        xy[valid] = q[:, :2] / q[:, 2, None]
    return xy, valid


def draw_wire(image, points, faces, R, t, K, color, max_faces=650):
    overlay = image.copy()
    uv, keep = project(points, R, t, K)
    step = max(1, math.ceil(len(faces) / max_faces))
    for tri in faces[::step]:
        if not np.all(keep[tri]):
            continue
        xy = uv[tri]
        if not np.isfinite(xy).all() or np.abs(xy).max() > 30000:
            continue
        cv2.polylines(overlay, [np.round(xy).astype(np.int32)],
                      True, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, .88, image, .12, 0)


def crop_bounds(image, points, pose_list, K):
    height, width = image.shape[:2]
    projected = []
    stride = max(1, len(points) // 3000)
    for R, t in pose_list:
        xy, keep = project(points[::stride], R, t, K)
        ok = keep & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        if ok.any():
            projected.append(xy[ok])
    if not projected:
        return 0, 0, width, height
    xy = np.concatenate(projected)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    max_size = max(*(mx - mn), 100.)
    margin = max(max_size * .20, 20.)
    x0 = max(0, math.floor(mn[0] - margin))
    y0 = max(0, math.floor(mn[1] - margin))
    x1 = min(width, math.ceil(mx[0] + margin))
    y1 = min(height, math.ceil(mx[1] + margin))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else (0, 0, width, height)


def draw_case(image, points, faces, Rgt, tgt, R0, t0, R1, t1, K, row,
              oracle=None):
    green, red = (55, 205, 55), (35, 40, 230)  # BGR
    coords = [(Rgt, tgt), (R0, t0), (R1, t1)]
    if oracle is not None:
        coords.append(oracle)
    bounds = crop_bounds(image, points, coords, K)
    x0, y0, x1, y1 = bounds
    gt_panel = draw_wire(image, points, faces, Rgt, tgt, K, green)
    pre_panel = draw_wire(gt_panel, points, faces, R0, t0, K, red)
    post_panel = draw_wire(gt_panel, points, faces, R1, t1, K, red)
    panels = [(image, 'RGB'), (pre_panel, 'INITIAL + GT'),
              (post_panel, 'REFINED + GT'), (gt_panel, 'GT (green)')]
    if oracle is not None:
        oracle_p = draw_wire(gt_panel, points, faces, oracle[0], oracle[1], K, red)
        panels.append((oracle_p, 'ORACLE + GT (diagnostic)'))
    display = []
    for p, name in panels:
        p = cv2.resize(p[y0:y1, x0:x1], (345, 265), interpolation=cv2.INTER_AREA)
        caption = np.full((40, 345, 3), 255, np.uint8)
        cv2.putText(caption, name, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .64, (20, 20, 20), 2, cv2.LINE_AA)
        display.append(np.concatenate([caption, p], axis=0))
    canvas = np.concatenate(display, axis=1)
    foot = np.full((90, canvas.shape[1], 3), 255, np.uint8)
    t1s = (f"{row['category']} | scene {row['scene_id']} / image {row['im_id']} / "
           f"obj {row['obj_id']} / GT {row['gt_id']} / hypothesis {row['hypothesis_index']} / {row['init_type']}")
    t2s = (f"MSSD: {row['mssd_before_d']:.3f} -> {row['mssd_after_d']:.3f} x diameter "
           f"({row['mssd_before_mm']:.1f} -> {row['mssd_after_mm']:.1f} mm)")
    t3s = (f"R sym: {row['rot_before_sym_deg']:.1f} -> {row['rot_after_sym_deg']:.1f} deg  |  "
           f"t raw: {row['trans_before_mm']:.1f} -> {row['trans_after_mm']:.1f} mm  |  "
           f"visibility: {row['visib_fract']:.2f}  IoU: {row['mask_iou']:.2f}")
    for ii, line in enumerate((t1s, t2s, t3s)):
        cv2.putText(foot, line, (14, 23 + 27 * ii), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (25, 25, 25), 1, cv2.LINE_AA)
    return np.concatenate([canvas, foot])


def select_cases(all_rows, n_per_category):
    groups = defaultdict(list)
    for r in all_rows:
        pre, post = r['mssd_before_d'], r['mssd_after_d']
        delta = r['improvement_d']
        if r['mode'] == 'controlled':
            if pre <= .15 and delta <= -.015:
                groups['fine_degraded'].append((-delta, r))
            if .025 <= pre <= .20 and delta >= .015:
                groups['fine_improved'].append((delta, r))
            if .025 <= pre <= .20 and abs(delta) < .012:
                groups['fine_stalled'].append((-.0 + pre, r))
        if r['mode'] == 'real':
            if pre >= .30 and post <= .25 and delta >= .09:
                groups['coarse_recovered'].append((delta, r))
            if pre >= .40 and post >= .35 and abs(delta) < .06:
                groups['coarse_not_recovered'].append((post, r))
            if pre >= .20 and delta <= -.07:
                groups['coarse_degraded'].append((-delta, r))
    # An explicit selection-failure case per detection: energy chose a pose with
    # much higher geometric error than the GT oracle among the same refined pool.
    real_by_det = defaultdict(list)
    for r in all_rows:
        if r['mode'] == 'real':
            real_by_det[r['detection_index']].append(r)
    for rows in real_by_det.values():
        chosen = next((r for r in rows if r['chosen_energy'] == 1), None)
        best = min(rows, key=lambda x: x['mssd_after_d'])
        if chosen is not None and chosen['mssd_after_d'] - best['mssd_after_d'] > .15:
            groups['energy_selection_miss'].append((chosen['mssd_after_d'] - best['mssd_after_d'], chosen))

    selected = []
    for category, pool in groups.items():
        pool.sort(key=lambda e: e[0], reverse=True)
        seen_detections = set()
        per_obj = Counter()
        for _, row in pool:
            d = row['detection_index']
            if d in seen_detections or per_obj[row['obj_id']] >= 2:
                continue
            item = dict(row)
            item['category'] = category
            selected.append(item)
            seen_detections.add(d)
            per_obj[row['obj_id']] += 1
            if len(seen_detections) >= n_per_category:
                break
    return selected


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contacts(images_root, n_per_category):
    for group in sorted(p for p in images_root.iterdir() if p.is_dir()):
        paths = sorted(group.glob('*.png'))
        thumbs = []
        for path in paths[:n_per_category]:
            img = cv2.imread(str(path))
            if img is not None:
                # Resize with aspect ratio intact to preserve geometry.
                tw = 1100
                th = round(img.shape[0] * tw / img.shape[1])
                thumbs.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))
        if not thumbs:
            continue
        max_h = max(x.shape[0] for x in thumbs)
        normalized = []
        for im in thumbs:
            out = np.full((max_h, 1100, 3), 255, np.uint8)
            out[:im.shape[0]] = im
            normalized.append(out)
        if len(normalized) % 2:
            normalized.append(np.full_like(normalized[-1], 255))
        lines = [np.concatenate(normalized[i:i+2], axis=1)
                 for i in range(0, len(normalized), 2)]
        cv2.imwrite(str(images_root / f'CONTACT_{group.name}.png'), np.concatenate(lines, axis=0))


def main():
    args = parse_args()
    if args.no_real and args.no_controlled:
        raise ValueError('Both --no-real and --no-controlled were supplied')
    if args.chunk < 1 or args.n < 1 or args.max_detections < 1:
        raise ValueError('chunk, n, max-detections must be positive')
    if not args.root.is_dir() or not args.cnos.is_file():
        raise FileNotFoundError('Check --root and --cnos; both must exist')
    args.out.mkdir(parents=True, exist_ok=True)
    # Load the current inference first: it sets CUDA_VISIBLE_DEVICES before torch import.
    mod = load_inference(args.inference_script, args.bop_toolkit_dir)
    import torch
    from pycocotools import mask as mask_utils
    assets_dir = args.assets_dir or (args.root / 'assets3d')
    # Latest inference module supplies the *actual* current refiner implementation:
    # GroupNorm, left SO(3) updates, per-object scale for translation updates.
    config = mod.load_cfg(args.config_path, args.config_name, [])
    net = mod.EnergyNet(config, str(args.ckpt), str(assets_dir), device=args.device, amp=True)
    refiner, rot_norm = mod.load_refiner(str(args.refiner_ckpt),
                                         net.encoder2d.output_dim[-1], args.device)
    targets = args.root / 'test_targets_bop19.json'
    ds = mod.CnosDetDataset(str(args.root), str(args.cnos), str(targets), args.split,
                            net.img_size, margin=args.crop_ratio,
                            min_score=args.min_score, det_per_target=args.det_per_target)
    indices = select_indices(ds, args.cases_csv, args.max_detections, args.max_per_object)
    if not indices:
        raise RuntimeError('No detections passed the filters (check --cases-csv / CNOS / min-score)')
    print(f'Analyzing {len(indices)} BOP detections; inference module: {args.inference_script}', flush=True)
    model_data = ModelData(args.root, args.sym_disc_step)
    grid = mod.super_fibonacci(args.n, args.device) if not args.no_real else None
    scenes = {}
    all_rows, pose_lookup = [], {}
    for count, ds_idx in enumerate(indices, 1):
        det = ds.dets[ds_idx]
        scene, im, obj = (int(det['scene_id']), int(det['image_id']), int(det['category_id']))
        sc = load_scene(args.root, args.split, scene, scenes)
        matched = match_detection_gt(ds, ds_idx, scene, im, obj, sc, mask_utils,
                                     args.min_mask_iou)
        if matched is None:
            print(f'[skip] {scene}/{im}/{obj}: no reliable detection↔GT match', flush=True)
            continue
        gt_id, mask_iou, Rgt, tgt, visib_fract = matched
        points, faces, diameter, syms = model_data.get(obj)
        item = ds[ds_idx]
        b = to_batch(item, torch, args.device)
        if not bool(b['valid'][0].item()):
            print(f'[skip] {scene}/{im}/{obj}: invalid t0/mask', flush=True)
            continue
        print(f'[{count}/{len(indices)}] scene={scene} im={im} obj={obj} GT={gt_id} maskIoU={mask_iou:.2f}', flush=True)
        with torch.inference_mode():
            net.prepare(b)
            pools = []
            if not args.no_real:
                pre_R, pre_t = sample_real(mod, b, grid, args.seed, ds_idx, torch)
                post_R, post_t = refine_in_chunks(mod, net, refiner, rot_norm, b,
                                                  pre_R, pre_t, args.chunk, args.steps, torch)
                energy_chunks = [mod._energy(net, b, post_R[:, q:q+args.chunk],
                                              post_t[:, q:q+args.chunk])
                                 for q in range(0, args.n, args.chunk)]
                post_E = torch.cat(energy_chunks, dim=1)[0].cpu().numpy()
                pools.append(('real', pre_R, pre_t, post_R, post_t, post_E, None))
            if not args.no_controlled:
                cR, ct, descriptions = sample_controlled(Rgt, tgt, diameter,
                                                          (scene, im, obj), torch, args.device)
                pR, pt = refine_in_chunks(mod, net, refiner, rot_norm, b,
                                          cR, ct, args.chunk, args.steps, torch)
                pools.append(('controlled', cR, ct, pR, pt, None, descriptions))
            for mode, pre_R, pre_t, post_R, post_t, E, descriptors in pools:
                pre_R = pre_R[0].detach().cpu().numpy()
                pre_t = pre_t[0].detach().cpu().numpy()
                post_R = post_R[0].detach().cpu().numpy()
                post_t = post_t[0].detach().cpu().numpy()
                rows = candidate_rows(scene, im, obj, gt_id, mask_iou, visib_fract,
                                      ds_idx, mode, pre_R, pre_t, post_R, post_t,
                                      E, diameter, points, syms, Rgt, tgt, descriptors)
                all_rows.extend(rows)
                # Keep only the small sampled set required for eventual figures;
                # poses are not written into the CSV because standard BOP CSV has
                # only the selected final pose, not the unselected hypotheses.
                for local_i in range(len(rows)):
                    pose_lookup[(ds_idx, mode, local_i)] = (
                        pre_R[local_i], pre_t[local_i], post_R[local_i], post_t[local_i])
        # After each detection: save a checkpoint to avoid losing the metrics.
        write_csv(args.out / 'all_hypotheses.csv', all_rows)

    if not all_rows:
        raise RuntimeError('No detections could be matched and evaluated')
    selected = select_cases(all_rows, args.n_per_category)
    write_csv(args.out / 'selected_examples.csv', selected)
    images_root = args.out / 'images'
    images_root.mkdir(exist_ok=True, parents=True)
    by_group = Counter()
    for row in selected:
        ds_idx = row['detection_index']
        key = (ds_idx, row['mode'], row['hypothesis_index'])
        if key not in pose_lookup:
            continue
        R0, t0, R1, t1 = pose_lookup[key]
        scene, im, obj, gt_id = [row[x] for x in ('scene_id', 'im_id', 'obj_id', 'gt_id')]
        sc = scenes[scene]
        gt = sc['gt'][str(im)][gt_id]
        Rgt = np.asarray(gt['cam_R_m2c']).reshape(3, 3)
        tgt = np.asarray(gt['cam_t_m2c']).reshape(3)
        K = np.asarray(sc['camera'][str(im)]['cam_K']).reshape(3, 3)
        pts, faces, _, _ = model_data.get(obj)
        rgb = read_rgb(args.root, args.split, scene, im)
        oracle = None
        if row['category'] == 'energy_selection_miss':
            candidates = [r for r in all_rows if r['mode'] == 'real' and r['detection_index'] == ds_idx]
            best = min(candidates, key=lambda r: r['mssd_after_d'])
            oracle = pose_lookup[(ds_idx, 'real', best['hypothesis_index'])][2:]
        picture = draw_case(rgb, pts, faces, Rgt, tgt, R0, t0, R1, t1, K, row, oracle)
        folder = images_root / row['category']
        folder.mkdir(exist_ok=True, parents=True)
        out_path = folder / (f"s{scene:06d}_i{im:06d}_o{obj:06d}_d{ds_idx}"
                             f"_{row['mode']}_h{row['hypothesis_index']}.png")
        if not cv2.imwrite(str(out_path), picture):
            raise IOError(f'Could not save {out_path}')
        by_group[row['category']] += 1
    if by_group:
        make_contacts(images_root, args.n_per_category)

    messages = [
        'POSEJEPA REFINER BOP DIAGNOSTICS (NOT OFFICIAL BOP AR)',
        f'Inference module: {args.inference_script}',
        f'Dataset: {args.root}, split={args.split}',
        f'Detections analyzed: {len(set(r["detection_index"] for r in all_rows))}',
        f'Real candidates per detection: {args.n}; refiner steps: {args.steps}',
        f'Symmetry transforms: BOP models_info, discretization={args.sym_disc_step}',
        'MSSD is symmetry-aware and calculated using ALL models_eval vertices.',
        'R error is symmetry-aware; translation is the raw Euclidean center translation error.',
        'Matching each detection to its GT is by visible-mask IoU, not CSV estimate order.',
        'Controlled/GT-initialized hypotheses are artificial diagnostic probes, NOT real inference.',
        'All before/after comparisons track the SAME hypothesis and the SAME GT instance.',
        '',
    ]
    for mode in ('real', 'controlled'):
        rows = [r for r in all_rows if r['mode'] == mode]
        if not rows:
            continue
        improved = np.mean([r['improvement_d'] > .01 for r in rows]) * 100.
        worsened = np.mean([r['improvement_d'] < -.01 for r in rows]) * 100.
        messages.append(f'{mode}: {len(rows)} hypotheses, {improved:.1f}% improved >1% diameter, '
                        f'{worsened:.1f}% worsened >1% diameter')
    messages.extend(['', 'Selected qualitative examples:'])
    messages.extend(f'  {k}: {v}' for k, v in sorted(by_group.items()))
    messages.append('WARNING: A low count in one category does NOT prove absence of that failure mode.')
    messages.append('Subset results do not estimate official BOP AR or full-dataset population statistics.')
    (args.out / 'summary.txt').write_text('\n'.join(messages) + '\n', encoding='utf8')
    print('\n'.join(messages), flush=True)
    print(f'Output directory: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
