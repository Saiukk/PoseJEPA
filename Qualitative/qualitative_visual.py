"""
Crea immagini GT (verde) vs PoseJEPA (rosso) a partire da selected_cases.csv.
Rendering leggero di wireframe (NON occlusion-aware, non equivalente a nvdiffrast).
Uso: python make_visuals.py
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh

import config_qualitative as cfg


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def csv_poses(path):
    """ID stima BOP = indice nella lista del medesimo scene_id, im_id, obj_id."""
    result, counts = {}, defaultdict(int)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k = (int(row['scene_id']), int(row['im_id']), int(row['obj_id']))
            idx = counts[k]
            counts[k] += 1
            R = np.fromstring(row['R'].replace(',', ' '), sep=' ', dtype=np.float64)
            t = np.fromstring(row['t'].replace(',', ' '), sep=' ', dtype=np.float64)
            if R.size != 9 or t.size != 3:
                raise ValueError(f'R/t non validi nel CSV per {k}, est_id={idx}')
            result[(*k, idx)] = (R.reshape(3, 3), t.reshape(3))
    return result


def project(vertices, R, t, K):
    cam = vertices @ R.T + t.reshape(1, 3)
    keep = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 10)  # BOP in mm
    img = np.zeros((len(vertices), 2), dtype=np.float64)
    if keep.any():
        v = cam[keep] @ K.T
        img[keep] = v[:, :2] / v[:, 2, None]
    return img, keep


def load_mesh(root, obj):
    # Per visualizzare preferiamo modelli originali; ripiega su models_eval.
    for folder in ('models_eval', 'models'):
        path = root / folder / f'obj_{obj:06d}.ply'
        if path.exists():
            mesh = trimesh.load(str(path), force='mesh', process=False)
            return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    raise FileNotFoundError(f'Mesh obj_{obj:06d}.ply in models/ oppure models_eval/')


def wireframe(image, vertices, faces, R, t, K, color):
    img, keep = project(vertices, R, t, K)
    overlay = image.copy()
    if len(faces):
        step = max(1, int(np.ceil(len(faces) / cfg.MAX_TRIANGLES)))
        subset = faces[::step]
        for face in subset:
            if not keep[face].all():
                continue
            xy = img[face]
            if not np.isfinite(xy).all() or np.max(np.abs(xy)) > 10000:
                continue
            cv2.polylines(overlay, [np.round(xy).astype(np.int32)],
                          isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    else:
        for xy in img[keep][::max(1, len(vertices) // 1200)]:
            x, y = np.round(xy).astype(int)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(overlay, (x, y), 1, color, -1)
    return cv2.addWeighted(overlay, 0.85, image, 0.15, 0)


def get_crop(img, coords_a, valid_a, coords_b, valid_b):
    h, w = img.shape[:2]
    visible = []
    for coords, valid in ((coords_a, valid_a), (coords_b, valid_b)):
        in_frame = valid & (coords[:, 0] >= 0) & (coords[:, 0] < w) & (coords[:, 1] >= 0) & (coords[:, 1] < h)
        if in_frame.any(): visible.append(coords[in_frame])
    if not visible:
        return (0, 0, w, h)
    xy = np.concatenate(visible)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    size = max(x1 - x0, y1 - y0, 100)
    margin = max(30, cfg.CROP_MARGIN * size)
    x0, y0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
    x1, y1 = min(w, int(x1 + margin)), min(h, int(y1 + margin))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, w, h)
    return (x0, y0, x1, y1)


def text_panel(image, title):
    h, w = image.shape[:2]
    out = np.full((h + 42, w, 3), 255, np.uint8)
    out[42:] = image
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def image_path(root, split, scene, im):
    base = root / split / f'{scene:06d}'
    for folder in ('rgb', 'gray'):
        for extension in ('.png', '.jpg', '.jpeg', '.tif'):
            p = base / folder / f'{im:06d}{extension}'
            if p.is_file(): return p
    raise FileNotFoundError(f'RGB mancante: {base}/rgb/{im:06d}.*')


def render_case(row, pose_est, scene_cache, mesh_cache):
    scene, im, obj, gt_id = (int(row[k]) for k in ('scene_id', 'im_id', 'obj_id', 'gt_id'))
    img = cv2.imread(str(image_path(cfg.DATASET_ROOT, cfg.SPLIT, scene, im)), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f'RGB non leggibile: {scene}/{im}')
    if scene not in scene_cache:
        base = cfg.DATASET_ROOT / cfg.SPLIT / f'{scene:06d}'
        scene_cache[scene] = (load_json(base / 'scene_gt.json'),
                              load_json(base / 'scene_camera.json'))
    gt, cam = scene_cache[scene]
    pose_gt = gt[str(im)][gt_id]
    Rgt = np.asarray(pose_gt['cam_R_m2c'], np.float64).reshape(3, 3)
    tgt = np.asarray(pose_gt['cam_t_m2c'], np.float64).reshape(3)
    K = np.asarray(cam[str(im)]['cam_K'], np.float64).reshape(3, 3)
    Re, te = pose_est
    if obj not in mesh_cache:
        mesh_cache[obj] = load_mesh(cfg.DATASET_ROOT, obj)
    vertices, faces = mesh_cache[obj]

    # Colori BGR: GT verde, stima rossa.
    green, red = (50, 205, 60), (40, 55, 235)
    gt_vis = wireframe(img, vertices, faces, Rgt, tgt, K, green)
    est_vis = wireframe(img, vertices, faces, Re, te, K, red)
    both = wireframe(gt_vis, vertices, faces, Re, te, K, red)

    pgt, okgt = project(vertices, Rgt, tgt, K)
    pest, okest = project(vertices, Re, te, K)
    x0, y0, x1, y1 = get_crop(img, pgt, okgt, pest, okest)
    panels = [img, gt_vis, est_vis, both]
    panels = [p[y0:y1, x0:x1] for p in panels]
    new_size = (440, 330)
    panels = [cv2.resize(p, new_size, interpolation=cv2.INTER_AREA) for p in panels]
    names = ['Input RGB', 'Ground Truth (green)', 'PoseJEPA (red)', 'GT + PoseJEPA']
    strip = np.concatenate([text_panel(p, n) for p, n in zip(panels, names)], axis=1)
    footer = np.full((70, strip.shape[1], 3), 255, np.uint8)
    t1 = (f'scene {scene} / image {im} / obj {obj} / est {row["est_id"]} / gt {gt_id} '
          f'| {row["group"]}')
    t2 = (f'VSD={float(row["vsd_passed"]):.2f}/100   '
          f'MSSD={float(row["mssd_passed"]):.2f}/10  '
          f'MSPD={float(row["mspd_passed"]):.2f}/10  ')
    cv2.putText(footer, t1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .61, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(footer, t2, (15, 53), cv2.FONT_HERSHEY_SIMPLEX, .61, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([strip, footer])


# Used to save all the render in one image
def contact_sheet(files, out_path):
    if not files: return
    thumbs = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None: continue
        w, h = 860, 205
        thumbs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    if not thumbs: return
    if len(thumbs) % 2:
        thumbs.append(np.full_like(thumbs[-1], 255))
    pairs = [np.concatenate(thumbs[i:i + 2], axis=1) for i in range(0, len(thumbs), 2)]
    cv2.imwrite(str(out_path), np.concatenate(pairs, axis=0))


def main():
    # Open the file that was create in the qualitative_analysis, used to pick the example that we want to analyze
    selected_path = cfg.OUTPUT_DIR / 'selected_cases.csv'
    if not selected_path.exists():
        raise FileNotFoundError(f'Manca {selected_path}. Prima esegui python analyze_bop.py')
    if not cfg.PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f'PREDICTIONS_CSV errato: {cfg.PREDICTIONS_CSV}')
    estimates = csv_poses(cfg.PREDICTIONS_CSV)

    # Read the CSV
    rows = list(csv.DictReader(selected_path.open(newline='', encoding='utf-8')))
    out = cfg.OUTPUT_DIR / 'images'
    out.mkdir(exist_ok=True, parents=True)
    scene_cache, mesh_cache = {}, {}
    group_paths = defaultdict(list)
    failures = []
    # For each row of the CSV we render the image
    for row in rows:
        s, i, o, est = (int(row[k]) for k in ('scene_id', 'im_id', 'obj_id', 'est_id'))
        key = (s, i, o, est)
        # Check if the CSV and the inference BOP file are the same
        if key not in estimates:
            failures.append(f'{row["group"]} {key}: est_id non trovato nel CSV. Usa lo stesso CSV della valutazione!')
            continue

        # Create the path and the directory for saving the images
        path = out / row['group'] / f's{s:06d}_i{i:06d}_o{o:06d}_e{est}_gt{row["gt_id"]}.png'
        path.parent.mkdir(exist_ok=True, parents=True)
        try:
            # Render the image and save it
            montage = render_case(row, estimates[key], scene_cache, mesh_cache)
            if not cv2.imwrite(str(path), montage): raise RuntimeError('cv2.imwrite fallito')
            group_paths[row['group']].append(path)
        except Exception as exc:
            failures.append(f'{row["group"]} {key}: {exc}')
    for group, paths in group_paths.items():
        contact_sheet(paths, out / f'CONTACT_{group}.png')
        print(f'{group}: {len(paths)} visualizzazioni')
    if failures:
        (cfg.OUTPUT_DIR / 'render_errors.txt').write_text('\n'.join(failures) + '\n', encoding='utf-8')
        print(f'[attenzione] {len(failures)} visualizzazioni non generate: render_errors.txt')
    print(f'Immagini: {out.resolve()}')


if __name__ == '__main__':
    main()
