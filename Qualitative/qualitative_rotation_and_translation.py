"""Qualitative dei RESIDUI di posa dopo PoseJEPA, a partire dai risultati BOP.

NON misura direttamente se il refiner ha migliorato/peggiorato una posa:
serve una posa prima del refinement per quello. Non esegue inferenza.

Metti questo file vicino a `config_qualitative.py` e `qualitative_visual.py`.
Uso: python qualitative_refiner_bop.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config_qualitative as cfg
import qualitative_visual as visual


# ---------------------- MODIFICA QUI -----------------------
N_PER_GROUP = 10
GOOD_TRANSLATION_DIAM = 0.05   # errore di traslazione < 5% diametro
BAD_TRANSLATION_DIAM = 0.15    # errore di traslazione >= 15% diametro
GOOD_ROTATION_DEG = 10.0
BAD_ROTATION_DEG = 45.0
GOOD_MSSD_PASSED = 8           # su 10 soglie
BAD_MSSD_PASSED = 3
GOOD_MSPD_PASSED = 8           # su 10 soglie
GOOD_VSD_PASSED = 70           # su 100 (con 10 tau)
OUT_DIR = cfg.OUTPUT_DIR / 'bop_pose_failure_modes'
# ----------------------------------------------------------


def load_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    # Errore geodetico SO(3) NON symmetry-aware.
    c = (np.trace(R_est @ R_gt.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def as_float(row, name, fallback=np.nan) -> float:
    v = row.get(name, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(fallback)


def build_rows():
    source = cfg.OUTPUT_DIR / 'all_cases.csv'
    if not source.is_file():
        raise FileNotFoundError(f'Manca {source}: esegui prima qualitative_analysis.py')
    if not cfg.PREDICTIONS_CSV.is_file():
        raise FileNotFoundError(f'Manca il CSV BOP: {cfg.PREDICTIONS_CSV}')

    pred = visual.csv_poses(cfg.PREDICTIONS_CSV)
    info_path = cfg.DATASET_ROOT / 'models_eval' / 'models_info.json'
    if not info_path.exists():
        info_path = cfg.DATASET_ROOT / 'models' / 'models_info.json'
    models = load_json(info_path)
    gt_cache = {}
    rows = []
    skipped = 0

    with source.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            scene, im, obj, est, gt_id = [int(r[k]) for k in ('scene_id', 'im_id', 'obj_id', 'est_id', 'gt_id')]
            pose_key = (scene, im, obj, est)
            if pose_key not in pred:
                skipped += 1
                continue
            if scene not in gt_cache:
                gt_path = cfg.DATASET_ROOT / cfg.SPLIT / f'{scene:06d}' / 'scene_gt.json'
                gt_cache[scene] = load_json(gt_path)
            candidates = gt_cache[scene][str(im)]
            if not (0 <= gt_id < len(candidates)) or int(candidates[gt_id]['obj_id']) != obj:
                skipped += 1
                continue
            target = candidates[gt_id]
            R_gt = np.array(target['cam_R_m2c'], dtype=float).reshape(3, 3)
            t_gt = np.array(target['cam_t_m2c'], dtype=float).reshape(3)
            R_est, t_est = pred[pose_key]
            diameter = float(models[str(obj)]['diameter'])
            if diameter <= 0:
                skipped += 1
                continue

            # ATTENZIONE: per oggetti con simmetrie la rotazione raw puo' essere
            # grande anche per pose geometricamente equivalenti.
            meta = models[str(obj)]
            has_sym = bool(meta.get('symmetries_discrete') or meta.get('symmetries_continuous'))
            dr = rotation_error_deg(R_est, R_gt)
            dt = float(np.linalg.norm(t_est - t_gt))
            result = dict(r)
            result.update({
                'rot_error_raw_deg': round(dr, 4),
                'translation_error_mm': round(dt, 4),
                'translation_error_over_diameter': round(dt / diameter, 5),
                'has_annotated_symmetry': int(has_sym),
            })
            # Compatibilita' con una eventuale versione precedente del CSV.
            for metric in ('vsd', 'mssd', 'mspd'):
                passed = f'{metric}_passed'
                denom = 100 if metric == 'vsd' else 10
                if passed not in result or result[passed] == '':
                    agreement = as_float(r, f'{metric}_agreement')
                    result[passed] = round(agreement * denom) if np.isfinite(agreement) else ''
                if f'{metric}_agreement' not in result:
                    count = as_float(result, passed)
                    result[f'{metric}_agreement'] = count / denom if np.isfinite(count) else ''
                if f'{metric}_passed_str' not in result:
                    result[f'{metric}_passed_str'] = f'{result[passed]}/{denom}'
            rows.append(result)
    print(f'Pose caricate: {len(rows)}; mancanti o non valide: {skipped}')
    return rows


def score_for_group(r, group):
    rot = as_float(r, 'rot_error_raw_deg')
    trans = as_float(r, 'translation_error_over_diameter')
    mssd = as_float(r, 'mssd_passed')
    mspd = as_float(r, 'mspd_passed')
    vsd = as_float(r, 'vsd_passed')
    sym = int(r['has_annotated_symmetry']) != 0

    if group == 'translation_good_rotation_bad':
        # NON fare affermazioni sulla rotazione per oggetti simmetrici.
        return rot if not sym and trans <= GOOD_TRANSLATION_DIAM and rot >= BAD_ROTATION_DEG and mssd <= BAD_MSSD_PASSED else None
    if group == 'rotation_good_translation_bad':
        return trans if not sym and rot <= GOOD_ROTATION_DEG and trans >= BAD_TRANSLATION_DIAM and mssd <= BAD_MSSD_PASSED else None
    if group == 'both_bad':
        return rot / 180 + trans if not sym and rot >= BAD_ROTATION_DEG and trans >= BAD_TRANSLATION_DIAM else None
    if group == 'both_good':
        return mssd + mspd + vsd / 10 if mssd >= GOOD_MSSD_PASSED and mspd >= GOOD_MSPD_PASSED and vsd >= GOOD_VSD_PASSED else None
    if group == 'small_residual_errors':
        # Prove di pose quasi corrette ma non ancora abbastanza precise (MSSD intermedio).
        return -abs(mssd - 6.0) if not sym and 4 <= mssd <= 7 and trans <= GOOD_TRANSLATION_DIAM and rot <= 20 else None
    if group == 'vsd_good_mssd_bad':
        return vsd / 100 - mssd / 10 if vsd >= GOOD_VSD_PASSED and mssd <= BAD_MSSD_PASSED else None
    if group == 'mspd_good_mssd_bad':
        return mspd / 10 - mssd / 10 if mspd >= GOOD_MSPD_PASSED and mssd <= BAD_MSSD_PASSED else None
    if group == 'all_metrics_bad':
        return as_float(r, 'mssd_over_diameter', 0) + as_float(r, 'mspd_px_640', 0) / 50 if mssd == 0 and mspd == 0 and vsd <= 20 else None
    raise ValueError(group)


GROUPS = [
    'translation_good_rotation_bad',
    'rotation_good_translation_bad',
    'small_residual_errors',
    'both_bad',
    'both_good',
    'vsd_good_mssd_bad',
    'mspd_good_mssd_bad',
    'all_metrics_bad',
]


def select_balanced(rows, group):
    by_obj = defaultdict(list)
    for r in rows:
        score = score_for_group(r, group)
        if score is not None and np.isfinite(score):
            by_obj[int(r['obj_id'])].append((score, r))
    for subset in by_obj.values():
        subset.sort(key=lambda p: p[0], reverse=True)
    result, seen = [], set()
    while len(result) < N_PER_GROUP:
        changed = False
        for obj_id in sorted(by_obj, key=lambda x: len(by_obj[x]), reverse=True):
            while by_obj[obj_id]:
                _, row = by_obj[obj_id].pop(0)
                key = (row['scene_id'], row['im_id'], row['obj_id'])
                if key in seen:
                    continue
                seen.add(key)
                result.append({'group': group, **row})
                changed = True
                break
            if len(result) >= N_PER_GROUP:
                break
        if not changed:
            break
    return result


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def make_images(selected):
    pred = visual.csv_poses(cfg.PREDICTIONS_CSV)
    images_dir = OUT_DIR / 'images'
    scene_cache, mesh_cache = {}, {}
    sheets = defaultdict(list)
    failures = []
    for row in selected:
        scene, im, obj, est = [int(row[x]) for x in ('scene_id','im_id','obj_id','est_id')]
        group = row['group']
        key = (scene, im, obj, est)
        path = images_dir / group / f's{scene:06d}_i{im:06d}_o{obj:06d}_e{est}_gt{row["gt_id"]}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = visual.render_case(row, pred[key], scene_cache, mesh_cache)
            # Aggiunge sintomi di posa sotto al footer BOP originale.
            h, w = image.shape[:2]
            footer = np.full((52, w, 3), 255, dtype=np.uint8)
            status = 'RAW rotation (symmetry-aware only when has_symmetry=0)'
            line = (f'Rotation raw: {float(row["rot_error_raw_deg"]):.1f} deg   |   '
                    f'Translation: {float(row["translation_error_mm"]):.1f} mm '
                    f'({100*float(row["translation_error_over_diameter"]):.1f}% diameter)   |   '
                    f'Symmetry: {row["has_annotated_symmetry"]}')
            cv2.putText(footer, line, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10,10,10), 1, cv2.LINE_AA)
            cv2.putText(footer, status, (15, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60,60,60), 1, cv2.LINE_AA)
            result = np.concatenate([image, footer], axis=0)
            if not cv2.imwrite(str(path), result):
                raise IOError('cv2.imwrite failed')
            sheets[group].append(path)
        except Exception as e:
            failures.append(f'{path.name}: {type(e).__name__}: {e}')
    for group, paths in sheets.items():
        visual.contact_sheet(paths, images_dir / f'CONTACT_{group}.png')
    if failures:
        (OUT_DIR / 'render_errors.txt').write_text('\n'.join(failures), encoding='utf-8')
    print(f'Immagini create: {sum(map(len,sheets.values()))}; errori: {len(failures)}')


def main():
    rows = build_rows()
    if not rows:
        raise RuntimeError('Nessuna posa valida nel CSV.')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    write_csv(OUT_DIR / 'all_pose_diagnostics.csv', rows, columns)
    selected = []
    for group in GROUPS:
        picked = select_balanced(rows, group)
        selected += picked
        print(f'{group:33s}: {len(picked)} esempi')
    write_csv(OUT_DIR / 'selected_pose_cases.csv', selected, ['group'] + columns)
    make_images(selected)
    print(f'Output: {OUT_DIR}')
    print('ATTENZIONE: questo script mostra errori FINALI; non dimostra che li abbia creati il refiner.')


if __name__ == '__main__':
    main()
