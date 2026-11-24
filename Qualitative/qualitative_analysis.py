"""
Analizza i file errors_*.json GIA' CALCOLATI da BOP Toolkit.
Esporta: all_cases.csv, selected_cases.csv, correlations.png, summary.txt.
Non lancia inferenza, non ricalcola VSD/MSSD/MSPD e non modifica BOP.

Uso: python analyze_bop.py
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import config_qualitative as cfg

THRESHOLDS_01 = np.arange(0.05, 0.51, 0.05)
THRESHOLDS_PX = np.arange(5, 51, 5)

# Open a JSON
def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# Reads the BOP File
def read_error_folder(folder):
    """chiave: (scene, image, object, estimation_index, gt_index)"""
    values = {}
    files = sorted(folder.glob('errors_*.json'))
    if not files:
        raise FileNotFoundError(f'Nessun errors_*.json in {folder}')
    for path in files:
        for row in read_json(path):
            for gt_id, errors in row['errors'].items():
                key = (int(row['scene_id']), int(row['im_id']), int(row['obj_id']),
                       int(row['est_id']), int(gt_id))
                values[key] = float(errors[0])
    return values

## RETURN THE DIRECTORY OF BOP
def metric_dirs(eval_dir, prefix):
    # Solo folders della BOP19 localization (n_top=-1), non mescolare settings.
    dirs = sorted(p for p in eval_dir.iterdir()
                  if p.is_dir() and p.name.startswith(f'error={prefix}_ntop=-1')
                  and list(p.glob('errors_*.json')))
    if not dirs:
        raise FileNotFoundError(f'Manca error={prefix}_ntop=-1* in {eval_dir}. '
                                'Riesegui eval_bop19_pose.py senza --cleanup_eval.')
    return dirs


def width_for_scene(dataset_root, split, scene, image, cache):
    if scene not in cache:
        base = dataset_root / split / f'{scene:06d}'
        paths = [base / d / f'{image:06d}{ext}'
                 for d in ('rgb', 'gray') for ext in ('.png', '.jpg', '.jpeg', '.tif')]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f'Immagine RGB non trovata in {base}/rgb')
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Impossibile aprire {path}')
        cache[scene] = int(img.shape[1])
    return cache[scene]


def visible_gt_indices(root, split, scene, im, obj, target_count, cache):
    """GT visibili: stessa selezione BOP19 per localization se visib_gt_min=-1."""
    if scene not in cache:
        base = root / split / f'{scene:06d}'
        cache[scene] = (read_json(base / 'scene_gt.json'),
                        read_json(base / 'scene_gt_info.json'))
    gt, info = cache[scene]
    all_ids = [j for j, pose in enumerate(gt[str(im)]) if int(pose['obj_id']) == obj]
    all_ids.sort(key=lambda j: float(info[str(im)][j].get('visib_fract', 0.0)), reverse=True)
    return set(all_ids[:target_count]) if target_count is not None else set(all_ids)


# Return the value of that group for that specific row.
def group_score(row, group):
    vals = [row['vsd_agreement'], row['mssd_agreement'], row['mspd_agreement']]
    average = sum(vals) / 3
    if group == 'overall_best': return min(vals) if min(vals) >= cfg.GOOD else None
    if group == 'overall_worst': return -max(vals) if max(vals) <= cfg.BAD else None
    if group.endswith('_best'):
        return row[group[:-5] + '_agreement']
    if group.endswith('_worst'):
        return -row[group[:-6] + '_agreement']
    if group == 'mspd_good_mssd_bad':
        return row['mspd_agreement'] - row['mssd_agreement'] if row['mspd_agreement'] >= cfg.GOOD and row['mssd_agreement'] <= cfg.BAD else None
    if group == 'vsd_good_mssd_bad':
        return row['vsd_agreement'] - row['mssd_agreement'] if row['vsd_agreement'] >= cfg.GOOD and row['mssd_agreement'] <= cfg.BAD else None
    raise ValueError(f'Gruppo non riconosciuto: {group}')


def balanced_top(rows, group, limit):
    """Preferisce oggetti diversi invece di 10 frame quasi uguali dello stesso oggetto."""
    per_obj = defaultdict(list)
    for row in rows:
        value = group_score(row, group)
        if value is not None:
            per_obj[row['obj_id']].append((value, row))
    # ordered for score
    for items in per_obj.values():
        items.sort(key=lambda item: item[0], reverse=True)
    result, seen_im = [], set()
    while len(result) < limit:
        changed = False
        # Reorder for object in image
        for obj_id in sorted(per_obj, key=lambda obj: len(per_obj[obj]), reverse=True):
            while per_obj[obj_id]:
                _, row = per_obj[obj_id].pop(0)
                key = (row['scene_id'], row['im_id'], row['obj_id'])
                if key in seen_im:
                    continue
                seen_im.add(key)
                result.append({'group': group, **row})
                changed = True
                # Used to skip to other object to have variety
                break
            if len(result) >= limit:
                break
        # If we do not add nothing, we exit
        if not changed:
            break
    return result


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot_correlations(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[info] matplotlib assente: salto correlations.png')
        return
    pairs = [('mssd_agreement', 'mspd_agreement'),
             ('mssd_agreement', 'vsd_agreement'),
             ('mspd_agreement', 'vsd_agreement')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (x, y) in zip(axes, pairs):
        a = np.array([r[x] for r in rows]); b = np.array([r[y] for r in rows])
        jitter = np.random.default_rng(42).normal(0, 0.006, size=(len(a), 2))
        ax.scatter(np.clip(a + jitter[:, 0], 0, 1),
                   np.clip(b + jitter[:, 1], 0, 1), s=10, alpha=0.4)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(rows) >= 2 and a.std() > 0 and b.std() > 0 else float('nan')
        ax.set(xlabel=x.replace('_agreement', ''), ylabel=y.replace('_agreement', ''),
               title=f'Pearson r = {corr:.2f}', xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        ax.grid(alpha=0.18)
    fig.suptitle('Per-pose threshold agreement (exploratory, NOT official BOP AR)')
    fig.tight_layout()
    fig.savefig(out_dir / 'correlations.png', dpi=180)
    plt.close(fig)


def main():
    # takes root and inference directory and create the output dir.
    root, eval_dir = cfg.DATASET_ROOT, cfg.EVAL_DIR
    if not root.is_dir(): raise FileNotFoundError(f'DATASET_ROOT errato: {root}')
    if not eval_dir.is_dir(): raise FileNotFoundError(f'EVAL_DIR errato: {eval_dir}')
    output = cfg.OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    # Takes the model directory and information
    model_info_path = root / 'models_eval' / 'models_info.json'
    if not model_info_path.exists(): model_info_path = root / 'models' / 'models_info.json'
    model_info = read_json(model_info_path)
    diameters = {int(k): float(v['diameter']) for k, v in model_info.items()}

    # Data from the instance and object present in the image
    target_file = root / 'test_targets_bop19.json'
    if not target_file.exists():
        raise FileNotFoundError(f'Manca {target_file}; serve per selezionare le GT corrette')
    target_counts = {(int(t['scene_id']), int(t['im_id']), int(t['obj_id'])): int(t['inst_count'])
                     for t in read_json(target_file)}

    # Return for each images: (scene, img_id, obj_id ----> Error) for all the 3 error
    mssd = read_error_folder(metric_dirs(eval_dir, 'mssd')[0])
    mspd = read_error_folder(metric_dirs(eval_dir, 'mspd')[0])
    vsd_dirs = metric_dirs(eval_dir, 'vsd')
    vsd = {folder.name.split('_tau=')[-1]: read_error_folder(folder) for folder in vsd_dirs}
    print(f'Caricate: MSSD={len(mssd)}, MSPD={len(mspd)}, VSD={len(vsd)} tau')
    if len(vsd) < 10:
        print('[avviso] Sono disponibili meno dei 10 tau BOP VSD: l\'agreement usa solo quelli disponibili.')

    # Starting to takes the first image
    width_cache, scene_cache = {}, {}
    groups = defaultdict(list)
    # If is not in all of the 3 error, skip
    for key, err3d in mssd.items():
        if key not in mspd or any(key not in m for m in vsd.values()):
            continue
        scene, image, obj, est, gt = key
        target_key = (scene, image, obj)
        # if is not present in the target, skip
        if target_key not in target_counts: continue

        diameter = diameters[obj]
        width = width_for_scene(root, cfg.SPLIT, scene, image, width_cache)
        # MSSD: diameter used for normalization. MSPD: error_px * (640 / image_width).
        e3 = err3d / diameter
        e2 = mspd[key] * 640.0 / width
        e_vsd = [v[key] for v in vsd.values()]
        row = {
            'scene_id': scene, 'im_id': image, 'obj_id': obj,
            'est_id': est, 'gt_id': gt,
            'mssd_mm': err3d, 'mssd_over_diameter': e3,
            'mspd_px': mspd[key], 'mspd_px_640': e2,
            'vsd_mean_error': float(np.mean(e_vsd)),
            'vsd_agreement': float(np.mean([np.mean(e < THRESHOLDS_01) for e in e_vsd])),
            'mssd_agreement': float(np.mean(e3 < THRESHOLDS_01)),
            'mspd_agreement': float(np.mean(e2 < THRESHOLDS_PX)),
            'vsd_passed': int(sum(np.sum(e < THRESHOLDS_01) for e in e_vsd)),
            'mssd_passed': int(np.sum(e3 < THRESHOLDS_01)),
            'mspd_passed': int(np.sum(e2 < THRESHOLDS_PX)),
        }
        row['mean_agreement'] = (row['vsd_agreement'] + row['mssd_agreement'] + row['mspd_agreement']) / 3
        groups[target_key].append(row)

    # Associazione 1:1 stimata<->GT, basata su MSSD. Serve SOLO per scegliere visualizzazioni.
    # Non e' il matching BOP ad ogni soglia e non altera i risultati ufficiali.
    rows = []
    for (scene, image, obj), candidates in groups.items():
        # return only the N valid instance in the image 
        valid_gt = visible_gt_indices(root, cfg.SPLIT, scene, image, obj,
                                       target_counts[(scene, image, obj)], scene_cache)
        candidates = [r for r in candidates if r['gt_id'] in valid_gt]
        candidates.sort(key=lambda r: r['mssd_over_diameter'])
        used_est, used_gt = set(), set()
        for r in candidates:
            if r['est_id'] in used_est or r['gt_id'] in used_gt:
                continue
            rows.append(r)
            used_est.add(r['est_id']); used_gt.add(r['gt_id'])

    if not rows:
        raise RuntimeError('Nessuna stima confrontabile: controlla root/eval/targets e le cartelle BOP.')
    rows.sort(key=lambda r: (r['scene_id'], r['im_id'], r['obj_id'], r['est_id']))
    columns = list(rows[0].keys())
    write_csv(output / 'all_cases.csv', rows, columns)

    # We want only a portion to visualize so we takes some example, and we save another csv with a smaller amount of data
    selected = []
    for group in cfg.GROUPS:
        # We picked randomly a variety of objects related to the group
        picked = balanced_top(rows, group, cfg.N_PER_GROUP)
        selected.extend(picked)
        print(f'{group:24s}  {len(picked)} esempi')
    write_csv(output / 'selected_cases.csv', selected, ['group'] + columns)
    plot_correlations(rows, output)

    summary = [f'Pose confrontate: {len(rows)}',
               f'Tau VSD trovati: {len(vsd)}',
               'ATTENZIONE: agreement = frazione di soglie BOP superate per singola coppia stimata/GT.',
               'NON e\' Average Recall ufficiale BOP: non include il matching per ogni soglia',
               'e la normalizzazione per oggetti/scenes delle metriche aggregate.',
               'Matching per le figure: greedy 1-a-1 per MSSD sullo stesso (scene,image,obj).',
               'Per diagnosticare search/refiner/scorer servono anche le pose intermedie.',
               'Categorie selezionate:']
    summary.extend(f'  {g}: {sum(1 for r in selected if r["group"] == g)}' for g in cfg.GROUPS)
    (output / 'summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(f'\nFatto: {output.resolve()}')


if __name__ == '__main__':
    main()
