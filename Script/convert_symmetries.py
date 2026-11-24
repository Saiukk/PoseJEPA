import argparse
import copy
import json
from pathlib import Path

import numpy as np


def convert_record(record, scale):
    out = copy.deepcopy(record)
    # Keep ALL length-valued metadata in the same coordinate system.
    for field in ('diameter', 'min_x', 'min_y', 'min_z', 'size_x', 'size_y', 'size_z'):
        if field in out:
            out[field] = float(out[field]) * scale

    for idx, flat in enumerate(out.get('symmetries_discrete', [])):
        S = np.asarray(flat, dtype=float).reshape(4, 4).copy()
        if not np.allclose(S[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f'Invalid homogeneous matrix at index {idx}')
        S[:3, 3] *= scale
        out['symmetries_discrete'][idx] = S.reshape(-1).tolist()

    for symmetry in out.get('symmetries_continuous', []):
        # Axis is a DIRECTION (dimensionless); offset is a POINT (length units).
        symmetry['offset'] = (np.asarray(symmetry['offset'], dtype=float) * scale).tolist()
    return out


def check_mesh(name, path, record, scale, tol):
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit('For --check, install trimesh: pip install trimesh') from e
    mesh = trimesh.load(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'{path} did not load as a single mesh')
    actual_min = mesh.bounds[0]
    actual_size = mesh.extents
    expected_min = scale * np.array([record[f'min_{a}'] for a in 'xyz'])
    expected_size = scale * np.array([record[f'size_{a}'] for a in 'xyz'])
    diameter = np.linalg.norm(actual_size)
    max_error = max(np.max(abs(actual_min - expected_min)),
                    np.max(abs(actual_size - expected_size)))
    relative = max_error / max(diameter, 1e-12)
    print(f'[{name}] max bbox discrepancy: {max_error:.6f} mesh units '
          f'({100*relative:.5f}% of bbox diagonal)')
    if relative > tol:
        raise ValueError(f'BBox mismatch for {name}: {relative:.5%} > {tol:.5%}. '
                         'Check object identity, scale, and coordinate frame.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default="/data/riccardop02/datasets/gso/gso.json", help='GSO-SAD gso.json')
    parser.add_argument('--output', type=Path, default= "/data/riccardop02/datasets/gso_MP/google_scanned_objects/models_bop-renderer_scale=0.1/gso.json", help='Converted JSON output path')
    parser.add_argument('--scale', type=float, default=100.0)
    parser.add_argument('--check', nargs=2, action='append', metavar=('OBJECT_NAME', 'PLY_PATH'),
                        help='Check object bounding box against mesh; repeatable')
    parser.add_argument('--bbox-tolerance', type=float, default=0.001,
                        help='Max bbox mismatch / bbox diagonal (default 0.001 = 0.1%%)')
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error('--scale must be positive')
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from input (keep the original as backup)')
    with args.input.open(encoding='utf-8') as f:
        source = json.load(f)
    if args.check:
        for name, ply in args.check:
            if name not in source:
                parser.error(f'Object not found in JSON: {name}')
            check_mesh(name, Path(ply), source[name], args.scale, args.bbox_tolerance)
    converted = {name: convert_record(rec, args.scale) for name, rec in source.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Converted {len(converted)} annotated objects to {args.output}')
    print('Unlisted objects are NOT annotated as asymmetric.')


if __name__ == '__main__':
    main()

