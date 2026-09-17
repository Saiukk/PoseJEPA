"""
convert_npz_to_npy.py
Uso:
    python convert_npz_to_npy.py /path/to/dataset_source
    python convert_npz_to_npy.py /path/to/dataset_source --workers 8
    python convert_npz_to_npy.py /path/to/dataset_source --dry-run

"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

NPZ_FILENAME = "embeddingDINO.npz"
NPY_SUBDIR = "embeddingDINO"
ARRAY_KEYS = ("features", "centers", "centroid", "scale")


def convert_one(model_dir: str, overwrite: bool = False) -> tuple[str, str]:
    """Converte un singolo modello. Ritorna (model_dir, status)."""
    npz_path = os.path.join(model_dir, NPZ_FILENAME)
    out_dir = os.path.join(model_dir, NPY_SUBDIR)

    if not os.path.isfile(npz_path):
        return model_dir, "skip: npz mancante"

    marker = os.path.join(out_dir, ".done")
    if os.path.isfile(marker) and not overwrite:
        return model_dir, "skip: gia' convertito"

    try:
        with np.load(npz_path) as data:
            arrays = {k: np.array(data[k]) for k in ARRAY_KEYS}
    except Exception as e:
        return model_dir, f"ERRORE lettura npz: {e}"

    os.makedirs(out_dir, exist_ok=True)
    try:
        for k, arr in arrays.items():
            # np.save scrive sempre .npy non compresso: e' esattamente
            # il formato che supporta mmap_mode.
            np.save(os.path.join(out_dir, f"{k}.npy"), arr)
        # marker per sapere che la conversione e' completa (evita file
        # parziali se il processo viene interrotto a meta')
        with open(marker, "w") as f:
            f.write("ok")
    except Exception as e:
        return model_dir, f"ERRORE scrittura npy: {e}"

    return model_dir, "ok"


def list_model_dirs(dataset_source: str) -> list[str]:
    return sorted(
        os.path.join(dataset_source, d)
        for d in os.listdir(dataset_source)
        if os.path.isdir(os.path.join(dataset_source, d))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_source", nargs="?", default="/data/riccardop02/datasets/gso", help="Cartella dataset sorgente")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--overwrite", action="store_true", help="Riconverte anche se gia' fatto")
    ap.add_argument("--dry-run", action="store_true", help="Elenca solo cosa farebbe")
    args = ap.parse_args()

    model_dirs = list_model_dirs(args.dataset_source)
    print(f"Trovati {len(model_dirs)} modelli in {args.dataset_source}")

    if args.dry_run:
        for d in model_dirs:
            npz_path = os.path.join(d, NPZ_FILENAME)
            print(f"  {'trovato' if os.path.isfile(npz_path) else 'MANCANTE'}: {npz_path}")
        return

    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(convert_one, d, args.overwrite): d for d in model_dirs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            model_dir, status = fut.result()
            name = os.path.basename(model_dir)
            if status == "ok":
                n_ok += 1
            elif status.startswith("skip"):
                n_skip += 1
            else:
                n_err += 1
                print(f"[{i}/{len(model_dirs)}] {name}: {status}", file=sys.stderr)

            if i % 50 == 0 or i == len(model_dirs):
                print(f"[{i}/{len(model_dirs)}] ok={n_ok} skip={n_skip} err={n_err}")

    print(f"\nCompletato. ok={n_ok} skip={n_skip} err={n_err}")
    if n_err:
        print("Alcuni modelli hanno dato errore, controlla i messaggi sopra.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()