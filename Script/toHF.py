import argparse
import io
import json
import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import hydra
import numpy as np
import pyrender
import torch
import webdataset as wds
from hydra import compose, initialize_config_dir
from PIL import Image
from huggingface_hub import HfApi, create_repo

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/riccardop02/JEPA")
from src.data.Render_Dataset import (  # noqa: E402
    DINOV2_MEAN,
    DINOV2_STD,
    _stable_seed,
    compute_context,
    list_model_names,
    split_model_names,
)
from src.data.render_utils import build_pyrender_mesh, render_random_view  # noqa: E402
from dotenv import load_dotenv

load_dotenv("/home/riccardop02/JEPA/.env")

# Quante immagini accumulare prima di fare un forward batched su encoder2d
# (solo parametro di efficienza, non tocca il modello/preprocessing).
TARGET_BATCH_SIZE = 8


def image_float_to_png_bytes(image_chw: np.ndarray) -> bytes:
    """image_chw: (3,H,W) float32 in [0,1], gia' mascherato. Ricostruisce
    l'uint8 RGB HWC originale (round-trip esatto dato che il valore di
    partenza era uint8/255)."""
    rgb_hwc = np.transpose(image_chw, (1, 2, 0))  # CHW -> HWC
    rgb_uint8 = np.clip(np.round(rgb_hwc * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def mask_float_to_png_bytes(mask_hw: np.ndarray) -> bytes:
    """mask_hw: (H,W) float32 0/1 -> PNG 8-bit 0/255."""
    mask_uint8 = (mask_hw * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


class SharedRenderer:
    """Stessa logica di RenderDataset._get_renderer: un renderer persistente
    per processo, ridimensionato quando cambia la viewport richiesta."""

    def __init__(self):
        self._renderer: pyrender.OffscreenRenderer | None = None
        self._size: tuple[int, int] | None = None

    def get(self, width: int, height: int) -> pyrender.OffscreenRenderer:
        if self._renderer is None:
            self._renderer = pyrender.OffscreenRenderer(width, height)
            self._size = (width, height)
        elif self._size != (width, height):
            self._renderer.viewport_width = width
            self._renderer.viewport_height = height
            self._size = (width, height)
        return self._renderer

    def delete(self):
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception:
                pass


def build_encoder2d(config_dir: str, config_name: str, overrides: list[str], device_override: str | None):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    encoder2d = hydra.utils.instantiate(cfg.encoder2d)
    device = torch.device(device_override or cfg.get("device", "cuda:0"))
    encoder2d = encoder2d.to(device).eval()
    for p in encoder2d.parameters():
        p.requires_grad_(False)

    trainable_components = list(cfg.get("train", {}).get("trainable_components", ["predictor"]))
    if "encoder2d" in trainable_components:
        print(
            "ATTENZIONE: 'encoder2d' e' tra i trainable_components del config caricato: "
            "encoder2d cambia durante il training, quindi un target precomputato "
            "qui diventa disallineato dopo il primo update. Verifica che sia corretto "
            "precomputarlo per questo esperimento."
        )

    return encoder2d, device


def image_to_encoder_tensor(image: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    image_t = torch.from_numpy(image).float()
    image_t = (image_t - mean) / std
    return image_t


@torch.no_grad()
def encode_batch(encoder2d, image_tensors: list[torch.Tensor], device: torch.device) -> np.ndarray:
    batch = torch.stack(image_tensors, dim=0).to(device)
    target = encoder2d(batch)
    return target.cpu().numpy()


def render_shard_to_tar(
    idx_range: range,
    model_names: list[str],
    epoch: int,
    image_size: int,
    dataset_source: str,
    mesh_cache: dict,
    shared_renderer: SharedRenderer,
    encoder2d,
    device: torch.device,
    dino_mean: torch.Tensor,
    dino_std: torch.Tensor,
    local_path_base: str,
    local_path_dino: str,
) -> tuple[int, int]:
    
    n_models = len(model_names)
    n_written_base = 0
    n_written_dino = 0

    # Inizializziamo il file di base. Il file DINO viene inizializzato SOLO se encoder2d non è None.
    sink_base = wds.TarWriter(local_path_base)
    sink_dino = wds.TarWriter(local_path_dino) if encoder2d is not None else None

    try:
        pending_keys: list[str] = []
        pending_meta_bytes: list[bytes] = []
        pending_tensors: list[torch.Tensor] = []

        def flush_target():
            nonlocal n_written_dino
            if not pending_keys or sink_dino is None:
                return
            feats = encode_batch(encoder2d, pending_tensors, device)
            for key, meta_bytes, feat in zip(pending_keys, pending_meta_bytes, feats):
                sink_dino.write({
                    "__key__": key,
                    "meta.json": meta_bytes,
                    "dino.npy": npy_bytes(feat),
                })
                n_written_dino += 1
            pending_keys.clear()
            pending_meta_bytes.clear()
            pending_tensors.clear()

        for idx in idx_range:
            model_idx = idx % n_models
            model_name = model_names[model_idx]

            if model_name not in mesh_cache:
                model_dir = os.path.join(dataset_source, model_name.replace("gso_", ""))
                mesh_cache[model_name] = build_pyrender_mesh(model_dir)
            mesh = mesh_cache[model_name]

            ctx = compute_context(model_name, dataset_source, cache_context=True)
            if ctx is None:
                continue
            _, _, centroid, scale = ctx

            seed = _stable_seed(model_name, epoch, idx) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

            image, mask, K, R, t = render_random_view(
                render_mesh=mesh,
                centroid=np.asarray(centroid),
                scale=float(scale),
                image_size=image_size,
                rng=rng,
                get_renderer_fn=shared_renderer.get,
            )

            meta = {
                "model_name": model_name,
                "K": np.asarray(K, dtype=np.float32).flatten().tolist(),
                "R": np.asarray(R, dtype=np.float32).flatten().tolist(),
                "t": np.asarray(t, dtype=np.float32).flatten().tolist(),
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            key = f"{idx:08d}"

            sink_base.write({
                "__key__": key,
                "image.png": image_float_to_png_bytes(image),
                "mask.png": mask_float_to_png_bytes(mask),
                "meta.json": meta_bytes,
            })
            n_written_base += 1

            # Accumuliamo per DINO solo se l'encoder è presente
            if encoder2d is not None:
                pending_keys.append(key)
                pending_meta_bytes.append(meta_bytes)
                pending_tensors.append(image_to_encoder_tensor(image, dino_mean, dino_std))
                if len(pending_keys) >= TARGET_BATCH_SIZE:
                    flush_target()

        if encoder2d is not None:
            flush_target()

    finally:
        sink_base.close()
        if sink_dino is not None:
            sink_dino.close()

    return n_written_base, n_written_dino


def chunk_range(total: int, n_chunks: int):
    size = max(1, total // n_chunks + (1 if total % n_chunks else 0))
    for start in range(0, total, size):
        yield range(start, min(start + size, total))


def fetch_existing_remote_files(api: HfApi, repo_id: str, token: str | None) -> set[str]:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        return set(files)
    except Exception as e:
        print(f"ATTENZIONE: impossibile elencare i file remoti gia' presenti ({e}); "
              f"si riparte senza skip, tutto verra' ri-renderizzato/ricaricato.")
        return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_source", nargs="?", default="/data/riccardop02/datasets/gso")
    ap.add_argument("--repo_id", nargs="?", default="Saiukk/POSEJEPA_Training")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--views_per_epoch", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--shards_per_epoch", type=int, default=16,
                     help="Piu' shard = piu' parallelismo lato streaming/worker a train time")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--token", default=None, help="Default: variabile d'ambiente HF_TOKENPRO")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--start_epoch", type=int, default=0,
                     help="Riprendi da questa epoca se il job si interrompe")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])

    ap.add_argument("--config_dir", default="/home/riccardop02/JEPA/configs")
    ap.add_argument("--config_name", default="config")
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    token = args.token or os.getenv("HF_TOKENPRO")
    if not token:
        print("ATTENZIONE: nessun HF_TOKEN trovato, upload rischia rate limit basso.")

    all_names = list_model_names(args.dataset_source)
    train_names, val_names = split_model_names(
        all_names, val_ratio=args.val_ratio, split_seed=args.split_seed
    )
    names_by_split = {"train": train_names, "val": val_names}
    print(f"train={len(train_names)} val={len(val_names)} modelli totali={len(all_names)}")

    create_repo(args.repo_id, repo_type="dataset", private=args.private,
                token=token, exist_ok=True)
    api = HfApi(token=token)

    existing_remote_files = fetch_existing_remote_files(api, args.repo_id, token)
    print(f"Trovati {len(existing_remote_files)} file gia' presenti sul repo remoto.")

    mesh_cache: dict = {}
    shared_renderer = SharedRenderer()


    # encoder2d, device = build_encoder2d(args.config_dir, args.config_name, args.overrides, args.device)
    # print(f"encoder2d caricato via hydra ({args.config_dir}/{args.config_name}), device={device}")
    encoder2d = None


    device = torch.device("cpu") # Fallback dummy se encoder2d è None
    
    dino_mean = torch.tensor(DINOV2_MEAN).view(3, 1, 1)
    dino_std = torch.tensor(DINOV2_STD).view(3, 1, 1)

    try:
        for epoch in range(args.start_epoch, args.epochs):
            for split_name in args.splits:
                names = names_by_split[split_name]
                n_total_idx = len(names) * args.views_per_epoch

                for shard_idx, idx_range in enumerate(
                    chunk_range(n_total_idx, args.shards_per_epoch)
                ):
                    remote_path_base = (
                        f"data/{split_name}/epoch_{epoch:03d}/shard-{shard_idx:05d}.tar"
                    )
                    remote_path_dino = (
                        f"data_dino/{split_name}/epoch_{epoch:03d}/shard-{shard_idx:05d}.tar"
                    )


                    skip_cond_base = remote_path_base in existing_remote_files
                    skip_cond_dino = remote_path_dino in existing_remote_files
                    
                    if skip_cond_base and (encoder2d is None or skip_cond_dino):
                        print(f"[{split_name}] epoch {epoch} shard {shard_idx}: gia' presente su HF, skip.")
                        continue

                    local_path_base = f"/tmp/shard-{shard_idx:05d}.tar"
                    local_path_dino = f"/tmp/shard-{shard_idx:05d}-dino.tar"

                    n_rows_base, n_rows_dino = render_shard_to_tar(
                        idx_range, names, epoch, args.image_size,
                        args.dataset_source, mesh_cache, shared_renderer,
                        encoder2d, device, dino_mean, dino_std,
                        local_path_base, local_path_dino,
                    )

                    api.upload_file(
                        path_or_fileobj=local_path_base,
                        path_in_repo=remote_path_base,
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        token=token,
                    )
                    os.remove(local_path_base)
                    existing_remote_files.add(remote_path_base)

                    # Upload e cleanup del file DINO (avviene solo se sono stati scritti dati)
                    if n_rows_dino != 0:
                        api.upload_file(
                            path_or_fileobj=local_path_dino,
                            path_in_repo=remote_path_dino,
                            repo_id=args.repo_id,
                            repo_type="dataset",
                            token=token,
                        )
                        os.remove(local_path_dino)
                        existing_remote_files.add(remote_path_dino)
                    
                    print(f"[{split_name}] epoch {epoch} shard {shard_idx}: "
                          f"{n_rows_base} righe -> {remote_path_base} | "
                          f"{n_rows_dino} righe -> {remote_path_dino}")
    finally:
        shared_renderer.delete()

    print("Completato.")


if __name__ == "__main__":
    main()