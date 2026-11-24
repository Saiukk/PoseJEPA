import hashlib
import logging
import multiprocessing
import os
import random

import numpy as np
import pyrender
import torch
from torch.utils.data import Dataset

from .render_utils import build_pyrender_mesh, render_random_view

log = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def _stable_seed(*parts: object) -> int:
    s = "_".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def compute_context(model_name, dataset_source, cache_context):
    """
    Compute the 3D context for a given model name.
    """
    model_name = model_name.replace("gso_", "")
    if cache_context:
        context_path = f"{dataset_source}/{model_name}/embedding.npz"
        try:
            context_data = np.load(context_path)
            context_3d = context_data["features"]
            center_3d = context_data["centers"]
            centroid = context_data["centroid"]
            scale = context_data["scale"]
        except FileNotFoundError:
            logging.warning(f"Cached context for {model_name} not found at {context_path}.")
            return None
    else:
        raise NotImplementedError("3D context computation is not implemented yet.")

    return context_3d, center_3d, centroid, scale


def list_model_names(dataset_source: str) -> list[str]:
    """List all model names in the dataset source, used to divide Train and Validation."""
    return sorted(
        d for d in os.listdir(dataset_source)
        if os.path.isdir(os.path.join(dataset_source, d))
    )


def split_model_names(
    model_names: list[str],
    val_count: int | None = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> tuple[list[str], list[str]]:

    rng = random.Random(split_seed)
    shuffled = list(model_names)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        raise ValueError(f"Nessun oggetto trovato per lo split (n={n}).")

    if val_count is not None:
        n_val = int(val_count)
    else:
        n_val = int(round(n * val_ratio))

    n_val = max(0, min(n_val, n - 1))  # tieni almeno 1 oggetto in train
    val_names = shuffled[:n_val]
    train_names = shuffled[n_val:]

    return train_names, val_names


class RenderDataset(Dataset):
    """
    Render Dataset: return a random view of a 3D model, along with its 3D context and camera parameters.
    """

    def __init__(
        self,
        dataset_source: str,
        split: str = "train",
        cache_context: bool = True,
        val_count: int | None = None,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        grid_size: int = 16,
        patch_size: int = 14,
        image_size: int = 224,
        views_per_epoch: int = 1,
        preload_meshes: bool = True,
    ):
        assert split in ("train", "val"), f"split non valido: {split}"

        self.dataset_source = dataset_source
        self.split = split
        self.cache_context = cache_context
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.views_per_epoch = max(1, views_per_epoch)

        all_names = list_model_names(dataset_source)
        train_names, val_names = split_model_names(
            all_names, val_count=val_count, val_ratio=val_ratio, split_seed=split_seed
        )
        self.model_names = train_names if split == "train" else val_names

        if len(self.model_names) == 0:
            raise ValueError(
                f"Split '{split}' vuoto: controlla val_count/val_ratio "
                f"rispetto al numero totale di oggetti ({len(all_names)})."
            )

        self.shard_paths = self.model_names
        self._epoch_counter = multiprocessing.Value("i", 0)
        self._context_cache: dict[str, tuple] = {}

        # Cache pyrender mesh to reduce the overhead of loading
        self._mesh_cache: dict[str, pyrender.Mesh] = {}

        if preload_meshes:
            self.preload_meshes()

        # Each worker will have its own renderer instance, created on demand only once and reused
        self._renderer: pyrender.OffscreenRenderer | None = None
        self._renderer_size: tuple[int, int] | None = None

        # normalizzazione DINOv2, precomputata come tensori (broadcast su C,H,W)
        self._dino_mean = torch.tensor(DINOV2_MEAN).view(3, 1, 1)
        self._dino_std = torch.tensor(DINOV2_STD).view(3, 1, 1)

        log.info(
            "[%s] RenderDataset: %d oggetti totali -> train=%d val=%d (seed=%d), "
            "views_per_epoch=%d",
            split, len(all_names), len(train_names), len(val_names),
            split_seed, self.views_per_epoch,
        )

    def preload_meshes(self) -> None:
        """
        Preload all meshes into memory to reduce the overhead of loading during training.
        """
        log.info("[%s] Preload di %d mesh in corso...", self.split, len(self.model_names))
        for name in self.model_names:
            model_dir = os.path.join(self.dataset_source, name.replace("gso_", ""))
            self._mesh_cache[name] = build_pyrender_mesh(model_dir)
        log.info("[%s] Preload completato (%d mesh in cache).", self.split, len(self._mesh_cache))

    def __len__(self) -> int:
        return len(self.model_names) * self.views_per_epoch

    def _get_context(self, model_name: str):
        if model_name not in self._context_cache:
            data = compute_context(model_name, self.dataset_source, self.cache_context)
            if data is None:
                raise FileNotFoundError(
                    f"Context 3D mancante per '{model_name}': esegui prima cacheEmb3d.py"
                )
            self._context_cache[model_name] = data
        return self._context_cache[model_name]

    def _get_mesh(self, model_name: str) -> pyrender.Mesh:
        mesh = self._mesh_cache.get(model_name)
        if mesh is None:
            model_dir = os.path.join(self.dataset_source, model_name.replace("gso_", ""))
            mesh = build_pyrender_mesh(model_dir)
            self._mesh_cache[model_name] = mesh
        return mesh

    def _get_renderer(self, width: int, height: int) -> pyrender.OffscreenRenderer:
        """Return a persistent renderer instance for the current worker
        """
        if self._renderer is None:
            self._renderer = pyrender.OffscreenRenderer(width, height)
            self._renderer_size = (width, height)
        elif self._renderer_size != (width, height):
            self._renderer.viewport_width = width
            self._renderer.viewport_height = height
            self._renderer_size = (width, height)
        return self._renderer

    def __del__(self):
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.delete()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> dict:
        model_idx = idx % len(self.model_names)         # use module to restart model selection every views_per_epoch
        model_name = self.model_names[model_idx]

        context_3d, center_3d, centroid, scale = self._get_context(model_name)
        render_mesh = self._get_mesh(model_name)

        epoch = self._epoch_counter.value
        seed = _stable_seed(model_name, epoch, idx) & 0xFFFFFFFF        # Per la riproducibilità
        rng = np.random.default_rng(seed)

        image, mask, K, R, t = render_random_view(
            render_mesh=render_mesh,
            centroid=centroid,
            scale=scale,
            image_size=self.image_size,
            rng=rng,
            get_renderer_fn=self._get_renderer,
        )

        # Dino setup
        image_t = torch.from_numpy(image).float()
        image_t = (image_t - self._dino_mean) / self._dino_std

        return {
            "model_name": model_name,
            "image": image_t,                                   # (3, H, W)
            "mask": torch.from_numpy(mask).float(),             # (H, W) 
            "context_3d": torch.from_numpy(context_3d).float(),
            "center_3d": torch.from_numpy(center_3d).float(),
            "centroid": torch.from_numpy(centroid).float(),
            "scale": torch.tensor(float(scale)),
            "K": torch.from_numpy(K).float(),
            "rot_matrix": torch.from_numpy(R).float(),
            "trans": torch.from_numpy(t).float(),
        }


def make_dataloader(
    dataset: RenderDataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
):
    """Factory di comodo con le opzioni di prefetch/persistenza consigliate.

    persistent_workers=True e' cruciale: senza, i worker vengono ri-forkati
    ad ogni epoca e sia la mesh cache che il renderer persistente per
    worker verrebbero ricreati da zero, vanificando gran parte del guadagno.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )