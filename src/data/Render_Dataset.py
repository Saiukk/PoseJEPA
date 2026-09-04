import hashlib
import logging
import multiprocessing
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .render_utils import render_random_view

log = logging.getLogger(__name__)

# Statistiche di normalizzazione DINOv2 (== ImageNet mean/std, usate da
# AutoImageProcessor.from_pretrained("facebook/dinov2-base")).
# NOTA: se in futuro cambi encoder2d.model_name con un checkpoint che usa
# statistiche diverse, queste costanti vanno riallineate di conseguenza.
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def _stable_seed(*parts: object) -> int:
    """
    Hash deterministico cross-processo (a differenza di hash() built-in,
    che e' salato per processo via PYTHONHASHSEED e quindi NON riproducibile
    dopo un resume/restart del training).
    """
    s = "_".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def compute_context(model_name, dataset_source, cache_context):
    """
    Compute the 3D context for a given model name.
    Ritorna None se l'embedding non c'e' -> l'oggetto viene skippato.
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
    """Elenca gli oggetti disponibili nella root del dataset (una sotto-cartella per oggetto)."""
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
    """
    Split riproducibile train/val sulla lista di oggetti.
    Priorita': se val_count e' specificato, usa quello (numero assoluto di oggetti
    in validation). Altrimenti usa val_ratio (frazione del totale).
    Lo split dipende solo da split_seed, quindi e' identico ad ogni run/epoca/worker.
    """
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

        # usato da Engine._build_dataloader per calcolare num_workers
        # (un "shard" = un oggetto, dato che ogni oggetto e' indipendente)
        self.shard_paths = self.model_names

        # contatore epoca condiviso tra i worker del DataLoader, per far variare
        # il sampling delle pose ad ogni epoca in modo deterministico
        self._epoch_counter = multiprocessing.Value("i", 0)

        # cache in RAM: features+centers+centroid+scale pesano poco, tienili in
        # memoria per oggetto invece di rileggere il .npz ad ogni __getitem__
        self._context_cache: dict[str, tuple] = {}

        # normalizzazione DINOv2, precomputata come tensori (broadcast su C,H,W)
        self._dino_mean = torch.tensor(DINOV2_MEAN).view(3, 1, 1)
        self._dino_std = torch.tensor(DINOV2_STD).view(3, 1, 1)

        log.info(
            "[%s] RenderDataset: %d oggetti totali -> train=%d val=%d (seed=%d), "
            "views_per_epoch=%d",
            split, len(all_names), len(train_names), len(val_names),
            split_seed, self.views_per_epoch,
        )

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

    def __getitem__(self, idx: int) -> dict:
        model_idx = idx % len(self.model_names)
        model_name = self.model_names[model_idx]

        context_3d, center_3d, centroid, scale = self._get_context(model_name)

        # seed deterministico cross-processo: stessa epoca + stesso idx ->
        # stessa posa campionata, ANCHE dopo uno stop/resume del training
        # (a differenza di hash() built-in, che e' salato per processo).
        epoch = self._epoch_counter.value
        seed = _stable_seed(model_name, epoch, idx) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        image, mask, K, R, t = render_random_view(
            model_name=model_name,
            dataset_source=self.dataset_source,
            centroid=centroid,
            scale=scale,
            image_size=self.image_size,
            rng=rng,
        )

        # image e' RGB grezzo in [0,1] (3, H, W), sfondo forzato a zero da
        # render_random_view. Normalizza con le statistiche DINOv2 PRIMA di
        # passarla all'encoder2d (che si aspetta input gia' normalizzato,
        # niente resize qui: la griglia patch deve restare allineata ai raggi).
        image_t = torch.from_numpy(image).float()
        image_t = (image_t - self._dino_mean) / self._dino_std

        return {
            "model_name": model_name,
            "image": image_t,                                   # (3, H, W) normalizzata DINOv2
            "mask": torch.from_numpy(mask).float(),             # (H, W) silhouette, non usata nella loss di Fase 1
            "context_3d": torch.from_numpy(context_3d).float(),
            "center_3d": torch.from_numpy(center_3d).float(),
            "centroid": torch.from_numpy(centroid).float(),
            "scale": torch.tensor(float(scale)),
            "K": torch.from_numpy(K).float(),
            "rot_matrix": torch.from_numpy(R).float(),
            "trans": torch.from_numpy(t).float(),
        }