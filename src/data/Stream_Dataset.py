from __future__ import annotations

import io
import json
import logging
import os
import shlex
import sys
sys.path.insert(0, "/home/riccardop02/JEPA")
import numpy as np
import torch
import webdataset as wds
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader

from src.data.Render_Dataset import compute_context, list_model_names, split_model_names

log = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


class RenderDatasetStreaming(IterableDataset):

    def __init__(
        self,
        repo_id: str,
        dataset_source: str,
        split: str = "train",
        cache_context: bool = True,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        shuffle_buffer_size: int = None,
        token: str | None = None,
    ):
        self.repo_id = repo_id
        self.dataset_source = dataset_source
        self.split = split
        self.cache_context = cache_context
        self.shuffle_buffer_size = shuffle_buffer_size
        self.token = token or os.getenv("HF_TOKENPRO")
        self.epoch = 0

        self._api = HfApi(token=self.token)

        all_names = list_model_names(dataset_source)
        train_names, val_names = split_model_names(
            all_names, val_ratio=val_ratio, split_seed=split_seed
        )
        self.model_names = train_names if split == "train" else val_names
        self.shard_paths = self.model_names

        self._dino_mean = torch.tensor(DINOV2_MEAN).view(3, 1, 1)
        self._dino_std = torch.tensor(DINOV2_STD).view(3, 1, 1)

        log.info(
            "[%s] RenderDatasetStreaming(wds): repo=%s, %d modelli attesi, epoch=%d",
            split, repo_id, len(self.model_names), self.epoch,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


    def _epoch_prefix(self, epoch: int) -> str:
        return f"data/{self.split}/epoch_{epoch:03d}/"

    def _list_remote_shards(self, epoch: int) -> list[str]:
        prefix = self._epoch_prefix(epoch)
        try:
            all_files = self._api.list_repo_files(self.repo_id, repo_type="dataset")
        except Exception as e:
            log.warning("Impossibile listare i file del repo %s: %s", self.repo_id, e)
            return []
        shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
        return shards

    def get_n_shards(self, epoch: int | None = None) -> int:
        return len(self._list_remote_shards(epoch if epoch is not None else self.epoch))

    def _shard_url(self, remote_path: str) -> str:

        base = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{remote_path}"
        cmd = [
            "curl", "-s", "-L", "--fail",
            "--http1.1",              # evita HTTP2_STREAM_ERROR su connessioni lunghe/parallele
            "--retry", "5",            # riprova su errori transitori (reset, timeout, 5xx)
            "--retry-delay", "2",
            "--retry-connrefused",
        ]
        if self.token:
            cmd += ["-H", f"Authorization: Bearer {self.token}"]
        cmd += [base]
        # shlex.quote per sicurezza sui path/token, poi ricomposto come stringa
        # per il formato "pipe:" atteso da webdataset.
        quoted = " ".join(shlex.quote(c) for c in cmd)
        return f"pipe:{quoted}"

    # ---------- decodifica dei sample ----------

    def _decode_sample(self, sample: dict) -> dict:
        required = ("image.png", "mask.png", "meta.json")
        missing = [k for k in required if k not in sample]
        if missing:
            raise ValueError(
                f"sample {sample.get('__key__', '?')} da {sample.get('__url__', '?')} "
                f"incompleto: mancano {missing} (chiavi presenti: {list(sample.keys())})"
            )

        image_bytes = sample["image.png"]
        mask_bytes = sample["mask.png"]
        meta = json.loads(sample["meta.json"])
        model_name = meta["model_name"]

        context_3d, center_3d, centroid, scale = compute_context(
            model_name, self.dataset_source, self.cache_context
        )

        rgb_hwc = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        mask_hw = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))

        image_f = rgb_hwc.astype(np.float32) / 255.0
        mask_f = (mask_hw > 127).astype(np.float32)

        K = np.asarray(meta["K"], dtype=np.float32).reshape(3, 3)
        R = np.asarray(meta["R"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(meta["t"], dtype=np.float32)

        image_t = torch.from_numpy(image_f).permute(2, 0, 1)
        image_t = (image_t - self._dino_mean) / self._dino_std

        return {
            "model_name": model_name,
            "image": image_t,
            "mask": torch.from_numpy(mask_f).float(),
            "context_3d": torch.from_numpy(np.array(context_3d)).float(),
            "center_3d": torch.from_numpy(np.array(center_3d)).float(),
            "centroid": torch.from_numpy(np.array(centroid)).float(),
            "scale": torch.tensor(float(scale)),
            "K": torch.from_numpy(K).float(),
            "rot_matrix": torch.from_numpy(R).float(),
            "trans": torch.from_numpy(t).float(),
        }

    def __iter__(self):
        remote_shards = self._list_remote_shards(self.epoch)
        if not remote_shards:
            log.warning(
                "[%s] epoch=%d: nessuno shard .tar trovato (prefix %s). Iteratore vuoto.",
                self.split, self.epoch, self._epoch_prefix(self.epoch),
            )
            return iter(())

        urls = [self._shard_url(p) for p in remote_shards]

        # shardshuffle=True mescola l'ordine degli shard (non il contenuto)
        # ad ogni epoca; il seed lega il mescolamento all'epoca per riproducibilita'.
        def _skip_and_log(exn):
            log.warning(
                "[%s] epoch=%d: shard/sample saltato per errore: %s",
                self.split, self.epoch, exn,
            )
            return True  # True = continua con il prossimo shard/sample invece di rilanciare

        pipeline = wds.WebDataset(
            urls,
            shardshuffle=len(urls),  # int = buffer di shuffle sugli shard (non True/False, deprecato)
            nodesplitter=wds.split_by_node,   # multi-GPU/nodo, se applicabile
            workersplitter=wds.split_by_worker,  # split automatico tra worker DataLoader
            handler=_skip_and_log,
        )

        # if self.shuffle_buffer_size:
        #     pipeline = pipeline.shuffle(self.shuffle_buffer_size)

        pipeline = pipeline.map(self._decode_sample, handler=_skip_and_log)

        return iter(pipeline)


def make_streaming_dataloader(
    dataset: RenderDatasetStreaming,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    """
    IterableDataset non supporta shuffle=True nel DataLoader (lo shuffle
    e' gia' fatto a livello di stream/shard, vedi __iter__).
    persistent_workers=False perche' il dataset va ri-ancorato a una nuova
    epoca (set_epoch + nuovo DataLoader) ad ogni epoca.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=pin_memory,
    )