from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, "/home/riccardop02/JEPA")

# ENV for HF
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

import numpy as np
import torch
import torchvision
import webdataset as wds
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import IterableDataset

from src.data.Render_Dataset import compute_context, list_model_names, split_model_names

log = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

_AGGREGATOR_CONTEXT_CACHE: dict[str, tuple] = {}


def compute_aggregator_context(
    model_name: str,
    dataset_source: str,
    cache_context: bool = False,
    dirname: str = "3DEmbedd",
):
    if cache_context and model_name in _AGGREGATOR_CONTEXT_CACHE:
        return _AGGREGATOR_CONTEXT_CACHE[model_name]

    base = os.path.join(dataset_source, model_name, dirname)
    points = np.load(os.path.join(base, "points.npy"), mmap_mode="r")
    view_feats = np.load(os.path.join(base, "view_feats.npy"), mmap_mode="r")
    view_dirs = np.load(os.path.join(base, "view_dirs.npy"), mmap_mode="r")
    mask = np.load(os.path.join(base, "mask.npy"), mmap_mode="r")
    centroid = np.load(os.path.join(base, "centroid.npy"))
    scale = float(np.load(os.path.join(base, "scale.npy")))

    result = (points, view_feats, view_dirs, mask, centroid, scale)
    if cache_context:
        _AGGREGATOR_CONTEXT_CACHE[model_name] = result
    return result


class RenderDatasetStreaming(IterableDataset):
    """
    Scarica shard .tar da HF 
    """

    def __init__(
        self,
        repo_id: str,
        dataset_source: str,
        split: str = "train",
        context_mode: str = "uni3d",          # "uni3d" | "aggregator"
        cache_context: bool = True,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        shuffle_buffer_size: int = None,
        token: str | None = None,
        local_cache_dir: str = "/tmp/shard_cache",
        prefetch_depth: int = 6,  # Impostato a 1 per evitare OOM quando si usano molti worker
        log_transfer_stats: bool = True,
    ):
        if context_mode not in ("uni3d", "aggregator"):
            raise ValueError(f"context_mode deve essere 'uni3d' o 'aggregator', ricevuto: {context_mode}")
        self.context_mode = context_mode

        self.repo_id = repo_id
        self.dataset_source = dataset_source
        self.split = split
        self.cache_context = cache_context
        self.shuffle_buffer_size = int(shuffle_buffer_size) if shuffle_buffer_size else None
        self.token = token or os.getenv("HF_TOKENPRO") or os.getenv("HF_TOKEN")
        self.epoch = 0
        self.local_cache_dir = local_cache_dir
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.log_transfer_stats = log_transfer_stats

        # Cache della lista shard per epoca: {epoch: [remote_path, ...]}.
        # Popolata nel processo padre da set_epoch() PRIMA che il DataLoader
        # forki i worker.
        self._shard_cache: dict[int, list[str]] = {}

        if not self.token:
            log.warning(
                "Nessun HF token trovato (HF_TOKENPRO / HF_TOKEN non settate). "
                "Se il repo %s e' privato, il download degli shard fallira'.",
                repo_id,
            )

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
            "[%s] RenderDatasetStreaming(wds, prefetch_depth=%d, context_mode=%s): repo=%s, %d modelli attesi, epoch=%d",
            split, self.prefetch_depth, self.context_mode, repo_id, len(self.model_names), self.epoch,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if epoch not in self._shard_cache:
            self._shard_cache[epoch] = self._fetch_remote_shards(epoch)

    def _epoch_prefix(self, epoch: int) -> str:
        return f"data/{self.split}/epoch_{epoch:03d}/"

    def _fetch_remote_shards(self, epoch: int) -> list[str]:
        prefix = self._epoch_prefix(epoch)
        try:
            all_files = self._api.list_repo_files(self.repo_id, repo_type="dataset")
        except Exception as e:
            log.warning("Impossibile listare i file del repo %s: %s", self.repo_id, e)
            return []
        shards = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
        return shards

    def _list_remote_shards(self, epoch: int) -> list[str]:
        if epoch not in self._shard_cache:
            log.debug(
                "[%s] cache shard mancante per epoch=%d, fetch diretto (fallback)",
                self.split, epoch,
            )
            self._shard_cache[epoch] = self._fetch_remote_shards(epoch)
        return self._shard_cache[epoch]

    def get_n_shards(self, epoch: int | None = None) -> int:
        return len(self._list_remote_shards(epoch if epoch is not None else self.epoch))

    # ---------- cache locale e download ----------

    def _worker_id(self) -> int | str:
        info = torch.utils.data.get_worker_info()
        return info.id if info is not None else "main"

    def _worker_cache_dir(self) -> str:
        wid = self._worker_id()
        path = os.path.join(self.local_cache_dir, f"worker_{wid}")
        os.makedirs(path, exist_ok=True)
        return path

    def _download_shard(self, remote_path: str, cache_dir: str) -> str:
        wid = self._worker_id()
        t0 = time.time()
        print(f"[download-start] worker={wid} {remote_path} @ {t0:.2f}", flush=True)
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=self.token,
                local_dir=cache_dir,
            )
        except Exception as e:
            print(
                f"[download-ERROR] worker={wid} {remote_path}: {type(e).__name__}: {e}",
                flush=True,
            )
            raise
        dt = time.time() - t0
        if self.log_transfer_stats:
            try:
                size_mb = os.path.getsize(local_path) / 1e6
                rate = size_mb / dt if dt > 0 else float("inf")
                print(
                    f"[download] worker={wid} {remote_path}: {size_mb:.1f}MB in {dt:.2f}s ({rate:.1f} MB/s)",
                    flush=True,
                )
            except OSError:
                pass
        else:
            print(f"[download-end] worker={wid} {remote_path} dt={dt:.2f}s", flush=True)
        return local_path

   
    # SKip error
    def _log_and_skip(self, exn: Exception) -> bool:
        wid = self._worker_id()
        log.warning(
            "[%s][worker %s] errore ignorato nella pipeline WebDataset: %r",
            self.split, wid, exn,
        )
        return True

    # change raw file to tensors
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

        img_tensor_uint8 = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image_t = torchvision.io.decode_image(img_tensor_uint8, mode=torchvision.io.image.ImageReadMode.RGB)
        image_t = image_t.float() / 255.0
        image_t = (image_t - self._dino_mean) / self._dino_std

        mask_tensor_uint8 = torch.frombuffer(mask_bytes, dtype=torch.uint8)
        mask_hw = torchvision.io.decode_image(mask_tensor_uint8, mode=torchvision.io.image.ImageReadMode.GRAY)
        mask_f = (mask_hw > 127).float().squeeze(0)

        K = np.asarray(meta["K"], dtype=np.float32).reshape(3, 3)
        R = np.asarray(meta["R"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(meta["t"], dtype=np.float32)

        out = {
            "model_name": model_name,
            "image": image_t,
            "mask": mask_f,
            "K": torch.from_numpy(K).float(),
            "rot_matrix": torch.from_numpy(R).float(),
            "trans": torch.from_numpy(t).float(),
        }

        if self.context_mode == "uni3d":
            context_3d, center_3d, centroid, scale = compute_context(
                model_name, self.dataset_source, self.cache_context
            )
            out["context_3d"] = torch.from_numpy(np.array(context_3d)).float()
            out["center_3d"] = torch.from_numpy(np.array(center_3d)).float()
            out["centroid"] = torch.from_numpy(np.array(centroid)).float()
            out["scale"] = torch.tensor(float(scale))

        else:  # "aggregator"
            points, view_feats, view_dirs, vis_mask, centroid, scale = compute_aggregator_context(
                model_name, self.dataset_source, self.cache_context
            )
            out["point_xyz"] = torch.from_numpy(points).float()
            out["view_feats"] = torch.from_numpy(view_feats).float()
            out["view_dirs"] = torch.from_numpy(view_dirs).float()
            out["view_mask"] = torch.from_numpy(vis_mask).bool()
            out["centroid"] = torch.from_numpy(np.asarray(centroid)).float()
            out["scale"] = torch.tensor(float(scale))

        return out

    # Iteration 
    def __iter__(self):
        # Check shard and assign to workers
        remote_shards = self._list_remote_shards(self.epoch)
        if not remote_shards:
            log.warning(
                "[%s] epoch=%d: nessuno shard .tar trovato (prefix %s). Iteratore vuoto.",
                self.split, self.epoch, self._epoch_prefix(self.epoch),
            )
            return iter(())

        # for each worker, take only the shards assigned 
        info = torch.utils.data.get_worker_info()
        wid = self._worker_id()
        if info is not None:
            total_before = len(remote_shards)
            remote_shards = remote_shards[info.id::info.num_workers]
            log.info(
                "[%s][worker %d/%d] shard totali=%d assegnati=%d -> %s",
                self.split, info.id, info.num_workers, total_before, len(remote_shards), remote_shards,
            )

        if not remote_shards:
            return iter(())

        # USED TO using workers to prefetch shards in parallel
        cache_dir = self._worker_cache_dir()
        depth = min(self.prefetch_depth, len(remote_shards))
        executor = ThreadPoolExecutor(max_workers=depth)

        def _fetch(path):
            return self._download_shard(path, cache_dir)

        # Generator that yields samples from shards, downloading them in parallel
        def _generator():
            # create the queue of futures shards to download
            futures = deque(
                executor.submit(_fetch, p) for p in remote_shards[:depth]
            )
            next_idx = depth

            try:
                for i, remote_path in enumerate(remote_shards):
                    t_wait = time.time()
                    try:
                        local_path = futures.popleft().result()
                    except Exception as e:
                        log.warning(
                            "[%s][worker %s] Download fallito per %s: %r",
                            self.split, wid, remote_path, e,
                        )
                        if next_idx < len(remote_shards):
                            futures.append(executor.submit(_fetch, remote_shards[next_idx]))
                            next_idx += 1
                        continue
                    if self.log_transfer_stats:
                        wait_s = time.time() - t_wait
                        print(
                            f"[wait] worker={wid} atteso {wait_s:.2f}s per shard pronto (in coda: {len(futures)})",
                            flush=True,
                        )

                    # Adding to the queue the next shard to download, if any
                    if next_idx < len(remote_shards):
                        futures.append(executor.submit(_fetch, remote_shards[next_idx]))
                        next_idx += 1

                    # Connection with the WebDataset pipeline
                    pipeline = wds.WebDataset(
                        local_path, 
                        handler=self._log_and_skip, 
                        empty_check=False,
                        shardshuffle=False,
                        nodesplitter=lambda src: src,   # FONDAMENTALE per i multi-worker
                        workersplitter=lambda src: src  # FONDAMENTALE per i multi-worker
                    ).map(self._decode_sample, handler=self._log_and_skip)

                    n_yielded = 0
                    try:
                        for item in pipeline:
                            n_yielded += 1
                            yield item
                    finally:
                        log.info(
                            "[%s][worker %s] shard=%s sample_emessi=%d",
                            self.split, wid, remote_path, n_yielded,
                        )
                        if n_yielded == 0:
                            log.warning(
                                "[%s][worker %s] shard=%s ha prodotto ZERO sample: "
                                "controlla i WARNING 'errore ignorato' sopra per la causa.",
                                self.split, wid, remote_path,
                            )
                        if os.path.exists(local_path):
                            try:
                                os.remove(local_path)
                            except OSError:
                                pass

                        metadata_path = local_path.replace("/data/", "/.cache/huggingface/download/data/") + ".metadata"

                        if os.path.exists(metadata_path):
                            try:
                                os.remove(metadata_path)
                            except OSError:
                                pass
            finally:
                executor.shutdown(wait=False)

        return _generator()