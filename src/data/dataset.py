from __future__ import annotations

import glob
import hashlib
import io
import json
import logging
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms.functional as TF
import webdataset as wds
from braceexpand import braceexpand
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

# Cache per processo (ogni worker/rank ha la sua). Evita di ricaricare da
# disco lo stesso context aggregator per ogni oggetto/ogni epoca: la npy
# viene mmap-pata una volta e riusata finche' il worker vive.
_AGG_CACHE: dict[str, dict] = {}


def _pose_to_mat(pose):
    """[[qx,qy,qz,qw], [tx,ty,tz]] -> 4x4 omogenea (megapose serializza gia' in xyzw)"""
    quat, trans = pose
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R.from_quat(quat).as_matrix()
    m[:3, 3] = trans
    return m


def _inv_rigid(T):
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def get_aggregator_context(
    model_name: str,
    dataset_source: str,
    cache_context: bool = True,
    dirname: str = "3DEmbedd",
) -> dict | None:
    """
    Ritorna i tensori del context aggregator per un modello, SENZA cast
    (il cast a float/bool si fa a valle, sulla GPU). Ritorna None se il
    context non esiste su disco: l'oggetto va skippato dal chiamante.
    """
    model_name = model_name.replace("gso_", "")

    if cache_context:
        ctx = _AGG_CACHE.get(model_name)
        if ctx is not None:
            return ctx

    base = os.path.join(dataset_source, model_name, dirname)

    def _mm(name: str):
        path = os.path.join(base, name)
        return torch.from_numpy(np.load(path, mmap_mode="c"))

    try:
        ctx = {
            "point_xyz": _mm("points.npy"),
            "view_feats": _mm("view_feats.npy"),
            "view_dirs": _mm("view_dirs.npy"),
            "view_mask": _mm("mask.npy"),
            "centroid": torch.from_numpy(np.load(os.path.join(base, "centroid.npy"))).float(),
            "scale": torch.tensor(float(np.load(os.path.join(base, "scale.npy")))),
        }
    except FileNotFoundError:
        log.warning("Aggregator context non trovato per %s in %s", model_name, base)
        return None

    if cache_context:
        _AGG_CACHE[model_name] = ctx
    return ctx


def _train_val_split(model_name: str, val_ratio: float, split_seed: int) -> str:
    """
    Split deterministico PER OGGETTO (non per shard): stesso model_name
    finisce sempre nello stesso split, in ogni shard e in ogni epoca.
    """
    if val_ratio <= 0:
        return "train"
    h = hashlib.md5(f"{split_seed}_{model_name}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if frac < val_ratio else "train"


class GSOMegaposeDataset(IterableDataset):
    """
    Streaming WebDataset per MegaPose-GSO
    """

    def __init__(
        self,
        shard_source: str,
        dataset_source: str,
        crop_ratio: float = 1.2,
        out_size: int = 224,
        patch_size: int = 14,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        cache_context: bool = True,
        val_ratio: float = 0.1,
        split: str = "train",
        split_seed: int = 42,
        shuffle_buffer: int = 2000,
        # --- curriculum learning su visible_frac ---
        visible_frac: float = 0.8,
        visible_frac_start: float | None = None,
        visible_frac_end: float | None = None,
        visible_frac_decay_epochs: int = 0,
        val_visible_frac: float | None = None,
        # --------------------------------------------
        local_cache_dir: str = "/tmp/gso_shard_cache",
        prefetch_depth: int = 2,  # abbassa a 1-2 se hai OOM/tmpfs pieno
        log_transfer_stats: bool = True,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split deve essere 'train' o 'val', ricevuto: {split}")

        self.shard_source = shard_source
        self.dataset_source = dataset_source
        self.crop_ratio = crop_ratio
        self.out_size = out_size
        self.patch_size = patch_size
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.cache_context = cache_context
        self.val_ratio = val_ratio
        self.split = split
        self.split_seed = split_seed
        self.shuffle_buffer = shuffle_buffer

        # --- curriculum learning su visible_frac ---
        # Se visible_frac_start/end non sono forniti, il comportamento e'
        # quello legacy: soglia fissa = visible_frac, sia per train che val.
        self._curriculum_enabled = (
            visible_frac_start is not None and visible_frac_end is not None
        )
        self._visible_frac_static = visible_frac
        self.visible_frac_start = visible_frac_start
        self.visible_frac_end = visible_frac_end
        self.visible_frac_decay_epochs = max(0, int(visible_frac_decay_epochs))
        # Soglia usata sul val: di default la piu' severa (visible_frac_end
        # se il curriculum e' attivo, altrimenti visible_frac statico).
        if val_visible_frac is not None:
            self._val_visible_frac = val_visible_frac
        elif self._curriculum_enabled:
            self._val_visible_frac = visible_frac_end
        else:
            self._val_visible_frac = visible_frac

        if self._curriculum_enabled and self.visible_frac_decay_epochs == 0:
            log.warning(
                "visible_frac_start/end impostati ma visible_frac_decay_epochs=0: "
                "la soglia sara' fissa a visible_frac_end fin dall'epoca 0."
            )
        # --------------------------------------------

        self.local_cache_dir = local_cache_dir
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.log_transfer_stats = log_transfer_stats

        if out_size % patch_size != 0:
            raise ValueError(f"out_size ({out_size}) non divisibile per patch_size ({patch_size})")
        self.grid = out_size // patch_size
        self.num_tokens = self.grid * self.grid

        # Epoca condivisa tra padre e worker (deve esistere prima del fork).
        self._epoch_counter = mp.Value("i", 0)

        self.shard_paths = self._resolve_shards(shard_source)
        if len(self.shard_paths) == 0:
            raise ValueError(f"Nessuno shard trovato per shard_source={shard_source}")

        log.info(
            "[GSOMegaposeDataset:%s] %d shard totali (split per oggetto, val_ratio=%.2f), "
            "type: %s, prefetch_depth=%d, visible_frac_curriculum=%s",
            split, len(self.shard_paths), val_ratio,
            "remote" if self._is_remote(shard_source) else "local",
            self.prefetch_depth,
            f"{visible_frac_start}->{visible_frac_end} in {self.visible_frac_decay_epochs} epoche"
            if self._curriculum_enabled else "disabilitato (soglia fissa)",
        )

    # ---------- epoca ----------

    @property
    def epoch(self) -> int:
        return self._epoch_counter.value

    def set_epoch(self, epoch: int) -> None:
        self._epoch_counter.value = int(epoch)

    # ---------- curriculum: visible_frac ----------

    @property
    def visible_frac(self) -> float:
        """
        return the visible fraction threshold for the current epoch (curriculum learning)
        """
        if self.split != "train":
            return self._val_visible_frac

        if not self._curriculum_enabled:
            return self._visible_frac_static

        if self.visible_frac_decay_epochs <= 0:
            return self.visible_frac_end

        t = min(max(self.epoch / self.visible_frac_decay_epochs, 0.0), 1.0)
        return self.visible_frac_start + (self.visible_frac_end - self.visible_frac_start) * t

    # ---------- shard: resolve, split node/worker ----------

    @staticmethod
    def _is_remote(source: str) -> bool:
        return source.startswith(("http://", "https://", "s3://", "gs://"))

    @staticmethod
    def _is_http(path: str) -> bool:
        return path.startswith(("http://", "https://"))

    def _resolve_shards(self, source: str):
        if self._is_remote(source):
            return list(braceexpand(source))
        return sorted(glob.glob(source))

    def _worker_id(self) -> int | str:
        info = torch.utils.data.get_worker_info()
        return info.id if info is not None else "main"

    def _worker_cache_dir(self) -> str:
        wid = self._worker_id()
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        path = os.path.join(self.local_cache_dir, f"rank_{rank}", f"worker_{wid}")
        os.makedirs(path, exist_ok=True)
        return path

    def _split_for_node_and_worker(self, shards: list[str]) -> list[str]:
        # 1) split tra rank DDP, se inizializzato
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            shards = shards[rank::world_size]

        # 2) split tra worker del DataLoader
        info = torch.utils.data.get_worker_info()
        if info is not None:
            shards = shards[info.id::info.num_workers]

        return shards

    # ---------- download con prefetch ----------

    def _local_target_path(self, remote_path: str, cache_dir: str) -> str:
        name = os.path.basename(urlparse(remote_path).path) or hashlib.md5(remote_path.encode()).hexdigest()
        return os.path.join(cache_dir, name)

    def _download_shard(self, remote_path: str, cache_dir: str) -> tuple[str, bool]:
        """Ritorna (local_path, downloaded). downloaded=False se il file era gia' locale."""
        if not self._is_http(remote_path):
            # locale, o protocollo (s3/gs) che deleghiamo a webdataset/gopen
            return remote_path, False

        wid = self._worker_id()
        local_path = self._local_target_path(remote_path, cache_dir)
        t0 = time.time()
        try:
            with requests.get(remote_path, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp_path = local_path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, local_path)
        except Exception as e:
            log.warning("[worker %s] download fallito per %s: %r", wid, remote_path, e)
            raise

        dt = time.time() - t0
        if self.log_transfer_stats:
            try:
                size_mb = os.path.getsize(local_path) / 1e6
                rate = size_mb / dt if dt > 0 else float("inf")
                log.info(
                    "[worker %s] scaricato %s: %.1fMB in %.2fs (%.1f MB/s)",
                    wid, remote_path, size_mb, dt, rate,
                )
            except OSError:
                pass
        return local_path, True

    @staticmethod
    def _remove_if_downloaded(local_path: str, downloaded: bool) -> None:
        if not downloaded:
            return
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass

    # ---------- crop utils ----------

    def _get_crop_box(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(x1 - x0, y1 - y0) * self.crop_ratio / 2.0
        return (
            int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)),
        )

    # ---------- error handling ----------

    def _log_and_skip(self, exn: Exception) -> bool:
        log.warning("[%s] errore ignorato nella pipeline WebDataset: %r", self.split, exn)
        return True

    # ---------- core: scena -> oggetti ----------

    def _expand_objects(self, src):
        # Letta UNA volta per shard/generatore: coerente per tutta la
        # scansione dello shard corrente anche se e' una property dinamica.
        visible_frac_threshold = self.visible_frac

        for sample in src:
            try:
                img = Image.open(io.BytesIO(sample["rgb.png"])).convert("RGB")
                seg = Image.open(io.BytesIO(sample["segmentation.png"]))
                object_datas = json.loads(sample["object_datas.json"])
                cam = json.loads(sample["camera_data.json"])
                scene_key = sample["__key__"]
            except Exception as e:
                self._log_and_skip(e)
                continue

            seg_np = np.array(seg)
            TCW = _inv_rigid(_pose_to_mat(cam["TWC"]))
            K_orig = np.array(cam["K"], dtype=np.float32)

            for obj in object_datas:
                model_name = obj["label"]

                if _train_val_split(model_name, self.val_ratio, self.split_seed) != self.split:
                    continue
                if visible_frac_threshold > float(obj["visib_fract"]):
                    continue

                mask_bool = (seg_np == obj["unique_id"])
                if not mask_bool.any():
                    continue

                box = self._get_crop_box(mask_bool)
                if box is None:
                    continue

                context = get_aggregator_context(model_name, self.dataset_source, self.cache_context)
                if context is None:
                    continue

                try:
                    img_cropped = img.crop(box).resize((self.out_size, self.out_size), Image.BICUBIC)
                    mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255).crop(box)
                    mask_cropped = mask_pil.resize((self.out_size, self.out_size), Image.NEAREST)
                    mask_grid = mask_pil.resize((self.grid, self.grid), Image.BOX)

                    img_t = TF.normalize(TF.to_tensor(img_cropped), self.mean, self.std)
                    mask_t = TF.to_tensor(mask_cropped)
                    mask_tokens = TF.to_tensor(mask_grid).reshape(-1)

                    TCO = TCW @ _pose_to_mat(obj["TWO"])
                    rot_matrix = TCO[:3, :3].astype(np.float32)
                    trans = TCO[:3, 3].astype(np.float32) * 1000.0          # convertito in mm per coerenza con il context aggregator
                    quat = R.from_matrix(rot_matrix).as_quat().astype(np.float32)

                    xmin, ymin, xmax, ymax = box
                    crop_w, crop_h = xmax - xmin, ymax - ymin
                    scale_x = self.out_size / float(crop_w)
                    scale_y = self.out_size / float(crop_h)

                    K_crop = K_orig.copy()
                    K_crop[0, 0] = K_orig[0, 0] * scale_x
                    K_crop[1, 1] = K_orig[1, 1] * scale_y
                    K_crop[0, 2] = (K_orig[0, 2] - xmin) * scale_x
                    K_crop[1, 2] = (K_orig[1, 2] - ymin) * scale_y
                except Exception as e:
                    self._log_and_skip(e)
                    continue

                out = {
                    "image": img_t,
                    "mask": mask_t,
                    "mask_tokens": mask_tokens,
                    "quat": quat,
                    "rot_matrix": rot_matrix,
                    "trans": trans,
                    "K": K_crop,
                    "model_name": model_name,
                    "visib_fract": float(obj["visib_fract"]),
                    "scene_key": scene_key,
                }
                out.update(context)
                yield out

    # ---------- iterazione con prefetch ----------

    def __iter__(self):
        is_train = self.split == "train"
        shard_paths = list(self.shard_paths)

        if is_train:
            epoch = self.epoch
            random.Random(epoch).shuffle(shard_paths)
            log.info(
                "[%s] epoch=%d, first_shard=%s, visible_frac=%.4f",
                self.split, epoch, shard_paths[0], self.visible_frac,
            )

        shard_paths = self._split_for_node_and_worker(shard_paths)
        if not shard_paths:
            return iter(())

        wid = self._worker_id()
        cache_dir = self._worker_cache_dir()
        depth = min(self.prefetch_depth, len(shard_paths))
        executor = ThreadPoolExecutor(max_workers=depth)

        def _fetch(path):
            return self._download_shard(path, cache_dir)

        def _generator():
            futures = deque(executor.submit(_fetch, p) for p in shard_paths[:depth])
            next_idx = depth

            try:
                for remote_path in shard_paths:
                    t_wait = time.time()
                    try:
                        local_path, downloaded = futures.popleft().result()
                    except Exception:
                        # shard scartato, si passa al successivo
                        if next_idx < len(shard_paths):
                            futures.append(executor.submit(_fetch, shard_paths[next_idx]))
                            next_idx += 1
                        continue

                    if self.log_transfer_stats:
                        wait_s = time.time() - t_wait
                        log.info(
                            "[worker %s] atteso %.2fs per shard pronto (in coda: %d)",
                            wid, wait_s, len(futures),
                        )

                    if next_idx < len(shard_paths):
                        futures.append(executor.submit(_fetch, shard_paths[next_idx]))
                        next_idx += 1

                    pipeline = wds.WebDataset(
                        local_path,
                        handler=self._log_and_skip,
                        empty_check=False,
                        shardshuffle=False,
                        nodesplitter=lambda src: src,   # split gia' fatto a mano sopra
                        workersplitter=lambda src: src,
                    ).compose(self._expand_objects)

                    if is_train and self.shuffle_buffer:
                        pipeline = pipeline.shuffle(self.shuffle_buffer)

                    n_yielded = 0
                    try:
                        for item in pipeline:
                            n_yielded += 1
                            yield item
                    finally:
                        if n_yielded == 0:
                            log.warning(
                                "[%s][worker %s] shard=%s ha prodotto ZERO oggetti "
                                "(split=%s): controlla warning sopra.",
                                self.split, wid, remote_path, self.split,
                            )
                        self._remove_if_downloaded(local_path, downloaded)
            finally:
                for fut in futures:
                    if not fut.cancel():
                        fut.add_done_callback(
                            lambda f: self._remove_if_downloaded(*f.result())
                            if not f.cancelled() and f.exception() is None
                            else None
                        )
                executor.shutdown(wait=False)

        return _generator()