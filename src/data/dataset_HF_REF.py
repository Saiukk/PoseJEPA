from __future__ import annotations

import hashlib

import http.client

import io

import json

import logging

import os

import random

import re

import tarfile

import time

from collections import deque

from urllib.parse import urlparse

import numpy as np

import requests

import urllib3

import torch

import torch.distributed as dist

import torch.multiprocessing as mp

import torchvision.transforms.functional as TF

from braceexpand import braceexpand

from PIL import Image

from scipy.spatial.transform import Rotation

from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

_AGG_CACHE: dict[str, dict] = {}

_HF_BASE = "https://huggingface.co/datasets/bop-benchmark/megapose/resolve/main/MegaPose-GSO"

DEFAULT_SHARDS = _HF_BASE + "/shard-{000000..001039}.tar"

DEFAULT_MODELS = _HF_BASE + "/gso_models.json"

def get_aggregator_context(model_name: str, dataset_source: str,

cache_context: bool = True, dirname: str = "3DEmbedd") -> dict | None:

    name = model_name.removeprefix("gso_")

    key = os.path.join(dataset_source, name, dirname)

    if cache_context and key in _AGG_CACHE:

        return _AGG_CACHE[key]

    base = key

    try:

        def mm(filename):

            return torch.from_numpy(np.load(os.path.join(base, filename), mmap_mode="c"))

        ctx = {

            "point_xyz": mm("points.npy"),

            "view_feats": mm("view_feats.npy"),

            "view_dirs": mm("view_dirs.npy"),

            "view_mask": mm("mask.npy"),

            "centroid": torch.from_numpy(np.load(os.path.join(base, "centroid.npy"))).float(),

            "scale": torch.tensor(float(np.load(os.path.join(base, "scale.npy")))),

        }

    except (FileNotFoundError, NotADirectoryError):

        log.warning("Context Aggregator mancante: %s", base)

        return None

    if cache_context:

        _AGG_CACHE[key] = ctx

    return ctx

def _train_val_split(name: str, val_ratio: float, seed: int) -> str:

    if val_ratio <= 0:

        return "train"

    hash8 = hashlib.md5(f"{seed}_{name}".encode()).hexdigest()[:8]

    return "val" if int(hash8, 16) / 0xFFFFFFFF < val_ratio else "train"

def _parse_models(data) -> dict[int, str]:

    """Accetta dizionari id->nome, liste di stringhe, liste di record BOP.

    L'indicizzazione di liste senza obj_id segue l'ordine BOP (id 1-based).

    """

    if isinstance(data, dict):

        if "models" in data and isinstance(data["models"], (list, dict)):

            return _parse_models(data["models"])

        if "objects" in data and isinstance(data["objects"], (list, dict)):

            return _parse_models(data["objects"])

        if data and all(str(k).isdigit() for k in data):

            pairs = ((int(k), v) for k, v in data.items())

        else:

            raise ValueError(f"Schema gso_models.json sconosciuto: chiavi={list(data)[:8]}")

    elif isinstance(data, list):

        pairs = ((i, v) for i, v in enumerate(data, 1))

    else:

        raise ValueError("gso_models.json deve essere lista o dizionario")

    result = {}

    for ident, item in pairs:

        if isinstance(item, str):

            name = item

        elif isinstance(item, dict):

            ident = int(item.get("obj_id", item.get("id", ident)))

            name = next((item[k] for k in ("gso_id", "gso_name", "model_name", "name", "label", "path")

                         if isinstance(item.get(k), str) and item[k]), None)

            if name is None:

                raise ValueError(f"Impossibile leggere il nome GSO dal record: {str(item)[:220]}")

        else:

            raise ValueError(f"Record gso_models.json non riconosciuto: {str(item)[:140]}")

        name = os.path.basename(str(name).rstrip("/"))

        if name in ("model.ply", "models") or not name:

            raise ValueError(f"Nome oggetto ambiguo nel mapping: {item}")

        result[int(ident)] = name

    return result

def _rle_decode(rle: dict) -> np.ndarray:

    """RLE COCO non compresso (counts lista); ordine Fortran, [H,W]."""

    h, w = map(int, rle["size"])

    counts = rle["counts"]

    if not isinstance(counts, list):

        raise ValueError("RLE compresso: questo file supporta counts come lista")

    flat = np.zeros(h * w, dtype=np.uint8)

    pos = 0

    for j, run in enumerate(counts):

        nxt = pos + int(run)

        if nxt > flat.size:

            raise ValueError("RLE eccede la dimensione dell'immagine")

        if j % 2:

            flat[pos:nxt] = 1

        pos = nxt

    if pos != flat.size:

        raise ValueError(f"RLE incompleto: {pos} != {flat.size}")

    return flat.reshape((h, w), order="F").astype(bool)

def _iter_tar_samples(stream):

    """Raggruppa file consecutivi per prefisso `scene_frame` senza estrarre il TAR."""

    current_key = None

    sample = {}

    with tarfile.open(fileobj=stream, mode="r|*") as tar:

        for entry in tar:

            if not entry.isfile():

                continue

            filename = os.path.basename(entry.name)

            if "." not in filename:

                continue

            key, suffix = filename.split(".", 1)

            if current_key is not None and key != current_key:

                sample["__key__"] = current_key

                yield sample

                sample = {}

            current_key = key

            # Streaming: depth necessaria per replicare t0 dell'inference.

            if suffix in {"rgb.jpg", "depth.png", "gt.json", "gt_info.json", "camera.json", "mask_visib.json"}:

                handle = tar.extractfile(entry)

                if handle is not None:

                    sample[suffix] = handle.read()

    if current_key is not None:

        sample["__key__"] = current_key

        yield sample

class _RangeNotSupported(IOError):

    """Il server ha ignorato una richiesta Range: serve riaprire il TAR da zero."""

class _ResumableHTTPStream:

    """File-like HTTP sequenziale che recupera interruzioni con Range: bytes=N-.

    Non conserva lo shard su disco. Il byte offset conta solo i byte

    effettivamente consegnati al lettore TAR; se urllib3 perde una lettura

    parziale, il server rimanda esattamente i byte non consegnati.

    """

    _NETWORK_ERRORS = (

        requests.RequestException,

        urllib3.exceptions.HTTPError,

        http.client.IncompleteRead,

        ConnectionError,

        TimeoutError,

        OSError,

        EOFError,

    )

    def __init__(self, url: str, *, max_reconnects: int = 12,

        backoff_s: float = 1.0, connect_timeout: float = 20.0,

        read_timeout: float = 20.0):

        self.url = url

        self.max_reconnects = max(0, int(max_reconnects))

        self.backoff_s = max(0.0, float(backoff_s))

        self.timeout = (connect_timeout, read_timeout)

        self.offset = 0

        self.total = None

        self.reconnects = 0

        self.response = None

        self.session = requests.Session()

        self._open()

    def _open(self):

        headers = {"Accept-Encoding": "identity"}

        if self.offset:

            headers["Range"] = f"bytes={self.offset}-"

        response = self.session.get(

                        self.url, headers=headers, stream=True,

                        timeout=self.timeout, allow_redirects=True,

        )

        try:

            response.raise_for_status()

            encoding = response.headers.get("Content-Encoding", "identity").lower()

            if encoding not in ("identity", ""):

                raise IOError(f"Content-Encoding non supportato per resume: {encoding}")

            if self.offset:

                if response.status_code != 206:

                    raise _RangeNotSupported(

                        f"Il server ha risposto HTTP {response.status_code} anziche' 206 "

                        f"a Range: bytes={self.offset}-"

                    )

                cr = response.headers.get("Content-Range", "")

                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr.strip())

                if match is None or int(match.group(1)) != self.offset:

                    raise _RangeNotSupported(

                        f"Content-Range non valido per offset {self.offset}: {cr!r}"

                    )

                if match.group(3) != "*":

                    length = int(match.group(3))

                    if self.total is not None and self.total != length:

                        raise IOError("Lo shard remoto ha cambiato dimensione durante il download")

                    self.total = length

            else:

                if response.status_code not in (200, 206):

                    raise IOError(f"HTTP inatteso: {response.status_code}")

                if response.status_code == 206:

                    cr = response.headers.get("Content-Range", "")

                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr.strip())

                    if match is None or int(match.group(1)) != 0:

                        raise IOError(f"Content-Range iniziale inatteso: {cr!r}")

                    if match.group(3) != "*":

                        self.total = int(match.group(3))

                elif response.headers.get("Content-Length", "").isdigit():

                    self.total = int(response.headers["Content-Length"])

            self.response = response

        except BaseException:

            response.close()

            raise

    def _reconnect(self, error):

        self._close_response()

        while self.reconnects < self.max_reconnects:

            self.reconnects += 1

            delay = min(self.backoff_s * (2 ** min(self.reconnects - 1, 4)), 15.0)

            log.warning(

                "[HF Range] stream interrotto: offset=%d/%s, reconnect %d/%d, "

                "attesa %.1fs; causa: %s",

                        self.offset, self.total, self.reconnects,

                        self.max_reconnects, delay, error,

            )

            if delay:

                time.sleep(delay)

            try:

                self._open()

                return

            except _RangeNotSupported:

                raise

            except self._NETWORK_ERRORS as exc:

                error = exc

        raise IOError(

                        f"Impossibile recuperare stream HTTP dopo {self.max_reconnects} "

                        f"riconnessioni, offset={self.offset}: {error}"

        ) from error

    def read(self, size=-1):

        if size is None or size < 0:

            # tarfile r|* richiede letture finite; implementiamo comunque read(-1).

            parts = []

            while True:

                block = self.read(1024 * 1024)

                if not block:

                    return b"".join(parts)

                parts.append(block)

        if size == 0:

            return b""

        while True:

            if self.total is not None and self.offset >= self.total:

                return b""

            try:

                data = self.response.raw.read(size)

                if data:

                    self.offset += len(data)

                    return data

                if self.total is None or self.offset == self.total:

                    return b""

                self._reconnect(EOFError("EOF prima del Content-Length dichiarato"))

            except _RangeNotSupported:

                raise

            except self._NETWORK_ERRORS as exc:

                self._reconnect(exc)

    def _close_response(self):

        if self.response is not None:

            self.response.close()

            self.response = None

    def close(self):

        self._close_response()

        self.session.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        self.close()

class GSOMegaposeDataset(IterableDataset):

    """Dataset alternativo Hugging Face, con interfaccia del dataset INRIA."""

    def __init__(

        self,

        shard_source: str = DEFAULT_SHARDS,

        dataset_source: str = "",

        crop_ratio: float = 1.2,

        out_size: int = 256,

        patch_size: int = 16,

        mean: tuple = (0.485, 0.456, 0.406),

        std: tuple = (0.229, 0.224, 0.225),

        cache_context: bool = True,

        val_ratio: float = 0.1,

        split: str = "train",

        split_seed: int = 42,

        shuffle_buffer: int = 2000,

        visible_frac: float = 0.8,

        visible_frac_start: float | None = None,

        visible_frac_end: float | None = None,

        visible_frac_decay_epochs: int = 0,

        val_visible_frac: float | None = None,

        local_cache_dir: str = "/tmp/gso_shard_cache",  # solo compatibilita' YAML; non usato

        prefetch_depth: int = 2,  # solo compatibilita' YAML; non usato

        log_transfer_stats: bool = True,

        models_json: str = DEFAULT_MODELS,

        stream_retries: int = 3,

        retry_backoff_s: float = 3.0,

        range_reconnects: int = 12,

        range_backoff_s: float = 1.0,

        min_mask_px: int = 50,
        context_dirname: str = "3DEmbedd",

        **kwargs,

    ):

        super().__init__()

        if split not in {"train", "val"}:

            raise ValueError(f"split non valido: {split}")

        if out_size % patch_size:

            raise ValueError("out_size deve essere divisibile per patch_size")

        if not dataset_source:

            raise ValueError("dataset_source richiesto: directory locale dei context 3DEmbedd")

        self.shard_source = shard_source

        self.dataset_source = dataset_source

        self.crop_ratio = crop_ratio

        self.out_size = out_size

        self.patch_size = patch_size

        self.mean = tuple(map(float, mean))

        self.std = tuple(map(float, std))

        self.cache_context = cache_context

        self.val_ratio = val_ratio

        self.split = split

        self.split_seed = split_seed

        self.shuffle_buffer = max(0, int(shuffle_buffer))

        self.log_transfer_stats = log_transfer_stats

        self.stream_retries = max(0, int(stream_retries))

        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

        self.range_reconnects = max(0, int(range_reconnects))

        self.range_backoff_s = max(0.0, float(range_backoff_s))
        self.min_mask_px = int(min_mask_px)
        self.context_dirname = str(context_dirname)

        self.grid = out_size // patch_size

        self.num_tokens = self.grid ** 2

        self._epoch_counter = mp.Value("i", 0)

        self._curriculum_enabled = visible_frac_start is not None and visible_frac_end is not None

        self._visible_frac_static = float(visible_frac)

        self.visible_frac_start = visible_frac_start

        self.visible_frac_end = visible_frac_end

        self.visible_frac_decay_epochs = max(0, int(visible_frac_decay_epochs))

        self._val_visible_frac = (val_visible_frac if val_visible_frac is not None

                                  else visible_frac_end if self._curriculum_enabled else visible_frac)

        if kwargs:

            log.warning("Parametri non utilizzati: %s", sorted(kwargs))

        self.shard_paths = list(braceexpand(shard_source)) if shard_source.startswith(("http://", "https://")) else __import__("glob").glob(shard_source)

        if not self.shard_paths:

            raise ValueError(f"Nessuno shard per {shard_source}")

        if models_json.startswith(("http://", "https://")):

            response = requests.get(models_json, timeout=90)

            response.raise_for_status()

            mapping_data = response.json()

        else:

            with open(models_json, encoding="utf8") as f:

                mapping_data = json.load(f)

        self.objid_to_name = _parse_models(mapping_data)

        if not self.objid_to_name:

            raise ValueError("Mapping GSO vuoto")

        log.info("[HF %s] %d shards, %d modelli; TAR letto in streaming senza cache su disco",

    self.split, len(self.shard_paths), len(self.objid_to_name))

    @property

    def epoch(self) -> int:

        return self._epoch_counter.value

    def set_epoch(self, epoch: int) -> None:

        self._epoch_counter.value = int(epoch)

    @property

    def visible_frac(self) -> float:

        if self.split != "train":

            return float(self._val_visible_frac)

        if not self._curriculum_enabled:

            return self._visible_frac_static

        if self.visible_frac_decay_epochs <= 0:

            return float(self.visible_frac_end)

        ratio = min(max(self.epoch / self.visible_frac_decay_epochs, 0.0), 1.0)

        return float(self.visible_frac_start + (self.visible_frac_end - self.visible_frac_start) * ratio)

    def _split_shards(self, shards: list[str]) -> list[str]:

        if dist.is_available() and dist.is_initialized():

            shards = shards[dist.get_rank()::dist.get_world_size()]

        info = torch.utils.data.get_worker_info()

        if info is not None:

            shards = shards[info.id::info.num_workers]

        return shards

    def _crop_box(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:

        ys, xs = np.where(mask)

        if xs.size == 0:

            return None

        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        half = max(x1 - x0 + 1, y1 - y0 + 1) * self.crop_ratio / 2

        xmin, ymin = int(round(cx - half)), int(round(cy - half))

        xmax, ymax = int(round(cx + half)), int(round(cy + half))

        return xmin, ymin, max(xmax, xmin + 1), max(ymax, ymin + 1)

    def _expand_objects(self, samples):

        threshold = self.visible_frac

        for sample in samples:

            try:

                img = Image.open(io.BytesIO(sample["rgb.jpg"])).convert("RGB")

                gt = json.loads(sample["gt.json"])

                gt_info = json.loads(sample["gt_info.json"])

                camera = json.loads(sample["camera.json"])

                masks = json.loads(sample["mask_visib.json"])
                # BOP depth is stored as 16-bit PNG. depth_scale converts its
                # values into the SAME millimetre units as cam_t_m2c.
                depth_raw = np.asarray(Image.open(io.BytesIO(sample["depth.png"])), dtype=np.float32)
                depth_mm = depth_raw * float(camera.get("depth_scale", 1.0))
                K_orig = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)

            except (KeyError, ValueError, OSError, TypeError) as e:

                log.warning("Campione incompleto %s: %s", sample.get("__key__"), e)

                continue

            for i, obj in enumerate(gt):

                try:

                    model_name = self.objid_to_name[int(obj["obj_id"])]

                    if _train_val_split(model_name, self.val_ratio, self.split_seed) != self.split:

                        continue

                    visibility = float(gt_info[i]["visib_fract"])

                    if visibility < threshold:

                        continue

                    rle = masks[str(i)]

                    mask_bool = _rle_decode(rle)
                    # Exactly the inference guess_translation: bbox centre and
                    # median valid depth BEFORE the crop, with original K.
                    ys, xs = np.where(mask_bool)
                    valid_depth = mask_bool & np.isfinite(depth_mm) & (depth_mm > 0)
                    if xs.size == 0 or np.count_nonzero(valid_depth) < self.min_mask_px:
                        continue
                    u_center = (float(xs.min()) + float(xs.max())) * 0.5
                    v_center = (float(ys.min()) + float(ys.max())) * 0.5
                    z_center = float(np.median(depth_mm[valid_depth]))
                    t0 = (np.linalg.inv(K_orig) @ np.array(
                        [u_center, v_center, 1.0], dtype=np.float32)) * z_center
                    t0 = t0.astype(np.float32)

                    box = self._crop_box(mask_bool)

                    if box is None:

                        continue

                    ctx = get_aggregator_context(model_name, self.dataset_source, self.cache_context,
                                                 dirname=self.context_dirname)

                    if ctx is None:

                        continue

                    image_crop = img.crop(box).resize((self.out_size, self.out_size), Image.Resampling.BICUBIC)

                    mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255).crop(box)

                    mask_crop = mask_pil.resize((self.out_size, self.out_size), Image.Resampling.NEAREST)

                    mask_grid = mask_pil.resize((self.grid, self.grid), Image.Resampling.BOX)

                    R_mat = np.asarray(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)

                    trans = np.asarray(obj["cam_t_m2c"], dtype=np.float32)  # gia' mm!

                    quat = Rotation.from_matrix(R_mat).as_quat().astype(np.float32)

                    xmin, ymin, xmax, ymax = box

                    sx, sy = self.out_size / (xmax - xmin), self.out_size / (ymax - ymin)

                    K_crop = K_orig.copy()

                    K_crop[0, 0] *= sx

                    K_crop[1, 1] *= sy

                    K_crop[0, 2] = (K_orig[0, 2] - xmin) * sx

                    K_crop[1, 2] = (K_orig[1, 2] - ymin) * sy

                    output = {

                        "image": TF.normalize(TF.to_tensor(image_crop), self.mean, self.std),

                        "mask": TF.to_tensor(mask_crop),

                        "mask_tokens": TF.to_tensor(mask_grid).reshape(-1),

                        "quat": quat,

                        "rot_matrix": R_mat,

                        "trans": trans,
                        "t0": torch.from_numpy(t0.copy()),
                        "valid": torch.tensor(True),
                        "K": K_crop,

                        "model_name": model_name,

                        "visib_fract": visibility,

                        "scene_key": sample["__key__"],

                    }

                    output.update(ctx)

                    yield output

                except (KeyError, IndexError, TypeError, ValueError, OSError) as e:

                    log.warning("Oggetto ignorato (%s, instance=%d): %s", sample.get("__key__"), i, e)

    def __iter__(self):

        shards = list(self.shard_paths)

        if self.split == "train":

            random.Random(self.epoch).shuffle(shards)

        shards = self._split_shards(shards)

        if not shards:

            return

        info = torch.utils.data.get_worker_info()

        worker = info.id if info else 0

        rng = random.Random(self.split_seed + self.epoch * 100003 + worker)

        def examples():

            # Recupero rapido via HTTP Range dentro _ResumableHTTPStream.

            # Fallback: se Range non e' supportato, riapri lo shard dall'inizio

            # senza riemettere i campioni gia' consegnati al DataLoader.

            network_errors = (

                requests.RequestException,

                urllib3.exceptions.HTTPError,

                http.client.IncompleteRead,

                ConnectionError,

                TimeoutError,

                OSError,

                EOFError,

                tarfile.TarError,

                _RangeNotSupported,

            )

            for url in shards:

                t0 = time.monotonic()

                completed_keys = set()

                finished = False

                for attempt in range(self.stream_retries + 1):

                    try:

                        def unseen_samples(stream):

                            for sample in _iter_tar_samples(stream):

                                key = sample["__key__"]

                                if key in completed_keys:

                                    continue

                                # Segna il campione prima del yield: _expand_objects

                                # puo' emettere piu' oggetti per la stessa immagine.

                                completed_keys.add(key)

                                yield sample

                        if url.startswith(("http://", "https://")):

                            with _ResumableHTTPStream(

                                url,

                                                max_reconnects=self.range_reconnects,

                                                backoff_s=self.range_backoff_s,

                            ) as stream:

                                yield from self._expand_objects(unseen_samples(stream))

                        else:

                            with open(url, "rb") as f:

                                yield from self._expand_objects(unseen_samples(f))

                        finished = True

                        break

                    except network_errors as exc:

                        if attempt < self.stream_retries:

                            delay = self.retry_backoff_s * (2 ** attempt)

                            log.warning(

                                "[HF %s worker=%s] Stream interrotto su %s "

                                "(tentativo %d/%d, %d campioni gia' elaborati): %s. "

                                "Riprovo tra %.1fs",

                                                self.split, worker, url, attempt + 1,

                                                self.stream_retries + 1, len(completed_keys), exc, delay,

                            )

                            time.sleep(delay)

                        else:

                            log.error(

                                "[HF %s worker=%s] Shard %s abbandonato dopo %d "

                                "tentativi; passo al successivo. Errore: %s",

                                                self.split, worker, url, self.stream_retries + 1, exc,

                            )

                if finished and self.log_transfer_stats:

                    log.info(

                        "[HF %s worker=%s] %s letto in %.1fs (%d campioni)",

                                        self.split, worker, url, time.monotonic() - t0,

                                        len(completed_keys),

                    )

        if self.split != "train" or self.shuffle_buffer <= 1:

            yield from examples()

            return

        buffer = []

        for example in examples():

            if len(buffer) < self.shuffle_buffer:

                buffer.append(example)

                continue

            j = rng.randrange(len(buffer))

            yield buffer[j]

            buffer[j] = example

        rng.shuffle(buffer)

        yield from buffer
