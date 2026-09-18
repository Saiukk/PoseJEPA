import webdataset as wds
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import json
import io
import glob
import logging
import random
import os
from braceexpand import braceexpand
from scipy.spatial.transform import Rotation as R


def _pose_to_mat(pose):
    """[[qx,qy,qz,qw], [tx,ty,tz]] -> 4x4 omogenea (megapose serializza già in xyzw)"""
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


NPY_SUBDIR = "embeddingDINO"  # output di convert_npz_to_npy.py
LEGACY_NPZ_FILENAME = "embeddingDINO.npz"

def compute_context(model_name, dataset_source, cache_context):

    model_name = model_name.replace("gso_", "")
    if not cache_context:
        raise NotImplementedError("3D context computation is not implemented yet.")

    model_dir = os.path.join(dataset_source, model_name)
    npy_dir = os.path.join(model_dir, NPY_SUBDIR)
    marker = os.path.join(npy_dir, ".done")

    if os.path.isfile(marker):
        try:
            # mmap_mode="r": mappa il file, non lo carica subito in RAM.
            # La copia reale avviene solo quando i dati vengono
            # consumati (es. torch.from_numpy(...).float() in
            # __getitem__), che e' esattamente il punto in cui li
            # vogliamo "vivi" per il training.
            context_3d = np.load(os.path.join(npy_dir, "features.npy"))             #, mmap_mode="r"
            center_3d = np.load(os.path.join(npy_dir, "centers.npy"))             #, mmap_mode="r"
            centroid = np.load(os.path.join(npy_dir, "centroid.npy"))             #, mmap_mode="r"
            scale = np.load(os.path.join(npy_dir, "scale.npy"))             #, mmap_mode="r"
            return context_3d, center_3d, centroid, scale
        except FileNotFoundError:
            logging.warning(
                f"Marker .done trovato ma file .npy mancanti per {model_name} in {npy_dir}, "
                f"fallback su .npz."
            )

    # Fallback: vecchio percorso .npz, non ancora convertito.
    context_path = os.path.join(model_dir, LEGACY_NPZ_FILENAME)
    try:
        with np.load(context_path) as context_data:
            context_3d = np.array(context_data["features"])
            center_3d = np.array(context_data["centers"])
            centroid = np.array(context_data["centroid"])
            scale = np.array(context_data["scale"])
    except FileNotFoundError:
        logging.warning(f"Cached context for {model_name} not found at {context_path}.")
        return None

    return context_3d, center_3d, centroid, scale

class GSOMegaposeDataset(IterableDataset):
    def __init__(
        self,
        shard_source: str,
        dataset_source: str = None, 
        crop_ratio: float = 1.2,
        shardshuffle: bool = False,
        out_size: int = 224,
        patch_size: int = 14,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        cache_context: bool = False,
        val_shards: int = 0,
        split: str = "train",
        split_seed: int = 42,
        shuffle_buffer: int = 2000,
        visible_frac: float = 0.3,
        
    ):
        super().__init__()
        self.shard_source = shard_source
        self.crop_ratio = crop_ratio
        self.shardshuffle = shardshuffle
        self.out_size = out_size
        self.patch_size = patch_size
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.cache_context = cache_context
        self.dataset_source = dataset_source
        self.shuffle_buffer = shuffle_buffer
        self.visible_frac = visible_frac
        self._epoch_counter = mp.Value("i", 0)
        if out_size % patch_size != 0:
            raise ValueError(f"out_size ({out_size}) patch size ({patch_size})")
        self.grid = out_size // patch_size
        self.num_tokens = self.grid * self.grid

        if split not in ("train", "val"):
            raise ValueError(f"split should be 'train' or 'val', received: {split}")
        self.split = split
        self.val_shards = val_shards

        all_shards = self._resolve_shards(shard_source)
        if len(all_shards) == 0:
            raise ValueError(f"No shard found for shard_source={shard_source}")

        self.shard_paths = self._split_shards(all_shards, val_shards, split, split_seed)
        if split == "train" and len(self.shard_paths) == 0:
            raise ValueError(f"No shard found for shard_source={shard_source}")

        logging.info(
            f"[GSOMegaposeDataset:{split}] {len(self.shard_paths)}/{len(all_shards)} shards, "
            f"type: ({'remote' if self._is_remote(shard_source) else 'local'})"
        )

    @staticmethod
    def _split_shards(all_shards, val_shards: float, split: str, seed: int):

        if val_shards == 0:
            return list(all_shards) if split == "train" else []

        if not 0 < val_shards < 1:
            raise ValueError(
                f"val_shards should be a fraction in (0, 1), received: {val_shards}"
            )

        n_total = len(all_shards)
        n_val = max(1, round(n_total * val_shards))
        if n_val >= n_total:
            raise ValueError(
                f"val_shards={val_shards} resulting in {n_val}/{n_total} shard: "
                f" No for train split"
            )

        shuffled = list(all_shards)
        random.Random(seed).shuffle(shuffled)

        val_set = sorted(shuffled[:n_val])
        train_set = sorted(shuffled[n_val:])
        return val_set if split == "val" else train_set

    @staticmethod
    def _is_remote(source: str) -> bool:
        return source.startswith(("http://", "https://", "s3://", "gs://"))

    def _resolve_shards(self, source: str):
        if self._is_remote(source):
            return list(braceexpand(source))
        return sorted(glob.glob(source))

    def _get_crop_box(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = max(x1 - x0, y1 - y0) * self.crop_ratio / 2.0
        return (int(round(cx - half)), int(round(cy - half)),
                int(round(cx + half)), int(round(cy + half)))

    def _expand_objects(self, src):
        for sample in src:
            try:
                img = Image.open(io.BytesIO(sample["rgb.png"])).convert("RGB")
                seg = Image.open(io.BytesIO(sample["segmentation.png"]))
                object_datas = json.loads(sample["object_datas.json"])
                cam = json.loads(sample["camera_data.json"])
                scene_key = sample["__key__"]
            except Exception as e:
                logging.warning(f"Error parsing {sample.get('__key__')}: {e}")
                continue

            seg_np = np.array(seg)

            # camera->world inversa, poi camera->oggetto per ciascun oggetto
            TCW = _inv_rigid(_pose_to_mat(cam["TWC"]))
            K_orig = np.array(cam["K"], dtype=np.float32)

            for obj in object_datas:
                # Object too occluded, skip it
                if self.visible_frac > float(obj["visib_fract"]):
                    continue
                
                mask_bool = (seg_np == obj["unique_id"])
                if not mask_bool.any():
                    continue

                box = self._get_crop_box(mask_bool)
                if box is None:
                    continue

                img_cropped = img.crop(box).resize(
                    (self.out_size, self.out_size), Image.BICUBIC)

                mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255).crop(box)
                mask_cropped = mask_pil.resize((self.out_size, self.out_size), Image.NEAREST)
                mask_grid = mask_pil.resize((self.grid, self.grid), Image.BOX)

                img_t = TF.normalize(TF.to_tensor(img_cropped), self.mean, self.std)
                mask_t = TF.to_tensor(mask_cropped)
                mask_tokens = TF.to_tensor(mask_grid).reshape(-1)

                # Find the relative pose of the object with respect to the camera
                TCO = TCW @ _pose_to_mat(obj["TWO"])
                rot_matrix = TCO[:3, :3].astype(np.float32)   
                trans = TCO[:3, 3].astype(np.float32)          
                quat = R.from_matrix(rot_matrix).as_quat().astype(np.float32)  

                # Comoput the right K intrinci (after cropping)
                xmin, ymin, xmax, ymax = box
                crop_w, crop_h = xmax - xmin, ymax - ymin

                # NOTE: I compute the scale after the cropping so is not H'/H but H'/crop_h, same for W
                scale_x = self.out_size / float(crop_w)
                scale_y = self.out_size / float(crop_h)

                K_crop = K_orig.copy()

                # update focal and the center
                K_crop[0, 0] = K_orig[0, 0] * scale_x  
                K_crop[1, 1] = K_orig[1, 1] * scale_y  
                K_crop[0, 2] = (K_orig[0, 2] - xmin) * scale_x 
                K_crop[1, 2] = (K_orig[1, 2] - ymin) * scale_y 


                # Passing the 3d context to the predictor
                context_3d, center_3d, centroid, scale = compute_context(obj["label"], self.dataset_source, self.cache_context)  

                yield {
                    "image": img_t,
                    "mask": mask_t,
                    "mask_tokens": mask_tokens,
                    "quat": quat,
                    "rot_matrix": rot_matrix,
                    "trans": trans,
                    "K": K_crop,
                    "model_name": obj["label"],
                    "context_3d": context_3d,
                    "center_3d": center_3d,
                    "centroid": centroid,
                    "scale": scale,
                    "visib_fract": float(obj["visib_fract"]),
                    "scene_key": scene_key,
                }

    def __iter__(self):
        is_train = self.split == "train"

        shard_paths = list(self.shard_paths)

        # Shuffle the shards to show all the data during training
        if is_train:
            epoch = self._epoch_counter.value       # Takes from the shared memory
            logging.info(f"[{self.split}] worker sees epoch={epoch}, first_shard_before_shuffle={self.shard_paths[0]}")
            random.Random(epoch).shuffle(shard_paths)                                           
            logging.info(f"[{self.split}] epoch={epoch}, first_shard_after_shuffle={shard_paths[0]}")


        pipeline = wds.WebDataset(
            shard_paths,
            shardshuffle=False,  # self.shardshuffle if is_train else False, 
            #resampled=is_train,
        ).compose(self._expand_objects)

        # Suffle the object obtained by expand objects.
        if is_train and self.shuffle_buffer:
            pipeline = pipeline.shuffle(self.shuffle_buffer)

        return iter(pipeline)



