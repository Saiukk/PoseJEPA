import webdataset as wds
from torch.utils.data import IterableDataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import json
import io
import glob
import logging
import random
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

def compute_context(model_name, dataset_source, cache_context):
    """
    Compute the 3D context for a given model name.
    """

    model_name = model_name.replace("gso_", "") 
    if cache_context:
        # Load the 3D context from a precomputed cache
        context_path = f"{dataset_source}/{model_name}/embedding.npy"
        try:
            context3d = np.load(context_path)
            logging.info(f"Loaded cached context for {model_name} from {context_path}")
        except FileNotFoundError:
            logging.warning(f"Cached context for {model_name} not found at {context_path}.")

            
    # Compute !TODO
    else:
        raise NotImplementedError("3D context computation is not implemented yet.")

    return context3d

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
        cache_context: bool = True,
        val_shards: int = 0,
        split: str = "train",
        split_seed: int = 42,
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
            K = np.array(cam["K"], dtype=np.float32)

            for obj in object_datas:
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


                # Passing the 3d context to the predictor
                context_3d = compute_context(obj["label"], self.dataset_source, self.cache_context)  

                yield {
                    "image": img_t,
                    "mask": mask_t,
                    "mask_tokens": mask_tokens,
                    "quat": quat,
                    "rot_matrix": rot_matrix,
                    "trans": trans,
                    "K": K,
                    "model_name": obj["label"],
                    "context_3d": context_3d,
                    "visib_fract": float(obj["visib_fract"]),
                    "scene_key": scene_key,
                }

    def __iter__(self):
        pipeline = (
            wds.WebDataset(self.shard_paths, shardshuffle=self.shardshuffle)
            .compose(self._expand_objects)
        )
        return iter(pipeline)