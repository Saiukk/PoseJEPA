import webdataset as wds
from torch.utils.data import IterableDataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import json
import io
import glob
import os
import logging
from braceexpand import braceexpand
from scipy.spatial.transform import Rotation as R


class GSOMegaposeDataset(IterableDataset):
    def __init__(
        self,
        shard_source: str,
        crop_ratio: float = 1.2,
        shardshuffle: bool = False,
        out_size: int = 224,
        patch_size: int = 14,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.shard_source = shard_source
        self.crop_ratio = crop_ratio
        self.shardshuffle = shardshuffle
        self.out_size = out_size
        self.patch_size = patch_size
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        if out_size % patch_size != 0:
            raise ValueError(
                f"out_size ({out_size}) patch size ({patch_size})")
        self.grid = out_size // patch_size          # 224/14 = 16
        self.num_tokens = self.grid * self.grid     # -> 256

        self.shard_paths = self._resolve_shards(shard_source)

        if len(self.shard_paths) == 0:
            raise ValueError(f"No shard found for shard_source={shard_source}")

        logging.info(f"[GSOMegaposeDataset] {len(self.shard_paths)} founded, type: "
              f"({'remote' if self._is_remote(shard_source) else 'local'}) ")


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
        logging.info(f"[GSOMegaposeDataset] open Streaming")
        for sample in src:
            print("CIAO")
            try:
                img = Image.open(io.BytesIO(sample["rgb.png"])).convert("RGB")
                seg = Image.open(io.BytesIO(sample["segmentation.png"]))
                object_datas = json.loads(sample["object_datas.json"])
                scene_key = sample["__key__"]
            except Exception as e:
                print(f"Error parsing {sample.get('__key__')}: {e}")
                continue

            seg_np = np.array(seg)
            
            for obj in object_datas:
                mask_bool = (seg_np == obj["unique_id"])
                if not mask_bool.any():
                    continue

                box = self._get_crop_box(mask_bool)
                if box is None:
                    continue

                # stesso box su entrambi -> crop allineati anche fuori bordo
                img_cropped = img.crop(box).resize(
                    (self.out_size, self.out_size), Image.BICUBIC)

                mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255).crop(box)

                # full-res: NEAREST -> bordi netti, resta binaria
                mask_cropped = mask_pil.resize(
                    (self.out_size, self.out_size), Image.NEAREST)

                # per-token: BOX -> media d'area = frazione di pixel oggetto per patch
                mask_grid = mask_pil.resize((self.grid, self.grid), Image.BOX)

                img_t = TF.normalize(TF.to_tensor(img_cropped), self.mean, self.std)
                mask_t = TF.to_tensor(mask_cropped)
                mask_tokens = TF.to_tensor(mask_grid).reshape(-1)

                quat = np.array(obj["TWO"][0], dtype=np.float32)
                trans = np.array(obj["TWO"][1], dtype=np.float32)
                rot_matrix = R.from_quat(
                    (quat[1], quat[2], quat[3], quat[0])
                ).as_matrix().astype(np.float32)

                logging.debug(f"[GSOMegaposeDataset] yield  - ")
                yield {
                    "image": img_t,            
                    "mask": mask_t,          
                    "mask_tokens": mask_tokens,
                    "quat": quat,
                    "rot_matrix": rot_matrix,
                    "trans": trans,
                    "model_name": obj["label"],
                    "visib_fract": float(obj["visib_fract"]),
                    "scene_key": scene_key,
                }

    def __iter__(self):
        pipeline = (
            wds.WebDataset(self.shard_paths, shardshuffle=self.shardshuffle)
            .compose(self._expand_objects)
        )
        return iter(pipeline)