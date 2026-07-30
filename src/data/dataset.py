import webdataset as wds
from torch.utils.data import IterableDataset
import numpy as np
from PIL import Image
import json
import io
import glob
import os
from braceexpand import braceexpand
from scipy.spatial.transform import Rotation as R


class GSOMegaposeDataset(IterableDataset):
    def __init__(
        self,
        shard_source: str,
        padding_ratio: float = 0.2,
        shardshuffle: bool = False,
    ):

        super().__init__()
        self.shard_source = shard_source
        self.padding_ratio = padding_ratio
        self.shardshuffle = shardshuffle
        self.shard_paths = self._resolve_shards(shard_source)

        if len(self.shard_paths) == 0:
            raise ValueError(f"No shard found for shard_source={shard_source}")

        print(f"[GSOMegaposeDataset] {len(self.shard_paths)} founded, type: "
              f"({'remote' if self._is_remote(shard_source) else 'local'})")

    @staticmethod
    def _is_remote(source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://") or source.startswith("s3://") or source.startswith("gs://")

    def _resolve_shards(self, source: str):
        if self._is_remote(source):
            return list(braceexpand(source))
        
        else:
            # local path
            return sorted(glob.glob(source))

    def _get_padded_bbox(self, mask_bool, img_w, img_h):
        # Get the bounding box of the object in the mask 
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0, y1 - y0
        pad_x, pad_y = w * self.padding_ratio, h * self.padding_ratio
        # Apply padding
        x0 = max(0, int(x0 - pad_x))
        y0 = max(0, int(y0 - pad_y))
        x1 = min(img_w, int(x1 + pad_x))
        y1 = min(img_h, int(y1 + pad_y))
        return x0, y0, x1, y1

    def _expand_objects(self, src):
        for sample in src:
            try:
                img = Image.open(io.BytesIO(sample["rgb.png"])).convert("RGB")
                seg = Image.open(io.BytesIO(sample["segmentation.png"]))
                object_datas = json.loads(sample["object_datas.json"])
                scene_key = sample["__key__"]
            except Exception as e:
                print(f"Error parsing {sample.get('__key__')}: {e}")
                continue

            seg_np = np.array(seg)
            img_w, img_h = img.size

            for obj in object_datas:
                unique_id = obj["unique_id"]
                mask_bool = (seg_np == unique_id)
                if not mask_bool.any():
                    continue

                bbox = self._get_padded_bbox(mask_bool, img_w, img_h)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox

                img_cropped = img.crop((x0, y0, x1, y1))
                mask_cropped_np = mask_bool[y0:y1, x0:x1].astype(np.uint8) * 255
                mask_cropped = Image.fromarray(mask_cropped_np)

                quat = np.array(obj["TWO"][0], dtype=np.float32)
                trans = np.array(obj["TWO"][1], dtype=np.float32)

                rot_matrix = R.from_quat(quat[1],quat[2],quat[3],quat[0]).as_matrix().astype(np.float32)

                yield {
                    "image": img_cropped,
                    "mask": mask_cropped,
                    "quat": quat,
                    "rot_matrix": rot_matrix,
                    "trans": trans,
                    "model_name": obj["label"],
                    "visib_fract": obj["visib_fract"],
                    "scene_key": scene_key,
                }

    def __iter__(self):
        pipeline = wds.WebDataset(self.shard_paths, shardshuffle=self.shardshuffle).compose(self._expand_objects)
        return iter(pipeline)