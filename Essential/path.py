"""
Path module — /Experiments/riccardop02/JEPA/Essential/path.py
Unico posto che conosce la geografia del disco.
Le funzioni prendono il NOME del dataset come parametro: valgono per qualsiasi dataset BOP.
"""
from pathlib import Path

# --- radici (assolute, sul server) ---
DATA       = Path("/Experiments/riccardop02/datasets")        # dataset BOP (sola lettura)
OUTPUTS    = Path("/Experiments/riccardop02/JEPA/Renders")    # output rigenerabili
RENDERS    = OUTPUTS / "renders"
DETECTIONS = Path("/Experiments/riccardop02/detection")       # detection (cnos, ...)

RENDERS.mkdir(parents=True, exist_ok=True)

# Aggiungi in /Experiments/riccardop02/JEPA/Essential/path.py


DETECTION_REGISTRY = {
    "cnos": {
        "ycbv_test": "cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json",
        "hb_test_primesense": "cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json",
        "tless_test_primesense": "cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json",
    }
}


def get_registered_detection(method: str, name: str, split: str) -> Path:
    """Obtain file of a detection"""
    
    if method not in DETECTION_REGISTRY:
        available_methods = list(DETECTION_REGISTRY.keys())
        raise KeyError(
            f"'{method}' not found. "
            f"Available: {available_methods}"
        )
        
    dataset_key = f"{name}_{split}"
    if dataset_key not in DETECTION_REGISTRY[method]:
        available_datasets = list(DETECTION_REGISTRY[method].keys())
        raise KeyError(
            f"Not found '{method}'. "
            f"Available: {available_datasets}"
        )
        
    filename = DETECTION_REGISTRY[method][dataset_key]
    return detection_file(filename, method=method)

# ---------- mesh / modelli ----------
def models_dir(name: str) -> Path:
    """Mesh di un dataset: .../<name>/models/obj_000001.ply"""
    return DATA / name / "models"

def model_file(name: str, obj_id: int) -> Path:
    """Una mesh specifica: .../<name>/models/obj_000004.ply"""
    return models_dir(name) / f"obj_{obj_id:06d}.ply"

def models_info(name: str) -> Path:
    """models_info.json (diametri, simmetrie)."""
    return models_dir(name) / "models_info.json"



def split_dir(name: str, split: str = "test") -> Path:
    """Cartella di uno split: .../<name>/test   (HB usa split='test_primesense')"""
    return DATA / name / split

def scene_dir(name: str, scene_id: int, split: str = "test") -> Path:
    """Cartella di UNA scena: .../<name>/test/000003/"""
    return split_dir(name, split) / f"{scene_id:06d}"

def rgb_file(name: str, scene_id: int, image_id: int, split: str = "test") -> Path:
    """Una immagine RGB: .../000003/rgb/000130.png"""
    return scene_dir(name, scene_id, split) / "rgb" / f"{image_id:06d}.png"

def scene_gt(name: str, scene_id: int, split: str = "test") -> Path:
    """Pose ground-truth della scena: .../000003/scene_gt.json"""
    return scene_dir(name, scene_id, split) / "scene_gt.json"

def scene_camera(name: str, scene_id: int, split: str = "test") -> Path:
    """Intrinseci per immagine: .../000003/scene_camera.json"""
    return scene_dir(name, scene_id, split) / "scene_camera.json"

def mask_visib(name: str, scene_id: int, image_id: int, obj_idx: int, split: str = "test") -> Path:
    """Maschera GT visibile: .../000003/mask_visib/000130_000000.png"""
    return scene_dir(name, scene_id, split) / "mask_visib" / f"{image_id:06d}_{obj_idx:06d}.png"


# ---------- camera di default del dataset ----------
def camera_file(name: str, sensor: str = "") -> Path:
    """Intrinseci di default: .../<name>/base/camera.json  (HB: sensor='_primesense')"""
    return DATA / name / "base" / f"camera{sensor}.json"


# ---------- detection (CNOS, ...) ----------
def detection_file(filename: str, method: str = "cnos") -> Path:
    """Un file di detection: .../detection/cnos/<filename>"""
    return DETECTIONS / method / filename