"""
Genera render sintetici di oggetti BOP da pose campionate uniformemente su SO(3),
e salva la ground-truth (rotazione + traslazione) in un CSV.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYOPENGL_PLATFORM"] = "egl"
 

import sys
sys.path.append("/Experiments/riccardop02")  

import csv
import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation

from JEPA.Essential.path import models_dir, RENDERS

# ----------------------------- CONFIG -----------------------------
DATASET_NAME = "ycbv"
OBJECT       = "all"      # "all" = tutti gli obj_*.ply, oppure "obj_000001" / "1"
POSES_NUMBER = 100
IMG_SIZE     = 224        # 14 * 16 -> pronto per DINOv2
YFOV         = np.pi / 4  # campo visivo verticale
FILL         = 2        # quanto "stretto" sta l'oggetto nel frame (>1 = piu' margine)
BG_GRAY      = 0.5        # sfondo grigio neutro
SEED         = 0
MODE         = "both"    # "rot" = solo rotazione | "trans" = solo traslazione | "both" = entrambe
ROT_AXIS     = "y"      # "all", "x", "y", "z" 
TRANS_FRAC = 0.30     # ampiezza max di tx,ty 
DEPTH_FRAC = 0.20    # variazione max di tz 
# ------------------------------------------------------------------


def add_raymond_lights(scene):
    """3 luci direzionali (setup standard pyrender): l'ombreggiatura rivela la
    geometria, fondamentale perche' DINOv2 abbia un segnale di forma da leggere."""
    thetas = np.pi * np.array([1/6, 1/6, 1/6])
    phis   = np.pi * np.array([0.0, 2/3, 4/3])
    for theta, phi in zip(thetas, phis):
        z = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, :3] = np.c_[x, y, z]
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=pose)


def sample_poses(n, axis = "all", seed=0):
    """n rotazioni uniformi su SO(3)"""
    if axis in ("x", "y", "z"):
        # angoli equispaziati su tutto il giro, endpoint escluso -> nessun doppione
        angles = np.linspace(0.0, 360.0, n, endpoint=False).reshape(-1, 1)
        return Rotation.from_euler(axis, angles, degrees=True)
    elif axis == "all":
        return Rotation.random(n, random_state=seed)
    else:
        raise ValueError(f"axis non valido: {axis}")


# Taken the object
def get_object_paths(dataset, obj):
    """Lista dei .ply da renderizzare."""
    mdir = models_dir(dataset)
    if obj == "all":
        paths = sorted(mdir.glob("obj_*.ply"))
        if not paths:
            raise FileNotFoundError(f"Nessun obj_*.ply in {mdir}")
        return paths
   
    if obj.endswith(".ply"):
        name = obj
    else:
        name = f"obj_{int(obj):06d}.ply"
    p = mdir / name
    if not p.exists():
        raise FileNotFoundError(p)
    return [p]


def render_object(mesh_path, rotations, off_xy, off_z, renderer, csv_writer, dataset):
    """Renderizza un oggetto da tutte le pose e appende le GT al CSV."""
    obj_name = mesh_path.stem                      # es. "obj_000001"
    out_dir  = RENDERS / f"{dataset}_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    # carica + centra la mesh sull'origine (rotazione attorno al suo centro)
    tm = trimesh.load(mesh_path, force="mesh")
    tm.apply_translation(-tm.centroid)
    radius = float(np.linalg.norm(tm.vertices, axis=1).max())   # in mm (BOP)

    # distanza camera-oggetto perche' riempia il frame con un po' di margine
    dist = FILL * radius / np.tan(YFOV / 2.0)

    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
    scene = pyrender.Scene(bg_color=np.array([BG_GRAY] * 3),
                           ambient_light=np.array([0.35] * 3))
    scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=np.eye(4))
    add_raymond_lights(scene)
    obj_node = scene.add(mesh, pose=np.eye(4))

    half_extent = dist * np.tan(YFOV / 2.0) 

    for i, R in enumerate(rotations):
        Rm = R.as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rm
        tx = off_xy[i, 0] * half_extent
        ty = off_xy[i, 1] * half_extent
        tz = -dist * (1.0 + off_z[i])
        T[:3,  3] = [tx, ty, tz]      # davanti alla camera
        scene.set_pose(obj_node, T)

        color, _ = renderer.render(scene)  # color: HxWx3 uint8
        rx, ry, rz = R.as_euler("xyz", degrees=True)
        img_name = (
            f"{obj_name}_pose{i:04d}"
            f"_rx{rx:+04.0f}_ry{ry:+04.0f}_rz{rz:+04.0f}"
            f"_tx{tx:+04.0f}_ty{ty:+04.0f}_tz{tz:+04.0f}.png"
        )
        trimesh.util.log.disabled = True
        from PIL import Image
        Image.fromarray(color).save(out_dir / img_name)

        
        r = Rm.flatten()                   # r00 r01 r02 r10 ... r22
        rel_path = (out_dir / img_name).relative_to(RENDERS).as_posix()
        csv_writer.writerow(
            [dataset, obj_name, i, rel_path,
             *r,
             tx, ty, tz,              # tx ty tz (object->camera)
             dist]
        )
    print(f"  {obj_name}: {len(rotations)} render -> {out_dir}")


def main():

    import torch
    print("GPU corrente:", torch.cuda.current_device())

    obj_paths = get_object_paths(DATASET_NAME, OBJECT)
    
    # GENERARE UNA POSA
    rng = np.random.default_rng(SEED)

    # rotazioni: variano in "rot"/"both", restano identità in "trans"
    if MODE in ("rot", "both"):
        rotations = sample_poses(POSES_NUMBER, axis=ROT_AXIS, seed=SEED)
    else:
        rotations = Rotation.identity(POSES_NUMBER)

    # offset traslazione NORMALIZZATI in [-1,1]: variano in "trans"/"both", zero in "rot"
    if MODE in ("trans", "both"):
        off_xy = rng.uniform(-TRANS_FRAC, TRANS_FRAC, size=(POSES_NUMBER, 2))
        off_z  = rng.uniform(-DEPTH_FRAC, DEPTH_FRAC, size=POSES_NUMBER)
    else:
        off_xy = np.zeros((POSES_NUMBER, 2))
        off_z  = np.zeros(POSES_NUMBER)


    csv_path = RENDERS / f"{DATASET_NAME}_all" / "gt_poses.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()

    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            header = (["dataset", "object", "pose_idx", "image"]
                      + [f"r{a}{b}" for a in range(3) for b in range(3)]
                      + ["tx", "ty", "tz", "dist"])
            writer.writerow(header)

        print(f"Render di {len(obj_paths)} oggetti x {POSES_NUMBER} pose ({DATASET_NAME})")
        for mp in obj_paths:
            render_object(mp, rotations, off_xy, off_z, renderer, writer, DATASET_NAME)

    renderer.delete()
    print(f"\nGT salvate in: {csv_path}")


if __name__ == "__main__":

    main()