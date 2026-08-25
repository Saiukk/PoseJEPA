import os
import logging
import numpy as np
import torch
import trimesh
from PIL import Image
import hydra
from omegaconf import DictConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add the parent directory to sys.path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


''' REMEMBER TO LAUNCHE THE SCRIPT with :
PYTHONPATH=/home/riccardop02/JEPA HYDRA_FULL_ERROR=1 /home/riccardop02/miniconda3/envs/poseJepa/bin/python /home/riccardop02/JEPA/Script/cacheEmb3d.py encoder3d._target_=src.models.encoder3d.Uni3D.Uni3DEncoder '''


def load_gso_mesh(model_dir: str) -> trimesh.Trimesh:
    """ Load meshes and texture from a GSO model """

    obj_path = os.path.join(model_dir, "meshes", "model.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ non trovato: {obj_path}")
    # Load mesh
    loaded = trimesh.load(obj_path, process=False)

    def fix_texture(mesh):
        vis = mesh.visual
        if hasattr(vis, "material") and getattr(vis.material, "image", None) is None:
            tex_path = os.path.join(model_dir, "materials", "textures", "texture.png")
            if os.path.exists(tex_path):
                vis.material.image = Image.open(tex_path)
            else:
                log.warning(f"Texture non trovata per {model_dir}: {tex_path}")
        return mesh

    # Search for texture and return the mesh
    if isinstance(loaded, trimesh.Scene):
        geoms = [fix_texture(g) for g in loaded.geometry.values()]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = fix_texture(loaded)

    return mesh



def sample_pointcloud_with_color(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """
    sample n_point from mesh
    """

    mesh = mesh.copy()
    if hasattr(mesh.visual, "to_color"):
        mesh.visual = mesh.visual.to_color()

    if not hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = np.full((len(mesh.vertices), 4), 200, dtype=np.uint8)
    else:
        vertex_colors = mesh.visual.vertex_colors

    # Sample points on the mesh surface
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)

    # Operation to get colors for the sampled points (uses triangles and baricentric coordinates)
    tri_verts_idx = mesh.faces[face_idx]              # (n_points, 3)
    tri_points = mesh.triangles[face_idx]              # (n_points, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tri_points, points)  # (n_points, 3)

    tri_colors = vertex_colors[tri_verts_idx][:, :, :3].astype(np.float32)  # (n_points, 3, 3)
    colors = (bary[:, :, None] * tri_colors).sum(axis=1)  # (n_points, 3), 0-255

    # return all the points with their colors (pointclouds)
    pc = np.concatenate([points.astype(np.float32), colors], axis=-1)  # (n_points, 6)
    return pc



def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    """
    pc: (N, 6) con xyz grezzo e rgb in [0, 255].
    Ritorna (N, 6) con xyz in sfera unitaria e rgb in [-1, 1].
    """
    xyz = pc[:, :3].copy()
    rgb = pc[:, 3:].copy()

    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale

    rgb = (rgb / 255.0 - 0.5) / 0.5

    return np.concatenate([xyz, rgb], axis=-1).astype(np.float32)



def process_model(model_dir: str, encoder3d, n_points: int, device: torch.device) -> np.ndarray:
    """ The brain of the processing: load mesh, sample point cloud, normalize, and extract features with encoder3d. """

    mesh = load_gso_mesh(model_dir)
    pc = sample_pointcloud_with_color(mesh, n_points)
    pc = normalize_pointcloud(pc)

    pc_tensor = torch.from_numpy(pc).unsqueeze(0).to(device)  # (1, N, 6)

    with torch.no_grad():
        feats = encoder3d.extractFeatures(pc_tensor)  # (1, num_group, trans_dim)

    return feats.squeeze(0).cpu().numpy()  # (num_group, trans_dim)



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Encoder3d
    encoder3d = hydra.utils.instantiate(cfg.encoder3d)
    encoder3d.to(device)
    encoder3d.eval()
    log.info(f"Encoder3d uploaded, output_dim={encoder3d.output_dim}")

    # Setting parameters from config
    root = cfg.gso.root
    n_points = cfg.gso.get("n_points", 8192)
    output_name = cfg.gso.get("output_name", "embedding.npy")
    overwrite = cfg.gso.get("overwrite", True)

    model_dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    log.info(f"Found {len(model_dirs)} models in {root}")

    n_ok, n_skip, n_fail = 0, 0, 0

    # Cicle to compute embeddings for each model and save them to disk
    for name in model_dirs:
        model_dir = os.path.join(root, name)
        out_path = os.path.join(model_dir, output_name)

        if os.path.exists(out_path) and not overwrite:
            log.info(f"[SKIP] {name}: embedding already present")
            n_skip += 1
            continue

        try:
            emb = process_model(model_dir, encoder3d, n_points, device)
            np.save(out_path, emb)
            log.info(f"[OK]   {name}: shape={emb.shape} -> {out_path}")
            n_ok += 1
        except Exception as e:
            log.error(f"[FAIL] {name}: {e}")
            n_fail += 1


        log.info(f"OK={n_ok} SKIP={n_skip} FAIL={n_fail}")
    
    log.info(f"Done.")

if __name__ == "__main__":
    main()