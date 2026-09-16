import hashlib
import math
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import logging
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
import hydra
from omegaconf import DictConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add the parent directory to sys.path

print("sys.path:", sys.path)

from src.data.render_utils import build_pyrender_mesh  # <-- adatta il path al tuo repo
# NOTA: render_random_view NON e' piu' usato: il rendering delle view e' ora fatto
# direttamente in questo script (vedi render_views_systematic), con pose deterministiche
# e dimensione immagine SEMPRE image_size x image_size, cosi' np.stack non fallisce piu'.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def _stable_seed(*parts: object) -> int:
    s = "_".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def load_gso_mesh(model_dir: str) -> trimesh.Trimesh:
    """ Load meshes and texture from a GSO model """

    obj_path = os.path.join(model_dir, "meshes", "model.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ non trovato: {obj_path}")
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

    if isinstance(loaded, trimesh.Scene):
        geoms = [fix_texture(g) for g in loaded.geometry.values()]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = fix_texture(loaded)

    return mesh


def sample_pointcloud_with_color(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """ sample n_point from mesh """

    mesh = mesh.copy()
    if hasattr(mesh.visual, "to_color"):
        mesh.visual = mesh.visual.to_color()

    if not hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = np.full((len(mesh.vertices), 4), 200, dtype=np.uint8)
    else:
        vertex_colors = mesh.visual.vertex_colors

    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)

    tri_verts_idx = mesh.faces[face_idx]
    tri_points = mesh.triangles[face_idx]
    bary = trimesh.triangles.points_to_barycentric(tri_points, points)

    tri_colors = vertex_colors[tri_verts_idx][:, :, :3].astype(np.float32)
    colors = (bary[:, :, None] * tri_colors).sum(axis=1)

    pc = np.concatenate([points.astype(np.float32), colors], axis=-1)
    return pc


def normalize_pointcloud(pc: np.ndarray) -> np.ndarray:
    """
    pc: (N, 6) con xyz grezzo e rgb in [0, 255].
    Ritorna (N, 6) con xyz in sfera unitaria e rgb in [-1, 1], + centroid, scale.
    """
    xyz = pc[:, :3].copy()
    rgb = pc[:, 3:].copy()

    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale

    rgb = (rgb / 255.0 - 0.5) / 0.5

    return np.concatenate([xyz, rgb], axis=-1).astype(np.float32), centroid, scale


def compute_normalization(model_dir: str, n_points: int) -> tuple[np.ndarray, float]:
    """
    Ricalcola centroid/scale con LA STESSA logica dello script UNI3D (sample + normalize).
    """
    mesh = load_gso_mesh(model_dir)
    pc = sample_pointcloud_with_color(mesh, n_points)
    _, centroid, scale = normalize_pointcloud(pc)
    return centroid.squeeze().astype(np.float32), np.float32(scale)


# ---------------------------------------------------------------------------
# Rendering sistematico (sostituisce render_random_view)
# ---------------------------------------------------------------------------

def _canonical_view_dirs() -> list[np.ndarray]:
    """6 direzioni assiali (fronte/retro/sinistra/destra/sopra/sotto) + 8 angoli
    del cubo, tutte normalizzate. Danno una copertura molto uniforme e prevedibile
    della sfera di vista attorno all'oggetto."""
    dirs = [
        np.array([0.0, 0.0, 1.0]),   # fronte
        np.array([0.0, 0.0, -1.0]),  # retro
        np.array([1.0, 0.0, 0.0]),   # destra
        np.array([-1.0, 0.0, 0.0]),  # sinistra
        np.array([0.0, 1.0, 0.0]),   # sopra
        np.array([0.0, -1.0, 0.0]),  # sotto
    ]
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                v = np.array([sx, sy, sz])
                dirs.append(v / np.linalg.norm(v))
    return dirs


def _fibonacci_sphere(n: int) -> list[np.ndarray]:
    """Punti extra distribuiti in modo (quasi) uniforme sulla sfera, usati solo
    se n_views supera le 14 viste canoniche."""
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(max(n - 1, 1))) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        pts.append(np.array([x, y, z]))
    return pts


def generate_view_directions(n_views: int) -> list[np.ndarray]:
    """
    Genera n_views direzioni di vista distribuite in modo uniforme sulla sfera.
    """
    dirs = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_views):
        y = 1.0 - (i / float(max(n_views - 1, 1))) * 2.0  # da +1 a -1
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        dirs.append(np.array([x, y, z]))
    return dirs


def _look_at_pose(eye: np.ndarray, target: np.ndarray, up_hint=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Pose camera->mondo (4x4), convenzione OpenGL/pyrender: la camera guarda
    lungo -Z, +Y e' up, +X e' right."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-8 else np.array([0.0, 0.0, -1.0])

    up = up_hint
    if abs(np.dot(forward, up)) > 0.99:  # viste dall'alto/basso: up degenere
        up = np.array([1.0, 0.0, 0.0])

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def render_views_systematic(
    render_mesh,
    centroid: np.ndarray,
    scale: float,
    image_size: int,
    n_views: int,
    renderer: pyrender.OffscreenRenderer,
    camera_dist_factor: float = 2.2,
    yfov_deg: float = 50.0,
):
    """
    Renderizza n_views viste deterministiche dell'oggetto (fronte, retro, lati,
    sopra, sotto, angoli...) usando SEMPRE un OffscreenRenderer di dimensione
    fissa (image_size, image_size): questo garantisce che ogni immagine abbia
    esattamente la stessa shape, eliminando il ValueError di np.stack.

    Le pose (R, t) restituite sono espresse nello spazio NORMALIZZATO (la stessa
    sfera unitaria usata per il point cloud), coerentemente con centroid/scale
    calcolati da compute_normalization. Il rendering vero e proprio invece avviene
    nello spazio "raw" della mesh (centroid + direzione*scale), perche' la mesh
    caricata da build_pyrender_mesh non e' normalizzata.
    """
    yfov = math.radians(yfov_deg)
    fy = (image_size / 2.0) / math.tan(yfov / 2.0)
    fx = fy
    cx = cy = image_size / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    view_dirs = generate_view_directions(n_views)
    assert len(view_dirs) == n_views

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[0.35, 0.35, 0.35])
    mesh_node = scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    cam_node = scene.add(camera, pose=np.eye(4))
    light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
    light_node = scene.add(light, pose=np.eye(4))

    images, masks, poses = [], [], []
    try:
        for d in view_dirs:
            # --- pose nello spazio normalizzato (quella che salviamo) ---
            eye_n = d * camera_dist_factor
            target_n = np.zeros(3)
            pose_n = _look_at_pose(eye_n, target_n)
            R_n = pose_n[:3, :3].T
            t_n = -R_n @ eye_n

            # --- pose nello spazio "raw" della mesh (quella che usiamo per renderizzare) ---
            eye_world = centroid + eye_n * scale
            target_world = centroid
            pose_world = _look_at_pose(eye_world, target_world)

            scene.set_pose(cam_node, pose_world)
            scene.set_pose(light_node, pose_world)

            color, depth = renderer.render(scene)  # color: (image_size, image_size, 3) uint8, SEMPRE

            img = color.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3, H, W), come si aspetta encoder2d

            images.append(img)
            masks.append(depth > 0)
            poses.append(np.concatenate([R_n.reshape(-1), t_n.reshape(-1)]).astype(np.float32))
    finally:
        scene.remove_node(mesh_node)
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

    return images, masks, K, poses


@torch.no_grad()
def process_model(
    model_dir: str,
    model_name: str,
    encoder2d,
    n_views: int,
    n_points: int,
    image_size: int,
    device: torch.device,
    renderer: pyrender.OffscreenRenderer,
    dino_mean: torch.Tensor,
    dino_std: torch.Tensor,
):
    """
    Calcola centroid/scale (come UNI3D), renderizza n_views view sistematiche
    (fronte/retro/lati/sopra/sotto/angoli) del mesh, ed estrae feature DINOv2
    patch-level per ciascuna, in batch.
    """
    centroid, scale = compute_normalization(model_dir, n_points)
    render_mesh = build_pyrender_mesh(model_dir)

    images, masks, K, poses = render_views_systematic(
        render_mesh=render_mesh,
        centroid=centroid,
        scale=scale,
        image_size=image_size,
        n_views=n_views,
        renderer=renderer,
    )

    imgs_t = torch.from_numpy(np.stack(images)).float().to(device)  # (N, 3, H, W)
    imgs_t = (imgs_t - dino_mean) / dino_std

    patch_tokens = encoder2d(imgs_t)  # (N, P, D) -- STESSO encoder usato come target in training
    feats = patch_tokens.cpu().numpy()

    poses_arr = np.stack(poses)  # (N, 12), shape ora sempre coerente

    return feats, poses_arr, centroid, scale  # (N,P,D), (N,12), (3,), ()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Encoder2d (DINOv2) -- deve essere la STESSA config usata come target encoder in training
    encoder2d = hydra.utils.instantiate(cfg.encoder2d)
    encoder2d.to(device)
    encoder2d.eval()
    for p in encoder2d.parameters():
        p.requires_grad_(False)
    log.info(f"Encoder2d uploaded, output_dim={encoder2d.output_dim}")

    dino_mean = torch.tensor(DINOV2_MEAN).view(3, 1, 1).to(device)
    dino_std = torch.tensor(DINOV2_STD).view(3, 1, 1).to(device)

    # Setting parameters from config
    root = cfg.gso.root
    n_points = cfg.gso.get("n_points", 8192)
    n_views = cfg.gso.get("n_views", 30)  
    image_size = cfg.gso.get("image_size", 224)
    output_name = cfg.gso.get("output_name", "embeddingDINO.npz")
    overwrite = cfg.gso.get("overwrite", True)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    model_dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    log.info(f"Found {len(model_dirs)} models in {root}")

    n_ok, n_skip, n_fail = 0, 0, 0

    for name in model_dirs:
        model_dir = os.path.join(root, name)
        out_path = os.path.join(model_dir, output_name)

        if os.path.exists(out_path) and not overwrite:
            log.info(f"[SKIP] {name}: embedding already present")
            n_skip += 1
            continue

        try:
            feats, poses, centroid, scale = process_model(
                model_dir, name, encoder2d, n_views, n_points, image_size,
                device, renderer, dino_mean, dino_std,
            )
            np.savez(out_path, features=feats, centers=poses, centroid=centroid, scale=scale)
            log.info(
                f"[OK]   {name}: feats={feats.shape}, centers(poses)={poses.shape} -> {out_path}, "
                f"(centroid, scale) = ({centroid}, {scale})"
            )
            n_ok += 1
        except Exception as e:
            log.exception(f"[FAIL] {name}: {e}")
            n_fail += 1

        log.info(f"OK={n_ok} SKIP={n_skip} FAIL={n_fail}")

    renderer.delete()
    log.info(f"Done.")


if __name__ == "__main__":
    main()