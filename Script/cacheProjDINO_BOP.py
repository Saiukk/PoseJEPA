import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pyrender
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/riccardop02/JEPA/")
from src.models.encoder3d.uni3d_backbone import fps  # ritorna già i punti, non indici

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_gso_mesh_bop(model_dir: str) -> trimesh.Trimesh:
    """
    Carica una mesh GSO in formato BOP-renderer (models_bop-renderer_scale=0.1).
    Struttura attesa: {model_dir}/meshes/model.ply

    Sostituisce load_gso_mesh: qui usiamo sempre e solo il .ply, perché è la
    stessa mesh con cui sono state renderizzate le immagini di riferimento
    (target render, pose annotate) usate nel resto della pipeline. Centroid e
    scale calcolati su questa mesh restano quindi coerenti con qualunque altro
    punto della pipeline (es. render_gso_object) che carichi lo stesso .ply.
    """
    ply_path = os.path.join(model_dir, "meshes", "model.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Mesh BOP non trovata: {ply_path}")

    mesh = trimesh.load(ply_path, process=False, force="mesh")

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Mesh vuota: {ply_path}")

    return mesh


def fps_torch(x: torch.Tensor, n: int) -> torch.Tensor:

    """x: (N,3) -> (n,3). FPS deterministico, tutto su device."""
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    idx[0] = (x - x.mean(0)).norm(dim=-1).argmax()      # start deterministico: punto più lontano dal centroide
    dist = (x - x[idx[0]]).norm(dim=-1)

    # for every point we take the farthest and we update the distance to the closest selected point
    for i in range(1, n):
        idx[i] = dist.argmax()
        dist = torch.minimum(dist, (x - x[idx[i]]).norm(dim=-1))
    return x[idx]

def fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    """n posizioni camera quasi-uniformi su una sfera di raggio `radius`."""
    idx = np.arange(n)
    golden = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * idx / golden
    z = 1 - (2 * idx + 1) / n
    r_xy = np.sqrt(np.maximum(0.0, 1 - z * z))
    x, y = r_xy * np.cos(theta), r_xy * np.sin(theta)
    return (np.stack([x, y, z], axis=-1) * radius).astype(np.float32)


def look_at_RT(cam_pos: np.ndarray, target: np.ndarray = np.zeros(3),
               up_hint: np.ndarray = np.array([0., 1., 0.])):
    """R, trans mondo->camera in convenzione OpenGL (camera guarda lungo -Z, y su): p_gl = R @ p_world + trans."""
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if abs(np.dot(forward, up_hint)) > 0.99:
        up_hint = np.array([1., 0., 0.])
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    R_w2c = np.stack([right, up, -forward], axis=0)
    trans = -R_w2c @ cam_pos
    return R_w2c.astype(np.float32), trans.astype(np.float32)


def render_view(mesh: trimesh.Trimesh, R: np.ndarray, trans: np.ndarray,
                K: np.ndarray, image_size: int):
    """Colore + depth con pyrender. Camera->world costruita invertendo (R, trans)."""
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.6, 0.6, 0.6])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    cam_pos = -R.T @ trans
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = cam_pos

    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    scene.add(camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=pose)

    r = pyrender.OffscreenRenderer(image_size, image_size)
    try:
        color, depth = r.render(scene)
    finally:
        r.delete()
    return np.ascontiguousarray(color[:, :, :3]), np.ascontiguousarray(depth)


_GL2CV = torch.tensor([1., -1., -1.])


def project_points(points_world: torch.Tensor, R: torch.Tensor, trans: torch.Tensor, K: torch.Tensor):
    """points_world: (P,3) -> uv pixel (P,2), depth camera-space (P,). R/trans sono OpenGL, K è OpenCV."""
    p_gl = points_world @ R.T + trans
    p_cam = p_gl * _GL2CV.to(p_gl.device)          # GL -> CV: y verso il basso
    z = p_cam[:, 2].clamp(min=1e-6)
    uv = ((p_cam / z.unsqueeze(-1)) @ K.T)[:, :2]
    return uv, z


def sample_feature_grid(feat_grid: torch.Tensor, uv: torch.Tensor, image_size: int) -> torch.Tensor:
    """feat_grid: (1,D,Hp,Wp). uv in pixel coord. -> (P, D) via bilinear grid_sample."""
    uv_norm = (uv / image_size) * 2 - 1
    grid = uv_norm.view(1, -1, 1, 2)
    sampled = F.grid_sample(feat_grid, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(0).squeeze(-1).T


def visibility_mask(uv: torch.Tensor, z_point: torch.Tensor, depth_map: torch.Tensor,
                    image_size: int, tol: float = 0.03) -> torch.Tensor:
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < image_size) & (uv[:, 1] >= 0) & (uv[:, 1] < image_size)
    ix = uv[:, 0].long().clamp(0, image_size - 1)
    iy = uv[:, 1].long().clamp(0, image_size - 1)
    z_render = depth_map[iy, ix]
    return in_bounds & (z_render > 0) & ((z_point - z_render).abs() < tol)


def get_target_size(encoder2d) -> int:
    """Risoluzione di output del processor DINOv2 (il render viene fatto già a questa risoluzione)."""
    proc = encoder2d.processor
    size = getattr(proc, "crop_size", None) or proc.size
    return size["height"]


@torch.no_grad()
def extract_dinov2_grid(encoder2d, image_uint8: np.ndarray, device: torch.device):
    """
    image_uint8: (H, W, 3) uint8, il render.
    Ritorna (1, D, Hp, Wp), H_proc, W_proc. Richiede encoder2d.token == "PatchToken".
    """
    if encoder2d.token != "PatchToken":
        raise ValueError(
            f"encoder2d.token='{encoder2d.token}', serve 'PatchToken' per una griglia spaziale campionabile"
        )

    proc = encoder2d.processor
    # niente resize/crop: il render ha già la risoluzione giusta, così K resta coerente con la griglia.
    # La normalizzazione (mean/std) resta identica al training.
    inputs = proc(images=image_uint8, return_tensors="pt",
                  do_resize=False, do_center_crop=False)
    pixel_values = inputs["pixel_values"].to(device)

    tokens = encoder2d.extractFeatures(pixel_values)           # (1, N, D)
    N, D = tokens.shape[1], tokens.shape[2]

    patch_size = encoder2d.model.config.patch_size
    H_proc, W_proc = pixel_values.shape[-2], pixel_values.shape[-1]
    Hp, Wp = H_proc // patch_size, W_proc // patch_size
    assert Hp * Wp == N, f"grid {Hp}x{Wp}={Hp*Wp} non combacia con N={N} token"

    return tokens.transpose(1, 2).reshape(1, D, Hp, Wp), H_proc, W_proc


def process_model(model_dir: str, encoder2d, n_points: int, n_views: int,
                  cam_radius: float, fov_deg: float, device: torch.device):

    # Load the mesh (ora sempre dal .ply BOP)
    mesh = load_gso_mesh_bop(model_dir)

    # Sample points on the mesh surface and apply FPS to get n_points
    dense_xyz = trimesh.sample.sample_surface(mesh, 20000)[0].astype(np.float32)
    points = fps_torch(torch.from_numpy(dense_xyz).to(device), n_points)

    #compute centroid and scale to normalize points and mesh to canonical frame
    centroid = points.mean(dim=0, keepdim=True)
    points_c = points - centroid
    scale = points_c.norm(dim=-1).max().clamp(min=1e-6)
    points_n = points_c / scale                                 # frame canonico

    mesh_c = mesh.copy()
    mesh_c.vertices = (mesh_c.vertices - centroid.cpu().numpy()) / scale.item()

    # Compute the intrinsic matrix
    image_size = get_target_size(encoder2d)
    f = 0.5 * image_size / np.tan(np.radians(fov_deg) / 2)
    K_np = np.array([[f, 0, image_size / 2], [0, f, image_size / 2], [0, 0, 1]], dtype=np.float32)
    K = torch.from_numpy(K_np).to(device)

    # Obtain camera positions on a sphere around the object to render multiple views that cover entirely the object 
    cam_positions = fibonacci_sphere(n_views, radius=cam_radius)
    view_feats, view_dirs, masks = [], [], []

    for cam_pos in cam_positions:
        # Compute the camera extrinsic matrix (R, trans) using the look_at_RT function
        R_np, trans_np = look_at_RT(cam_pos)
        R, trans = torch.from_numpy(R_np).to(device), torch.from_numpy(trans_np).to(device)

        # Render the view of the mesh from the camera position and obtain the color and depth images
        color, depth = render_view(mesh_c, R_np, trans_np, K_np, image_size)
        depth_t = torch.from_numpy(depth).to(device)


        feat_grid, H_proc, W_proc = extract_dinov2_grid(encoder2d, color, device)
        assert H_proc == image_size and W_proc == image_size, \
            "il processor ha ritagliato/scalato: K non è più coerente con la griglia di feature"

        # Project the 3D points into the camera view and compute the visibility mask
        uv, z_point = project_points(points_n, R, trans, K)
        vis = visibility_mask(uv, z_point, depth_t, image_size)

        feats = sample_feature_grid(feat_grid, uv, image_size) * vis.unsqueeze(-1)

        cam_pos_t = torch.from_numpy(cam_pos).to(device)
        d = F.normalize(cam_pos_t.unsqueeze(0) - points_n, dim=-1) * vis.unsqueeze(-1)

        view_feats.append(feats.cpu().numpy())
        view_dirs.append(d.cpu().numpy())
        masks.append(vis.cpu().numpy())

    return (points_n.cpu().numpy(),
            np.stack(view_feats, axis=1),    # (P, n_views, D)
            np.stack(view_dirs, axis=1),     # (P, n_views, 3)
            np.stack(masks, axis=1),         # (P, n_views)
            centroid.cpu().numpy(), scale.item())


def _atomic_save(path: str, arr: np.ndarray):
    """Scrive su file temporaneo e poi rinomina: un crash a metà non lascia file corrotti."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def save_aggregator_context(out_dir: str, points, view_feats, view_dirs, mask, centroid, scale,
                            fp16: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    feat_dtype = np.float16 if fp16 else np.float32
    _atomic_save(os.path.join(out_dir, "points.npy"), points.astype(np.float32))
    _atomic_save(os.path.join(out_dir, "view_feats.npy"), view_feats.astype(feat_dtype))
    _atomic_save(os.path.join(out_dir, "view_dirs.npy"), view_dirs.astype(np.float32))
    _atomic_save(os.path.join(out_dir, "mask.npy"), mask.astype(np.bool_))
    _atomic_save(os.path.join(out_dir, "centroid.npy"), np.asarray(centroid, dtype=np.float32))
    # scale.npy per ultimo: fa da marker di completamento
    _atomic_save(os.path.join(out_dir, "scale.npy"), np.array(scale, dtype=np.float32))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model
    encoder2d = hydra.utils.instantiate(cfg.encoder2d).to(device).eval()
    for p in encoder2d.parameters():
        p.requires_grad_(False)
    log.info(f"Encoder2d (DINOv2) caricato per il precompute, token={encoder2d.token}")

    root = cfg.gso.root   # deve puntare a .../models_bop-renderer_scale=0.1
    n_points = cfg.gso.get("n_points", 512)
    n_views = cfg.gso.get("n_views", 24)
    cam_radius = cfg.gso.get("cam_radius", 2.5)
    fov_deg = cfg.gso.get("fov_deg", 40.0)
    aggregator_dirname = cfg.gso.get("aggregator_dirname", "3DEmbedd")
    overwrite = cfg.gso.get("overwrite", True)   # False: salta i modelli già completati
    save_fp16 = cfg.gso.get("save_fp16", False)   # True: view_feats in float16 (metà spazio)

    # look at the available model
    model_dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    log.info(f"Trovati {len(model_dirs)} modelli in {root}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for name in model_dirs:
        model_dir = os.path.join(root, name)
        out_dir = os.path.join(model_dir, aggregator_dirname)
        marker = os.path.join(out_dir, "scale.npy")

        if os.path.exists(marker) and not overwrite:
            log.info(f"[SKIP] {name}: già presente")
            n_skip += 1
            continue

        try:

            # Compute the 3D context features
            points, view_feats, view_dirs, mask, centroid, scale = process_model(
                model_dir, encoder2d, n_points, n_views, cam_radius, fov_deg, device
            )
            #save it as npy
            save_aggregator_context(out_dir, points, view_feats, view_dirs, mask, centroid, scale,
                                    fp16=save_fp16)
            log.info(f"[OK]   {name}: points={points.shape} view_feats={view_feats.shape} "
                     f"vis_media={mask.mean():.2f} -> {out_dir}")
            n_ok += 1
        except Exception:
            log.exception(f"[FAIL] {name}")
            n_fail += 1

    log.info(f"Done. OK={n_ok} SKIP={n_skip} FAIL={n_fail}")


if __name__ == "__main__":
    main()