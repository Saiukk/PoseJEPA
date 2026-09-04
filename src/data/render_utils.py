"""
Rendering on-the-fly per RenderDataset.

ATTENZIONE: questo e' uno scheletro funzionante ma con scelte di default
(range distanza/elevazione camera, luci, sfondo) che vanno tarate sul tuo
setup specifico -- non sono state discusse nel dettaglio, quindi trattale
come punto di partenza, non come valori definitivi.
"""

import logging
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image

log = logging.getLogger(__name__)


def load_gso_mesh(model_dir: str) -> trimesh.Trimesh:
    """Load mesh and texture from a GSO model."""

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


def _sample_camera_pose(centroid: np.ndarray, scale: float, rng: np.random.Generator):
    """
    Campiona una posa random della camera su una shell sferica attorno al
    centroide dell'oggetto (frame mesh originale), guardando verso il centroide.

    TODO: tara questi range sul tuo dataset (dimensioni reali GSO, FOV camera).
    """
    radius = rng.uniform(2.0 * scale, 4.0 * scale)          # distanza camera-oggetto
    azimuth = rng.uniform(0, 2 * np.pi)
    elevation = rng.uniform(-0.3, 0.9)                       # rad, evita viste da sotto

    cam_pos = centroid + radius * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])

    forward = (centroid - cam_pos)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # R, t tali che: punto_camera = R @ punto_mondo + t  (convenzione OpenCV)
    R_cam_to_world = np.stack([right, -up, forward], axis=1)  # colonne = assi camera in frame mondo
    R = R_cam_to_world.T
    t = -R @ cam_pos

    return R.astype(np.float32), t.astype(np.float32)


def _sample_intrinsics(image_size: int, rng: np.random.Generator, fov_range=(35, 55), cxcy_jitter_frac=0.03):
    """
    Campiona intrinseci random per la camera: FOV variabile (quindi focale variabile)
    e piccolo jitter sul principal point.

    fov_range: (min_deg, max_deg) — TODO: allinealo al FOV reale delle camere
    che userai in inference (BOP/CNOS), non lasciarlo un range arbitrario.
    cxcy_jitter_frac: jitter di cx,cy come frazione di image_size (0 = sempre
    centrato esatto, valori tipici 0.02-0.05 simulano principal point reale
    non perfettamente centrato).
    """
    fov_y = np.deg2rad(rng.uniform(*fov_range))
    f = image_size / (2.0 * np.tan(fov_y / 2.0))

    jitter = cxcy_jitter_frac * image_size
    cx = image_size / 2.0 + rng.uniform(-jitter, jitter)
    cy = image_size / 2.0 + rng.uniform(-jitter, jitter)

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return K, fov_y


def render_random_view(
    model_name: str,
    dataset_source: str,
    centroid: np.ndarray,
    scale: float,
    image_size: int = 224,
    rng: np.random.Generator = None,
    fov_range=(35, 55),
    cxcy_jitter_frac: float = 0.03,
):
    """
    Renderizza una vista random dell'oggetto (frame mesh originale, non normalizzato).

    Ritorna:
        image: (3, H, W) float32, RGB in [0,1], sfondo forzato a 0 (nero)
        mask:  (H, W) float32, silhouette in [0,1] (da depth > 0)
        K:     (3, 3) float32, intrinseci camera
        R:     (3, 3) float32, rotazione mondo->camera
        t:     (3,)   float32, traslazione mondo->camera
    """
    if rng is None:
        rng = np.random.default_rng()

    model_dir = os.path.join(dataset_source, model_name)
    mesh = load_gso_mesh(model_dir)

    # TODO: cache della mesh trimesh->pyrender per oggetto se il caricamento
    # ripetuto ad ogni __getitem__ diventa il collo di bottiglia.
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(render_mesh)

    R, t = _sample_camera_pose(centroid.astype(np.float64), float(scale), rng)
    K, fov_y = _sample_intrinsics(image_size, rng, fov_range=fov_range, cxcy_jitter_frac=cxcy_jitter_frac)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]

    # pyrender.PerspectiveCamera centra sempre il principal point sul canvas.
    # Per ottenere cx,cy != centro (e restare coerenti con la K dichiarata),
    # renderizziamo un canvas piu' grande, centrato per costruzione, e poi
    # ritagliamo la finestra image_size x image_size nel punto giusto.
    pad = int(np.ceil(cxcy_jitter_frac * image_size)) + 1
    render_w = render_h = image_size + 2 * pad

    # stessa focale f (in pixel) del canvas finale, ricavata sul canvas grande
    fov_y_padded = 2.0 * np.arctan(render_h / (2.0 * f))
    camera = pyrender.PerspectiveCamera(yfov=fov_y_padded, aspectRatio=render_w / render_h)

    # pyrender vuole la posa camera->mondo (inversa di R, t)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = -R.T @ t
    # pyrender: asse camera guarda lungo -Z locale, va allineato con la
    # convenzione OpenCV usata sopra (+Z forward) -> flip
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float64)
    cam_pose = cam_pose @ cv_to_gl

    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)  # TODO: tara intensita'/randomizza
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(render_w, render_h)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    # crop: il centro del canvas padded (render_w/2, render_h/2) e' il vero
    # principal point del render pyrender; vogliamo che dopo il crop il
    # principal point cada in (cx, cy) dentro la finestra image_size x image_size
    ox = int(round(render_w / 2.0 - cx))
    oy = int(round(render_h / 2.0 - cy))
    ox = np.clip(ox, 0, render_w - image_size)
    oy = np.clip(oy, 0, render_h - image_size)

    color_crop = color[oy:oy + image_size, ox:ox + image_size]
    depth_crop = depth[oy:oy + image_size, ox:ox + image_size]

    mask = (depth_crop > 0).astype(np.float32)

    rgb = color_crop[..., :3].astype(np.float32) / 255.0
    # bg_color/alpha di pyrender non e' affidabile per garantire sfondo nero
    # (bg_color qui e' bianco, l'alpha viene ignorato a valle): azzeriamo
    # esplicitamente i pixel di sfondo usando la mask, cosi' il nero e'
    # garantito indipendentemente dal comportamento interno del renderer.
    rgb = rgb * mask[..., None]

    image = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    return image, mask, K, R, t