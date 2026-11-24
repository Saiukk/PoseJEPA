import logging
import os
import numpy as np
import pyrender
import trimesh
from PIL import Image

log = logging.getLogger(__name__)


MAX_TEXTURE_SIZE = 512  

def load_gso_mesh(model_dir: str) -> trimesh.Trimesh:
    obj_path = os.path.join(model_dir, "meshes", "model.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ non trovato: {obj_path}")

    loaded = trimesh.load(obj_path, process=False)

    def fix_texture(mesh):
        vis = mesh.visual
        if hasattr(vis, "material") and getattr(vis.material, "image", None) is None:
            tex_path = os.path.join(model_dir, "materials", "textures", "texture.png")
            if os.path.exists(tex_path):
                img = Image.open(tex_path).convert("RGB")  # niente alpha se non serve
                if max(img.size) > MAX_TEXTURE_SIZE:
                    img.thumbnail((MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE), Image.LANCZOS)
                vis.material.image = img
            else:
                log.warning(f"Texture non trovata per {model_dir}: {tex_path}")
        return mesh

    if isinstance(loaded, trimesh.Scene):
        geoms = [fix_texture(g) for g in loaded.geometry.values()]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = fix_texture(loaded)

    return mesh

# Used to reduce the overhead of creating a pyrender.Mesh from a trimesh.
def build_pyrender_mesh(model_dir: str) -> pyrender.Mesh:
    mesh = load_gso_mesh(model_dir)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def _sample_camera_pose(
    centroid: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    fov_y: float,
    frac: float,
):
    # compute the camera position 
    theta = np.clip(frac * fov_y, 1e-3, np.pi - 1e-3)
    radius = scale / np.tan(theta / 2.0)

    # Used to sample a random point on the sphere around the centroid
    azimuth = rng.uniform(0, 2 * np.pi)
    elevation = np.arcsin(rng.uniform(-1.0, 1.0))  # rad, full-sphere uniforme
    cam_pos = centroid + radius * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])

    # Where the camera is looking at (the centroid of the object)
    forward = (centroid - cam_pos)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    # OPENCV convention: camera looks along +Z, Y is down, X is right
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # compute the camera-to-world rotation matrix and translation vector
    R_cam_to_world = np.stack([right, -up, forward], axis=1)  # colonne = assi camera in frame mondo
    R = R_cam_to_world.T
    t = -R @ cam_pos

    return R.astype(np.float32), t.astype(np.float32)


def _sample_intrinsics(image_size: int, rng: np.random.Generator, fov_range=(25, 70), cxcy_jitter_frac=0.35):

    # compute the focal length in pixels
    fov_y = np.deg2rad(rng.uniform(*fov_range))
    f = image_size / (2.0 * np.tan(fov_y / 2.0))

    # compute the center point 
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
    render_mesh: pyrender.Mesh,
    centroid: np.ndarray,
    scale: float,
    image_size: int = 224,
    rng: np.random.Generator = None,
    fov_range=(40, 70),
    frac_range=(0.85, 0.9),
    shift_safety=0.9,
    light_intensity_range=(2.5, 3.5),
    renderer: "pyrender.OffscreenRenderer | None" = None,
    get_renderer_fn=None,
):
    """
    Render a view of a 3D model.
    """
    if rng is None:
        rng = np.random.default_rng()

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(render_mesh)

    # Generate a causal frac between the range and compute margin to ensure the object is fully visible
    frac = rng.uniform(*frac_range)
    margin_frac = max(0.0, (1.0 - frac) / 2.0 * shift_safety)
    cxcy_jitter_frac = rng.uniform(0.0, margin_frac)

    # Sample camera intrinsics and pose
    K, fov_y = _sample_intrinsics(image_size, rng, fov_range=fov_range, cxcy_jitter_frac=cxcy_jitter_frac)
    R, t = _sample_camera_pose(
        centroid.astype(np.float64), float(scale), rng, fov_y=fov_y, frac=frac
    )

    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]

    # Enlarge the canvas to use jittered principal point.
    pad = int(np.ceil(cxcy_jitter_frac * image_size)) + 4
    render_w = render_h = image_size + 2 * pad

    # compute the correct fov_y
    fov_y_padded = 2.0 * np.arctan(render_h / (2.0 * f))
    camera = pyrender.PerspectiveCamera(yfov=fov_y_padded, aspectRatio=render_w / render_h)

    # compute the camera pose in world coordinates 
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.T
    cam_pose[:3, 3] = -R.T @ t

    # OpenGL convention: camera looks along -Z, Y is up, X is right
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float64)
    cam_pose = cam_pose @ cv_to_gl

    # Add camera and light to the scene
    scene.add(camera, pose=cam_pose)
    light_intensity = rng.uniform(*light_intensity_range)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    scene.add(light, pose=cam_pose)

    # Render the scene using either a provided renderer or a temporary one
    owns_renderer = False
    if get_renderer_fn is not None:
        active_renderer = get_renderer_fn(render_w, render_h)
    elif renderer is not None:
        if renderer.viewport_width != render_w or renderer.viewport_height != render_h:
            renderer.viewport_width = render_w
            renderer.viewport_height = render_h
        active_renderer = renderer
    else:
        active_renderer = pyrender.OffscreenRenderer(render_w, render_h)
        owns_renderer = True

    try:
        color, depth = active_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        if owns_renderer:
            active_renderer.delete()

    # Crop the image to the desired size, centering on the jittered principal point
    ox = int(round(render_w / 2.0 - cx))
    oy = int(round(render_h / 2.0 - cy))
    ox = np.clip(ox, 0, render_w - image_size)
    oy = np.clip(oy, 0, render_h - image_size)
    color_crop = color[oy:oy + image_size, ox:ox + image_size]
    depth_crop = depth[oy:oy + image_size, ox:ox + image_size]

    # Create a mask and normalize the RGB image
    mask = (depth_crop > 0).astype(np.float32)
    rgb = color_crop[..., :3].astype(np.float32) / 255.0
    rgb = rgb * mask[..., None]
    image = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    return image, mask, K, R, t