## !TODO SCRIPT NON UTILIZZABILE PER COSTO DI MEMORIA / Utilizzabile per costruire al volo ply, poi sampling e salvare embedding encoder3d

import trimesh
import os
from PIL import Image

TARGET_EDGE = 0.002  # prova 2-3mm invece di 1mm: molto meno vertici, dettaglio comunque buono

base = '/data/riccardop02/datasets/gso/Brisk_Iced_Tea_Lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt'
obj_path = os.path.join(base, 'meshes', 'model.obj')

scene = trimesh.load(obj_path)

def fix_texture_path(mesh):
    if hasattr(mesh.visual, 'material') and mesh.visual.material.image is None:
        texture_path = os.path.join(base, 'materials', 'textures', 'texture.png')
        if os.path.exists(texture_path):
            mesh.visual.material.image = Image.open(texture_path)
    return mesh

def densify_and_bake(mesh):
    mesh = fix_texture_path(mesh)
    n_before = len(mesh.vertices)

    mesh = mesh.subdivide_to_size(max_edge=TARGET_EDGE)
    n_after = len(mesh.vertices)
    print(f"Vertici: {n_before} -> {n_after}")

    if hasattr(mesh.visual, 'to_color'):
        mesh.visual = mesh.visual.to_color()

    return mesh

if isinstance(scene, trimesh.Scene):
    processed = [densify_and_bake(g) for g in scene.geometry.values()]
    final_mesh = trimesh.util.concatenate(processed)
else:
    final_mesh = densify_and_bake(scene)

final_mesh.export('model_dense.ply', encoding='binary')
print("Dimensione:", os.path.getsize('model_dense.ply') / 1e6, "MB")