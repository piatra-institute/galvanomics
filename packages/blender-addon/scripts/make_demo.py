"""Build a ready-to-open .blend from a bundle: tissue imported, camera and viewport set up.

Run inside Blender with the extension installed and enabled:

    Blender -b --python-exit-code 1 -P scripts/make_demo.py -- <bundle_dir> <out.blend> [fps]

The .blend is saved before the import so the bundle path is stored relative to it; keep the
two side by side.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

args = sys.argv[sys.argv.index("--") + 1 :]
bundle_dir, out = Path(args[0]).resolve(), Path(args[1]).resolve()
fps = int(args[2]) if len(args) > 2 else 8
view = args[3] if len(args) > 3 else "auto"  # auto, all, or wound
module = "bl_ext.user_default.bioelectric_playback"
if module not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module=module)

bpy.ops.wm.read_homefile(use_empty=True)
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out))

result = bpy.ops.bioelectric.import_bundle(filepath=str(bundle_dir / "bundle.json"))
assert result == {"FINISHED"}, result
tissue = bpy.context.view_layer.objects.active
scene = bpy.context.scene
scene.render.fps = fps
scene.view_settings.view_transform = "Standard"

# Make the cell closest to the wound edge active, so the overlay plot has something to show.
reader = importlib.import_module(f"{module}.reader")
bundle = reader.open_bundle(bundle_dir)
centres = np.asarray(bundle.array("cell_centres"))
removed = bundle.removed()
if removed.any():
    distance = np.min(np.linalg.norm(centres[:, None] - centres[None, removed], axis=2), axis=1)
    distance[removed] = np.inf
    target = int(np.argmin(distance))
    if bundle.kind == "polyhedra":  # polygons are faces: pick the cell's first face
        target = int(np.asarray(bundle.array("cell_offsets"))[target])
    tissue.data.polygons.active = target

# A camera framing the sheet: top-down for 2D bundles, oblique for 3D ones.
verts, _ = bundle.polygons()
verts = np.asarray(verts)[:, :2]
lo, hi = verts.min(axis=0) / 10.0, verts.max(axis=0) / 10.0
wound_x = bundle.constants.get("wound_x_um")
wide = (hi[0] - lo[0]) > 3 * (hi[1] - lo[1])
if wound_x is not None and (view == "wound" or (view == "auto" and wide)):
    # Frame the wound region of a wide block: 320 um from the wound centre.
    x0 = float(wound_x) / 10.0
    lo, hi = np.array([max(lo[0], x0 - 32.0), lo[1]]), np.array([min(hi[0], x0 + 32.0), hi[1]])
centre = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = float(max(hi - lo)) * 1.15
camera = bpy.data.objects.new("Camera", camera_data)
camera.location = (*centre, 50.0)
extent = float(max(hi - lo))
poly = bundle.kind == "polyhedra"
three_d = bundle.extrusion_height_um is not None or poly
tilt = float(np.arctan2(1.5, 0.75)) if poly else float(np.arctan2(1.4, 1.3))
if three_d:  # an oblique perspective view shows the prisms' sides
    camera_data.type = "PERSP"
    camera.location = (
        (centre[0], centre[1] - 1.5 * extent, 0.75 * extent)
        if poly
        else (centre[0], centre[1] - 1.4 * extent, 1.3 * extent)
    )
    camera.rotation_euler = (tilt, 0.0, 0.0)
camera_data.clip_end = 20.0 * extent
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.engine = "BLENDER_WORKBENCH"
if poly:  # glyph colours come from materials (Geometry Nodes output): use EEVEE
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
scene.display.shading.light = "STUDIO" if poly else "FLAT"
scene.display.shading.color_type = "VERTEX"
scene.display.shading.show_cavity = poly

# Every 3D viewport: flat attribute colours, framed like the camera.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        space.shading.type = "MATERIAL" if poly else "SOLID"
        space.shading.light = "STUDIO" if poly else "FLAT"
        space.shading.show_cavity = poly
        space.shading.cavity_type = "BOTH"
        space.shading.color_type = "VERTEX"
        space.overlay.show_floor = False
        space.overlay.show_axis_x = space.overlay.show_axis_y = False
        space.clip_end = 20.0 * extent
        region = space.region_3d
        region.view_location = (*centre, 0.0)
        if three_d:
            region.view_perspective = "CAMERA"
            region.view_rotation = mathutils.Euler((tilt, 0.0, 0.0)).to_quaternion()
            region.view_distance = (1.7 if poly else 1.6) * extent
        else:
            region.view_perspective = "ORTHO"
            region.view_rotation = (1.0, 0.0, 0.0, 0.0)
            region.view_distance = 0.9 * extent
        for sidebar in area.regions:
            if sidebar.type == "UI":
                space.show_region_ui = True

prefs = bpy.context.preferences.addons[module].preferences
config = (
    Path(bpy.path.abspath(prefs.solver_project or ".")) / "configs" / f"{bundle_dir.stem}.yaml"
)
if prefs.solver_project and config.exists():
    scene.bioelectric_config = str(config)  # what the Simulate button will rerun

scene.frame_set(0)
bpy.ops.wm.save_mainfile()
print("DEMO", out)
