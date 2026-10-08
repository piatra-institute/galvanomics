"""Headless check of the nanoscope: import a skin bundle, open a nanoscope on the apical face with
the most Na channels, check that every drawn molecule is an instance point and that the open
states follow the frame, render the close-up camera.

    Blender -b --factory-startup --python-exit-code 1 \
        -P scripts/smoke_nanoscope.py -- <bundle> <png>
"""

from __future__ import annotations

import importlib
import json
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1 :]
bundle_dir, png = args[0], args[1]
module = "bl_ext.user_default.bioelectric_playback"
if module not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module=module)
bpy.ops.wm.read_homefile(use_empty=True)
assert bpy.ops.bioelectric.import_bundle(filepath=f"{bundle_dir}/bundle.json") == {"FINISHED"}
playback = importlib.import_module(f"{module}.playback")
nanoscope = importlib.import_module(f"{module}.nanoscope")
tissue = next(o for o in bpy.context.scene.objects if o.get("tbundle_role") == "tissue_poly")
bundle = playback.bundle_for(tissue)
domain = np.asarray(bundle.array("face_domain"))[0]
g = np.asarray(bundle.array("chan_g_naleak"))[0]
apical = np.flatnonzero(domain == 1)
face = int(apical[np.argmax(g[apical])])
tissue.data.polygons.active = face
bpy.context.view_layer.objects.active = tissue
bpy.context.scene.bioelectric_nano_size = 1000.0  # 1 µm²: a couple of ENaC expected
assert bpy.ops.bioelectric.open_nanoscope() == {"FINISHED"}
scene = next(s for s in bpy.data.scenes if s.get("tb_follow"))
root = nanoscope.root_of(scene)
summary = json.loads(root["tb_summary"])
for label, m in summary["molecules"].items():
    obj = scene.objects[f"Nanoscope {label}"]
    assert len(obj.data.vertices) == m["drawn"], (label, len(obj.data.vertices), m)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    assert evaluated.modifiers[0].node_group is not None
assert summary["molecules"], summary
for frame in (0, bundle.n_frames - 1):  # the tissue timeline drives the nanoscope
    bpy.context.window.scene = next(s for s in bpy.data.scenes if not s.get("tb_follow"))
    bpy.context.scene.frame_set(frame)
    assert scene.frame_current == frame
    assert root["tb_frame_index"] == min(frame, bundle.n_frames - 1)
lines = nanoscope.legend_lines(scene)
assert any("ions/s" in line for line in lines), lines
close = next(o for o in scene.objects if o.name.startswith("Nanoscope close-up camera"))
scene.camera = close
scene.render.resolution_x, scene.render.resolution_y = 640, 400
scene.render.filepath = png
with bpy.context.temp_override(scene=scene):
    bpy.ops.render.render(write_still=True, scene=scene.name)
print("NANOSCOPE OK", json.dumps({k: v["drawn"] for k, v in summary["molecules"].items()}))
