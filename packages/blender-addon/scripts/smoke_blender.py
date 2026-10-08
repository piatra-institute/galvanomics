"""Headless check inside Blender: import a bundle, step frames, compare, render.

Usage (the extension must already be installed in the active user resources):

    Blender -b --factory-startup --python-exit-code 1 -P scripts/smoke_blender.py -- \
        <bundle_dir> <out_dir> [module]

Writes ``<out_dir>/frame_<n>.png`` for the baseline, first post-wound and last frames, and
``<out_dir>/smoke.json`` with what was checked. Raises (non-zero exit) on any mismatch.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1 :]
bundle_dir, out_dir = Path(args[0]).resolve(), Path(args[1]).resolve()
module = args[2] if len(args) > 2 else "bl_ext.user_default.bioelectric_playback"
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.preferences.addon_enable(module=module)
core = importlib.import_module(f"{module}.core")
reader = importlib.import_module(f"{module}.reader")

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

result = bpy.ops.bioelectric.import_bundle(filepath=str(bundle_dir / "bundle.json"))
assert result == {"FINISHED"}, result
bundle = reader.open_bundle(bundle_dir)
tissue = bpy.data.objects[bundle.name]
quantity = tissue["tbundle_quantity"]
removed = bundle.removed()
_, offsets = bundle.polygons()
scene = bpy.context.scene

post = [i for i in range(bundle.n_frames) if i not in bundle.baseline_frames]
frames = sorted({0, *(post[:1]), bundle.n_frames // 2, bundle.n_frames - 1})
three_d = tissue["tbundle_role"] in ("tissue3d", "tissue_poly")
checked = []
for frame in frames:
    scene.frame_set(frame)
    mesh = tissue.data
    if tissue["tbundle_role"] == "tissue_poly":
        co = np.empty(3 * len(mesh.vertices), dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        vcell = np.empty(len(mesh.vertices), dtype=np.int32)
        mesh.attributes["tb_vcell"].data.foreach_get("value", vcell)
        centres = np.asarray(bundle.array("cell_centres"), np.float32) / core.UM_PER_UNIT
        at_centre = np.all(np.isclose(co.reshape(-1, 3), centres[vcell], atol=1e-4), axis=1)
        gone = removed & (frame in post)
        assert np.array_equal(at_centre, gone[vcell]), (
            "removed cells must collapse after the wound"
        )
        checked.append(
            {"frame": frame, "cells": int(bundle.n_cells), "collapsed": int(gone.sum())}
        )
        continue
    if three_d:
        n_faces = len(mesh.polygons)
        got = np.empty(n_faces, dtype=np.float32)
        mesh.attributes["tb_value"].data.foreach_get("value", got)
        face_cell = np.empty(n_faces, dtype=np.int32)
        face_domain = np.empty(n_faces, dtype=np.int32)
        mesh.attributes["tb_cell"].data.foreach_get("value", face_cell)
        mesh.attributes["tb_domain"].data.foreach_get("value", face_domain)
        apical, basolateral = core.domain_pair(quantity, set(bundle.quantities))
        a, b = bundle.frame(apical, frame), bundle.frame(basolateral, frame)
        want = np.where(face_domain == core.APICAL, a[face_cell], b[face_cell]).astype(np.float32)
    else:
        got = np.empty(bundle.n_cells, dtype=np.float32)
        mesh.attributes[quantity].data.foreach_get("value", got)
        want = bundle.frame(quantity, frame).astype(np.float32)
        face_cell = np.arange(bundle.n_cells)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    rgba = np.empty(4 * len(mesh.loops), dtype=np.float32)
    mesh.color_attributes["tbundle_rgba"].data.foreach_get("color", rgba)
    starts = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    face_rgba = rgba.reshape(-1, 4)[starts]
    if frame in post and removed.any():
        gone = removed[face_cell]
        np.testing.assert_allclose(face_rgba[gone], np.tile(core.REMOVED_RGBA, (gone.sum(), 1)))
    checked.append({"frame": frame, "faces": int(len(got)), "nan": int(np.isnan(got).sum())})

extra = [o.name for o in bpy.data.objects if o.get("tbundle_role") not in (None, "tissue")]
# Skin views: every frame must update the Geometry Nodes point sets without error.
for obj in bpy.data.objects:
    if obj.get("tbundle_role") in ("channels_gn", "ions"):
        for frame in (0, bundle.n_frames - 1):
            scene.frame_set(frame)
            assert len(obj.data.vertices) > 0, obj.name
# Top-down orthographic render with flat attribute colours.
verts, _ = bundle.polygons()
verts = np.asarray(verts)[:, :2]
lo, hi = verts.min(axis=0) / core.UM_PER_UNIT, verts.max(axis=0) / core.UM_PER_UNIT
camera_data = bpy.data.cameras.new("smoke_camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = float(max(hi - lo)) * 1.15
camera = bpy.data.objects.new("smoke_camera", camera_data)
camera.location = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), 50.0)
if three_d:  # an oblique view shows the prisms' sides
    camera_data.type = "PERSP"
    extent = float(max(hi - lo))
    camera.location = (
        float((lo[0] + hi[0]) / 2),
        float((lo[1] + hi[1]) / 2) - 1.4 * extent,
        1.3 * extent,
    )
    camera.rotation_euler = (float(np.arctan2(1.4, 1.3)), 0.0, 0.0)
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "VERTEX"
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = scene.render.resolution_y = 800
scene.render.image_settings.file_format = "PNG"
renders = []
for frame in sorted({0, *(post[:1]), bundle.n_frames - 1}):
    scene.frame_set(frame)
    scene.render.filepath = str(out_dir / f"frame_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)
    renders.append(scene.render.filepath)

(out_dir / "smoke.json").write_text(
    json.dumps(
        {
            "bundle": str(bundle_dir),
            "quantity": quantity,
            "checked": checked,
            "objects": extra,
            "renders": renders,
        },
        indent=2,
    )
    + "\n"
)
print("SMOKE OK", json.dumps(checked))
