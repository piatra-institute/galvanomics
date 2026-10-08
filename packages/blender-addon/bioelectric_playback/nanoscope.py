"""The nanoscope: one membrane face at nanometre scale, in a scene of its own.

It draws a square patch of the face: the lipid bilayer, the channel and pump molecules (meshes
built from Protein Data Bank structures, ``assets/channels``) and, frame by frame, which
channels are open and the ions crossing them. Nothing here is physics. The counts come from the
solver's conductance of the face (``chan_g_*``) divided by sourced single-channel conductances,
the pumps from the pump rate divided by a sourced turnover (derived); positions are a Poisson or
clustered pattern (assumed); open or closed is drawn every frame at the solver's open fraction
(``chan_open_*``; leak channels are always open). Sources: ``docs/nanoscope.md``.

Scene unit: 1 Blender unit = 10 nm, so molecules stay at sensible coordinates.
"""

from __future__ import annotations

import json
import math
import zlib
from functools import cache
from pathlib import Path

import bpy
import numpy as np

from . import core

ASSETS = Path(__file__).parent / "assets" / "channels"
NM_PER_UNIT = 10.0
ROLE = "nanoscope"
PROP_PATH = "tbundle_path"
COLOURS = {
    "naleak": (1.0, 0.55, 0.15, 1.0),
    "kleak": (0.35, 0.6, 1.0, 1.0),
    "kv3p4": (0.55, 0.85, 1.0, 1.0),
    "clleak": (0.6, 1.0, 0.5, 1.0),
    "pump": (0.35, 0.8, 0.4, 1.0),
}
CLOSED_DIM = 0.45  # closed channels are drawn darker


@cache
def index() -> dict:
    return json.loads((ASSETS / "index.json").read_text())


@cache
def structure_info(key: str) -> dict:
    return json.loads((ASSETS / f"{key}.json").read_text())


def _material(name: str, rgba, alpha: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.55
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        try:
            mat.surface_render_method = "BLENDED"
        except (AttributeError, TypeError):
            pass
    mat.diffuse_color = rgba
    return mat


def structure_object(key: str, rgba, collection) -> bpy.types.Object:
    """A hidden object holding one structure's mesh (nm → scene units), for instancing."""
    name = f"tb structure {key} {rgba[:3]}"
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    data = np.load(ASSETS / f"{key}.npz")
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata((data["high_verts"] / NM_PER_UNIT).tolist(), [], data["high_faces"].tolist())
    mesh.shade_smooth()
    mesh.materials.append(_material(name, rgba))
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.hide_render = obj.hide_viewport = True
    return obj


def _instancer_group(name: str, open_obj, closed_obj) -> bpy.types.NodeTree:
    """Points → instances: open points get ``open_obj``, closed ones ``closed_obj``; each point
    turned about z by its ``tb_rot`` attribute. Instances are not realized (memory)."""
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, link = group.nodes, group.links.new
    g_in, g_out = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    is_open = n.new("GeometryNodeInputNamedAttribute")
    is_open.data_type = "BOOLEAN"
    is_open.inputs["Name"].default_value = "tb_open"
    rot = n.new("GeometryNodeInputNamedAttribute")
    rot.data_type = "FLOAT_VECTOR"
    rot.inputs["Name"].default_value = "tb_rot"
    closed = n.new("FunctionNodeBooleanMath")
    closed.operation = "NOT"
    link(is_open.outputs["Attribute"], closed.inputs[0])
    join = n.new("GeometryNodeJoinGeometry")
    for source, selection in (
        (open_obj, is_open.outputs["Attribute"]),
        (closed_obj, closed.outputs[0]),
    ):
        info = n.new("GeometryNodeObjectInfo")
        info.inputs["Object"].default_value = source
        inst = n.new("GeometryNodeInstanceOnPoints")
        link(g_in.outputs["Geometry"], inst.inputs["Points"])
        link(info.outputs["Geometry"], inst.inputs["Instance"])
        link(selection, inst.inputs["Selection"])
        link(rot.outputs["Attribute"], inst.inputs["Rotation"])
        link(inst.outputs["Instances"], join.inputs["Geometry"])
    link(join.outputs["Geometry"], g_out.inputs["Geometry"])
    return group


def _points(name: str, xy_nm: np.ndarray, z_nm: float, collection) -> bpy.types.Object:
    pts = np.column_stack((xy_nm, np.full(len(xy_nm), z_nm))) / NM_PER_UNIT
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(pts.tolist(), [], [])
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def face_patch_data(bundle, face: int) -> dict:
    """What the solver says about one face: its cell and layer, side, conductance per channel
    type if all were open (frame 0), open fraction per frame for gated channels, pump rate,
    membrane voltage per frame."""
    face_offsets = np.asarray(bundle.array("face_offsets"))
    cell_offsets = np.asarray(bundle.array("cell_offsets"))
    face_cell = np.repeat(np.arange(bundle.n_cells), np.diff(cell_offsets))
    cell = int(face_cell[face])
    layer = int(np.asarray(bundle.array("cell_layer"))[cell])
    domain = float(np.asarray(bundle.array("face_domain"))[0][face])
    if not np.isfinite(domain) or domain == 0:
        raise ValueError(f"face {face} is not living membrane (cell {cell}, layer {layer})")
    side = "apical" if domain == 1 else "basolateral"
    g, series = {}, {}
    for q in bundle.quantities:
        if q.startswith("chan_g_"):
            value = float(np.asarray(bundle.array(q))[0][face])
            if np.isfinite(value) and value > 0:
                g[q[7:]] = value
    for name in g:
        key = f"chan_open_{name}"
        series[name] = (
            np.nan_to_num(np.asarray(bundle.array(key))[:, face], nan=0.0).tolist()
            if key in bundle.quantities
            else [1.0] * bundle.n_frames
        )
    pump_alpha = float(bundle.constants.get(f"pump_alpha_l{layer}_{side}", 0.0))
    vmem = np.asarray(bundle.array("vmem_face"))[:, face].tolist()
    flux = {
        ion: np.nan_to_num(np.asarray(bundle.array(f"flux_{ion}_face"))[:, face]).tolist()
        for ion in ("na", "k", "cl")
        if f"flux_{ion}_face" in bundle.quantities
    }
    del face_offsets
    return {
        "face": face,
        "cell": cell,
        "layer": layer,
        "layer_name": (bundle.layers[layer] if layer < len(bundle.layers) else str(layer)),
        "side": side,
        "g_full": g,
        "open_series": series,
        "pump_alpha": pump_alpha,
        "vmem_mV": vmem,
        "flux": flux,
    }


def open_nanoscope(
    context,
    tissue,
    bundle,
    face: int,
    size_nm: float = 500.0,
    placement: str = "uniform",
    seed: int = 1,
) -> bpy.types.Scene:
    info = face_patch_data(bundle, face)
    idx = index()
    unitary = {k: v["unitary_pS"] for k, v in idx["channels"].items()}
    area_um2 = (size_nm / 1000.0) ** 2
    expected = core.expected_counts(
        info["g_full"], unitary, info["pump_alpha"], idx["pump"]["turnover_per_s"], area_um2
    )
    rng = np.random.default_rng(seed)
    tissue_scene = context.scene
    scene = bpy.data.scenes.new(f"Nanoscope: face {face}")
    scene["tb_follow"] = tissue_scene.name
    scene.frame_start, scene.frame_end = tissue_scene.frame_start, tissue_scene.frame_end
    scene.render.fps = tissue_scene.render.fps
    scene.frame_current = tissue_scene.frame_current
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = NM_PER_UNIT * 1e-9
    try:
        scene.unit_settings.length_unit = "NANOMETERS"
    except TypeError:  # older unit lists stop at micrometres
        scene.unit_settings.length_unit = "ADAPTIVE"
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("Nanoscope world")
    world.color = (0.02, 0.02, 0.03)
    scene.world = world
    coll = scene.collection
    hidden = bpy.data.collections.new("Nanoscope structures")
    coll.children.link(hidden)

    half = size_nm / 2.0
    root = bpy.data.objects.new("Nanoscope", None)
    coll.objects.link(root)
    root[ROLE] = "root"
    root[PROP_PATH] = tissue[PROP_PATH]
    root["tb_seed"], root["tb_face"], root["tb_size_nm"] = seed, face, size_nm

    # The bilayer: a 4 nm slab (hydrocarbon core plus head groups), cytoplasm below (z < 0).
    bpy_mesh = bpy.data.meshes.new("Lipid bilayer")
    s, t = half / NM_PER_UNIT, 2.0 / NM_PER_UNIT
    verts = [(x, y, z) for z in (-t, t) for y in (-s, s) for x in (-s, s)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    bpy_mesh.from_pydata(verts, [], faces)
    bpy_mesh.materials.append(_material("tb lipid", (0.93, 0.83, 0.55, 1.0), alpha=0.55))
    bilayer = bpy.data.objects.new("Lipid bilayer (4 nm)", bpy_mesh)
    coll.objects.link(bilayer)
    if size_nm <= 200:  # head groups as beads, one per 0.65 nm² per leaflet
        per_leaflet = int(size_nm**2 / 0.65)
        heads = np.vstack(
            [
                np.column_stack(
                    (rng.uniform(-half, half, (per_leaflet, 2)), np.full(per_leaflet, z))
                )
                for z in (-2.0, 2.0)
            ]
        )
        beads = _points("Lipid head groups", heads[:, :2], 0.0, coll)
        beads.data.vertices.foreach_set("co", (heads / NM_PER_UNIT).astype(np.float32).ravel())
        sphere = bpy.data.meshes.new("tb head bead")
        _ico(sphere, 0.4 / NM_PER_UNIT)
        sphere.materials.append(_material("tb head", (0.85, 0.75, 0.5, 1.0)))
        bead_obj = bpy.data.objects.new("tb head bead", sphere)
        hidden.objects.link(bead_obj)
        bead_obj.hide_render = bead_obj.hide_viewport = True
        group = _instancer_group("TB lipid heads", bead_obj, bead_obj)
        mod = beads.modifiers.new("Instances", "NODES")
        mod.node_group = group
        _set_attr(beads, "tb_open", np.ones(len(heads), bool))
        _set_attr(beads, "tb_rot", np.zeros((len(heads), 3), np.float32))

    summary = {
        "face": face,
        "cell": info["cell"],
        "layer": info["layer_name"],
        "side": info["side"],
        "size_nm": size_nm,
        "placement": placement,
        "molecules": {},
    }
    for name in sorted(info["g_full"]):
        if name not in idx["channels"]:
            continue
        spec = idx["channels"][name]
        xy = core.sample_positions(expected[name], size_nm, rng, placement) - half
        rgba = COLOURS.get(name, (0.8, 0.8, 0.8, 1.0))
        dim = (*[c * CLOSED_DIM for c in rgba[:3]], 1.0)
        obj = _points(f"Nanoscope {spec['label']}", xy, 0.0, coll)
        obj[ROLE], obj["tb_channel"] = "channels", name
        obj["tb_open_series"] = info["open_series"][name]
        mod = obj.modifiers.new("Instances", "NODES")
        mod.node_group = _instancer_group(
            f"TB nanoscope {name}",
            structure_object(spec["open"], rgba, hidden),
            structure_object(spec["closed"], dim, hidden),
        )
        _set_attr(
            obj,
            "tb_rot",
            np.column_stack((np.zeros((len(xy), 2)), rng.uniform(0, 2 * math.pi, len(xy)))).astype(
                np.float32
            ),
        )
        _set_attr(obj, "tb_open", np.ones(len(xy), bool))
        summary["molecules"][spec["label"]] = {
            "expected": expected[name],
            "drawn": int(len(xy)),
            "pdb": structure_info(spec["open"])["pdb_id"],
            "unitary_pS": spec["unitary_pS"],
            "conductance_S_per_m2": info["g_full"][name],
        }
    if expected.get("pump", 0) > 0:
        pump = idx["pump"]
        xy = core.sample_positions(expected["pump"], size_nm, rng, placement) - half
        obj = _points(f"Nanoscope {pump['label']}", xy, 0.0, coll)
        obj[ROLE] = "pumps"
        mod = obj.modifiers.new("Instances", "NODES")
        pump_obj = structure_object(pump["structure"], COLOURS["pump"], hidden)
        mod.node_group = _instancer_group("TB nanoscope pump", pump_obj, pump_obj)
        _set_attr(
            obj,
            "tb_rot",
            np.column_stack((np.zeros((len(xy), 2)), rng.uniform(0, 2 * math.pi, len(xy)))).astype(
                np.float32
            ),
        )
        _set_attr(obj, "tb_open", np.ones(len(xy), bool))
        summary["molecules"][pump["label"]] = {
            "expected": expected["pump"],
            "drawn": int(len(xy)),
            "pdb": structure_info(pump["structure"])["pdb_id"],
            "turnover_per_s": pump["turnover_per_s"],
            "pump_rate_mol_m2_s": info["pump_alpha"],
        }
    root["tb_summary"] = json.dumps(summary)
    root["tb_vmem"] = info["vmem_mV"]
    root["tb_area_m2"] = (size_nm * 1e-9) ** 2
    for ion, series in info["flux"].items():
        root[f"tb_flux_{ion}"] = series
    for ion, rgba in (
        ("na", (1.0, 0.85, 0.2, 1.0)),
        ("k", (0.75, 0.45, 1.0, 1.0)),
        ("cl", (0.4, 1.0, 0.45, 1.0)),
    ):
        obj = _points(f"Nanoscope {ion.upper()}+ ions", np.zeros((0, 2)), 0.0, coll)
        obj[ROLE], obj["tb_ion"] = "ions", ion
        sphere = bpy.data.meshes.new(f"tb ion {ion}")
        _ico(sphere, 0.35 / NM_PER_UNIT)  # a hydrated Na+ or K+ is about 0.7 nm across
        mat = _material(f"tb ion {ion}", rgba)
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = 2.0
        sphere.materials.append(mat)
        ion_obj = bpy.data.objects.new(f"tb ion {ion}", sphere)
        hidden.objects.link(ion_obj)
        ion_obj.hide_render = ion_obj.hide_viewport = True
        mod = obj.modifiers.new("Instances", "NODES")
        mod.node_group = _instancer_group(f"TB nanoscope ions {ion}", ion_obj, ion_obj)
    _scale_bars(coll, half)
    _camera_and_lights(scene, size_nm)
    _close_up_camera(scene)
    marker = bpy.data.objects.new(f"Nanoscope patch (face {face})", None)
    marker.empty_display_type = "CIRCLE"
    marker.empty_display_size = 0.15
    centroid = _face_centroid_units(bundle, face)
    marker.location = centroid
    tissue_scene.collection.objects.link(marker)
    marker[ROLE] = "marker"
    update_scene(scene)
    return scene


def _face_centroid_units(bundle, face: int):
    face_offsets = np.asarray(bundle.array("face_offsets"))
    verts = np.asarray(bundle.array("cell_verts"))[face_offsets[face] : face_offsets[face + 1]]
    return tuple(float(v) for v in verts.mean(axis=0) / core.UM_PER_UNIT)


def _ico(mesh, radius: float) -> None:
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius)
    bm.to_mesh(mesh)
    bm.free()


def _set_attr(obj, name: str, values: np.ndarray) -> None:
    mesh = obj.data
    kind = {np.dtype(bool): "BOOLEAN"}.get(
        values.dtype, "FLOAT_VECTOR" if values.ndim == 2 else "FLOAT"
    )
    attr = mesh.attributes.get(name) or mesh.attributes.new(name, kind, "POINT")
    if len(values):
        attr.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", values.ravel())
    mesh.update()


def _scale_bars(coll, half: float) -> None:
    mat = _material("tb bar", (0.95, 0.95, 0.95, 1.0))
    y0 = half + 30.0  # beyond the far edge, where the camera sees it
    for i, length in enumerate((10.0, 100.0)):
        if length > 2 * half:
            continue
        mesh = bpy.data.meshes.new(f"Scale bar {length:g} nm")
        x0, y = -half, y0 - 12.0 * i
        quad = [(x0, y, 0), (x0 + length, y, 0), (x0 + length, y + 2.0, 0), (x0, y + 2.0, 0)]
        mesh.from_pydata([tuple(c / NM_PER_UNIT for c in p) for p in quad], [], [(0, 1, 2, 3)])
        mesh.materials.append(mat)
        bar = bpy.data.objects.new(f"Scale bar {length:g} nm", mesh)
        coll.objects.link(bar)
        text = bpy.data.curves.new(f"{length:g} nm", "FONT")
        text.body = f"{length:g} nm"
        text.size = 6.0 / NM_PER_UNIT
        label = bpy.data.objects.new(f"Scale bar label {length:g} nm", text)
        label.location = ((x0 + length + 3.0) / NM_PER_UNIT, y / NM_PER_UNIT, 0.0)
        label.data.materials.append(mat)
        coll.objects.link(label)


def _camera_and_lights(scene, size_nm: float) -> None:
    cam_data = bpy.data.cameras.new("Nanoscope camera")
    cam = bpy.data.objects.new("Nanoscope camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    d = 0.9 * size_nm / NM_PER_UNIT
    cam.location = (0.0, -1.2 * d, 0.75 * d)
    cam.rotation_euler = (math.radians(58), 0.0, 0.0)
    cam_data.lens = 35
    cam_data.clip_start = 0.05 / NM_PER_UNIT
    cam_data.clip_end = 20.0 * d
    for name, rot, energy in (("Key", (0.8, 0.2, 0.7), 3.0), ("Fill", (1.7, -0.4, -1.4), 1.0)):
        light = bpy.data.objects.new(f"Nanoscope {name}", bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = rot
        scene.collection.objects.link(light)


def _close_up_camera(scene) -> None:
    """A second camera, 40 nm across, on the molecule nearest the patch centre."""
    best, where = None, None
    for obj in scene.objects:
        if obj.get(ROLE) in ("channels", "pumps") and len(obj.data.vertices):
            co = np.zeros(3 * len(obj.data.vertices))
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            i = int(np.argmin(np.linalg.norm(co[:, :2], axis=1)))
            d = float(np.linalg.norm(co[i, :2]))
            if (
                best is None
                or (obj.get(ROLE) == "channels") > (best.get(ROLE) == "channels")
                or (obj.get(ROLE) == best.get(ROLE) and d < where[1])
            ):
                best, where = obj, (co[i], d)
    if best is None:
        return
    data = bpy.data.cameras.new("Nanoscope close-up")
    cam = bpy.data.objects.new("Nanoscope close-up camera (40 nm)", data)
    scene.collection.objects.link(cam)
    target = where[0]
    d = 40.0 / NM_PER_UNIT
    cam.location = (target[0], target[1] - 1.3 * d, target[2] + 0.55 * d)
    cam.rotation_euler = (math.radians(70), 0.0, 0.0)
    data.lens = 50
    data.clip_start = 0.02 / NM_PER_UNIT
    data.clip_end = 50 * d


def _update_ions(scene, root, index_: int) -> None:
    """Beads moving through each open channel, in the direction of the solver's net flux on
    this face; the legend gives the rate (each bead is a cue, not one ion)."""
    open_xy = {"na": [], "k": [], "cl": []}
    for obj in scene.objects:
        if obj.get(ROLE) != "channels" or not len(obj.data.vertices):
            continue
        ion = {"naleak": "na", "clleak": "cl"}.get(obj["tb_channel"], "k")
        co = np.zeros(3 * len(obj.data.vertices))
        obj.data.vertices.foreach_get("co", co)
        is_open = np.zeros(len(obj.data.vertices), bool)
        obj.data.attributes["tb_open"].data.foreach_get("value", is_open)
        open_xy[ion].append(co.reshape(-1, 3)[is_open, :2])
    for obj in scene.objects:
        if obj.get(ROLE) != "ions":
            continue
        ion = obj["tb_ion"]
        xy = np.vstack(open_xy[ion]) if open_xy[ion] else np.zeros((0, 2))
        series = list(root.get(f"tb_flux_{ion}", []))
        inward = (series[min(index_, len(series) - 1)] if series else 0.0) > 0
        per_channel = 6
        phase = (np.arange(per_channel) / per_channel + 0.17 * index_) % 1.0
        z_nm = (8.0 - 16.0 * phase) if inward else (-8.0 + 16.0 * phase)
        pts = np.column_stack(
            (np.repeat(xy, per_channel, axis=0), np.tile(z_nm / NM_PER_UNIT, len(xy)))
        )
        mesh = obj.data
        mesh.clear_geometry()
        mesh.vertices.add(len(pts))
        if len(pts):
            mesh.vertices.foreach_set("co", pts.astype(np.float32).ravel())
        _set_attr(obj, "tb_open", np.ones(len(pts), bool))
        _set_attr(obj, "tb_rot", np.zeros((len(pts), 3), np.float32))


def is_nanoscope(scene) -> bool:
    return any(o.get(ROLE) == "root" for o in scene.objects)


def root_of(scene):
    return next((o for o in scene.objects if o.get(ROLE) == "root"), None)


def update_scene(scene) -> None:
    """Redraw which channels are open at the scene's frame."""
    root = root_of(scene)
    if root is None:
        return
    from .playback import bundle_for

    bundle = bundle_for(root)
    if bundle is None:
        return
    index_ = core.frame_index(scene.frame_current, 0, bundle.n_frames)
    seed = int(root.get("tb_seed", 1))
    for obj in scene.objects:
        if obj.get(ROLE) != "channels":
            continue
        series = list(obj["tb_open_series"])
        fraction = float(series[min(index_, len(series) - 1)])
        n = len(obj.data.vertices)
        stream = seed + zlib.crc32(obj["tb_channel"].encode()) % 100_000  # stable per type
        _set_attr(obj, "tb_open", core.sample_open(fraction, n, stream, index_))
    _update_ions(scene, root, index_)
    root["tb_frame_index"] = index_


def _expected_open(scene, summary: dict, ion: str, i: int) -> float:
    """Expected number of open channels of an ion's types in the patch (counts are Poisson
    draws, so the rate per channel uses the expectation)."""
    total = 0.0
    for obj in scene.objects:
        if obj.get(ROLE) != "channels":
            continue
        if {"naleak": "na", "clleak": "cl"}.get(obj["tb_channel"], "k") != ion:
            continue
        label = obj.name.removeprefix("Nanoscope ")
        series = list(obj["tb_open_series"])
        total += summary["molecules"][label]["expected"] * float(series[min(i, len(series) - 1)])
    return total


def _open_count(scene, ion: str) -> int:
    total = 0
    for obj in scene.objects:
        if obj.get(ROLE) != "channels" or not len(obj.data.vertices):
            continue
        if {"naleak": "na", "clleak": "cl"}.get(obj["tb_channel"], "k") != ion:
            continue
        values = np.zeros(len(obj.data.vertices), bool)
        obj.data.attributes["tb_open"].data.foreach_get("value", values)
        total += int(values.sum())
    return total


def legend_lines(scene) -> list[str]:
    root = root_of(scene)
    if root is None:
        return []
    summary = json.loads(root["tb_summary"])
    i = int(root.get("tb_frame_index", 0))
    vmem = list(root.get("tb_vmem", []))
    lines = [
        f"Nanoscope: face {summary['face']} of cell {summary['cell']} ({summary['layer']}, "
        f"{summary['side']}), patch {summary['size_nm']:g} x {summary['size_nm']:g} nm",
        f"Membrane voltage now: {vmem[i]:.1f} mV" if vmem else "",
    ]
    for label, m in summary["molecules"].items():
        open_now = ""
        obj = next((o for o in scene.objects if o.name == f"Nanoscope {label}"), None)
        if obj is not None and "tb_open" in obj.data.attributes and obj.get(ROLE) == "channels":
            values = np.zeros(len(obj.data.vertices), bool)
            obj.data.attributes["tb_open"].data.foreach_get("value", values)
            open_now = f", {int(values.sum())} open"
        lines.append(
            f"{label} (PDB {m['pdb']}): {m['drawn']} drawn, {m['expected']:.2f} expected{open_now}"
        )
    area = float(root.get("tb_area_m2", 0.0))
    for ion, label in (("na", "Na+"), ("k", "K+"), ("cl", "Cl-")):
        series = list(root.get(f"tb_flux_{ion}", []))
        if not series or area == 0:
            continue
        flux = series[min(i, len(series) - 1)]
        rate = abs(flux) * core.AVOGADRO * area
        expected_open = _expected_open(scene, summary, ion, i)
        per_channel = rate / expected_open if expected_open > 0 else 0.0
        per = (
            f", {per_channel:.3g} per open channel ({per_channel * 1.602e-7:.2g} pA)"
            if expected_open > 0
            else ""
        )
        lines.append(
            f"{label} {'into' if flux > 0 else 'out of'} the cell through this patch: "
            f"{rate:.3g} ions/s{per} (solver's flux; moving beads are a cue, not single ions)"
        )
    lines.append(
        "Counts derived (conductance / single-channel conductance; pump rate / turnover); "
        f"positions assumed ({summary['placement']}); open or closed drawn at the solver's "
        "open fraction. Sources: docs/nanoscope.md"
    )
    return [line for line in lines if line]
