"""Build objects from a bundle and keep their colours (and arrows) in step with the frame.

Roles, stored on each object as ``tbundle_role``:

- ``tissue``: 2D footprint, one polygon per cell.
- ``tissue3d``: extruded prisms; apical faces show the apical quantity, lateral and basal faces
  the basolateral one (``core.domain_pair``).
- ``grid``: one bath layer (or BETSE's environment grid), coloured per node.
- ``field``: arrows of the field in the apical bath.
- ``channels``: glyphs on the membrane domain a channel lives in, coloured by open fraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import core, skin_view
from .reader import Bundle

logger = logging.getLogger(__name__)

PROP_PATH = "tbundle_path"
PROP_QUANTITY = "tbundle_quantity"
PROP_VMIN = "tbundle_vmin"
PROP_VMAX = "tbundle_vmax"
PROP_OFFSET = "tbundle_frame_offset"
PROP_ROLE = "tbundle_role"
PROP_SCALE = "tbundle_arrow_scale"
COLOR_ATTR = "tbundle_rgba"
CELL_ATTR = "tb_cell"
DOMAIN_ATTR = "tb_domain"

_BUNDLES: dict[str, Bundle] = {}
_TOPOLOGY: dict[str, dict[str, np.ndarray]] = {}
_SKIN: dict[str, dict[str, np.ndarray]] = {}
_UPDATING = False


def bundle_for(obj: bpy.types.Object) -> Bundle | None:
    path = obj.get(PROP_PATH)
    if not path:
        return None
    resolved = str(Path(bpy.path.abspath(path)).resolve())
    if resolved not in _BUNDLES:
        try:
            _BUNDLES[resolved] = Bundle(resolved)
        except Exception:
            logger.exception("cannot open bundle %s", resolved)
            return None
    return _BUNDLES[resolved]


def cell_quantities(bundle: Bundle) -> list[str]:
    return [k for k, q in bundle.quantities.items() if q["location"] == "cell"]


def grid_quantities(bundle: Bundle, domain: str | None = None) -> list[str]:
    return [
        k
        for k, q in bundle.quantities.items()
        if q["location"] == "grid" and (domain is None or q.get("domain") == domain)
    ]


def _emission_material(name: str) -> bpy.types.Material:
    """Colour from the geometry's ``tbundle_rgba`` attribute, read by name so that it also
    works on Geometry Nodes output; lit (Principled) with some emission so colours stay
    readable in Material Preview."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if hasattr(material, "use_nodes") and not material.use_nodes:
        material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_type = "GEOMETRY"
    attribute.attribute_name = COLOR_ATTR
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
    links.new(attribute.outputs["Color"], shader.inputs["Emission Color"])
    shader.inputs["Emission Strength"].default_value = 0.6
    shader.inputs["Roughness"].default_value = 0.6
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _new_object(
    context: bpy.types.Context, name: str, xyz: np.ndarray, faces: list, domain: str
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(xyz).tolist(), [], faces)
    mesh.update(calc_edges=True)
    colors = mesh.color_attributes.new(COLOR_ATTR, "FLOAT_COLOR", domain)
    mesh.color_attributes.active_color = colors
    mesh.color_attributes.render_color_index = mesh.color_attributes.find(COLOR_ATTR)
    mesh.materials.append(_emission_material(f"{name}_material"))
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    return obj


def _tag(obj, role: str, path: str, quantity: str, vmin: float, vmax: float) -> None:
    obj[PROP_ROLE], obj[PROP_PATH], obj[PROP_QUANTITY] = role, path, quantity
    obj[PROP_OFFSET], obj[PROP_VMIN], obj[PROP_VMAX] = 0, vmin, vmax


def _int_attribute(mesh, name: str, domain: str, values: np.ndarray) -> None:
    attribute = mesh.attributes.new(name, "INT", domain)
    attribute.data.foreach_set("value", np.asarray(values, dtype=np.int32))


def _pair_range(bundle: Bundle, quantity: str) -> tuple[float, float]:
    a, b = core.domain_pair(quantity, set(bundle.quantities))
    data = np.concatenate((np.ravel(bundle.array(a)), np.ravel(bundle.array(b))))
    step = 5.0 if bundle.unit(quantity) == "mV" else 0.0
    if step:
        return core.default_range(data, step=step)
    finite = data[np.isfinite(data)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def _import_polyhedra(context, bundle: Bundle, stored: str) -> list:
    """Polyhedral cells (each face its own vertices), shrunk slightly toward their centroids
    so cell boundaries read like a section; non-cell anatomy from the bundle's context meshes."""
    verts = np.asarray(bundle.array("cell_verts", mmap=False))
    face_offsets = np.asarray(bundle.array("face_offsets", mmap=False))
    cell_offsets = np.asarray(bundle.array("cell_offsets", mmap=False))
    centres = np.asarray(bundle.array("cell_centres", mmap=False))
    face_cell = np.repeat(np.arange(bundle.n_cells), np.diff(cell_offsets))
    vertex_face = np.repeat(np.arange(len(face_offsets) - 1), np.diff(face_offsets))
    vertex_cell = face_cell[vertex_face]
    shrunk = centres[vertex_cell] + 0.94 * (verts - centres[vertex_cell])
    xyz = shrunk / core.UM_PER_UNIT
    faces = [list(range(int(a), int(b))) for a, b in zip(face_offsets[:-1], face_offsets[1:])]
    tissue = _new_object(context, bundle.name, xyz, faces, "CORNER")
    _int_attribute(tissue.data, CELL_ATTR, "FACE", face_cell)
    _int_attribute(tissue.data, "tb_vcell", "POINT", vertex_cell)
    rest = tissue.data.attributes.new("tb_rest", "FLOAT_VECTOR", "POINT")
    rest.data.foreach_set("vector", np.asarray(xyz, np.float32).ravel())
    names = cell_quantities(bundle)
    quantity = next((q for q in ("vmem", "layer") if q in names), names[0] if names else "")
    lo_hi = (
        (0.0, max(len(bundle.layers) - 1, 1))
        if quantity == "layer"
        else _pair_range(bundle, quantity)
    )
    _tag(tissue, "tissue_poly", stored, quantity, *lo_hi)
    created = [tissue]
    for key, mesh in bundle.context_meshes().items():
        obj = _new_object(
            context,
            f"{bundle.name}_{key}",
            mesh["verts"] / core.UM_PER_UNIT,
            [list(map(int, f)) for f in mesh["faces"]],
            "CORNER",
        )
        corners = sum(len(f) for f in mesh["faces"])
        _set_colors(obj.data, np.tile(np.asarray(mesh["color"], np.float32), (corners, 1)))
        obj[PROP_ROLE] = "context"
        created.append(obj)
    if "flux_na_face" in bundle.quantities:
        views = skin_view.create_skin_views(context, bundle, stored, _emission_material)
        created += views
    scene = context.scene
    scene.frame_start, scene.frame_end = 0, bundle.n_frames - 1
    for obj in created:
        update_object(obj, scene.frame_current)
    return created


def skin_cache(bundle: Bundle) -> dict[str, np.ndarray]:
    """Per-bundle arrays the skin views share: face geometry, face -> cell, cell y."""
    key = str(bundle.path)
    if key not in _SKIN:
        geo = skin_view.face_geometry(bundle)
        geo["cell_y"] = np.asarray(bundle.array("cell_centres", mmap=False))[:, 1]
        inert = bundle.manifest["geometry"].get("cell_inert")
        geo["inert"] = (
            np.asarray(bundle.array("cell_inert", mmap=False), bool)
            if inert
            else np.zeros(bundle.n_cells, bool)
        )
        _SKIN[key] = geo
    return _SKIN[key]


def _collapse_like(bundle: Bundle, index: int, cells: np.ndarray, cut_y, cell_y) -> np.ndarray:
    """Which of ``cells`` are hidden: removed by the wound, or in front of the section plane."""
    hidden = wounded_now(bundle, index)[cells]
    if cut_y is not None:
        hidden = hidden | (np.asarray(cell_y)[cells] < cut_y)
    return hidden


def layer_colors(bundle: Bundle) -> np.ndarray:
    colors = np.asarray(bundle.layer_colors or [(0.8, 0.8, 0.8)], np.float32)
    return np.concatenate((colors, np.ones((len(colors), 1), np.float32)), axis=1)


def _collapse_cells(obj, bundle: Bundle, index: int) -> None:
    """Removed cells shrink to their centroids after the wound (they leave the tissue), and
    so do cells in front of the section plane when it cuts away."""
    topo = _topology(obj)
    cut_y = skin_view.cutaway_y_um(bpy.context.scene)
    cell_y = np.asarray(bundle.array("cell_centres", mmap=False))[:, 1]
    gone = _collapse_like(bundle, index, np.arange(bundle.n_cells), cut_y, cell_y)
    key = f"{int(gone.sum())}:{None if cut_y is None else round(cut_y, 1)}"
    if obj.get("tb_collapsed", -1) == key:
        return
    mesh = obj.data
    rest = np.empty(3 * len(mesh.vertices), dtype=np.float32)
    mesh.attributes["tb_rest"].data.foreach_get("vector", rest)
    co = rest.reshape(-1, 3)
    vc = topo["tb_vcell"]
    hide = gone[vc]
    if hide.any():
        centres = np.asarray(bundle.array("cell_centres"), np.float32) / core.UM_PER_UNIT
        co[hide] = centres[vc[hide]]
    mesh.vertices.foreach_set("co", co.ravel())
    obj["tb_collapsed"] = key


def import_bundle(context: bpy.types.Context, path: str) -> list[bpy.types.Object]:
    bundle = Bundle(path)
    _BUNDLES[str(Path(path).resolve())] = bundle
    stored = bpy.path.relpath(path) if bpy.data.filepath else path
    if bundle.kind == "polyhedra":
        return _import_polyhedra(context, bundle, stored)
    verts, offsets = bundle.polygons()
    names = cell_quantities(bundle)
    height = bundle.extrusion_height_um
    created = []

    if height:
        xyz, faces, face_cell, face_domain = core.prism_mesh(verts, offsets, height)
        tissue = _new_object(context, bundle.name, xyz, faces, "CORNER")
        _int_attribute(tissue.data, CELL_ATTR, "FACE", face_cell)
        _int_attribute(tissue.data, DOMAIN_ATTR, "FACE", face_domain)
        vertex_cell = np.repeat(np.arange(bundle.n_cells), np.diff(offsets))
        _int_attribute(
            tissue.data, "tb_vcell", "POINT", np.concatenate((vertex_cell, vertex_cell))
        )
        _int_attribute(
            tissue.data,
            "tb_top",
            "POINT",
            np.concatenate((np.zeros(len(verts)), np.ones(len(verts)))),
        )
        quantity = "vmem_apical" if "vmem_apical" in names else (names[0] if names else "")
        _tag(tissue, "tissue3d", stored, quantity, *_pair_range(bundle, quantity))
    else:
        tissue = _new_object(
            context,
            bundle.name,
            core.to_blender_xyz(verts),
            core.faces_from_offsets(offsets),
            "CORNER",
        )
        quantity = "vmem" if "vmem" in names else (names[0] if names else "")
        lo_hi = core.default_range(bundle.array(quantity)) if quantity else (0.0, 1.0)
        _tag(tissue, "tissue", stored, quantity, *lo_hi)
    created.append(tissue)

    if bundle.grid_shape is not None:
        ny, nx = bundle.grid_shape
        points = np.column_stack(
            (np.ravel(bundle.array("grid_x")), np.ravel(bundle.array("grid_y")))
        )
        layers: dict[tuple, list[str]] = {}
        for key, q in bundle.quantities.items():
            if q["location"] == "grid":
                layers.setdefault((q.get("domain"), q.get("z_um")), []).append(key)
        for (domain, z_um), keys in layers.items():
            label = (domain or "env").replace("bath_", "")
            z_bu = -0.05 if z_um is None else float(z_um) / core.UM_PER_UNIT
            plane = _new_object(
                context,
                f"{bundle.name}_{label}",
                core.to_blender_xyz(points, z=z_bu),
                core.grid_faces(ny, nx),
                "POINT",
            )
            pick = next((k for k in keys if k.startswith(("phi_", "venv"))), keys[0])
            _tag(
                plane,
                "grid",
                stored,
                pick,
                *core.default_range(bundle.array(pick), step=0.5, symmetric=True),
            )
            if height and domain == "bath_apical":  # it sits above the cells; shown on request
                plane.hide_set(True)
                plane.hide_render = True
            created.append(plane)

        if height and {"efield_apical_x", "efield_apical_y"} <= set(bundle.quantities):
            stride = max(1, int(np.ceil(np.sqrt(nx * ny / 500))))
            sel = np.zeros((ny, nx), bool)
            sel[::stride, ::stride] = True
            arrows = _new_object(
                context,
                f"{bundle.name}_field",
                np.zeros((3 * sel.sum(), 3)),
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(int(sel.sum()))],
                "CORNER",
            )
            _int_attribute(arrows.data, "tb_node", "FACE", np.flatnonzero(sel.ravel()))
            mags = [
                np.hypot(bundle.array("efield_apical_x")[t], bundle.array("efield_apical_y")[t])
                for t in range(bundle.n_frames)
            ]
            peak = float(max(np.nanmax(m) for m in mags)) or 1.0
            _tag(arrows, "field", stored, "efield_apical", 0.0, peak)
            arrows[PROP_SCALE] = 0.9 * stride * float(np.diff(bundle.array("grid_x")[0, :2])[0])
            created.append(arrows)

    if height:
        rng = np.random.default_rng(0)
        _, faces, face_cell, face_domain = core.prism_mesh(verts, offsets, height)
        xyz = core.prism_mesh(verts, offsets, height)[0]
        for key in names:
            if not key.startswith("open_"):
                continue
            domain = bundle.domain(key)
            wanted = core.APICAL if domain == "apical" else core.LATERAL
            ids = np.flatnonzero(face_domain == wanted)
            points, owner = core.sample_on_faces(
                xyz, faces, ids, 3 if wanted == core.APICAL else 1, rng
            )
            normal_lift = np.zeros_like(points)
            normal_lift[:, 2] = 0.02 if wanted == core.APICAL else 0.0
            gxyz, gfaces = core.tetra_glyphs(points + normal_lift, 0.035)
            glyphs = _new_object(context, f"{bundle.name}_{key}", gxyz, gfaces, "CORNER")
            _int_attribute(glyphs.data, CELL_ATTR, "FACE", np.repeat(face_cell[owner], 4))
            _int_attribute(glyphs.data, "tb_vcell", "POINT", np.repeat(face_cell[owner], 4))
            rest = glyphs.data.attributes.new("tb_rest", "FLOAT_VECTOR", "POINT")
            rest.data.foreach_set("vector", np.asarray(gxyz, np.float32).ravel())
            _tag(glyphs, "channels", stored, key, 0.0, 1.0)
            created.append(glyphs)

    scene = context.scene
    scene.frame_start, scene.frame_end = 0, bundle.n_frames - 1
    for obj in created:
        update_object(obj, scene.frame_current)
    return created


def _topology(obj) -> dict[str, np.ndarray]:
    key = obj.data.name
    if key not in _TOPOLOGY:
        mesh, topo = obj.data, {}
        for name in (CELL_ATTR, DOMAIN_ATTR, "tb_node", "tb_vcell", "tb_top", "tb_face"):
            attribute = mesh.attributes.get(name)
            if attribute is not None:
                values = np.empty(len(attribute.data), dtype=np.int32)
                attribute.data.foreach_get("value", values)
                topo[name] = values
        counts = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", counts)
        topo["corners"] = counts
        _TOPOLOGY[key] = topo
    return _TOPOLOGY[key]


def wounded_now(bundle: Bundle, index: int) -> np.ndarray:
    """Cells that are gone at this frame (none in baseline frames)."""
    removed = bundle.removed()
    return removed & (index not in bundle.baseline_frames)


def _collapse_prisms(obj, bundle: Bundle, index: int) -> None:
    """Removed cells sink to the basal plane after the wound, leaving the wound visible."""
    topo = _topology(obj)
    gone = wounded_now(bundle, index)
    key = int(gone.sum())
    if obj.get("tb_collapsed", -1) == key:
        return
    mesh = obj.data
    co = np.empty(3 * len(mesh.vertices), dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    top = topo["tb_top"] == 1
    co[top, 2] = bundle.extrusion_height_um / core.UM_PER_UNIT
    co[top & gone[topo["tb_vcell"]], 2] = 0.0
    mesh.vertices.foreach_set("co", co.ravel())
    obj["tb_collapsed"] = key


def _collapse_glyphs(obj, bundle: Bundle, index: int) -> None:
    topo = _topology(obj)
    gone = wounded_now(bundle, index)
    key = int(gone.sum())
    if obj.get("tb_collapsed", -1) == key:
        return
    mesh = obj.data
    rest = np.empty(3 * len(mesh.vertices), dtype=np.float32)
    mesh.attributes["tb_rest"].data.foreach_get("vector", rest)
    co = rest.reshape(-1, 4, 3).copy()
    hide = gone[topo["tb_vcell"][::4]]
    co[hide] = co[hide].mean(axis=1, keepdims=True)
    mesh.vertices.foreach_set("co", co.ravel())
    obj["tb_collapsed"] = key


def _set_colors(mesh, rgba: np.ndarray) -> None:
    mesh.color_attributes[COLOR_ATTR].data.foreach_set("color", np.ascontiguousarray(rgba).ravel())


def update_object(obj: bpy.types.Object, scene_frame: int) -> None:
    bundle = bundle_for(obj)
    quantity = obj.get(PROP_QUANTITY, "")
    if bundle is None or not quantity:
        return
    index = core.frame_index(scene_frame, int(obj.get(PROP_OFFSET, 0)), bundle.n_frames)
    vmin, vmax = float(obj[PROP_VMIN]), float(obj[PROP_VMAX])
    role, mesh = obj.get(PROP_ROLE), obj.data

    if role == "tissue":
        values = bundle.frame(quantity, index).ravel()
        attribute = mesh.attributes.get(quantity) or mesh.attributes.new(quantity, "FLOAT", "FACE")
        attribute.data.foreach_set("value", values.astype(np.float32))
        _, offsets = bundle.polygons()
        _set_colors(mesh, core.corner_colors(core.colormap(values, vmin, vmax), offsets))
    elif role == "tissue3d":
        topo = _topology(obj)
        apical, basolateral = core.domain_pair(quantity, set(bundle.quantities))
        a = bundle.frame(apical, index).ravel()
        b = bundle.frame(basolateral, index).ravel()
        face_values = np.where(
            topo[DOMAIN_ATTR] == core.APICAL, a[topo[CELL_ATTR]], b[topo[CELL_ATTR]]
        )
        attribute = mesh.attributes.get("tb_value") or mesh.attributes.new(
            "tb_value", "FLOAT", "FACE"
        )
        attribute.data.foreach_set("value", face_values.astype(np.float32))
        rgba = core.colormap(face_values, vmin, vmax)
        _set_colors(mesh, np.repeat(rgba, topo["corners"], axis=0))
        _collapse_prisms(obj, bundle, index)
    elif role == "tissue_poly":
        topo = _topology(obj)
        values = bundle.frame(quantity, index).ravel()
        palette = layer_colors(bundle)
        layer = np.asarray(bundle.array("cell_layer", mmap=False))
        layer_rgba = palette[np.clip(layer, 0, len(palette) - 1)]
        if bundle.location(quantity) == "membrane":  # one value per face (e.g. chan_g_*)
            rgba = core.colormap(values, vmin, vmax)
            no_value = np.isnan(values)
            rgba[no_value] = layer_rgba[topo[CELL_ATTR]][no_value]
            _set_colors(mesh, np.repeat(rgba, topo["corners"], axis=0))
            _collapse_cells(obj, bundle, index)
            return
        if quantity == "layer":
            rgba = layer_rgba
            rgba[np.isnan(values)] = core.REMOVED_RGBA
        else:
            rgba = core.colormap(values, vmin, vmax)
            inert = skin_cache(bundle)["inert"]
            rgba[inert] = layer_rgba[inert]  # corneocytes carry no voltage: shown by layer
        _set_colors(mesh, np.repeat(rgba[topo[CELL_ATTR]], topo["corners"], axis=0))
        _collapse_cells(obj, bundle, index)
    elif role == "grid":
        values = bundle.frame(quantity, index).ravel()
        _set_colors(mesh, core.colormap(values, vmin, vmax))
    elif role == "channels_gn":
        cache = skin_cache(bundle)
        topo = dict(_topology(obj))
        topo["face_cell"], topo["cell_y"] = cache["face_cell"], cache["cell_y"]
        if "tb_face" not in topo:
            attribute = mesh.attributes["tb_face"]
            values = np.empty(len(attribute.data), dtype=np.int32)
            attribute.data.foreach_get("value", values)
            _TOPOLOGY[obj.data.name]["tb_face"] = values
            topo["tb_face"] = values
        skin_view.update_channels(
            obj, bundle, index, topo, skin_view.cutaway_y_um(bpy.context.scene)
        )
        return
    elif role == "ions":
        skin_view.update_ions(
            obj,
            bundle,
            index,
            scene_frame,
            skin_cache(bundle),
            skin_view.cutaway_y_um(bpy.context.scene),
        )
        return
    elif role == skin_view.SECTION_ROLE:
        tissue = next(
            (
                o
                for o in bpy.context.scene.objects
                if o.get(PROP_ROLE) == "tissue_poly" and bundle_for(o) is bundle
            ),
            None,
        )
        cache = skin_cache(bundle)
        if "planes" not in cache:
            cache["planes"] = skin_view.cell_planes(bundle)
        palette = layer_colors(bundle)
        layer = np.asarray(bundle.array("cell_layer", mmap=False))
        skin_view.update_section(
            obj,
            bundle,
            index,
            tissue,
            cache["planes"],
            cache["inert"],
            palette[np.clip(layer, 0, len(palette) - 1)],
        )
        return
    elif role == "field":
        topo = _topology(obj)
        nodes = topo["tb_node"]
        ex = bundle.frame("efield_apical_x", index).ravel()[nodes]
        ey = bundle.frame("efield_apical_y", index).ravel()[nodes]
        points = np.column_stack(
            (np.ravel(bundle.array("grid_x"))[nodes], np.ravel(bundle.array("grid_y"))[nodes])
        )
        z = (bundle.extrusion_height_um + 1.0) / core.UM_PER_UNIT
        scale = float(obj[PROP_SCALE]) / max(vmax, 1e-30)
        xyz = core.arrow_triangles(points, ex, ey, scale, z, 2.0)
        mesh.vertices.foreach_set("co", xyz.ravel())
        rgba = core.colormap(np.hypot(ex, ey), 0.0, vmax)
        _set_colors(mesh, np.repeat(rgba, 3, axis=0))
    elif role == "channels":
        topo = _topology(obj)
        _collapse_glyphs(obj, bundle, index)
        values = bundle.frame(quantity, index).ravel()[topo[CELL_ATTR]]
        hue = next(
            (
                h
                for k, h in core.CHANNEL_HUES.items()
                if f"_{k}" in quantity or quantity.startswith(f"open_{k}")
            ),
            (0.9, 0.9, 0.9),
        )
        _set_colors(mesh, np.repeat(core.glyph_colors(values, hue), 3, axis=0))
    mesh.update()


def set_quantity(obj: bpy.types.Object, quantity: str) -> None:
    bundle = bundle_for(obj)
    if bundle is None:
        return
    obj[PROP_QUANTITY] = quantity
    role = obj.get(PROP_ROLE)
    if role == "tissue_poly" and quantity == "layer":
        obj[PROP_VMIN], obj[PROP_VMAX] = 0.0, float(max(len(bundle.layers) - 1, 1))
    elif role in ("tissue3d", "tissue_poly"):
        obj[PROP_VMIN], obj[PROP_VMAX] = _pair_range(bundle, quantity)
    else:
        grid = role == "grid"
        obj[PROP_VMIN], obj[PROP_VMAX] = core.default_range(
            bundle.array(quantity), step=0.5 if grid else 5.0, symmetric=grid
        )
    update_object(obj, bpy.context.scene.frame_current)


@persistent
def on_frame_change(scene: bpy.types.Scene, _depsgraph: object = None) -> None:
    global _UPDATING
    was = _UPDATING
    _UPDATING = True
    try:
        for obj in scene.objects:
            if PROP_PATH in obj:
                update_object(obj, scene.frame_current)
        from . import nanoscope

        if nanoscope.is_nanoscope(scene):
            nanoscope.update_scene(scene)
        for other in bpy.data.scenes:  # nanoscopes follow their tissue's timeline
            if other is not scene and other.get("tb_follow") == scene.name:
                other.frame_current = scene.frame_current
                nanoscope.update_scene(other)
    finally:
        _UPDATING = was


@persistent
def on_depsgraph(scene: bpy.types.Scene, depsgraph) -> None:
    """When the section plane is moved, recolour it and redo the cutaway."""
    global _UPDATING
    if _UPDATING:
        return
    section = skin_view.section_object(scene)
    if section is None:
        return
    key = tuple(round(v, 4) for v in section.matrix_world.translation) + (
        section.visible_get(),
        bool(section.get("tb_cutaway", False)),
        bool(section.get("tb_show_cells", False)),
    )
    if section.get("tb_last") == str(key):
        return
    section["tb_last"] = str(key)
    _UPDATING = True
    try:
        on_frame_change(scene)
    finally:
        _UPDATING = False


@persistent
def on_load(_filepath: object = None) -> None:
    _BUNDLES.clear()
    _TOPOLOGY.clear()
    _SKIN.clear()
    on_frame_change(bpy.context.scene)


def register() -> None:
    bpy.app.handlers.frame_change_post.append(on_frame_change)
    bpy.app.handlers.load_post.append(on_load)
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph)


def unregister() -> None:
    for handlers, fn in (
        (bpy.app.handlers.frame_change_post, on_frame_change),
        (bpy.app.handlers.load_post, on_load),
        (bpy.app.handlers.depsgraph_update_post, on_depsgraph),
    ):
        if fn in handlers:
            handlers.remove(fn)
    _BUNDLES.clear()
    _TOPOLOGY.clear()
