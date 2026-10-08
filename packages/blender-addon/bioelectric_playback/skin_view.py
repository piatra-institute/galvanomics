"""Views for skin simulations (polyhedral bundles with a 3D extracellular volume).

- Channel glyphs: one point per channel site on the membrane faces where that channel lives,
  instanced by a Geometry Nodes modifier ("Bioelectric Glyphs": glyph size is a modifier input).
  Each glyph's size and brightness follow the computed flux through its face.
- Ions: Na+ and K+ particles streaming across the membrane faces, assigned to faces in
  proportion to the computed flux (inward for Na+ through apical channels, outward for K+).
  Illustrative of the fluxes, not individual ions.
- Section: a plane through the tissue coloured by the extracellular potential; cells, glyphs and
  ions in front of it are cut away so the plane can be moved through the tissue (G, then Y).
- Probe: an empty whose position reads the extracellular field, like the vibrating probe.
- Scale bar: 50 µm.
"""

from __future__ import annotations

import bpy
import numpy as np

from . import core

GLYPH_GROUP = "Bioelectric Glyphs"
ROLE = "tbundle_role"
SECTION_ROLE, PROBE_ROLE = "section", "probe"
CHANNEL_KINDS = (
    # key in the bundle, label, face domain (1 apical, 2 basolateral), sign of "active", hue
    ("flux_na_face", "Na channels (apical)", 1, +1.0, (1.0, 0.55, 0.10)),
    ("flux_k_face", "K channels (basolateral)", 2, -1.0, (0.15, 0.45, 1.00)),
    ("pump_face", "Na/K pumps (basolateral)", 2, +1.0, (0.20, 0.85, 0.35)),
)
ION_KINDS = (
    ("flux_na_face", "Na+", 1, +1.0, (1.0, 0.6, 0.15)),
    ("flux_k_face", "K+", 2, -1.0, (0.3, 0.6, 1.0)),
)
IONS_PER_KIND = 1500


def glyph_node_group() -> bpy.types.NodeTree:
    group = bpy.data.node_groups.get(GLYPH_GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(GLYPH_GROUP, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    size = group.interface.new_socket("Glyph size", in_out="INPUT", socket_type="NodeSocketFloat")
    size.default_value, size.min_value = 0.06, 0.0
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    g_in, g_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Subdivisions"].default_value = 1
    scale = nodes.new("GeometryNodeInputNamedAttribute")
    scale.data_type = "FLOAT"
    scale.inputs["Name"].default_value = "tb_scale"
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    real = nodes.new("GeometryNodeRealizeInstances")
    links.new(g_in.outputs["Glyph size"], ico.inputs["Radius"])
    links.new(g_in.outputs["Geometry"], inst.inputs["Points"])
    links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    links.new(scale.outputs["Attribute"], inst.inputs["Scale"])
    links.new(inst.outputs["Instances"], real.inputs["Geometry"])
    links.new(real.outputs["Geometry"], g_out.inputs["Geometry"])
    return group


def _point_object(context, name: str, points: np.ndarray, size: float, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(points).tolist(), [], [])
    colors = mesh.color_attributes.new("tbundle_rgba", "FLOAT_COLOR", "POINT")
    mesh.color_attributes.active_color = colors
    mesh.color_attributes.render_color_index = mesh.color_attributes.find("tbundle_rgba")
    mesh.attributes.new("tb_scale", "FLOAT", "POINT")
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    # Geometry Nodes output takes its materials from the instanced sphere (none), so the
    # material must be linked to the object, not to the mesh.
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = material
    group = glyph_node_group().copy()  # one per object, so each keeps its own glyph size
    group.name = f"{GLYPH_GROUP} ({name})"
    group.interface.items_tree["Glyph size"].default_value = size
    modifier = obj.modifiers.new("Glyphs", "NODES")
    modifier.node_group = group
    return obj


def face_geometry(bundle) -> dict[str, np.ndarray]:
    """Shrunk face vertices (as drawn), face centroids and outward normals, in Blender units."""
    verts = np.asarray(bundle.array("cell_verts", mmap=False))
    face_offsets = np.asarray(bundle.array("face_offsets", mmap=False))
    cell_offsets = np.asarray(bundle.array("cell_offsets", mmap=False))
    centres = np.asarray(bundle.array("cell_centres", mmap=False))
    face_cell = np.repeat(np.arange(bundle.n_cells), np.diff(cell_offsets))
    vertex_face = np.repeat(np.arange(len(face_offsets) - 1), np.diff(face_offsets))
    vc = face_cell[vertex_face]
    xyz = (centres[vc] + 0.94 * (verts - centres[vc])) / core.UM_PER_UNIT
    n_faces = len(face_offsets) - 1
    centroid = (
        np.stack(
            [np.bincount(vertex_face, weights=xyz[:, d], minlength=n_faces) for d in range(3)],
            axis=1,
        )
        / np.diff(face_offsets)[:, None]
    )
    nxt = np.arange(len(xyz)) + 1
    nxt[face_offsets[1:] - 1] = face_offsets[:-1]
    cross = np.cross(xyz, xyz[nxt])
    normal = np.stack(
        [np.bincount(vertex_face, weights=cross[:, d], minlength=n_faces) for d in range(3)],
        axis=1,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-30)[:, None]
    faces = [list(range(int(a), int(b))) for a, b in zip(face_offsets[:-1], face_offsets[1:])]
    return {
        "xyz": xyz,
        "faces": faces,
        "centroid": centroid,
        "normal": normal,
        "face_cell": face_cell,
        "cell_centre": centres / core.UM_PER_UNIT,
    }


def create_skin_views(context, bundle, stored: str, material_for) -> list[bpy.types.Object]:
    """All skin-simulation objects. ``material_for(name)`` makes an emission material."""
    from .playback import PROP_PATH, PROP_QUANTITY, PROP_VMAX, PROP_VMIN

    geo = face_geometry(bundle)
    domain = np.nan_to_num(bundle.frame("face_domain", 0), nan=0.0)
    rng = np.random.default_rng(1)
    created = []
    for key, label, dom, sign, hue in CHANNEL_KINDS:
        if key not in bundle.quantities:
            continue
        ids = np.flatnonzero(domain == dom)
        per_face = 3 if dom == 1 else 1
        points, owner = core.sample_on_faces(geo["xyz"], geo["faces"], ids, per_face, rng)
        points = points + geo["normal"][owner] * 0.02
        obj = _point_object(
            context,
            f"{bundle.name} {label}",
            points,
            0.16 if dom == 1 else 0.09,
            material_for(f"{bundle.name}_glyphs"),
        )
        face_attr = obj.data.attributes.new("tb_face", "INT", "POINT")
        face_attr.data.foreach_set("value", owner.astype(np.int32))
        data = np.asarray(bundle.array(key))[:, ids] * sign
        obj[ROLE], obj[PROP_PATH], obj[PROP_QUANTITY] = "channels_gn", stored, key
        obj[PROP_VMIN], obj[PROP_VMAX] = 0.0, float(np.nanmax(np.abs(data)) or 1.0)
        obj["tb_sign"], obj["tb_hue"] = sign, list(hue)
        created.append(obj)
    for key, label, dom, sign, hue in ION_KINDS:
        if key not in bundle.quantities:
            continue
        obj = _point_object(
            context,
            f"{bundle.name} {label} ions",
            np.zeros((IONS_PER_KIND, 3)),
            0.07,
            material_for(f"{bundle.name}_ions"),
        )
        obj[ROLE], obj[PROP_PATH], obj[PROP_QUANTITY] = "ions", stored, key
        obj["tb_sign"], obj["tb_hue"], obj["tb_domain"] = sign, list(hue), dom
        obj[PROP_VMIN], obj[PROP_VMAX] = 0.0, 1.0
        created.append(obj)
    if bundle.volume_shape is not None and "phi_ecs" in bundle.quantities:
        created.append(_section(context, bundle, stored, material_for(f"{bundle.name}_section")))
        created.append(_probe(context, bundle))
    created += _scale_bar(context, bundle, material_for(f"{bundle.name}_scale"))
    return created


def _section(context, bundle, stored: str, material):
    from .playback import PROP_PATH, PROP_QUANTITY, PROP_VMAX, PROP_VMIN

    xs, ys, zs = bundle.volume_coordinates()
    verts = np.asarray(bundle.array("cell_verts", mmap=False))
    top = float(verts[:, 2].max())
    # Columns: the solver's own x nodes (fine near the wound on a graded grid) plus an even set.
    cols = np.unique(np.round(np.concatenate((xs, np.linspace(xs[0], xs[-1], 120))), 6))
    nx, nz = len(cols), 90
    gx, gz = np.meshgrid(cols, np.linspace(zs[0], top, nz))
    local = np.column_stack((gx.ravel(), np.zeros(gx.size), gz.ravel())) / core.UM_PER_UNIT
    faces = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append([a, a + 1, a + nx + 1, a + nx])
    mesh = bpy.data.meshes.new(f"{bundle.name} section")
    mesh.from_pydata(local.tolist(), [], faces)
    colors = mesh.color_attributes.new("tbundle_rgba", "FLOAT_COLOR", "POINT")
    mesh.color_attributes.active_color = colors
    mesh.materials.append(material)
    obj = bpy.data.objects.new(f"{bundle.name} section (move with G Y)", mesh)
    obj.location.y = float(0.5 * (ys[0] + ys[-1])) / core.UM_PER_UNIT
    obj.lock_location[0] = obj.lock_location[2] = True
    context.collection.objects.link(obj)
    phi = np.asarray(bundle.array("phi_ecs"))
    post = phi[1:] if len(phi) > 1 else phi
    bound = float(np.nanpercentile(np.abs(post), 99)) or 1.0  # post-wound detail; frame 0 clips
    obj[ROLE], obj[PROP_PATH], obj[PROP_QUANTITY] = SECTION_ROLE, stored, "phi_ecs"
    obj[PROP_VMIN], obj[PROP_VMAX] = -bound, bound
    obj["tb_cutaway"] = True
    obj["tb_show_cells"] = False
    obj.hide_viewport = True  # shown from the panel; it covers the cells while visible
    return obj


def wound_x_um(bundle) -> float | None:
    """Where the incision is centred: written by the solver since 2026-10-08; for older bundles,
    the median of the removed cells."""
    if "wound_x_um" in bundle.constants:
        return float(bundle.constants["wound_x_um"])
    removed = bundle.removed()
    if removed.any():
        return float(np.median(np.asarray(bundle.array("cell_centres"))[removed, 0]))
    return None


def _probe(context, bundle):
    xs, ys, zs = bundle.volume_coordinates()
    probe = bpy.data.objects.new("Vibrating probe (move me)", None)
    probe.empty_display_type = "SPHERE"
    probe.empty_display_size = 0.25
    wound_x = wound_x_um(bundle)
    if wound_x is None:
        wound_x = float(xs.mean())
    tj = bundle.constants.get("tight_junction_z_um", float(zs[-1]) - 3)
    probe.location = (
        min(wound_x + 35.0, float(xs[-1]) - 5) / core.UM_PER_UNIT,
        float(ys.mean()) / core.UM_PER_UNIT,
        (tj - 8.0) / core.UM_PER_UNIT,
    )
    probe[ROLE] = PROBE_ROLE
    context.collection.objects.link(probe)
    return probe


def _scale_bar(context, bundle, material) -> list:
    verts = np.asarray(bundle.array("cell_verts", mmap=False))
    xs, ys, zs = (
        bundle.volume_coordinates() if bundle.volume_shape else (None, None, [verts[:, 2].min()])
    )
    z0 = float(np.min(zs)) / core.UM_PER_UNIT - 0.8
    um = 100 if float(np.ptp(verts[:, 0])) > 400 else 50
    length = um / core.UM_PER_UNIT
    bar_v = [(0, -0.3, z0), (length, -0.3, z0), (length, -0.3, z0 + 0.3), (0, -0.3, z0 + 0.3)]
    mesh = bpy.data.meshes.new(f"Scale bar {um} um")
    mesh.from_pydata(bar_v, [], [[0, 1, 2, 3]])
    colors = mesh.color_attributes.new("tbundle_rgba", "FLOAT_COLOR", "CORNER")
    colors.data.foreach_set("color", np.tile([0.95, 0.95, 0.95, 1.0], 4).astype(np.float32))
    mesh.materials.append(material)
    bar = bpy.data.objects.new(f"Scale bar {um} um", mesh)
    context.collection.objects.link(bar)
    text_data = bpy.data.curves.new("Scale bar label", "FONT")
    text_data.body = f"{um} µm"
    text_data.size = 1.0
    label = bpy.data.objects.new("Scale bar label", text_data)
    label.location = (0.0, -0.3, z0 - 1.3)
    label.rotation_euler = (np.pi / 2, 0.0, 0.0)
    context.collection.objects.link(label)
    return [bar, label]


# ---------------------------------------------------------------------------- per-frame updates
def section_object(scene):
    return next((o for o in scene.objects if o.get(ROLE) == SECTION_ROLE), None)


def cutaway_y_um(scene) -> float | None:
    """Cells in front of (y below) the visible section plane are cut away."""
    section = section_object(scene)
    if section is None or not section.get("tb_cutaway", False) or not section.visible_get():
        return None
    return float(section.matrix_world.translation.y) * core.UM_PER_UNIT


def cell_planes(bundle) -> dict[str, np.ndarray]:
    """Each cell's face planes (outward normal and a point, µm), padded to a common count."""
    verts = np.asarray(bundle.array("cell_verts", mmap=False))
    face_offsets = np.asarray(bundle.array("face_offsets", mmap=False))
    cell_offsets = np.asarray(bundle.array("cell_offsets", mmap=False))
    n_faces = len(face_offsets) - 1
    vertex_face = np.repeat(np.arange(n_faces), np.diff(face_offsets))
    nxt = np.arange(len(verts)) + 1
    nxt[face_offsets[1:] - 1] = face_offsets[:-1]
    cross = np.cross(verts, verts[nxt])
    normal = np.stack(
        [np.bincount(vertex_face, weights=cross[:, d], minlength=n_faces) for d in range(3)],
        axis=1,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-30)[:, None]
    point = (
        np.stack(
            [np.bincount(vertex_face, weights=verts[:, d], minlength=n_faces) for d in range(3)],
            axis=1,
        )
        / np.diff(face_offsets)[:, None]
    )
    counts = np.diff(cell_offsets)
    width = int(counts.max())
    slots = np.arange(width)[None, :] < counts[:, None]
    index = np.where(slots, cell_offsets[:-1, None] + np.arange(width)[None, :], 0)
    return {
        "normal": normal[index],
        "point": point[index],
        "valid": slots,
        "centre": np.asarray(bundle.array("cell_centres", mmap=False)),
    }


def containing_cell(planes: dict, points: np.ndarray, candidates: int = 6) -> np.ndarray:
    """The cell containing each point (-1 if none): nearest centroids, then a face-plane test
    (cells are convex up to the gentle papilla deformation, hence the small tolerance)."""
    centre = planes["centre"]
    out = np.full(len(points), -1)
    c2 = (centre**2).sum(axis=1)
    for start in range(0, len(points), 1000):
        chunk = points[start : start + 1000]
        d2 = c2[None, :] - 2.0 * chunk @ centre.T  # + |p|^2, the same for every candidate
        near = np.argpartition(d2, min(candidates, len(centre) - 1), axis=1)[:, :candidates]
        for j in range(near.shape[1]):
            c = near[:, j]
            side = np.einsum(
                "pfk,pfk->pf", planes["normal"][c], chunk[:, None, :] - planes["point"][c]
            )
            inside = np.all((side <= 0.15) | ~planes["valid"][c], axis=1)
            hit = inside & (out[start : start + len(chunk)] < 0)
            out[start : start + len(chunk)][hit] = c[hit]
    return out


_MEMBERSHIP: dict = {}


def update_section(
    obj, bundle, index: int, tissue=None, planes=None, inert=None, layer_rgba=None
) -> None:
    """A virtual histology section: cells cut by the plane show their membrane voltage, the
    space between them the extracellular potential."""
    mesh = obj.data
    co = np.empty(3 * len(mesh.vertices), dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    world = np.asarray(obj.matrix_world)
    pts = (co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]) * core.UM_PER_UNIT
    ecs = core.trilinear_sample(
        bundle.volume_coordinates(), np.asarray(bundle.array("phi_ecs")[index]), pts
    )
    rgba = core.colormap(
        ecs, float(obj["tbundle_vmin"]), float(obj["tbundle_vmax"]), nan_rgba=(0.2, 0.2, 0.2, 1.0)
    )
    if planes is not None and tissue is not None and obj.get("tb_show_cells", False):
        key = (obj.name, round(float(world[1, 3]), 4))
        if key not in _MEMBERSHIP:
            _MEMBERSHIP.clear()
            _MEMBERSHIP[key] = containing_cell(planes, pts)
        cell = _MEMBERSHIP[key]
        inside = cell >= 0
        quantity = tissue.get("tbundle_quantity", "vmem")
        values = np.asarray(bundle.array(quantity)[index], dtype=float)
        gone = bundle.removed() & (index not in bundle.baseline_frames)
        cell_rgba = core.colormap(
            values, float(tissue["tbundle_vmin"]), float(tissue["tbundle_vmax"])
        )
        if inert is not None and layer_rgba is not None:
            cell_rgba[inert] = layer_rgba[inert]
        show = inside & ~gone[np.maximum(cell, 0)]
        rgba[show] = cell_rgba[cell[show]]
    mesh.color_attributes["tbundle_rgba"].data.foreach_set("color", rgba.ravel())
    mesh.update()


def update_channels(obj, bundle, index: int, topo: dict, cut_y: float | None) -> None:
    from .playback import _collapse_like

    faces = topo["tb_face"]
    values = np.asarray(bundle.array(obj["tbundle_quantity"])[index])[faces] * float(
        obj["tb_sign"]
    )
    intensity = np.clip(np.nan_to_num(values, nan=0.0) / float(obj["tbundle_vmax"]), 0.0, 1.0)
    rgba = core.glyph_colors(intensity, tuple(obj["tb_hue"]))
    scale = (0.35 + 0.65 * intensity).astype(np.float32)
    hidden = np.isnan(values) | _collapse_like(
        bundle, index, topo["face_cell"][faces], cut_y, topo["cell_y"]
    )
    scale[hidden] = 0.0
    mesh = obj.data
    mesh.color_attributes["tbundle_rgba"].data.foreach_set("color", rgba.ravel())
    mesh.attributes["tb_scale"].data.foreach_set("value", scale)
    mesh.update()


def update_ions(obj, bundle, index: int, frame: int, geo: dict, cut_y: float | None) -> None:
    from .playback import _collapse_like

    domain = np.nan_to_num(np.asarray(bundle.array("face_domain")[index]), nan=0.0)
    values = np.asarray(bundle.array(obj["tbundle_quantity"])[index]) * float(obj["tb_sign"])
    weights = np.where(domain == int(obj["tb_domain"]), values, 0.0)
    gone = _collapse_like(
        bundle, index, geo["face_cell"], cut_y, geo["cell_centre"][:, 1] * core.UM_PER_UNIT
    )
    weights[gone] = 0.0
    count = len(obj.data.vertices)
    face = core.assign_particles(weights, count)
    phase = core.particle_offsets(count, frame)
    valid = face >= 0
    f = np.where(valid, face, 0)
    travel = 0.35  # Blender units each way across the membrane (3.5 µm)
    inward = float(obj["tb_sign"]) > 0 and obj["tbundle_quantity"] == "flux_na_face"
    along = (0.5 - phase) if inward else (phase - 0.5)
    pos = geo["centroid"][f] + geo["normal"][f] * (2 * travel * along)[:, None]
    pos[~valid] = 0.0
    mesh = obj.data
    mesh.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    hue = np.asarray(obj["tb_hue"], np.float32)
    rgba = np.tile(np.append(hue, 1.0).astype(np.float32), (count, 1))
    mesh.color_attributes["tbundle_rgba"].data.foreach_set("color", rgba.ravel())
    scale = np.where(valid, 1.0, 0.0).astype(np.float32)
    mesh.attributes["tb_scale"].data.foreach_set("value", scale)
    mesh.update()


def probe_series(bundle, probe) -> tuple[np.ndarray, np.ndarray]:
    """Field magnitude (mV/mm) and potential (mV) at the probe, for every frame."""
    p = np.asarray(probe.matrix_world.translation)[None, :] * core.UM_PER_UNIT
    axes = bundle.volume_coordinates()
    ex, ey, ez, phi = (
        np.array(
            [
                core.trilinear_sample(axes, np.asarray(bundle.array(k)[t]), p)[0]
                for t in range(bundle.n_frames)
            ]
        )
        for k in ("efield_x", "efield_y", "efield_z", "phi_ecs")
    )
    return np.sqrt(ex**2 + ey**2 + ez**2), phi


def battery_mv(bundle, index: int) -> float | None:
    """Mean extracellular potential below minus above the tight-junction band, away from the
    wound: the skin battery."""
    if "phi_ecs" not in bundle.quantities:
        return None
    xs, ys, zs = bundle.volume_coordinates()
    tj = bundle.constants.get("tight_junction_z_um")
    if tj is None:
        return None
    phi = np.asarray(bundle.array("phi_ecs")[index])
    region = np.asarray(bundle.array("region")[index])
    Z, _, X = np.meshgrid(zs, ys, xs, indexing="ij")
    wound_x = wound_x_um(bundle)
    if wound_x is None:
        wound_x = -1e9
    far = np.abs(X - wound_x) > 50.0
    inner = (region == 1) & (Z < tj - 2) & (Z > tj - 12) & far
    outer = (region == 1) & (Z > tj + 0.1) & far
    if not inner.any() or not outer.any():
        return None
    return float(phi[inner].mean() - phi[outer].mean())
