"""Procedural stratified epidermis over dermis: real 3D cell polyhedra, papillae, an incision.

Cells are Voronoi regions of seeds placed layer by layer (one seed per cell). A layer's
lateral and vertical seed spacing sets its cell shape: columnar basal cells, polyhedral
spinous cells, flattened granular cells, thin wide corneocytes. Each region is computed
exactly as an intersection of half-spaces (bisector planes with its Delaunay neighbours and
the domain box), then a smooth vertical deformation gives the dermal-epidermal junction its
papillae without changing any cell's topology. Dimensions are parameters; their sources are
in ``docs/skin.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

logger = logging.getLogger(__name__)

BASEMENT, SURFACE, SIDE = 1, 2, 3  # face_boundary codes; 0 = shared with another cell


class LayerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    rows: int = Field(ge=1)
    thickness_um: float = Field(gt=0)
    lateral_spacing_um: float = Field(gt=0)
    jitter: float = Field(0.25, ge=0, le=0.45, description="Fraction of the lateral spacing")
    row_sheets: bool = Field(
        False,
        description="Tessellate each row as its own flat sheet (flattened, stacked cells: "
        "corneocytes); otherwise rows interdigitate as 3D polyhedra",
    )
    packing: Literal["rows", "kelvin"] = Field(
        "rows",
        description="rows: a jittered hexagonal lattice per row. kelvin: a body-centred lattice "
        "(square per row, alternate rows offset by half a cell) tessellated where it is "
        "regular, so cells are flattened tetrakaidecahedra (Kelvin cells), as granular "
        "cells are (Yokouchi 2016); lateral spacing keeps the same cell area per row",
    )
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)

    @model_validator(mode="after")
    def _kelvin_is_3d(self) -> LayerSpec:
        if self.packing == "kelvin" and self.row_sheets:
            raise ValueError("kelvin packing tessellates the layer in 3D; row_sheets must be off")
        return self

    def square_spacing(self) -> float:
        """In-plane spacing of the kelvin lattice: a square cell of the hexagonal cell's area."""
        return self.lateral_spacing_um * np.sqrt(np.sqrt(3) / 2)


class WoundSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field("incision", description="incision (a V-shaped cut along y) or none")
    x_um: float = 80.0
    top_width_um: float = 40.0
    bottom_width_um: float = 16.0
    depth_into_dermis_um: float = 20.0


class SkinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Defaults: human dorsal forearm. Sources and verdicts in docs/skin.md.
    width_x_um: float = 160.0
    seed_width_x_um: float | None = Field(
        None,
        description="Lay the seed lattice out for a block this wide and keep the seeds inside "
        "width_x_um: cells away from the new wall are then identical to the wider block's",
    )
    width_y_um: float = 80.0
    dermis_depth_um: float = 60.0
    papilla_amplitude_um: float = 18.0
    papilla_wavelength_um: float = 160.0
    papilla_offset_x_um: float = Field(
        0.0, description="Shift of the papilla pattern along x (a trough sits at offset + λ/2)"
    )
    seed: int = 7
    layers: tuple[LayerSpec, ...] = (
        LayerSpec(
            name="stratum basale",
            rows=1,
            thickness_um=10.0,
            lateral_spacing_um=9.5,
            color=(0.55, 0.20, 0.45),
        ),
        LayerSpec(
            name="stratum spinosum",
            rows=4,
            thickness_um=37.6,
            lateral_spacing_um=13.5,
            color=(0.85, 0.45, 0.55),
        ),
        LayerSpec(
            name="stratum granulosum",
            rows=3,
            thickness_um=9.0,
            lateral_spacing_um=28.6,
            jitter=0.2,
            packing="kelvin",
            color=(0.55, 0.35, 0.75),
        ),
        LayerSpec(
            name="stratum corneum",
            rows=15,
            thickness_um=18.3,
            lateral_spacing_um=36.4,
            jitter=0.15,
            row_sheets=True,
            color=(0.92, 0.85, 0.65),
        ),
    )
    wound: WoundSpec = WoundSpec()


@dataclass
class Skin:
    """Polyhedral cells (faces with their own vertices, outward-ordered) and context."""

    spec: SkinSpec
    face_verts: np.ndarray  # [V, 3] µm
    face_offsets: np.ndarray  # [F + 1]
    cell_offsets: np.ndarray  # [N + 1] into faces
    face_neighbour_cell: np.ndarray  # [F] neighbour cell or -1
    face_boundary: np.ndarray  # [F] 0 shared, BASEMENT, SURFACE, SIDE
    cell_layer: np.ndarray  # [N]
    seeds: np.ndarray  # [N, 3] µm (deformed)
    wounded: np.ndarray  # [N]
    derived: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cell_offsets) - 1

    @property
    def epidermis_top_um(self) -> float:
        return float(sum(layer.thickness_um for layer in self.spec.layers))


def papilla_height(spec: SkinSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Height of the dermal-epidermal junction above its mean plane: one dermal papilla per
    wavelength square (a square lattice, density 1 / wavelength^2), peak to trough twice the
    amplitude. Its mean over a whole or half period in each direction is zero."""
    k = 2 * np.pi / spec.papilla_wavelength_um
    return (
        0.5
        * spec.papilla_amplitude_um
        * (np.cos(k * (x - spec.papilla_offset_x_um)) + np.cos(k * y))
    )


def deform(spec: SkinSpec, points: np.ndarray) -> np.ndarray:
    """Lift the junction into papillae; the lift fades to zero at the top of the spinous layer
    so the granular layer and the stratum corneum stay level. Monotonic in z while the
    amplitude is below the fade height, so no cell inverts."""
    fade = sum(layer.thickness_um for layer in spec.layers[:2])
    out = points.copy()
    w = np.clip(1.0 - points[:, 2] / fade, 0.0, 1.0)
    out[:, 2] = points[:, 2] + papilla_height(spec, points[:, 0], points[:, 1]) * w
    return out


def _seeds(spec: SkinSpec, rng: np.random.Generator):
    """Seeds row by row; returns (points, layer of each, slab of each, slab z-bounds)."""
    points, layer_of, slab_of, slabs = [], [], [], []
    lattice_w = spec.seed_width_x_um or spec.width_x_um
    z0 = 0.0
    for li, layer in enumerate(spec.layers):
        dz = layer.thickness_um / layer.rows
        a = layer.lateral_spacing_um
        row_pitch = a * np.sqrt(3) / 2
        if not layer.row_sheets:
            slabs.append((z0, z0 + layer.thickness_um, li))
        if layer.packing == "kelvin":
            sq = layer.square_spacing()
            nx = max(1, round(lattice_w / sq))
            ny = max(1, round(spec.width_y_um / sq))
            ax, ay = lattice_w / nx, spec.width_y_um / ny
            jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            offset = rng.uniform(0, 1, 2) * [ax, ay]
            for r in range(layer.rows):
                half = 0.5 * (r % 2)
                xy = np.column_stack(
                    (((ii + 0.25 + half) * ax).ravel(), ((jj + 0.25 + half) * ay).ravel())
                )
                xy = (xy + offset) % [lattice_w, spec.width_y_um]
                xy += rng.uniform(-layer.jitter * sq, layer.jitter * sq, xy.shape)
                xy = np.clip(xy, 0.02 * sq, [lattice_w - 0.02 * sq, spec.width_y_um - 0.02 * sq])
                z = z0 + (r + 0.5) * dz + rng.uniform(-0.1, 0.1, len(xy)) * dz * (layer.jitter > 0)
                points.append(np.column_stack((xy, z)))
                layer_of += [li] * len(xy)
                slab_of += [len(slabs) - 1] * len(xy)
            z0 += layer.thickness_um
            continue
        for r in range(layer.rows):
            if layer.row_sheets:
                slabs.append((z0 + r * dz, z0 + (r + 1) * dz, li))
            nx = max(1, round(lattice_w / a))
            ny = max(1, round(spec.width_y_um / row_pitch))
            ax, ay = lattice_w / nx, spec.width_y_um / ny  # exact fit, nominal density
            jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            offset = rng.uniform(0, 1, 2) * [ax, ay]
            xs = (ii + 0.5 * (jj % 2) + 0.25) * ax + offset[0]
            ys = (jj + 0.5) * ay + offset[1]
            xy = np.column_stack((xs.ravel(), ys.ravel())) % [lattice_w, spec.width_y_um]
            xy += rng.uniform(-layer.jitter * a, layer.jitter * a, xy.shape)
            xy = np.clip(xy, 0.02 * a, [lattice_w - 0.02 * a, spec.width_y_um - 0.02 * a])
            jitter_z = 0.0 if layer.row_sheets else 0.15
            z = z0 + (r + 0.5) * dz + rng.uniform(-jitter_z, jitter_z, len(xy)) * dz
            points.append(np.column_stack((xy, z)))
            layer_of += [li] * len(xy)
            slab_of += [len(slabs) - 1] * len(xy)
        z0 += layer.thickness_um
    points, layer_of, slab_of = np.concatenate(points), np.array(layer_of), np.array(slab_of)
    keep = points[:, 0] < spec.width_x_um
    return points[keep], layer_of[keep], slab_of[keep], slabs


def _order_polygon(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    u = points[0] - centre
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angle = np.arctan2((points - centre) @ v, (points - centre) @ u)
    return np.argsort(angle)


INTERFACE_BELOW, INTERFACE_ABOVE = -10, -11  # temporary tags, split into shared faces


def _cell_polyhedron(
    s: np.ndarray,
    others: np.ndarray,
    others_ids: np.ndarray,
    box: np.ndarray,
    bottom_tag: int,
    top_tag: int,
):
    """The Voronoi region of seed ``s`` among ``others``, clipped to ``box``; faces outward."""
    normals = others - s
    mids = 0.5 * (others + s)
    planes = np.column_stack((normals, -np.einsum("ij,ij->i", normals, mids)))
    lo, hi = box
    box_planes = np.array(
        [
            [-1, 0, 0, lo[0]],
            [1, 0, 0, -hi[0]],
            [0, -1, 0, lo[1]],
            [0, 1, 0, -hi[1]],
            [0, 0, -1, lo[2]],
            [0, 0, 1, -hi[2]],
        ],
        float,
    )
    halfspaces = np.vstack((planes, box_planes))
    tags = list(others_ids) + [-SIDE, -SIDE, -SIDE, -SIDE, bottom_tag, top_tag]
    verts = HalfspaceIntersection(halfspaces, s).intersections
    hull = ConvexHull(verts)
    unit = halfspaces[:, :3] / np.linalg.norm(halfspaces[:, :3], axis=1)[:, None]
    offs = halfspaces[:, 3] / np.linalg.norm(halfspaces[:, :3], axis=1)
    score = hull.equations[:, :3] @ unit.T - np.abs(offs[None, :] - hull.equations[:, 3:4])
    owner = np.argmax(score, axis=1)
    groups: dict[int, set[int]] = {}
    for h in np.unique(owner):
        groups[int(h)] = set(hull.simplices[owner == h].ravel().tolist())
    faces, face_tags = [], []
    for h, ids in groups.items():
        ids = np.array(sorted(ids))
        if len(ids) < 3:
            continue
        poly = verts[ids][_order_polygon(verts[ids], unit[h])]
        if _polygon_area(poly) < 1e-9:
            continue
        faces.append(poly)
        face_tags.append(tags[h])
    return faces, face_tags


def _polygon_area(poly: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(poly, np.roll(poly, -1, axis=0)).sum(axis=0)))


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the intersection of two convex polygons, both CCW in the xy plane."""
    out = subject
    for i in range(len(clip)):
        if len(out) == 0:
            break
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        side = edge[0] * (out[:, 1] - a[1]) - edge[1] * (out[:, 0] - a[0])
        kept = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            sp, sq = side[j], side[(j + 1) % len(out)]
            if sp >= 0:
                kept.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                kept.append(p + t * (q - p))
        out = np.array(kept) if kept else np.zeros((0, 2))
    return out


def _ccw(xy: np.ndarray) -> np.ndarray:
    signed = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    return xy if signed > 0 else xy[::-1]


def generate(spec: SkinSpec) -> Skin:
    """Each layer is tessellated in its own slab, so strata stay distinct (no cell crosses into
    another layer or reaches the surface from below). Where two layers meet, every overlap of a
    lower cell's top face with an upper cell's bottom face becomes one shared face pair."""
    rng = np.random.default_rng(spec.seed)
    seeds, layer_of, slab_of, slabs = _seeds(spec, rng)
    cell_faces: list[list[tuple[np.ndarray, int, int]]] = [[] for _ in range(len(seeds))]
    for k, (z_lo, z_hi, li) in enumerate(slabs):
        ids = np.flatnonzero(slab_of == k)
        layer = spec.layers[li]
        a = layer.lateral_spacing_um
        rows_in_slab = 1 if layer.row_sheets else layer.rows
        dz = (z_hi - z_lo) / rows_in_slab
        # Kelvin layers are tessellated in coordinates where the lattice is regular (body-centred
        # cubic), then mapped back: a uniform stretch keeps faces planar and the tiling exact.
        stretch = np.ones(3)
        if layer.packing == "kelvin":
            sq = layer.square_spacing()
            lattice_w = spec.seed_width_x_um or spec.width_x_um
            ax = lattice_w / max(1, round(lattice_w / sq))
            ay = spec.width_y_um / max(1, round(spec.width_y_um / sq))
            stretch = np.array([1.0, ax / ay, ax / (2 * dz)])
            radius = 2.0 * ax
        else:
            radius = 2.0 * np.hypot(a, dz)  # covers every Voronoi neighbour of a jittered row
        pts = seeds[ids] * stretch
        tree = cKDTree(pts)
        box = np.array([[0.0, 0.0, z_lo], [spec.width_x_um, spec.width_y_um, z_hi]]) * stretch
        bottom = -BASEMENT if k == 0 else INTERFACE_BELOW
        top = -SURFACE if k == len(slabs) - 1 else INTERFACE_ABOVE
        for local, cell in enumerate(ids):
            near = [j for j in tree.query_ball_point(pts[local], radius) if j != local]
            faces, tags = _cell_polyhedron(pts[local], pts[near], ids[near], box, bottom, top)
            faces = [f / stretch for f in faces]
            cell_faces[cell] = list(zip(faces, tags, [li] * len(faces), strict=True))
    for k in range(len(slabs) - 1):
        z = slabs[k][1]
        lower = [
            (c, i)
            for c in np.flatnonzero(slab_of == k)
            for i, (_, tag, _) in enumerate(cell_faces[c])
            if tag == INTERFACE_ABOVE
        ]
        upper = [
            (c, i)
            for c in np.flatnonzero(slab_of == k + 1)
            for i, (_, tag, _) in enumerate(cell_faces[c])
            if tag == INTERFACE_BELOW
        ]
        lo_xy = [_ccw(cell_faces[c][i][0][:, :2]) for c, i in lower]
        up_xy = [_ccw(cell_faces[c][i][0][:, :2]) for c, i in upper]
        lo_box = np.array([[p.min(0), p.max(0)] for p in lo_xy])
        up_box = np.array([[p.min(0), p.max(0)] for p in up_xy])
        pieces_lower: dict[int, list] = {c: [] for c, _ in lower}
        pieces_upper: dict[int, list] = {c: [] for c, _ in upper}
        for a_, (ca, _) in enumerate(lower):
            hit = np.flatnonzero(
                np.all(up_box[:, 0] <= lo_box[a_, 1], axis=1)
                & np.all(up_box[:, 1] >= lo_box[a_, 0], axis=1)
            )
            for b in hit:
                piece = _clip_convex(lo_xy[a_], up_xy[b])
                if len(piece) < 3:
                    continue
                poly = np.column_stack((piece, np.full(len(piece), z)))
                if _polygon_area(poly) < 1e-9:
                    continue
                cb = upper[b][0]
                pieces_lower[ca].append((poly, int(cb)))
                pieces_upper[cb].append((poly[::-1].copy(), int(ca)))
        for c, pieces in list(pieces_lower.items()) + list(pieces_upper.items()):
            drop = (
                INTERFACE_ABOVE
                if c in pieces_lower and pieces is pieces_lower[c]
                else (INTERFACE_BELOW)
            )
            kept = [f for f in cell_faces[c] if f[1] != drop]
            cell_faces[c] = kept + [(poly, nb, int(layer_of[c])) for poly, nb in pieces]
    all_verts, face_offsets, cell_offsets, neighbour_cell, boundary = [], [0], [0], [], []
    for faces in cell_faces:
        for poly, tag, _ in faces:
            all_verts.append(poly)
            face_offsets.append(face_offsets[-1] + len(poly))
            neighbour_cell.append(tag if tag >= 0 else -1)
            boundary.append(0 if tag >= 0 else -tag)
        cell_offsets.append(cell_offsets[-1] + len(faces))
    face_verts = deform(spec, np.concatenate(all_verts))
    skin = Skin(
        spec,
        face_verts,
        np.array(face_offsets),
        np.array(cell_offsets),
        np.array(neighbour_cell),
        np.array(boundary),
        layer_of,
        deform(spec, seeds),
        np.zeros(len(seeds), bool),
    )
    compute_geometry(skin)
    skin.wounded = in_wound(spec, skin.derived["centroid"])
    logger.info(
        "skin: %d cells (%s), %d faces, %d wounded",
        skin.n_cells,
        ", ".join(f"{(layer_of == k).sum()} {lyr.name}" for k, lyr in enumerate(spec.layers)),
        len(face_offsets) - 1,
        int(skin.wounded.sum()),
    )
    return skin


def in_wound(spec: SkinSpec, points: np.ndarray) -> np.ndarray:
    """Inside a V-shaped incision along y, from the skin surface into the dermis."""
    w = spec.wound
    if w.kind == "none":
        return np.zeros(len(points), bool)
    top = sum(layer.thickness_um for layer in spec.layers)
    bottom = -w.depth_into_dermis_um
    frac = np.clip((points[:, 2] - bottom) / (top - bottom), 0.0, 1.0)
    half = 0.5 * (w.bottom_width_um + frac * (w.top_width_um - w.bottom_width_um))
    return (np.abs(points[:, 0] - w.x_um) < half) & (points[:, 2] >= bottom)


def compute_geometry(skin: Skin) -> None:
    """Face areas, normals and centroids; cell volumes and centroids; contact pairs."""
    fv, fo = skin.face_verts, skin.face_offsets
    nxt = np.arange(len(fv)) + 1
    nxt[fo[1:] - 1] = fo[:-1]  # each face's last vertex wraps to its first
    vec = np.add.reduceat(np.cross(fv, fv[nxt]), fo[:-1], axis=0)
    area = 0.5 * np.linalg.norm(vec, axis=1)
    normal = vec / (2 * area[:, None])
    centre = np.add.reduceat(fv, fo[:-1], axis=0) / np.diff(fo)[:, None]
    face_cell = np.repeat(np.arange(skin.n_cells), np.diff(skin.cell_offsets))
    signed = np.einsum("ij,ij->i", normal, centre) * area / 3.0
    volume = np.bincount(face_cell, weights=signed, minlength=skin.n_cells)
    centroid = np.stack(
        [
            np.bincount(face_cell, weights=centre[:, d] * area, minlength=skin.n_cells)
            for d in range(3)
        ],
        axis=1,
    )
    centroid /= np.bincount(face_cell, weights=area, minlength=skin.n_cells)[:, None]
    skin.derived.update(
        face_area_um2=area,
        face_normal=normal,
        face_centre_um=centre,
        face_cell=face_cell,
        volume_um3=volume,
        centroid=centroid,
    )


def facing_faces(skin: Skin) -> np.ndarray:
    """For each face shared with a neighbour, the matching face of that neighbour, else -1."""
    face_cell = skin.derived["face_cell"]
    lookup: dict[tuple[int, int], int] = {}
    for f, (c, nb) in enumerate(zip(face_cell, skin.face_neighbour_cell, strict=True)):
        if nb >= 0:
            lookup[(int(c), int(nb))] = f
    out = np.full(len(face_cell), -1)
    for f, (c, nb) in enumerate(zip(face_cell, skin.face_neighbour_cell, strict=True)):
        if nb >= 0:
            out[f] = lookup.get((int(nb), int(c)), -1)
    return out
