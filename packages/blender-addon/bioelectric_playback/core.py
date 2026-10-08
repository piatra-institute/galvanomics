"""Mesh and colour logic with no ``bpy`` import, so it can be tested outside Blender."""

from __future__ import annotations

import numpy as np

UM_PER_UNIT = 10.0
"""Ten micrometres per Blender unit: a 150 µm sheet is 15 units across."""

REMOVED_RGBA = (0.08, 0.08, 0.08, 1.0)

# A blue-white-red diverging ramp (anchor colours of Moreland's cool-warm map, linear RGB).
_ANCHORS = np.array(
    [
        [0.0, 0.0507, 0.0624, 0.3851],
        [0.25, 0.1972, 0.3324, 0.7955],
        [0.5, 0.7529, 0.7529, 0.7529],
        [0.75, 0.8070, 0.2747, 0.1708],
        [1.0, 0.4931, 0.0, 0.0335],
    ]
)


def faces_from_offsets(offsets: np.ndarray) -> list[list[int]]:
    """One polygon per cell; cell ``i`` owns vertices ``offsets[i]:offsets[i + 1]``."""
    return [list(range(int(a), int(b))) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def grid_faces(ny: int, nx: int) -> list[list[int]]:
    """Quads over a ``[ny, nx]`` point grid flattened in C order, counter-clockwise."""
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append([a, a + 1, a + nx + 1, a + nx])
    return faces


def to_blender_xyz(points_um: np.ndarray, z: float = 0.0) -> np.ndarray:
    """``[n, 2]`` µm to ``[n, 3]`` Blender units."""
    xyz = np.zeros((len(points_um), 3), dtype=np.float32)
    xyz[:, :2] = np.asarray(points_um, dtype=np.float64) / UM_PER_UNIT
    xyz[:, 2] = z
    return xyz


def colormap(
    values: np.ndarray, vmin: float, vmax: float, nan_rgba: tuple[float, ...] = REMOVED_RGBA
) -> np.ndarray:
    """Map values to RGBA on the diverging ramp; NaN gets ``nan_rgba``. Out-of-range clips."""
    values = np.asarray(values, dtype=np.float64)
    span = vmax - vmin if vmax > vmin else 1.0
    s = np.clip((values - vmin) / span, 0.0, 1.0)
    rgba = np.ones((len(values), 4), dtype=np.float32)
    for channel in range(3):
        rgba[:, channel] = np.interp(s, _ANCHORS[:, 0], _ANCHORS[:, channel + 1])
    rgba[np.isnan(values)] = nan_rgba
    return rgba


def corner_colors(cell_rgba: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Repeat each cell's colour over its polygon corners: ``[V, 4]``."""
    return np.repeat(cell_rgba, np.diff(offsets), axis=0)


def default_range(data: np.ndarray, step: float = 5.0, symmetric: bool = False) -> tuple:
    """Finite min and max over every frame, rounded outward to ``step``."""
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    if symmetric:
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    lo, hi = step * np.floor(lo / step), step * np.ceil(hi / step)
    if hi <= lo:
        hi = lo + step
    return (float(lo), float(hi))


def frame_index(scene_frame: int, frame_offset: int, count: int) -> int:
    """Scene frame to bundle frame, clamped to the run."""
    return int(min(max(scene_frame - frame_offset, 0), count - 1))


# ---------------------------------------------------------------------------- 3D (prisms)
APICAL, BASAL, LATERAL = 0, 1, 2


def prism_mesh(verts_um: np.ndarray, offsets: np.ndarray, height_um: float) -> tuple:
    """Each footprint polygon extruded into a closed prism with outward-facing faces.

    Returns ``(xyz [2V, 3] in Blender units, faces, face_cell [F], face_domain [F])``. Bottom
    vertices are ``0..V-1`` (z = 0), top vertices ``V..2V-1``. Lateral face k of a cell is
    footprint edge (v[k-1], v[k]), the same edge as membrane k.
    """
    v = len(verts_um)
    xyz = np.concatenate(
        (to_blender_xyz(verts_um), to_blender_xyz(verts_um, z=height_um / UM_PER_UNIT))
    )
    faces, cell, domain = [], [], []
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        ring = list(range(int(a), int(b)))
        faces.append([v + k for k in ring])
        faces.append(ring[::-1])
        cell += [i, i]
        domain += [APICAL, BASAL]
        for j, k in enumerate(ring):
            prev = ring[j - 1]
            faces.append([prev, k, v + k, v + prev])
            cell.append(i)
            domain.append(LATERAL)
    return xyz, faces, np.array(cell), np.array(domain)


def face_corner_counts(faces: list[list[int]]) -> np.ndarray:
    return np.array([len(f) for f in faces])


def domain_pair(name: str, available: set[str]) -> tuple[str, str]:
    """The (apical, basolateral) quantities to show for a selected quantity on prism faces."""
    for a, b in (("_apical", "_basolateral"), ("_basolateral", "_apical")):
        if name.endswith(a):
            other = name[: -len(a)] + b
            if other in available:
                return (name, other) if a == "_apical" else (other, name)
    return name, name


def arrow_triangles(
    points_um: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    scale_um: float,
    z_bu: float,
    width_um: float,
) -> np.ndarray:
    """One flat triangle per point, tip along the field: ``[3n, 3]`` Blender coordinates.

    ``scale_um`` is the arrow length for a field equal to the run's maximum; zero field gives a
    degenerate (invisible) triangle.
    """
    mag = np.hypot(ex, ey)
    unit = np.column_stack((ex, ey)) / np.maximum(mag, 1e-30)[:, None]
    normal = np.column_stack((-unit[:, 1], unit[:, 0]))
    length = scale_um * mag
    half = 0.5 * np.minimum(width_um, length)[:, None]
    tip = points_um + unit * length[:, None]
    left, right = points_um + normal * half, points_um - normal * half
    tri = np.stack((left, right, tip), axis=1).reshape(-1, 2)
    return to_blender_xyz(tri, z=z_bu)


def sample_on_faces(
    xyz: np.ndarray,
    faces: list[list[int]],
    face_ids: np.ndarray,
    per_face: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over the given faces (fan-triangulated), and the face each came from."""
    points, owner = [], []
    for f in face_ids:
        ring = xyz[faces[f]]
        tris = np.stack((np.repeat(ring[:1], len(ring) - 2, axis=0), ring[1:-1], ring[2:]), axis=1)
        area = 0.5 * np.linalg.norm(
            np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
        )
        pick = rng.choice(len(tris), size=per_face, p=area / area.sum())
        r1, r2 = rng.random(per_face), rng.random(per_face)
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        t = tris[pick]
        points.append(
            t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])
        )
        owner += [f] * per_face
    return np.concatenate(points), np.array(owner)


def tetra_glyphs(centres: np.ndarray, size: float) -> tuple[np.ndarray, list[list[int]]]:
    """A small tetrahedron at each point: ``(xyz [4n, 3], faces)``."""
    base = size * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    xyz = (centres[:, None, :] + base[None, :, :]).reshape(-1, 3)
    faces = []
    for g in range(len(centres)):
        o = 4 * g
        faces += [[o, o + 1, o + 2], [o, o + 3, o + 1], [o, o + 2, o + 3], [o + 1, o + 3, o + 2]]
    return xyz, faces


CHANNEL_HUES = {
    "k": (0.15, 0.45, 1.0),
    "na": (1.0, 0.55, 0.1),
    "cl": (0.2, 0.85, 0.3),
    "hcn": (0.75, 0.3, 0.95),
}


def glyph_colors(open_fraction: np.ndarray, hue: tuple[float, float, float]) -> np.ndarray:
    """Dark when closed, the channel's hue when open; NaN (removed) is the removed colour."""
    f = np.clip(np.nan_to_num(open_fraction, nan=0.0), 0.0, 1.0)[:, None]
    rgb = 0.12 + (np.asarray(hue)[None, :] - 0.12) * f
    rgba = np.concatenate((rgb, np.ones((len(f), 1))), axis=1).astype(np.float32)
    rgba[np.isnan(open_fraction)] = REMOVED_RGBA
    return rgba


# ---------------------------------------------------------------------------- volumes, ions
def trilinear_sample(axes: tuple, data: np.ndarray, points_um: np.ndarray) -> np.ndarray:
    """Sample a [nz, ny, nx] grid with (possibly non-uniform) axes (x, y, z) at points (µm);
    NaN outside the grid."""
    xs, ys, zs = (np.asarray(a, float) for a in axes)
    idx, frac = [], []
    inside = np.ones(len(points_um), bool)
    for axis, nodes in enumerate((xs, ys, zs)):
        p = points_um[:, axis]
        inside &= (p >= nodes[0]) & (p <= nodes[-1])
        i = np.clip(np.searchsorted(nodes, p) - 1, 0, len(nodes) - 2)
        t = np.clip((p - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)
        idx.append(i)
        frac.append(t)
    (ix, iy, iz), (tx, ty, tz) = idx, frac
    out = np.zeros(len(points_um))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (tx if dx else 1 - tx) * (ty if dy else 1 - ty) * (tz if dz else 1 - tz)
                out += w * data[iz + dz, iy + dy, ix + dx]
    out[~inside] = np.nan
    return out


def assign_particles(weights: np.ndarray, count: int) -> np.ndarray:
    """Stratified assignment of ``count`` particles to items in proportion to ``weights``.
    Stable from frame to frame while the weights change smoothly (no flicker)."""
    w = np.nan_to_num(np.clip(weights, 0.0, None))
    total = w.sum()
    if total <= 0:
        return np.full(count, -1)
    cdf = np.cumsum(w) / total
    u = (np.arange(count) + 0.5) / count
    return np.minimum(np.searchsorted(cdf, u), len(w) - 1)


def particle_offsets(count: int, frame: int, speed: float = 0.12) -> np.ndarray:
    """Phase in [0, 1) of each particle along its path, advancing with the frame."""
    golden = 0.6180339887498949
    return (frame * speed + np.arange(count) * golden) % 1.0


# ---------------------------------------------------------------------------- nanoscope sampling
AVOGADRO = 6.02214076e23


def expected_counts(
    g_full_s_per_m2: dict[str, float],
    unitary_ps: dict[str, float],
    pump_rate_mol_m2_s: float,
    pump_turnover_per_s: float,
    area_um2: float,
) -> dict[str, float]:
    """Expected number of each channel type and of pumps in a patch (derived, not measured):
    channels = conductance if all were open × area ÷ single-channel conductance; pumps = pump
    rate × Avogadro × area ÷ turnover."""
    out = {
        name: max(g, 0.0) * area_um2 * 1e-12 / (unitary_ps[name] * 1e-12)
        for name, g in g_full_s_per_m2.items()
        if name in unitary_ps
    }
    out["pump"] = pump_rate_mol_m2_s * AVOGADRO * area_um2 * 1e-12 / pump_turnover_per_s
    return out


def sample_positions(
    expected: float,
    size_nm: float,
    rng: np.random.Generator,
    placement: str = "uniform",
    cluster_radius_nm: float = 50.0,
    per_cluster: float = 5.0,
) -> np.ndarray:
    """Positions (nm, in [0, size)²) of a Poisson number of molecules with the expected count:
    uniform (Poisson process) or clustered (Thomas process: Poisson parents, Gaussian
    offspring of the given radius, wrapped periodically). Placement is an assumption."""
    if placement == "uniform":
        return rng.uniform(0.0, size_nm, (rng.poisson(expected), 2))
    parents = rng.uniform(0.0, size_nm, (rng.poisson(expected / per_cluster), 2))
    children = rng.poisson(per_cluster, len(parents))
    pts = np.repeat(parents, children, axis=0)
    pts += rng.normal(0.0, cluster_radius_nm, pts.shape)
    return np.mod(pts, size_nm)


def sample_open(open_fraction: float, count: int, seed: int, frame: int) -> np.ndarray:
    """Open (True) or closed for each channel at one frame: independent draws, because gating
    (milliseconds) is much faster than the 50 ms between frames. Reproducible per seed."""
    rng = np.random.default_rng([seed, frame])
    return rng.random(count) < open_fraction
