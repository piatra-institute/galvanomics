"""Write a skin model as a polyhedral tissue bundle (schema 0.3), dermis included."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tissuebundle.writer import QuantityInput, SourceInfo, write_bundle

from episolver import __version__
from episolver.skin import Skin, SkinSpec, facing_faces, papilla_height


def wound_floor(spec: SkinSpec, x: np.ndarray) -> np.ndarray:
    """Height of the incision's wall at distance |x - x_w| (the V-shaped cut's surface)."""
    w = spec.wound
    top = sum(layer.thickness_um for layer in spec.layers)
    bottom = -w.depth_into_dermis_um
    r = np.abs(x - w.x_um)
    slope = (top - bottom) / max(w.top_width_um - w.bottom_width_um, 1e-9) * 2
    return np.where(
        r <= w.bottom_width_um / 2, bottom, bottom + (r - w.bottom_width_um / 2) * slope
    )


def dermis_mesh(spec: SkinSpec, step_um: float = 2.0) -> tuple[np.ndarray, list[list[int]]]:
    """A closed block: papillary top surface (with the incision cut into it), sides, bottom."""
    xs = np.linspace(0, spec.width_x_um, int(round(spec.width_x_um / step_um)) + 1)
    ys = np.linspace(0, spec.width_y_um, int(round(spec.width_y_um / step_um)) + 1)
    gx, gy = np.meshgrid(xs, ys)
    top = papilla_height(spec, gx, gy)
    if spec.wound.kind != "none":
        top = np.minimum(top, wound_floor(spec, gx))
    ny, nx = gx.shape
    bottom_z = -spec.dermis_depth_um
    verts = [
        np.column_stack((gx.ravel(), gy.ravel(), top.ravel())),
        np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, bottom_z))),
    ]
    verts = np.concatenate(verts)
    t = lambda j, i: j * nx + i  # noqa: E731
    b = lambda j, i: nx * ny + j * nx + i  # noqa: E731
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            faces.append([t(j, i), t(j, i + 1), t(j + 1, i + 1), t(j + 1, i)])
    faces.append([b(0, 0), b(ny - 1, 0), b(ny - 1, nx - 1), b(0, nx - 1)])
    for i in range(nx - 1):  # front (y = 0) and back (y = max) walls
        faces.append([b(0, i), b(0, i + 1), t(0, i + 1), t(0, i)])
        faces.append([b(ny - 1, i + 1), b(ny - 1, i), t(ny - 1, i), t(ny - 1, i + 1)])
    for j in range(ny - 1):  # left (x = 0) and right (x = max) walls
        faces.append([b(j + 1, 0), b(j, 0), t(j, 0), t(j + 1, 0)])
        faces.append([b(j, nx - 1), b(j + 1, nx - 1), t(j + 1, nx - 1), t(j, nx - 1)])
    return verts, faces


def export_skin_bundle(skin: Skin, out: Path, overwrite: bool = True) -> Path:
    spec = skin.spec
    d = skin.derived
    layer = np.stack([skin.cell_layer.astype(float)] * 2)
    layer[1, skin.wounded] = np.nan
    volume = np.stack([d["volume_um3"]] * 2)
    volume[1, skin.wounded] = np.nan
    verts, faces = dermis_mesh(spec)
    return write_bundle(
        out,
        name="skin_anatomy",
        description="Procedural human-like skin: stratified epidermis over dermis, with an "
        "incision. Geometry only (frame 0 intact, frame 1 wounded); no physics yet.",
        source=SourceInfo(
            solver="episolver.skin",
            solver_version=__version__,
            seed=spec.seed,
            command=tuple(sys.argv),
            producer="epithelium-solver",
            producer_version=__version__,
        ),
        ions=(),
        cell_verts=skin.face_verts,
        cell_offsets=skin.cell_offsets,
        cell_centres=d["centroid"],
        cell_removed=skin.wounded,
        time_integrated_s=np.array([0.0, 0.0]),
        time_reported_s=np.array([0.0, 0.0]),
        baseline_frames=(0,),
        quantities={
            "layer": QuantityInput(layer, "1", "cell", "Epidermal layer index (see layers)"),
            "cell_volume": QuantityInput(volume, "1", "cell", "Cell volume in cubic micrometres"),
        },
        mem_neighbour=facing_faces(skin),
        kind="polyhedra",
        face_offsets=skin.face_offsets,
        face_boundary=skin.face_boundary,
        cell_layer=skin.cell_layer,
        layers=[layer.name for layer in spec.layers],
        layer_colors=[layer.color for layer in spec.layers],
        context_meshes={
            "dermis": (
                verts,
                faces,
                (0.93, 0.62, 0.58, 1.0),
                "Dermis; its top is the dermal-epidermal junction",
            )
        },
        constants={
            "epidermis_thickness_um": skin.epidermis_top_um,
            "dermis_depth_um": spec.dermis_depth_um,
        },
        overwrite=overwrite,
    )
