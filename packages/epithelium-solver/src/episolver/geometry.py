"""Footprints extruded into prisms, and the bath grid under and over them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tissuebundle.reader import Bundle
from tissuebundle.synthetic import hex_neighbours, hex_sheet

from episolver.params import BETSE_ION_NAMES, RunConfig

logger = logging.getLogger(__name__)

UM = 1e-6


@dataclass
class Tissue:
    """A 2D footprint (µm) and everything derived from it, in SI units."""

    verts_um: np.ndarray
    offsets: np.ndarray
    centres_um: np.ndarray
    neighbour: np.ndarray
    wounded: np.ndarray
    height_m: float
    initial_conc: dict[str, np.ndarray] | None = None
    source: str = ""

    def __post_init__(self) -> None:
        n = len(self.offsets) - 1
        self.n_cells = n
        self.mem_cell = np.repeat(np.arange(n), np.diff(self.offsets))
        verts = self.verts_um * UM
        starts = self.offsets[:-1][self.mem_cell]
        local = np.arange(len(verts)) - starts
        prev = np.where(local == 0, self.offsets[1:][self.mem_cell] - 1, np.arange(len(verts)) - 1)
        edge = verts - verts[prev]
        self.edge_len_m = np.hypot(edge[:, 0], edge[:, 1])
        self.mem_mid_um = 0.5 * (self.verts_um + self.verts_um[prev])
        x, y = verts[:, 0], verts[:, 1]
        cross = x[prev] * y - x * y[prev]
        self.area_m2 = 0.5 * np.bincount(self.mem_cell, weights=cross, minlength=n)
        if np.any(self.area_m2 <= 0):
            raise ValueError("footprint polygons must be counter-clockwise with positive area")
        self.lateral_m2 = np.bincount(self.mem_cell, weights=self.edge_len_m, minlength=n) * (
            self.height_m
        )
        self.volume_m3 = self.area_m2 * self.height_m
        k = np.flatnonzero(self.neighbour > np.arange(len(self.neighbour)))
        kk = self.neighbour[k]
        self.pair_mem = np.column_stack((k, kk))
        self.pair_cells = np.column_stack((self.mem_cell[k], self.mem_cell[kk]))
        self.pair_len_m = 0.5 * (self.edge_len_m[k] + self.edge_len_m[kk])
        self.pair_area_m2 = self.pair_len_m * self.height_m
        self.pair_mid_um = 0.5 * (self.mem_mid_um[k] + self.mem_mid_um[kk])


def build_tissue(config: RunConfig) -> Tissue:
    fp = config.footprint
    height = config.height_um * UM
    if fp.kind == "hex":
        verts, offsets, centres = hex_sheet(fp.cols, fp.rows, fp.radius_um, shrink=0.97)
        neighbour = hex_neighbours(fp.cols, fp.rows)
        tissue = Tissue(
            verts,
            offsets,
            centres,
            neighbour,
            np.zeros(len(centres), bool),
            height,
            source=f"hex {fp.cols}x{fp.rows}, radius {fp.radius_um} um",
        )
    else:
        if fp.bundle is None:
            raise ValueError("footprint.kind betse_bundle needs footprint.bundle")
        bundle = Bundle(fp.bundle)
        if "mem_neighbour" not in bundle.manifest["geometry"] or (
            bundle.manifest["geometry"]["mem_neighbour"] is None
        ):
            raise ValueError(f"{fp.bundle} has no mem_neighbour; re-export it")
        verts, offsets = bundle.polygons()
        conc = None
        if "conc_cell" in bundle.quantities:
            baseline = bundle.frame("conc_cell", bundle.baseline_frames[0])
            conc = {
                BETSE_ION_NAMES.get(name, name): baseline[i] for i, name in enumerate(bundle.ions)
            }
        tissue = Tissue(
            np.asarray(verts, float),
            np.asarray(offsets),
            np.asarray(bundle.array("cell_centres")),
            np.asarray(bundle.array("mem_neighbour", mmap=False)),
            bundle.removed().copy(),
            height,
            initial_conc=conc,
            source=f"{Path(fp.bundle).name} ({bundle.manifest['provenance']['solver']})",
        )
    wound = config.wound
    if wound.kind == "none":
        tissue.wounded[:] = False
    elif wound.kind == "circle":
        tissue.wounded = np.hypot(*(tissue.centres_um - [wound.x_um, wound.y_um]).T) < wound.r_um
    elif wound.kind == "from_bundle" and fp.kind != "betse_bundle":
        raise ValueError("wound.kind from_bundle needs a betse_bundle footprint")
    logger.info(
        "tissue: %s; %d cells, %d wounded, %d junctions",
        tissue.source,
        tissue.n_cells,
        int(tissue.wounded.sum()),
        len(tissue.pair_len_m),
    )
    return tissue


@dataclass
class BathGrid:
    """A regular grid of bath nodes, node (j, i) at index j * nx + i."""

    x0_um: float
    y0_um: float
    spacing_um: float
    nx: int
    ny: int

    @classmethod
    def around(cls, tissue: Tissue, spacing_um: float, margin_um: float) -> BathGrid:
        lo = tissue.verts_um.min(axis=0) - margin_um
        hi = tissue.verts_um.max(axis=0) + margin_um
        nx, ny = (np.ceil((hi - lo) / spacing_um).astype(int) + 1).tolist()
        return cls(float(lo[0]), float(lo[1]), spacing_um, nx, ny)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0_um + self.spacing_um * np.arange(self.nx)
        y = self.y0_um + self.spacing_um * np.arange(self.ny)
        return np.meshgrid(x, y)

    def bilinear(self, points_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Node indices and weights ``[n, 4]`` interpolating at ``points``."""
        fx = (points_um[:, 0] - self.x0_um) / self.spacing_um
        fy = (points_um[:, 1] - self.y0_um) / self.spacing_um
        i = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx, ty = fx - i, fy - j
        idx = np.column_stack(
            (
                j * self.nx + i,
                j * self.nx + i + 1,
                (j + 1) * self.nx + i,
                (j + 1) * self.nx + i + 1,
            )
        )
        w = np.column_stack(((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty))
        return idx, w

    def edges(self) -> np.ndarray:
        """Index pairs of horizontally and vertically adjacent nodes."""
        ids = np.arange(self.size).reshape(self.ny, self.nx)
        return np.concatenate(
            (
                np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel())),
                np.column_stack((ids[:-1, :].ravel(), ids[1:, :].ravel())),
            )
        )

    def boundary(self) -> np.ndarray:
        mask = np.zeros((self.ny, self.nx), bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return mask.ravel()


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting: which of ``points [m, 2]`` lie inside ``polygon [n, 2]``."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x1, y1 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x2, y2 = np.roll(polygon[:, 0], -1)[None, :], np.roll(polygon[:, 1], -1)[None, :]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.count_nonzero(crosses & (x < x_at), axis=1) % 2 == 1
