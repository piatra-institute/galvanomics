"""A synthetic wounded hexagonal sheet, for tests and for Blender work without a solver.

The values are made up and carry no physics: a resting sheet near -45 mV, a circular wound
removed after frame 0, and a depolarization that spreads from the wound edge and decays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tissuebundle.writer import QuantityInput, SourceInfo, write_bundle


def hex_sheet(
    cols: int, rows: int, radius_um: float, shrink: float = 0.92
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointy-top hexagons on an offset lattice, each with its own (unshared) vertices.

    Returns ``(verts [6N, 2], offsets [N + 1], centres [N, 2])`` in µm, counter-clockwise.
    """
    width = np.sqrt(3.0) * radius_um
    centres = np.array(
        [
            ((c + 0.5 * (r % 2)) * width + width, r * 1.5 * radius_um + radius_um)
            for r in range(rows)
            for c in range(cols)
        ]
    )
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    ring = shrink * radius_um * np.column_stack((np.cos(angles), np.sin(angles)))
    verts = (centres[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    offsets = np.arange(0, 6 * len(centres) + 1, 6, dtype=np.int64)
    return verts, offsets, centres


def hex_neighbours(cols: int, rows: int) -> np.ndarray:
    """Facing membrane of every membrane of :func:`hex_sheet`, or -1 on the sheet's edge.

    Membrane k of a pointy-top hexagon is the edge (v[k-1], v[k]); it faces direction 60k
    degrees (east, north-east, north-west, west, south-west, south-east), and the facing
    membrane in the neighbour is k + 3 (mod 6).
    """
    out = np.full(6 * cols * rows, -1, dtype=np.int64)
    for r in range(rows):
        odd = r % 2
        steps = [(0, 1), (1, odd), (1, odd - 1), (0, -1), (-1, odd - 1), (-1, odd)]
        for c in range(cols):
            cell = r * cols + c
            for k, (dr, dc) in enumerate(steps):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    out[6 * cell + k] = 6 * (rr * cols + cc) + (k + 3) % 6
    return out


def make_synthetic(
    out: Path,
    cols: int = 14,
    rows: int = 14,
    frames: int = 12,
    radius_um: float = 5.0,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    verts, offsets, centres = hex_sheet(cols, rows, radius_um)
    n = len(centres)
    mem_cell = np.repeat(np.arange(n), 6)

    wound_centre = centres.mean(axis=0) + np.array([0.25, 0.0]) * np.ptp(centres[:, 0])
    distance = np.linalg.norm(centres - wound_centre, axis=1)
    removed = distance < 2.2 * radius_um

    times = np.linspace(0.0, 0.02, frames)
    rest = -45.0 + rng.normal(0.0, 0.3, n)
    edge = np.clip(distance - 2.2 * radius_um, 0.0, None)
    vmem = np.empty((frames, n))
    for t, time in enumerate(times):
        spread = 10.0 + 400.0 * time
        bump = 0.0 if t == 0 else 30.0 * np.exp(-edge / spread) * np.exp(-time / 0.05)
        vmem[t] = rest + bump
    vmem[1:, removed] = np.nan

    phase = np.tile(np.linspace(-1.0, 1.0, 6), n)
    vmem_mem = vmem[:, mem_cell] + 0.5 * phase[None, :]
    vmem_mem[np.isnan(vmem[:, mem_cell])] = np.nan
    # Keep the documented relation: cell value is the plain mean of its membrane values.
    vmem = np.where(np.isnan(vmem), np.nan, vmem_mem.reshape(frames, n, 6).mean(axis=2))

    lo, hi = verts.min(axis=0) - radius_um, verts.max(axis=0) + radius_um
    nx, ny = 20, 18
    grid_x, grid_y = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
    r_grid = np.hypot(grid_x - wound_centre[0], grid_y - wound_centre[1])
    venv = np.stack([(0.0 if t == 0 else 1.5) * np.exp(-r_grid / 25.0) for t in range(frames)])

    mids = 0.5 * (verts + np.roll(verts.reshape(n, 6, 2), 1, axis=1).reshape(-1, 2))
    gx = np.clip(np.rint((mids[:, 0] - lo[0]) / (hi[0] - lo[0]) * (nx - 1)), 0, nx - 1)
    gy = np.clip(np.rint((mids[:, 1] - lo[1]) / (hi[1] - lo[1]) * (ny - 1)), 0, ny - 1)
    mem_grid_index = (gy * nx + gx).astype(np.int64)

    return write_bundle(
        Path(out),
        name="synthetic_hex_wound",
        description="Synthetic wounded hexagonal sheet; made-up values, no physics.",
        source=SourceInfo(solver="synthetic", solver_version="0", seed=seed),
        ions=("na", "k"),
        ion_charges=(1, 1),
        cell_verts=verts,
        cell_offsets=offsets,
        cell_centres=centres,
        cell_removed=removed,
        time_integrated_s=times,
        time_reported_s=times,
        baseline_frames=(0,),
        quantities={
            "vmem": QuantityInput(vmem, "mV", "cell", "Membrane voltage, cell mean"),
            "vmem_membrane": QuantityInput(vmem_mem, "mV", "membrane", "Membrane voltage"),
            "venv": QuantityInput(venv, "mV", "grid", "Extracellular potential"),
        },
        grid_x=grid_x,
        grid_y=grid_y,
        mem_grid_index=mem_grid_index,
        mem_neighbour=hex_neighbours(cols, rows),
        overwrite=overwrite,
    )
