"""Export a BETSE init + sim pair as a tissue bundle.

Geometry and the baseline frame come from the init phase (the tissue before the wound). The
sim phase's cutting event fires at t = 0, so every stored sim frame is post-wound; its arrays
are scattered back into the pre-wound numbering with NaN on removed cells and membranes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tissuebundle.writer import QuantityInput, SourceInfo, sha256_file, write_bundle

from betse_adapter import __version__
from betse_adapter.betse_env import betse_version, import_betse
from betse_adapter.config import CONFIG_NAME
from betse_adapter.run import INIT_PICKLE, SIM_PICKLE

logger = logging.getLogger(__name__)

M_TO_UM = 1e6
V_TO_MV = 1e3


class ExportError(Exception):
    """BETSE's data broke an assumption the bundle depends on; nothing was written."""


@dataclass(frozen=True)
class Spec:
    """One exported quantity: BETSE's final-state attribute (baseline), its time series, and
    how to place and scale it."""

    final: str | None
    series: str
    location: str
    unit: str
    scale: float
    description: str


QUANTITIES: dict[str, Spec] = {
    "vmem": Spec(
        "vm_ave",
        "vm_ave_time",
        "cell",
        "mV",
        V_TO_MV,
        "Membrane voltage per cell, the plain mean over its membranes (BETSE vm_ave)",
    ),
    "vmem_membrane": Spec(
        "vm",
        "vm_time",
        "membrane",
        "mV",
        V_TO_MV,
        "Membrane voltage per membrane segment (BETSE vm)",
    ),
    "vmem_ghk": Spec(
        None,
        "vm_GHK_time",
        "cell",
        "mV",
        V_TO_MV,
        "BETSE's Goldman-Hodgkin-Katz estimate per cell, including any channels "
        "of the molecule network (BETSE vm_GHK); not stored for the baseline",
    ),
    "venv": Spec(
        "v_env",
        "venv_time",
        "grid",
        "mV",
        V_TO_MV,
        "Extracellular potential on the environment grid (BETSE v_env)",
    ),
    "efield_x": Spec(
        "E_env_x",
        "efield_ecm_x_time",
        "grid",
        "V/m",
        1.0,
        "Extracellular electric field, x component (BETSE E_env_x)",
    ),
    "efield_y": Spec(
        "E_env_y",
        "efield_ecm_y_time",
        "grid",
        "V/m",
        1.0,
        "Extracellular electric field, y component (BETSE E_env_y)",
    ),
    "current_env_x": Spec(
        "J_env_x",
        "I_tot_x_time",
        "grid",
        "A/m2",
        1.0,
        "Extracellular current density, x component (BETSE J_env_x)",
    ),
    "current_env_y": Spec(
        "J_env_y",
        "I_tot_y_time",
        "grid",
        "A/m2",
        1.0,
        "Extracellular current density, y component (BETSE J_env_y)",
    ),
    "current_membrane": Spec(
        "I_mem",
        "I_mem_time",
        "membrane",
        "A/m2",
        1.0,
        "Transmembrane current density per membrane (BETSE I_mem; BETSE's sign convention)",
    ),
    "conc_cell": Spec(
        "cc_cells",
        "cc_time",
        "cell_ion",
        "mol/m3",
        1.0,
        "Cytoplasmic concentration per ion (BETSE cc_cells)",
    ),
    "conc_env": Spec(
        "cc_env",
        "cc_env_time",
        "grid_ion",
        "mol/m3",
        1.0,
        "Extracellular concentration per ion on the grid (BETSE cc_env)",
    ),
    "diff_membrane": Spec(
        "Dm_cells",
        "dd_time",
        "membrane_ion",
        "m2/s",
        1.0,
        "Membrane diffusion constant per ion and membrane (BETSE Dm_cells); "
        "permeability is this over membrane_thickness_m",
    ),
}


@dataclass
class ExportReport:
    """What the exporter checked on the way, for the audit record."""

    n_cells_pre: int
    n_cells_post: int
    n_membranes_pre: int
    n_membranes_post: int
    removed_cells: list[int]
    checks: dict[str, bool] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def load_phase(path: Path) -> tuple[Any, Any, Any]:
    import_betse()
    from betse.science import filehandling as fh

    return fh.loadSim(str(path))


def _cell_mem_index(cells: Any) -> np.ndarray:
    """Offsets [N + 1], after asserting membranes are numbered contiguously cell by cell."""
    sizes = np.array([len(m) for m in cells.cell_to_mems], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    for i, mems in enumerate(cells.cell_to_mems):
        if not np.array_equal(np.asarray(mems), np.arange(offsets[i], offsets[i + 1])):
            raise ExportError(f"membranes of cell {i} are not numbered contiguously")
        if len(cells.cell_verts[i]) != sizes[i]:
            raise ExportError(f"cell {i} has {len(cells.cell_verts[i])} vertices, {sizes[i]} mems")
    if offsets[-1] != len(cells.mem_i):
        raise ExportError("membrane count does not match cell_to_mems")
    return offsets


def _edge_midpoints(cells: Any) -> np.ndarray:
    """Midpoint of edge (v[k-1], v[k]) for every membrane k, in BETSE's order."""
    return np.concatenate(
        [0.5 * (np.asarray(v) + np.roll(np.asarray(v), 1, axis=0)) for v in cells.cell_verts]
    )


def mapping(init_cells: Any, sim_cells: Any, sim: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    """Kept pre-wound cell and membrane indices, and the bitwise identity checks (C1)."""
    n_pre, m_pre = len(init_cells.cell_i), len(init_cells.mem_i)
    target_c = getattr(sim, "target_inds_cell_o", None)
    target_m = getattr(sim, "target_inds_mem_o", None)
    if len(sim_cells.cell_i) == n_pre:
        kept_c, kept_m = np.arange(n_pre), np.arange(m_pre)
    elif target_c is None or target_m is None:
        raise ExportError("cell count changed but BETSE recorded no removed indices")
    else:
        kept_c = np.delete(np.arange(n_pre), np.asarray(target_c, dtype=np.int64))
        kept_m = np.delete(np.arange(m_pre), np.asarray(target_m, dtype=np.int64))
    checks = {
        "centres_identical": np.array_equal(
            init_cells.cell_centres[kept_c], sim_cells.cell_centres
        ),
        "membrane_midpoints_identical": np.array_equal(
            init_cells.mem_mids_flat[kept_m], sim_cells.mem_mids_flat
        ),
        "polygons_identical": len(kept_c) == len(sim_cells.cell_verts)
        and all(
            np.array_equal(init_cells.cell_verts[k], sim_cells.cell_verts[j])
            for j, k in enumerate(kept_c)
        ),
        "mem_grid_map_identical": np.array_equal(
            np.asarray(init_cells.map_mem2ecm)[kept_m], np.asarray(sim_cells.map_mem2ecm)
        ),
    }
    return kept_c, kept_m, checks


def _scatter(values: np.ndarray, kept: np.ndarray, total: int) -> np.ndarray:
    """Place values over kept entities (last axis) into a NaN array of ``total`` entities."""
    out = np.full((*values.shape[:-1], total), np.nan)
    out[..., kept] = values
    return out


def export_run(
    work_dir: Path,
    out: Path,
    *,
    name: str,
    description: str,
    seed: int | None,
    command: tuple[str, ...] = (),
    overwrite: bool = False,
) -> tuple[Path, ExportReport]:
    s_init, c_init, p_init = load_phase(work_dir / INIT_PICKLE)
    s_sim, c_sim, p_sim = load_phase(work_dir / SIM_PICKLE)
    if not p_sim.is_ecm:
        raise ExportError("only runs with extracellular spaces are supported")

    offsets = _cell_mem_index(c_init)
    if not np.allclose(_edge_midpoints(c_init), c_init.mem_mids_flat, rtol=0, atol=1e-15):
        raise ExportError("membrane k is not the edge (v[k-1], v[k])")
    kept_c, kept_m, checks = mapping(c_init, c_sim, s_sim)
    n_pre, m_pre = len(c_init.cell_i), len(c_init.mem_i)
    ny, nx = c_init.X.shape
    checks["grid_c_order"] = np.array_equal(
        c_init.xypts, np.column_stack((c_init.X.ravel(), c_init.Y.ravel()))
    )
    report = ExportReport(
        n_pre,
        len(kept_c),
        m_pre,
        len(kept_m),
        sorted(set(range(n_pre)) - set(kept_c.tolist())),
        checks,
    )
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ExportError(f"mapping checks failed: {failed}")

    shapes: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "grid": lambda a: a.reshape(-1, ny, nx),
        "grid_ion": lambda a: a.reshape(a.shape[0], -1, ny, nx),
    }
    kept_for = {
        "cell": (kept_c, n_pre),
        "cell_ion": (kept_c, n_pre),
        "membrane": (kept_m, m_pre),
        "membrane_ion": (kept_m, m_pre),
    }
    n_post = len(s_sim.time)
    quantities: dict[str, QuantityInput] = {}
    for key, spec in QUANTITIES.items():
        series = getattr(s_sim, spec.series, None)
        if series is None or len(series) == 0:
            report.skipped.append(key)
            continue
        if len(series) != n_post:
            raise ExportError(f"{spec.series} has {len(series)} frames, time has {n_post}")
        post = np.asarray(series, dtype=np.float64) * spec.scale
        if spec.location in kept_for:
            kept, total = kept_for[spec.location]
            post = _scatter(post, kept, total)
        base_value = getattr(s_init, spec.final, None) if spec.final else None
        if base_value is None:
            base = np.full(post.shape[1:], np.nan)
        else:
            base = np.asarray(base_value, dtype=np.float64) * spec.scale
        if base.shape != post.shape[1:]:
            raise ExportError(f"{key}: baseline shape {base.shape} != frame {post.shape[1:]}")
        stacked = np.concatenate((base[None], post))
        if spec.location in shapes:
            stacked = shapes[spec.location](stacked)
        quantities[key] = QuantityInput(stacked, spec.unit, spec.location, spec.description)

    dt, n_steps = float(p_sim.dt), int(p_sim.sim_tsteps)
    reported = np.asarray(s_sim.time, dtype=np.float64)
    step = np.rint(reported * (n_steps - 1) / (n_steps * dt)).astype(np.int64)
    integrated = (step + 1) * dt

    verts = np.concatenate([np.asarray(v) for v in c_init.cell_verts]) * M_TO_UM
    facing = np.asarray(c_init.nn_i, dtype=np.int64).copy()
    facing[facing == np.arange(m_pre)] = -1  # BETSE marks sheet-edge membranes by self-reference
    paired = np.flatnonzero(facing >= 0)
    if np.any(facing[facing[paired]] != paired):
        raise ExportError("BETSE's membrane neighbour map is not symmetric")
    conf = work_dir / CONFIG_NAME
    path = write_bundle(
        out,
        name=name,
        description=description,
        source=SourceInfo(
            solver="BETSE",
            solver_version=betse_version(),
            config_file=str(conf),
            config_sha256=sha256_file(conf) if conf.exists() else None,
            seed=seed,
            command=command,
            producer="betse-adapter",
            producer_version=__version__,
        ),
        ions=[str(s_sim.ionlabel[i]) for i in range(len(s_sim.zs))],
        ion_charges=[int(z) for z in s_sim.zs],
        constants={
            "temperature_k": float(s_sim.T),
            "membrane_thickness_m": float(p_sim.tm),
            "gas_constant_j_per_k_mol": float(p_sim.R),
            "faraday_c_per_mol": float(p_sim.F),
            "sim_time_step_s": dt,
            "init_time_step_s": float(p_init.dt),
            "init_total_time_s": float(p_init.init_tsteps * p_init.dt),
        },
        cell_verts=verts,
        cell_offsets=offsets,
        cell_centres=np.asarray(c_init.cell_centres) * M_TO_UM,
        cell_removed=~np.isin(np.arange(n_pre), kept_c),
        time_integrated_s=np.concatenate(([0.0], integrated)),
        time_reported_s=np.concatenate(([np.nan], reported)),
        baseline_frames=(0,),
        quantities=quantities,
        grid_x=np.asarray(c_init.X) * M_TO_UM,
        grid_y=np.asarray(c_init.Y) * M_TO_UM,
        mem_grid_index=np.asarray(c_init.map_mem2ecm, dtype=np.int64),
        mem_neighbour=facing,
        overwrite=overwrite,
    )
    logger.info(
        "exported %s: %d cells (%d removed), %d frames, skipped %s",
        path,
        n_pre,
        n_pre - len(kept_c),
        n_post + 1,
        report.skipped,
    )
    return path, report
