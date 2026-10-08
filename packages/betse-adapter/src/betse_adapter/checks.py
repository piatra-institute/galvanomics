"""The audit checks of docs/plan.md, computed from bundles (and BETSE's own CSV exports).

Each function returns measurements; the verdict against the pre-fixed tolerance is made by
the caller in :mod:`betse_adapter.audit`, so the numbers are recorded whether or not they pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tissuebundle.reader import Bundle
from tissuebundle.validate import polygon_areas

from betse_adapter.ghk import ghk_from_bundle, membrane_mean


def post_frames(bundle: Bundle) -> list[int]:
    return [i for i in range(bundle.n_frames) if i not in bundle.baseline_frames]


def c2_mean_consistency(bundle: Bundle) -> float:
    """Max |plain membrane mean - exported cell V_mem| in volts, over all frames and cells."""
    _, offsets = bundle.polygons()
    worst = 0.0
    for frame in range(bundle.n_frames):
        mean = membrane_mean(bundle.frame("vmem_membrane", frame), offsets)
        diff = np.abs(mean - bundle.frame("vmem", frame))
        if np.any(np.isfinite(diff)):
            worst = max(worst, float(np.nanmax(diff)) * 1e-3)
    return worst


def c3_polygons(bundle: Bundle) -> int:
    """Number of cells that are not counter-clockwise with positive area."""
    verts, offsets = bundle.polygons()
    return int(np.sum(polygon_areas(np.asarray(verts), np.asarray(offsets)) <= 0))


def c4_csv(bundle: Bundle, csv_dir: Path) -> dict[str, float]:
    """Exported cell V_mem and centres against BETSE's own ``Vmem2D_<k>.csv`` (frame k + 1).
    Both come from the same BETSE array, so this catches ordering, unit and offset errors."""
    removed = bundle.removed()
    centres = bundle.array("cell_centres")[~removed]
    worst_v, worst_xy, frames = 0.0, 0.0, 0
    for k, frame in enumerate(post_frames(bundle)):
        path = csv_dir / f"Vmem2D_{k}.csv"
        if not path.exists():
            break
        table = np.loadtxt(path, delimiter=",", skiprows=1)
        ours = bundle.frame("vmem", frame)[~removed]
        if len(table) != len(ours):
            return {"frames": frames, "max_vmem_mV": np.inf, "max_centre_um": np.inf}
        worst_v = max(worst_v, float(np.max(np.abs(table[:, 2] - ours))))
        worst_xy = max(worst_xy, float(np.max(np.abs(table[:, :2] - centres))))
        frames += 1
    return {"frames": frames, "max_vmem_mV": worst_v, "max_centre_um": worst_xy}


def d1_differences(a: Bundle, b: Bundle) -> list[str]:
    """Names of arrays whose SHA-256 differs between two bundles."""
    differing = []
    for section in ("geometry", "frames", "quantities"):
        left, right = a.manifest[section], b.manifest[section]
        for key, record in left.items():
            if isinstance(record, dict) and "sha256" in record:
                other = right.get(key)
                if not isinstance(other, dict) or other.get("sha256") != record["sha256"]:
                    differing.append(key)
    return differing


def p1_ghk_reading(bundle: Bundle) -> float:
    """Max |our GHK - BETSE's vm_GHK| in volts over post-wound frames and live cells."""
    worst = 0.0
    for frame in post_frames(bundle):
        ours = ghk_from_bundle(bundle, frame)
        theirs = bundle.frame("vmem_ghk", frame) * 1e-3
        live = np.isfinite(ours) & np.isfinite(theirs)
        worst = max(worst, float(np.max(np.abs(ours[live] - theirs[live]))))
    return worst


def p2_passive(bundle: Bundle) -> dict[str, float]:
    """V_mem against GHK at the last frame, in mV, against our GHK and BETSE's."""
    last = bundle.n_frames - 1
    vmem = bundle.frame("vmem", last)
    ours = ghk_from_bundle(bundle, last) * 1e3
    out = {
        "max_abs_vs_our_ghk_mV": float(np.nanmax(np.abs(vmem - ours))),
        "mean_vs_our_ghk_mV": float(np.nanmean(vmem - ours)),
        "mean_vmem_mV": float(np.nanmean(vmem)),
        "time_s": float(bundle.array("time_integrated_s")[last]),
    }
    if "vmem_ghk" in bundle.quantities:
        out["max_abs_vs_betse_ghk_mV"] = float(
            np.nanmax(np.abs(vmem - bundle.frame("vmem_ghk", last)))
        )
    series = np.nanmean(bundle.array("vmem"), axis=1)
    out["mean_vmem_first_post_mV"] = float(series[1])
    return out


def p3_convergence(coarse: Bundle, mid: Bundle, fine: Bundle) -> dict[str, float]:
    """V_mem differences between time steps at matched sample index (mV)."""
    a, b, c = (np.asarray(x.array("vmem"))[1:] for x in (coarse, mid, fine))
    if not a.shape == b.shape == c.shape:
        raise ValueError(f"frame shapes differ: {a.shape} {b.shape} {c.shape}")
    e12 = float(np.nanmax(np.abs(a - b)))
    e23 = float(np.nanmax(np.abs(b - c)))
    t = np.asarray(coarse.array("time_integrated_s"))[1:]
    dvdt = np.abs(np.diff(a, axis=0)) / np.diff(t)[:, None]
    offset = float(coarse.constants["sim_time_step_s"] - mid.constants["sim_time_step_s"])
    return {
        "max_diff_coarse_mid_mV": e12,
        "max_diff_mid_fine_mV": e23,
        "observed_order": float(np.log2(e12 / e23)) if e23 > 0 else float("inf"),
        "time_offset_s": offset,
        "offset_bound_mV": float(np.nanmax(dvdt)) * offset,
    }


def r1_offset(bundle: Bundle) -> dict[str, float]:
    """V_mem - BETSE's GHK estimate over post-wound frames (mV): the electrogenic offset."""
    frames = post_frames(bundle)
    diff = np.asarray(bundle.array("vmem"))[frames] - np.asarray(bundle.array("vmem_ghk"))[frames]
    return {
        "mean_mV": float(np.nanmean(diff)),
        "min_mV": float(np.nanmin(diff)),
        "max_mV": float(np.nanmax(diff)),
    }


def r2_field(bundle: Bundle, frame: int | None = None) -> dict[str, float]:
    """-grad(venv) by centred differences against BETSE's extracellular field."""
    frame = bundle.n_frames - 1 if frame is None else frame
    gx = np.asarray(bundle.array("grid_x"))[0, :] * 1e-6
    gy = np.asarray(bundle.array("grid_y"))[:, 0] * 1e-6
    venv = bundle.frame("venv", frame) * 1e-3
    d_dy, d_dx = np.gradient(venv, gy, gx)
    ours = np.stack((-d_dx, -d_dy)).ravel()
    theirs = np.stack((bundle.frame("efield_x", frame), bundle.frame("efield_y", frame))).ravel()
    norm_ours, norm_theirs = np.linalg.norm(ours), np.linalg.norm(theirs)
    return {
        "frame": frame,
        "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
        "norm_ratio_ours_over_betse": float(norm_ours / norm_theirs) if norm_theirs else np.nan,
        "max_field_betse_V_per_m": float(np.max(np.abs(theirs))),
        "max_field_ours_V_per_m": float(np.max(np.abs(ours))),
    }


def r4_relaxation(bundle: Bundle) -> dict[str, list[float]]:
    """Time course of mean V_mem and max |V_mem - our GHK| (mV), every tenth frame."""
    t = np.asarray(bundle.array("time_integrated_s"))
    frames = sorted({0, 1, 2, 3, 5, *range(10, bundle.n_frames, 10), bundle.n_frames - 1})
    mean, gap = [], []
    for frame in frames:
        vmem = bundle.frame("vmem", frame)
        mean.append(float(np.nanmean(vmem)))
        gap.append(float(np.nanmax(np.abs(vmem - ghk_from_bundle(bundle, frame) * 1e3))))
    return {
        "time_s": [float(t[f]) for f in frames],
        "mean_vmem_mV": mean,
        "max_abs_vmem_minus_ghk_mV": gap,
    }
