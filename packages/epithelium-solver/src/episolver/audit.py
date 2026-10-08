"""Phase 1b audit (docs/plan.md): checks E1 to E8 and reported W1 to W3.

Tolerances are copied from docs/plan.md, where they were fixed before the solver's first run.
Closed forms and root-finding here are written independently of the time stepper.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, root
from tissuebundle.reader import Bundle
from tissuebundle.validate import validate_bundle

from episolver import __version__
from episolver.export import export_bundle
from episolver.flux import Constants, PumpParameters, electroflux, pump_nak
from episolver.geometry import build_tissue
from episolver.params import BETSE_ION_NAMES, RunConfig, load_config
from episolver.solver import Epithelium, Frame

logger = logging.getLogger(__name__)

PACKAGE = Path(__file__).resolve().parents[2]
CONFIGS = PACKAGE / "configs"
BETSE_GHK_BUNDLE = (
    PACKAGE.parent / "betse-adapter" / "data" / "work" / "ghk-audit" / "ghk-audit.tbundle"
)
NOTE = (
    "Tolerances were fixed in docs/plan.md (Phase 1b) on 2026-10-08, before the solver's "
    "first run. Machine-checked, not human-confirmed."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Record:
    def __init__(self, name: str, runs_dir: Path) -> None:
        self.path = runs_dir / name / "result.json"
        self.data: dict[str, Any] = {
            "run": name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": NOTE,
            "environment": {
                "episolver": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "checks": {},
            "reported": {},
        }

    def check(self, cid: str, passed: bool, value: Any, tolerance: str, detail: str) -> None:
        self.data["checks"][cid] = {
            "passed": bool(passed),
            "value": value,
            "tolerance": tolerance,
            "detail": detail,
        }
        logger.info("%s %s: %s (%s)", cid, "PASS" if passed else "FAIL", value, tolerance)

    def report(self, rid: str, value: Any, detail: str) -> None:
        self.data["reported"][rid] = {"value": value, "detail": detail}
        logger.info("%s: %s", rid, value)

    def save(self) -> Path:
        checks = self.data["checks"].values()
        self.data["all_passed"] = all(c["passed"] for c in checks) if checks else None
        self.data["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(_jsonable(self.data), indent=2) + "\n")
        return self.path


def simulate(config: RunConfig) -> tuple[Epithelium, list[Frame]]:
    model = Epithelium(config, build_tissue(config))
    frames, _ = model.run()
    return model, frames


def _update(config: RunConfig, **sections: dict) -> RunConfig:
    updates = {k: getattr(config, k).model_copy(update=v) for k, v in sections.items()}
    return config.model_copy(update=updates)


# ---------------------------------------------------------------------------- E1
def e1_passive_limit(rec: Record) -> None:
    config = RunConfig(
        name="e1-single-cell",
        footprint={"kind": "hex", "cols": 1, "rows": 1, "radius_um": 5.0},
        apical={"pump_alpha": 0.0},
        basolateral={"pump_alpha": 0.0},
        time={"init_dt_s": 5e-3, "init_s": 20.0, "dt_s": 5e-3, "sim_s": 0.0, "sample_s": 5e-3},
    )
    model, frames = simulate(config)
    last = frames[-1].data
    c = config.constants
    perm = (
        np.array([config.apical.base_d.get(i.name, 0.0) for i in config.ions])
        / c.membrane_thickness_m
    )
    z = np.array([i.z for i in config.ions])
    cin = last["conc_cell"][:, 0]
    cout = np.array([i.bath_apical for i in config.ions])
    cat, an = z > 0, z < 0
    num = (perm[cat] * cout[cat]).sum() + (perm[an] * cin[an]).sum()
    den = (perm[cat] * cin[cat]).sum() + (perm[an] * cout[an]).sum()
    ghk_mv = 1e3 * c.gas_constant * c.temperature_k / c.faraday * np.log(num / den)
    err = max(
        abs(last["vmem_apical"][0] * 1e3 - ghk_mv), abs(last["vmem_basolateral"][0] * 1e3 - ghk_mv)
    )
    rec.check(
        "E1",
        err <= 0.01,
        {
            "ghk_mV": ghk_mv,
            "vmem_apical_mV": last["vmem_apical"][0] * 1e3,
            "vmem_basolateral_mV": last["vmem_basolateral"][0] * 1e3,
            "max_abs_diff_mV": err,
        },
        "<= 0.01 mV",
        "one cell, pump off, no channels, 20 s; GHK in closed form",
    )


# ---------------------------------------------------------------------------- E4
def e4_betse_crosscheck(rec: Record, bundle_path: Path = BETSE_GHK_BUNDLE) -> None:
    if not bundle_path.exists():
        rec.check(
            "E4",
            False,
            f"missing {bundle_path}",
            "<= 0.5 mV",
            "run `uv run betse-adapter audit --only ghk-audit` first",
        )
        return
    b = Bundle(bundle_path)
    k = b.constants
    const = Constants(
        k["temperature_k"],
        k["gas_constant_j_per_k_mol"],
        k["faraday_c_per_mol"],
        k["membrane_thickness_m"],
    )
    pump = PumpParameters()
    names = [BETSE_ION_NAMES.get(n, n) for n in b.ions]
    z = np.asarray(b.ion_charges, float)
    i_na, i_k = names.index("na"), names.index("k")
    frame = b.baseline_frames[0]
    verts, offsets = b.polygons()
    lengths = np.empty(len(verts))
    for i in range(b.n_cells):
        v = verts[offsets[i] : offsets[i + 1]]
        lengths[offsets[i] : offsets[i + 1]] = np.linalg.norm(v - np.roll(v, 1, axis=0), axis=1)
    conc_in = b.frame("conc_cell", frame)
    conc_env = b.frame("conc_env", frame).reshape(len(names), -1)[
        :, b.array("mem_grid_index", mmap=False)
    ]
    diff = b.frame("diff_membrane", frame)
    vmem = b.frame("vmem", frame) * 1e-3
    worst, ours = 0.0, np.empty(b.n_cells)
    for i in range(b.n_cells):
        m = slice(offsets[i], offsets[i + 1])
        cin = np.repeat(conc_in[:, i : i + 1], m.stop - m.start, axis=1)
        cout, d, w = conc_env[:, m], diff[:, m], lengths[m]

        def current(v: float) -> float:
            vv = np.full(w.shape, v)
            f = electroflux(
                cout, cin, d, const.membrane_thickness_m, z[:, None], vv[None, :], const
            )
            f_na, f_k = pump_nak(cin[i_na], cout[i_na], cin[i_k], cout[i_k], vv, 1e-7, const, pump)
            q = (z[:, None] * f).sum(axis=0) + f_na + f_k
            return float((q * w).sum())

        ours[i] = brentq(current, -0.2, 0.2, xtol=1e-12)
        worst = max(worst, abs(ours[i] - vmem[i]))
    rec.check(
        "E4",
        worst * 1e3 <= 0.5,
        {
            "max_abs_diff_mV": worst * 1e3,
            "mean_betse_mV": float(vmem.mean() * 1e3),
            "mean_ours_mV": float(ours.mean() * 1e3),
        },
        "<= 0.5 mV",
        "zero-net-current voltage from episolver's flux and pump functions on "
        "BETSE's exported baseline state (ghk-audit bundle) vs BETSE's V_mem, every cell",
    )


# ---------------------------------------------------------------------------- E6
def kcl_residual(model: Epithelium) -> tuple[float, float]:
    """Nonlinear steady-state Kirchhoff residual at every free node, written without the solver's
    assembly: loops over branch types, currents accumulated with np.add.at."""
    n, g, t, c = model.n, model.g, model.tissue, model.const
    phi = model.phi_cell()
    pa = model.phi_bath("apical")
    pb = model.phi_bath("basolateral")
    res_cell = np.zeros(n)
    res_a = np.zeros(g)
    res_b = np.zeros(g)
    scale = 0.0
    idx, w = model.cell_nodes
    for d, bath_phi, res_bath in (("apical", pa, res_a), ("basolateral", pb, res_b)):
        for i in range(n):
            v = phi[i] - (bath_phi[idx[i]] * w[i]).sum()
            bath = model.bath[d]
            conc = model.conc[:, i]
            dd = model.diffusion(d)[:, i]
            f = electroflux(
                bath, conc, dd, c.membrane_thickness_m, model.z, np.full_like(dd, v), c
            )
            q = (model.z * f).sum()
            if model.alpha[d] > 0:
                fna, fk = pump_nak(
                    conc[model.i_na],
                    bath[model.i_na],
                    conc[model.i_k],
                    bath[model.i_k],
                    v,
                    model.alpha[d],
                    c,
                    model.pump,
                )
                q += fna + fk
            current = -c.faraday * q * model.area[d][i]  # out of the cell, into the bath
            res_cell[i] += current
            np.add.at(res_bath, idx[i], -w[i] * current)
            scale = max(scale, abs(current))
    a, b = t.pair_cells.T
    gj = model.config.gap_junctions
    for k in range(len(a)):
        dd = model.d_free * gj.surface * model.gj_open[k]
        f = electroflux(
            model.conc[:, b[k]],
            model.conc[:, a[k]],
            dd,
            gj.pore_m,
            model.z,
            np.full_like(dd, phi[a[k]] - phi[b[k]]),
            c,
        )
        current = -c.faraday * (model.z * f).sum() * t.pair_area_m2[k]  # from a to b
        res_cell[a[k]] += current
        res_cell[b[k]] -= current
    pidx, pw = model.pair_nodes
    for k in range(len(a)):
        current = (
            model.tj_per_length
            * t.pair_len_m[k]
            * ((pa[pidx[k]] * pw[k]).sum() - (pb[pidx[k]] * pw[k]).sum())
        )
        np.add.at(res_a, pidx[k], pw[k] * current)
        np.add.at(res_b, pidx[k], -pw[k] * current)
    for u, v in model.grid_edges:
        for arr, phi_b, gsh in (
            (res_a, pa, model.sheet["apical"]),
            (res_b, pb, model.sheet["basolateral"]),
        ):
            cur = gsh * (phi_b[u] - phi_b[v])
            arr[u] += cur
            arr[v] -= cur
    free_a = ~model.grid.boundary()
    r = np.concatenate((res_cell, res_a[free_a], res_b))
    return float(np.abs(r).max()), scale


def e6_column(rec: Record) -> None:
    config = load_config(CONFIGS / "hex-wound-polarized.yaml")
    config = _update(config, wound={"kind": "none"}, time={"sim_s": 0.0})
    model, frames = simulate(config)
    t = model.tissue
    edge = np.zeros(t.n_cells, bool)
    np.logical_or.at(edge, t.mem_cell[t.neighbour < 0], True)
    interior = ~edge
    for _ in range(2):
        bad = np.zeros(t.n_cells, bool)
        np.logical_or.at(bad, t.pair_cells[:, 0], ~interior[t.pair_cells[:, 1]])
        np.logical_or.at(bad, t.pair_cells[:, 1], ~interior[t.pair_cells[:, 0]])
        interior &= ~bad
    cells = np.flatnonzero(interior)
    data = frames[0].data
    const, pump = model.const, model.pump
    worst, worst_avg = 0.0, 0.0
    g_avg = t.area_m2 / (config.tight_junctions.resistance_ohm_cm2 * 1e-4)
    for i in cells[:: max(len(cells) // 20, 1)]:
        conc = data["conc_cell"][:, i]
        g_col = model.tj_per_length * t.edge_len_m[t.mem_cell == i].sum() / 2
        phi_a = model.bath_at_cells("apical")[i]

        def out_current(domain: str, v: float) -> float:
            bath = model.bath[domain]
            d = model.diffusion(domain)[:, i]
            f = electroflux(
                bath, conc, d, const.membrane_thickness_m, model.z, np.full_like(d, v), const
            )
            q = (model.z * f).sum()
            if model.alpha[domain] > 0:
                f_na, f_k = pump_nak(
                    conc[model.i_na],
                    bath[model.i_na],
                    conc[model.i_k],
                    bath[model.i_k],
                    v,
                    model.alpha[domain],
                    const,
                    pump,
                )
                q += f_na + f_k
            return -const.faraday * q

        def column(g_tj: float) -> np.ndarray:
            def equations(x: np.ndarray) -> list[float]:
                phi_i, phi_b = x
                i_ap = model.area["apical"][i] * out_current("apical", phi_i - phi_a)
                i_bl = model.area["basolateral"][i] * out_current("basolateral", phi_i - phi_b)
                return [i_ap + i_bl, i_bl - g_tj * (phi_b - phi_a)]

            phi_i, phi_b = root(equations, x0=[-0.05, 0.0], tol=1e-14).x
            return np.array([phi_i - phi_a, phi_i - phi_b, phi_a - phi_b]) * 1e3

        got = np.array([data["vmem_apical"][i], data["vmem_basolateral"][i], data["tep"][i]]) * 1e3
        worst = max(worst, float(np.max(np.abs(column(g_col) - got))))
        worst_avg = max(worst_avg, float(np.max(np.abs(column(g_avg[i]) - got))))
    rec.check(
        "E6",
        worst <= 0.05,
        {
            "max_abs_diff_mV": worst,
            "interior_cells": len(cells),
            "tep_mV": float(np.mean(data["tep"][cells]) * 1e3),
        },
        "<= 0.05 mV",
        "intact polarized hex sheet: V_ap, V_bl and TEP of interior cells vs "
        "the single-column zero-current root (scipy.optimize.root)",
    )
    residual, scale = kcl_residual(model)
    rec.report(
        "E6b",
        {
            "max_diff_vs_sheet_average_column_mV": worst_avg,
            "steady_state_kcl_residual_A": residual,
            "largest_membrane_current_A": scale,
            "relative_residual": residual / scale,
        },
        "Added after E6 failed, reported only. (a) The same column with the sheet-average "
        "junction density (1/R_t), which governs a sheet smaller than its length constant; "
        "(b) Kirchhoff's law at every free node at the final state, written independently of "
        "the solver's assembly; the residual is the capacitive current of the slow "
        "concentration drift",
    )


# ---------------------------------------------------------------------------- E7, E8, W
def _wound_geometry(model: Epithelium) -> tuple[np.ndarray, float]:
    t = model.tissue
    centre = t.centres_um[t.wounded].mean(axis=0)
    radius = float(np.max(np.linalg.norm(t.centres_um[t.wounded] - centre, axis=1))) + 5.0
    return centre, radius


def _field(frame: Frame, model: Epithelium, bath: str = "apical"):
    phi = frame.data[f"phi_bath_{bath}"]
    h = model.grid.spacing_um * 1e-6
    gy, gx = np.gradient(phi, h, h)
    x, y = model.grid.coordinates()
    return -gx, -gy, np.stack((x, y), axis=-1)


def wound_field(model: Epithelium, frames: list[Frame]) -> dict[str, Any]:
    """Field near the wound in each bath, in V/m (= mV/mm)."""
    centre, radius = _wound_geometry(model)
    out: dict[str, Any] = {"wound_centre_um": centre, "wound_radius_um": radius}
    for bath in ("apical", "basolateral"):
        peak, best = 0.0, None
        for frame in frames[1:]:
            ex, ey, _ = _field(frame, model, bath)
            m = float(np.hypot(ex, ey).max())
            if m > peak:
                peak, best = m, frame
        ex, ey, xy = _field(frames[-1], model, bath)
        r = xy - centre
        dist = np.linalg.norm(r, axis=-1)
        unit = r / np.maximum(dist[..., None], 1e-9)
        radial = ex * unit[..., 0] + ey * unit[..., 1]
        ring = (dist > radius) & (dist < radius + 30)
        mag = np.hypot(ex, ey)
        rings = {
            f"{a}-{a + 10}um": float(np.mean(mag[(dist - radius >= a) & (dist - radius < a + 10)]))
            for a in range(0, 70, 10)
        }
        out[bath] = {
            "peak_mV_per_mm": peak,
            "peak_time_s": best.time_s if best else None,
            "mean_radial_component_near_edge_mV_per_mm": float(radial[ring].mean()),
            "direction": "away from the wound" if radial[ring].mean() > 0 else "toward the wound",
            "mean_magnitude_by_distance_from_edge_mV_per_mm": rings,
        }
    return out


def e7_timestep(rec: Record) -> None:
    base = load_config(CONFIGS / "betse-wound-polarized.yaml")
    results = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        cfg = _update(base, time={"dt_s": dt, "sim_s": 0.5, "sample_s": 0.05})
        _, frames = simulate(cfg)
        results.append(
            np.stack(
                [
                    np.concatenate((f.data["vmem_apical"], f.data["vmem_basolateral"]))
                    for f in frames
                ]
            )
            * 1e3
        )
    e12 = float(np.nanmax(np.abs(results[0] - results[1])))
    e23 = float(np.nanmax(np.abs(results[1] - results[2])))
    rec.check(
        "E7",
        e23 <= 0.1,
        {
            "max_diff_dt_vs_half_mV": e12,
            "max_diff_half_vs_quarter_mV": e23,
            "observed_order": float(np.log2(e12 / e23)) if e23 > 0 else None,
        },
        "<= 0.1 mV (dt/2 vs dt/4)",
        "betse-wound-polarized, sim dt 1e-3, 5e-4, 2.5e-4 s, "
        "frames at identical times every 0.05 s for 0.5 s",
    )


def e8_grid(rec: Record) -> None:
    base = load_config(CONFIGS / "betse-wound-polarized.yaml")
    out = []
    for spacing in (5.0, 2.5):
        cfg = _update(base, baths={"spacing_um": spacing}, time={"sim_s": 0.5, "sample_s": 0.05})
        model, frames = simulate(cfg)
        centre, radius = _wound_geometry(model)
        ex, ey, xy = _field(frames[-1], model)
        near = np.linalg.norm(xy - centre, axis=-1) < radius + 30
        far = np.linalg.norm(model.tissue.centres_um - centre, axis=1)
        far_cells = far >= np.quantile(far, 0.8)
        out.append(
            (
                float(np.hypot(ex, ey)[near].max()),
                float(np.mean(frames[-1].data["tep"][far_cells]) * 1e3),
            )
        )
    rel = abs(out[1][0] - out[0][0]) / max(out[1][0], 1e-30)
    dtep = abs(out[1][1] - out[0][1])
    rec.check(
        "E8",
        rel <= 0.10 and dtep <= 0.1,
        {
            "peak_field_5um": out[0][0],
            "peak_field_2p5um": out[1][0],
            "relative_change": rel,
            "far_tep_5um_mV": out[0][1],
            "far_tep_2p5um_mV": out[1][1],
            "tep_change_mV": dtep,
        },
        "<= 10% and <= 0.1 mV (provisional)",
        "bath grid 5 vs 2.5 um, last frame at 0.5 s",
    )


def main_runs(rec: Record, work_root: Path) -> None:
    """The two BETSE-footprint runs, exported for Blender, with E2, E3 and W1 to W3."""
    tep_intact, fields = {}, {}
    worst_residual, worst_charge = 0.0, 0.0
    for name in ("betse-wound-polarized", "betse-wound-uniform", "hex-wound-polarized"):
        path = CONFIGS / f"{name}.yaml"
        config = load_config(path)
        model, frames = simulate(config)
        from tissuebundle.writer import sha256_file

        out = export_bundle(
            model,
            frames,
            work_root / name / f"{name}.tbundle",
            str(path),
            sha256_file(path),
            overwrite=True,
        )
        errors = validate_bundle(out)
        rec.check(f"contract:{name}", not errors, errors, "no violations", "tissuebundle validate")
        worst_residual = max(worst_residual, model.diag.max_kcl_residual)
        worst_charge = max(worst_charge, model.diag.max_charge_error)
        tep_intact[name] = float(np.mean(frames[0].data["tep"]) * 1e3)
        fields[name] = wound_field(model, frames)
        fields[name]["tep_after_mV"] = float(
            np.nanmean(np.where(model.tissue.wounded, np.nan, frames[-1].data["tep"])) * 1e3
        )
    rec.check(
        "E2",
        worst_charge <= 1e-6,
        worst_charge,
        "<= 1e-6 (of 1 mV of membrane charge)",
        "every step and live cell of the three exported runs",
    )
    rec.check(
        "E3",
        worst_residual <= 1e-9,
        worst_residual,
        "<= 1e-9",
        "every linear solve of the three exported runs",
    )
    rec.report("W1", tep_intact, "mean TEP of the intact sheet at the end of init (mV)")
    rec.report(
        "W2",
        {k: v for k, v in fields.items() if "uniform" not in k},
        "wound field in the apical bath, polarized membranes",
    )
    rec.report("W3", fields["betse-wound-uniform"], "the same wound, BETSE-uniform membranes")


def run_audit(only: list[str] | None, runs_dir: Path, work_root: Path) -> list[Path]:
    rec = Record("epithelium", runs_dir)
    rec.data["revision"] = (
        "v2: each removed cell's bath shunt is spread over the grid nodes inside its footprint. "
        "v1 used a point shunt at the cell centre; its record, in which E8 failed, is kept in "
        "runs/epithelium-v1-point-shunt/. Tolerances unchanged."
    )
    steps = {
        "E1": e1_passive_limit,
        "E4": e4_betse_crosscheck,
        "E6": e6_column,
        "E7": e7_timestep,
        "E8": e8_grid,
        "MAIN": lambda r: main_runs(r, work_root),
    }
    for key, fn in steps.items():
        if only and key not in only:
            continue
        fn(rec)
    return [rec.save()]
