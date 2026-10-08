"""Skin physics audit (docs/plan.md, Phase 2: checks K1 to K5, reported S1 to S3).

The tolerances are those fixed in docs/plan.md before the skin solver first ran. The runs are
independent, so they go in parallel processes.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from episolver.audit import Record
from episolver.flux import ghk_voltage
from episolver.params import Domain, TimeSpec
from episolver.skin import generate
from episolver.skin_physics import EPIDERMIS, Frame, SkinModel, SkinRunConfig

logger = logging.getLogger(__name__)

NOTE = (
    "Tolerances were fixed in docs/plan.md (Phase 2, skin physics) before the skin solver's "
    "first run. Machine-checked, not human-confirmed."
)
FULL = TimeSpec(init_dt_s=0.02, init_s=2.0, dt_s=0.01, sim_s=2.0, sample_s=0.05)
HALVED = TimeSpec(init_dt_s=0.01, init_s=2.0, dt_s=0.005, sim_s=2.0, sample_s=0.05)
# Passive limit: backward Euler, so the steady state does not depend on the step; 30 s is many
# membrane time constants.
PASSIVE = TimeSpec(init_dt_s=0.1, init_s=30.0, dt_s=0.1, sim_s=0.0, sample_s=0.1)


def _config(skin: dict | None = None, **physics) -> SkinRunConfig:
    base = SkinRunConfig(name="skin-audit")
    update = {"physics": base.physics.model_copy(update=physics)}
    if skin:
        update["skin"] = base.skin.model_copy(update=skin)
    return base.model_copy(update=update)


def battery_mv(model: SkinModel, frame: Frame, far_um: float = 50.0) -> float:
    """Mean ECS potential below minus above the tight-junction band, more than ``far_um`` from
    the wound centre (the definition used by the Blender legend)."""
    z, _, x = model.grid.coords()
    region = frame.data["region"]
    far = np.abs(x - model.config.skin.wound.x_um) > far_um
    inner = (region == EPIDERMIS) & (z < model.tj_z - 2) & (z > model.tj_z - 12) & far
    outer = (region == EPIDERMIS) & (z > model.tj_z + 0.1) & far
    phi, w = frame.data["phi_ecs"], node_volume(model)
    return float((_mean(phi, inner, w) - _mean(phi, outer, w)) * 1e3)


def node_volume(model: SkinModel) -> np.ndarray:
    """Dual-cell volume of every ECS node (µm³), for means over a graded grid."""
    g = model.grid
    wz, wy, wx = (g._dual(a) for a in (g.zs, g.ys, g.xs))
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def _mean(values: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def passive_ghk_gap_mv(model: SkinModel) -> np.ndarray:
    """|V_mem - GHK| per living membrane face [mV], GHK from the face's base permeabilities and
    the current concentrations (check K3)."""
    v = model.face_vmem()
    cin = model.conc[:, model.f_cell]
    cout = np.repeat(model.c_out[:, None], len(v), axis=1)
    perm = model.f_base_d / model.const.membrane_thickness_m
    return np.abs(v - ghk_voltage(cin, cout, perm, model.z, model.const)) * 1e3


def passive_config(
    base: SkinRunConfig, time: TimeSpec, gap_junctions: bool = True
) -> SkinRunConfig:
    """Channels and pumps off on every face, no wound; gap junctions optionally off too."""
    passive = Domain(pump_alpha=0.0, channels=())
    gj = base.physics.gap_junctions.model_copy(update={"enabled": gap_junctions})
    return base.model_copy(
        update={
            "skin": base.skin.model_copy(
                update={"wound": base.skin.wound.model_copy(update={"kind": "none"})}
            ),
            "physics": base.physics.model_copy(
                update={
                    "membranes": {
                        k: {"apical": passive, "basolateral": passive} for k in (0, 1, 2)
                    },
                    "time": time,
                    "gap_junctions": gj,
                }
            ),
        }
    )


SMALL_BANDS = tuple((lo, lo + 10) for lo in (20, 30, 40, 50, 60, 70))
WIDE_BANDS = ((20, 50), (50, 100), (100, 200), (200, 400))


def lateral_field(
    model: SkinModel, frame: Frame, layer: str, bands=SMALL_BANDS
) -> dict[str, float]:
    """Lateral field E_x in V/m (= mV/mm), signed positive toward the wound, by distance from
    the wound centre (the cut is 40 µm wide at the top, so its edge is at 20 µm). Means are
    weighted by node volume (since 2026-10-08; the first small-block records used node counts)."""
    z, _, x = model.grid.coords()
    region = frame.data["region"]
    ex = frame.data["efield_x"]
    offset = x - model.config.skin.wound.x_um
    toward = -ex * np.sign(offset)
    zone = region == EPIDERMIS
    if layer == "beneath_corneum":
        zone &= z > model.tj_z + 0.1
    else:  # viable epidermis below the barrier, above the papillae
        zone &= z < model.tj_z - 2
    out: dict[str, float] = {}
    w = node_volume(model)
    for lo, hi in bands:
        band = zone & (np.abs(offset) >= lo) & (np.abs(offset) < hi)
        if band.any():
            out[f"mean_{lo}_{hi}_um"] = _mean(toward, band, w)
    near = zone & (np.abs(offset) >= 20) & (np.abs(offset) < 50)
    out["peak_abs_20_50_um"] = float(np.abs(ex[near]).max())
    return out


def _simulate(cfg: SkinRunConfig) -> tuple[SkinModel, list[Frame]]:
    model = SkinModel(cfg, generate(cfg.skin))
    return model, model.run()


def run_job(job: str) -> tuple[str, dict]:
    if job in ("main", "halved", "grid3", "no_na"):
        overrides = {
            "main": {},
            "halved": {"time": HALVED},
            "grid3": {"spacing_um": 3.0},
            "no_na": {"apical_na_override": 0.0},
        }[job]
        model, frames = _simulate(_config(**({"time": FULL} | overrides)))
        return job, {
            "kcl": model.diag.max_kcl_residual,
            "charge": model.diag.max_charge_error,
            "vmem_mV": np.stack([f.data["vmem"] for f in frames]) * 1e3,
            "times_s": [f.time_s for f in frames],
            "battery_intact_mV": battery_mv(model, frames[0]),
            "battery_after_mV": battery_mv(model, frames[-1]),
            "field_viable": lateral_field(model, frames[-1], "viable"),
            "field_beneath_corneum": lateral_field(model, frames[-1], "beneath_corneum"),
        }
    if job in ("passive", "passive_uncoupled"):
        cfg = passive_config(SkinRunConfig(name="skin-audit"), PASSIVE, job == "passive")
        model, _ = _simulate(cfg)
        gap = passive_ghk_gap_mv(model)
        layer = model.skin.cell_layer[model.f_cell]
        return job, {
            "max_abs_mV": float(gap.max()),
            "mean_abs_mV": float(gap.mean()),
            "max_abs_by_layer_mV": {
                int(k): float(gap[layer == k].max()) for k in np.unique(layer)
            },
            "faces": len(gap),
        }
    raise ValueError(job)


def run_skin_audit(runs_dir: Path) -> Path:
    rec = Record("skin", runs_dir)
    rec.data["note"] = NOTE
    jobs = ["main", "halved", "grid3", "no_na", "passive", "passive_uncoupled"]
    os.environ.setdefault("OMP_NUM_THREADS", "2")  # inherited by the worker processes
    with ProcessPoolExecutor(len(jobs)) as pool:
        futures = {job: pool.submit(run_job, job) for job in jobs}
        results, errors = {}, {}
        for job, future in futures.items():
            try:
                results[job] = future.result()[1]
            except Exception as exc:  # recorded, then raised: no check is passed by omission
                errors[job] = repr(exc)
    if errors:
        rec.data["errors"] = errors
        rec.save()
        raise RuntimeError(f"skin audit jobs failed: {errors}")
    main, halved, grid3 = results["main"], results["halved"], results["grid3"]

    rec.check(
        "K1",
        main["kcl"] <= 1e-9,
        main["kcl"],
        "<= 1e-9",
        "max relative residual of every Kirchhoff solve, full run (2 s init, cut, 2 s)",
    )
    rec.check(
        "K2",
        main["charge"] <= 1e-6,
        main["charge"],
        "<= 1e-6 of 1 mV of membrane charge",
        "every step and living cell, full run",
    )
    p = results["passive"]
    rec.check(
        "K3",
        p["max_abs_mV"] <= 0.01,
        p,
        "<= 0.01 mV",
        "channels and pumps off, no wound, 30 s: every living membrane face against the "
        "closed-form GHK voltage of its base permeabilities and current concentrations",
    )
    rec.report(
        "K3b",
        results["passive_uncoupled"],
        "added after the design check on the small test skin (before this run) showed "
        "the K3 gap growing with time when gap junctions are on: K3 with gap junctions "
        "off as well. Cells of different surface-to-volume ratio drift apart in "
        "concentration without pumps, and gap junctions then hold each cell away from "
        "its own GHK voltage; uncoupled, only the lag behind the drifting GHK remains",
    )
    assert np.allclose(main["times_s"], halved["times_s"])
    k4 = float(np.nanmax(np.abs(main["vmem_mV"] - halved["vmem_mV"])))
    rec.check(
        "K4",
        k4 <= 0.1,
        {"max_abs_mV": k4, "frames": len(main["times_s"])},
        "<= 0.1 mV",
        "init 0.02 -> 0.01 s and sim 0.01 -> 0.005 s; cell V_mem at the end of init and "
        "every 0.05 s after the cut",
    )
    b4, b3 = main["battery_intact_mV"], grid3["battery_intact_mV"]
    f4, f3 = main["field_viable"]["peak_abs_20_50_um"], grid3["field_viable"]["peak_abs_20_50_um"]
    rel_b, rel_f = abs(b3 - b4) / abs(b3), abs(f3 - f4) / abs(f3)
    rec.check(
        "K5",
        rel_b <= 0.05 and rel_f <= 0.15,
        {
            "battery_4um_mV": b4,
            "battery_3um_mV": b3,
            "battery_rel_change": rel_b,
            "peak_field_4um_V_per_m": f4,
            "peak_field_3um_V_per_m": f3,
            "field_rel_change": rel_f,
            "field_profile_4um": main["field_viable"],
            "field_profile_3um": grid3["field_viable"],
        },
        "battery <= 5%, peak field <= 15% (provisional)",
        "ECS grid 4 vs 3 um laterally; battery before the cut, peak |E_x| in the viable "
        "epidermis 20-50 um from the wound centre at 2 s after the cut",
    )

    rec.report(
        "S1",
        {"intact_mV": b4, "2_s_after_cut_mV": main["battery_after_mV"]},
        "against 10-60 mV (inside positive) for human skin: calibrated, not predicted; "
        "the apical Na density (3e-16 m2/s) and the barrier (2000 ohm cm2) were chosen to "
        "land inside the range",
    )
    rec.report(
        "S2",
        {
            "viable_epidermis": main["field_viable"],
            "beneath_stratum_corneum": main["field_beneath_corneum"],
        },
        "lateral field 2 s after the cut, V/m = mV/mm, positive toward the wound; not "
        "comparable with 107 +- 13 (human) or 177 +- 14 mV/mm (mouse): the block is "
        "160 um wide, smaller than the field's 0.3-0.4 mm space constant, and the "
        "battery is calibrated",
    )
    n = results["no_na"]
    rec.report(
        "S3",
        {
            "battery_intact_mV": n["battery_intact_mV"],
            "battery_after_mV": n["battery_after_mV"],
            "viable_epidermis": n["field_viable"],
            "beneath_stratum_corneum": n["field_beneath_corneum"],
        },
        "the same skin without apical Na channels in the granular layer",
    )
    return rec.save()
