"""Wide skin block audit (docs/plan.md, "Wide skin block and held-out prediction": L1 to L5, S4,
S5). Tolerances are those written there before any wide run. Runs go in parallel processes."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from episolver.audit import Record
from episolver.skin_audit import (
    FULL,
    HALVED,
    WIDE_BANDS,
    _mean,
    _simulate,
    battery_mv,
    lateral_field,
    node_volume,
)
from episolver.skin_physics import EPIDERMIS, Frame, SkinModel, SkinRunConfig

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parents[2] / "configs" / "skin-forearm-wide.yaml"
NOTE = (
    "Tolerances were fixed in docs/plan.md (wide skin block) before any wide-block run. "
    "Machine-checked, not human-confirmed."
)


def wide_config(**changes) -> SkinRunConfig:
    base = SkinRunConfig.model_validate(yaml.safe_load(CONFIG.read_text()))
    skin = changes.pop("skin", {})
    physics = {"time": FULL} | changes
    return base.model_copy(
        update={
            "skin": base.skin.model_copy(
                update={k: v for k, v in skin.items() if k != "wound"}
                | (
                    {"wound": base.skin.wound.model_copy(update=skin["wound"])}
                    if "wound" in skin
                    else {}
                )
            ),
            "physics": base.physics.model_copy(update=physics),
        }
    )


def battery_beyond(model: SkinModel, frame: Frame, beyond_um: float) -> float:
    """The battery more than ``beyond_um`` from the wound centre (mV)."""
    return battery_mv(model, frame, far_um=beyond_um)


def potential_profile(
    model: SkinModel, frame: Frame, region_frame: Frame | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Volume-weighted ECS potential beneath the stratum corneum in 20 µm bands of distance
    from the wound centre (mV)."""
    z, _, x = model.grid.coords()
    d = np.abs(x - model.config.skin.wound.x_um)
    zone = ((region_frame or frame).data["region"] == EPIDERMIS) & (z > model.tj_z + 0.1)
    w = node_volume(model)
    edges = np.arange(0.0, d.max() + 20.0, 20.0)
    mids, values = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = zone & (d >= lo) & (d < hi)
        if band.any():
            mids.append(0.5 * (lo + hi))
            values.append(_mean(frame.data["phi_ecs"], band, w) * 1e3)
    return np.array(mids), np.array(values)


def space_constant_um(mids: np.ndarray, phi: np.ndarray) -> dict:
    """Exponential fit of the potential beneath the stratum corneum, relative to its value at
    the far wall, over the distances where it is above 5% of its largest value."""
    rel = phi - phi[-1]
    use = (mids >= 40.0) & (np.abs(rel) > 0.05 * np.abs(rel).max())
    slope, _ = np.polyfit(mids[use], np.log(np.abs(rel[use])), 1)
    return {
        "lambda_um": float(-1.0 / slope),
        "fit_from_um": float(mids[use].min()),
        "fit_to_um": float(mids[use].max()),
        "points": int(use.sum()),
    }


def run_job(job: str) -> tuple[str, dict]:
    cfg = {
        "main": lambda: wide_config(),
        "halved": lambda: wide_config(time=HALVED),
        "w640": lambda: wide_config(skin={"width_x_um": 640.0}),
        "grid3": lambda: wide_config(spacing_um=3.0),
        "half320": lambda: wide_config(skin={"width_x_um": 320.0}),
        "full640": lambda: wide_config(skin={"width_x_um": 640.0, "wound": {"x_um": 320.0}}),
    }[job]()
    model, frames = _simulate(cfg)
    first, last = frames[0], frames[-1]
    out = {
        "kcl": model.diag.max_kcl_residual,
        "charge": model.diag.max_charge_error,
        "max_cg_iterations": model.diag.max_cg_iterations,
        "field_beneath_corneum": lateral_field(model, last, "beneath_corneum", WIDE_BANDS),
        "field_viable": lateral_field(model, last, "viable", WIDE_BANDS),
    }
    if job == "main":
        out["battery_far_intact_mV"] = battery_beyond(model, first, 800.0)
        out["battery_far_after_mV"] = battery_beyond(model, last, 800.0)
        mids, phi = potential_profile(model, last)
        out["space_constant"] = space_constant_um(mids, phi)
    if job in ("main", "halved"):
        out["vmem_mV"] = np.stack([f.data["vmem"] for f in frames]) * 1e3
        out["times_s"] = [f.time_s for f in frames]
    return job, out


def _band_changes(a: dict, b: dict) -> dict:
    """Relative change of each band mean, |a - b| / |b|, per layer."""
    out = {}
    for layer in ("field_beneath_corneum", "field_viable"):
        for lo, hi in WIDE_BANDS:
            key = f"mean_{lo}_{hi}_um"
            out[f"{layer}:{key}"] = {
                "a": a[layer][key],
                "b": b[layer][key],
                "rel_change": abs(a[layer][key] - b[layer][key]) / abs(b[layer][key]),
            }
    return out


def run_wide_audit(runs_dir: Path) -> Path:
    rec = Record("skin-wide", runs_dir)
    rec.data["note"] = NOTE
    jobs = ["main", "halved", "w640", "grid3", "half320", "full640"]
    os.environ.setdefault("OMP_NUM_THREADS", "2")
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
        raise RuntimeError(f"wide audit jobs failed: {errors}")
    main = results["main"]

    b0, b1 = main["battery_far_intact_mV"], main["battery_far_after_mV"]
    rel = abs(b1 - b0) / abs(b0)
    rec.check(
        "L1",
        rel <= 0.05,
        {"intact_mV": b0, "after_mV": b1, "rel_change": rel},
        "within 5%",
        "battery more than 800 um from the wound centre, 2 s after the cut",
    )
    l2 = _band_changes(results["w640"], main)
    rec.check(
        "L2",
        all(v["rel_change"] <= 0.05 for v in l2.values()),
        l2,
        "within 5%",
        "band means at half-width 640 (a) vs 960 um (b), 2 s after the cut",
    )
    l3 = _band_changes(results["grid3"], main)
    rec.check(
        "L3",
        all(v["rel_change"] <= 0.10 for v in l3.values()),
        l3,
        "within 10%",
        "band means with fine spacing 3 (a) vs 4 um (b), 2 s after the cut",
    )
    a, b = main["vmem_mV"], results["halved"]["vmem_mV"]
    assert np.allclose(main["times_s"], results["halved"]["times_s"])
    k4 = float(np.nanmax(np.abs(a - b)))
    worst = max(r["kcl"] for r in results.values())
    charge = max(r["charge"] for r in results.values())
    rec.check(
        "L4",
        worst <= 1e-9 and charge <= 1e-6 and k4 <= 0.1,
        {
            "max_kcl_residual": worst,
            "max_charge_error": charge,
            "dt_halved_max_abs_mV": k4,
            "max_cg_iterations": max(r["max_cg_iterations"] for r in results.values()),
        },
        "as K1 (1e-9), K2 (1e-6), K4 (0.1 mV)",
        "every wide run; time step on the main run",
    )
    rec.report(
        "L5",
        _band_changes(results["half320"], results["full640"]),
        "half domain at half-width 320 um (a) against a full 640 um block with the wound "
        "centred (b); the cell arrangements differ, so this is reported, not judged",
    )
    rec.report(
        "S4",
        main["space_constant"] | {"reference": "0.3-0.4 mm, guinea pig (Barker 1982)"},
        "exponential fit of the potential beneath the stratum corneum 2 s after the cut",
    )
    rec.report(
        "S5",
        {
            "field_beneath_corneum": main["field_beneath_corneum"],
            "field_viable": main["field_viable"],
            "reference": "107 +- 13 mV/mm, young human forearm (Nuccitelli 2011)",
        },
        "band means in mV/mm (= V/m), positive toward the wound, 2 s after the cut; the "
        "battery is calibrated, so agreement would not be a prediction",
    )
    return rec.save()


# ---------------------------------------------------------------------------- follow-ups (L6, S6)
FOLLOWUP_NOTE = (
    "L6 and S6 were fixed in docs/plan.md (follow-ups to the wide audit) before these runs. "
    "Machine-checked, not human-confirmed."
)


def cable_space_constant_um(model: SkinModel) -> float:
    """One-dimensional cable estimate: sqrt(R_barrier / (r_outer + r_inner)), with the sheet
    resistances of the space above the tight junctions and of everything below them."""
    ph, spec = model.config.physics, model.config.skin
    r_barrier = ph.tight_junction_ohm_cm2 * 1e-4  # ohm m2
    sigma_epi = ph.sigma_fluid * ph.epidermal_ecs_factor
    t_outer = (model.granular_top - model.tj_z) * 1e-6
    t_viable = model.tj_z * 1e-6
    t_dermis = spec.dermis_depth_um * 1e-6
    r_outer = 1.0 / (sigma_epi * t_outer)
    r_inner = 1.0 / (sigma_epi * t_viable + ph.sigma_dermis * t_dermis)
    return float(np.sqrt(r_barrier / (r_outer + r_inner)) * 1e6)


def wound_change_space_constant_um(model: SkinModel, first: Frame, last: Frame) -> dict:
    """The space constant of what the wound changed: the potential beneath the stratum corneum
    after the cut minus before it, which removes the static variation the cell arrangement
    imposes on the intact profile; fitted where the change exceeds 5% of its largest value."""
    mids, after = potential_profile(model, last)
    _, before = potential_profile(model, first, region_frame=last)  # same nodes in both
    return space_constant_um(mids, after - before)


def run_followup_job(job: str) -> tuple[str, dict]:
    base = wide_config().physics
    cfg = {
        "main960": lambda: wide_config(),
        "crop640": lambda: wide_config(skin={"width_x_um": 640.0, "seed_width_x_um": 960.0}),
        "barrier_half": lambda: wide_config(
            tight_junction_ohm_cm2=base.tight_junction_ohm_cm2 / 2
        ),
        "barrier_double": lambda: wide_config(
            tight_junction_ohm_cm2=base.tight_junction_ohm_cm2 * 2
        ),
        "ecs_double": lambda: wide_config(epidermal_ecs_factor=base.epidermal_ecs_factor * 2),
        "dermis_double": lambda: wide_config(sigma_dermis=base.sigma_dermis * 2),
    }[job]()
    model, frames = _simulate(cfg)
    last = frames[-1]
    mids, phi = potential_profile(model, last)
    return job, {
        "field_beneath_corneum": lateral_field(model, last, "beneath_corneum", WIDE_BANDS),
        "field_viable": lateral_field(model, last, "viable", WIDE_BANDS),
        "space_constant": space_constant_um(mids, phi),
        "space_constant_of_change": wound_change_space_constant_um(model, frames[0], last),
        "cable_estimate_um": cable_space_constant_um(model),
        "battery_far_intact_mV": battery_beyond(model, frames[0], 800.0),
        "kcl": model.diag.max_kcl_residual,
        "charge": model.diag.max_charge_error,
    }


def run_followups(runs_dir: Path) -> Path:
    rec = Record("skin-wide-followup", runs_dir)
    rec.data["note"] = FOLLOWUP_NOTE
    jobs = ["main960", "crop640", "barrier_half", "barrier_double", "ecs_double", "dermis_double"]
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = dict(pool.map(run_followup_job, jobs))
    changes = _band_changes(results["crop640"], results["main960"])
    for change in changes.values():
        change["abs_change"] = abs(change["a"] - change["b"])
    ok = all(c["rel_change"] <= 0.05 or c["abs_change"] <= 0.1 for c in changes.values())
    rec.check(
        "L6",
        ok,
        changes,
        "each band within 5% or 0.1 mV/mm",
        "960 um half-block (b) against the same cells cut at 640 um (a), 2 s after the cut",
    )
    rec.report(
        "S6b",
        {k: v["space_constant_of_change"] for k, v in results.items()},
        "post hoc, added after S6 gave a non-monotonic profile for the halved barrier: the "
        "same fit applied to the change the wound made (after minus before), which removes "
        "the static variation from the cell arrangement",
    )
    rec.report(
        "S6",
        {
            k: {
                "space_constant": v["space_constant"],
                "cable_estimate_um": v["cable_estimate_um"],
                "battery_far_intact_mV": v["battery_far_intact_mV"],
                "field_beneath_corneum_20_50": v["field_beneath_corneum"]["mean_20_50_um"],
            }
            for k, v in results.items()
            if k != "crop640"
        },
        "space constant (exponential fit beneath the stratum corneum, 2 s after the cut) "
        "and its one-dimensional cable estimate, for the parameter changes named; "
        "reference 0.3-0.4 mm (guinea pig, Barker 1982)",
    )
    rec.data["conservation"] = {
        k: {"kcl": v["kcl"], "charge": v["charge"]} for k, v in results.items()
    }
    return rec.save()
