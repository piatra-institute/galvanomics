"""Rival mechanisms after the held-out test (docs/plan.md, "Rival mechanisms after H1": R1-R4).

Rival A: a Na-only battery, and drugs reach only cells within ``d`` of the wound centre.
Rival B: the same battery plus Ca2+-activated Cl- channels switched on at wounding on every
living face touching the wound fluid; drugs reach everything.

Both share the intact model (B's Cl- channels exist only after the cut), so they share the
apical Na density calibrated to the intact battery T; A's d and B's channel density are then
calibrated to the measured 64% amiloride reduction. Their predictions for interventions not
yet done are frozen (configuration hashes recorded) and the most discriminating one is named.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from episolver.audit import Record
from episolver.params import TimeSpec
from episolver.skin_amiloride import BLOCK, MEASURED, SEM, field_beneath_corneum
from episolver.skin_audit import FULL, _simulate, battery_mv
from episolver.skin_physics import Modifier, SkinRunConfig

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parents[2] / "configs" / "skin-mouse-wide.yaml"
RUNS = Path(__file__).parents[2] / "runs"
SIGMA = SEM * np.sqrt(11)  # per-wound SD of a percent change (amiloride arm, Table 1)
ANO1_PS = 2.63  # single-channel conductance, noise analysis (Hartzell & Whitlock 2016)
NOTE = (
    "R1-R4 were fixed in docs/plan.md (rival mechanisms after H1) after the chloride sources "
    "and before any run of either rival. Machine-checked, not human-confirmed."
)
INIT_ONLY = TimeSpec(init_dt_s=0.02, init_s=2.0, dt_s=0.01, sim_s=0.0, sample_s=0.05)
AMILORIDE = "amiloride"


def target_battery_mv() -> float:
    record = json.loads((RUNS / "skin-amiloride" / "result.json").read_text())
    return record["reported"]["H1-variants"]["value"]["runs"]["unblocked"]["battery_intact_mV"]


def config(
    a_na: float,
    extra: dict | None = None,
    time: TimeSpec = FULL,
    base_modifiers: tuple[Modifier, ...] = (),
) -> SkinRunConfig:
    base = SkinRunConfig.model_validate(yaml.safe_load(CONFIG.read_text()))
    extra = dict(extra or {})
    modifiers = base_modifiers + tuple(extra.pop("modifiers", ()))
    physics = {
        "ion_profile": "mammal_no_ca",
        "cell_conc_override": {"cl": 6.8, "m": 9.2},  # keratinocyte [Cl-]i (Yamanoi 2023)
        "apical_na_override": a_na,
        "time": time,
        "modifiers": modifiers,
    } | extra
    return base.model_copy(update={"physics": base.physics.model_copy(update=physics)})


def interventions(rival: str, d_um: float | None, g_cl: float) -> dict[str, dict]:
    """Physics changes per intervention. In A drugs reach cells within d; in B everywhere."""
    within = d_um if rival == "A" else None
    wound_cl = {"wound_cl_d": g_cl} if rival == "B" else {}
    amil = Modifier(
        target="NaLeak",
        side="apical",
        factor=1.0 - BLOCK,
        within_um=within,
        label="amiloride 1 mM",
    )
    cl_block = Modifier(
        target="ClLeak",
        side="any",
        factor=0.0,
        within_um=within,
        label="Cl- channel blocker (ANO1 inhibitor)",
    )
    pge2 = Modifier(
        target="ClLeak",
        side="any",
        factor=2.0,
        within_um=within,
        label="PGE2 as doubled Ca2+-activated Cl- conductance",
    )
    return {
        "none": wound_cl,
        AMILORIDE: wound_cl | {"modifiers": (amil,)},
        "I1_cl_blocker": wound_cl | {"modifiers": (cl_block,)},
        "I2_amiloride_plus_cl_blocker": wound_cl | {"modifiers": (amil, cl_block)},
        "I4_pge2": wound_cl | {"modifiers": (pge2,)},
    }


def sha(cfg: SkinRunConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()


def run_case(case: tuple[str, str]) -> tuple[str, dict]:
    """Run one configuration given as (key, JSON) and return the H1 measures."""
    key, payload = case
    cfg = SkinRunConfig.model_validate_json(payload)
    model, frames = _simulate(cfg)
    layer = model.skin.cell_layer
    vm = frames[0].data["vmem"] * 1e3
    out = {
        "battery_intact_mV": battery_mv(model, frames[0], far_um=300.0),
        "vmem_basal_spinous_mV": float(np.nanmean(vm[(layer == 0) | (layer == 1)])),
        "vmem_granular_mV": float(np.nanmean(vm[layer == 2])),
    }
    if len(frames) > 1:
        last = frames[-1]
        out["field_20_100"] = field_beneath_corneum(model, last, 20.0, 100.0)
        out["field_200_400"] = field_beneath_corneum(model, last, 200.0, 400.0)
        g = last.data.get("chan_g_clleak")
        if g is not None and cfg.physics.wound_cl_d > 0:
            k = next(i for i, ch in enumerate(model.channels) if ch[7] == "wound")
            on = model.channel_scale[k] > 0
            faces = model.f_face[on]
            out["wound_cl_S_per_m2"] = float(np.mean(g[faces])) if len(faces) else 0.0
            out["wound_cl_faces"] = int(on.sum())
    out["kcl"] = model.diag.max_kcl_residual
    out["charge"] = model.diag.max_charge_error
    return key, out


def run_batch(cases: dict[str, SkinRunConfig], workers: int = 8) -> dict[str, dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    payload = [(k, c.model_dump_json()) for k, c in cases.items()]
    with ProcessPoolExecutor(min(workers, len(payload))) as pool:
        return dict(pool.map(run_case, payload))


def _crossing(xs: list[float], ys: list[float], target: float) -> float:
    """Where a monotonic sampled curve crosses the target (linear interpolation)."""
    order = np.argsort(ys)
    return float(np.interp(target, np.asarray(ys)[order], np.asarray(xs)[order]))


def run_rivals(runs_dir: Path = RUNS) -> Path:
    rec = Record("skin-rivals", runs_dir)
    rec.data["note"] = NOTE
    target = target_battery_mv()
    rec.data["target_battery_mV"] = target

    # Phase 1: the shared apical Na density, on the intact block.
    grid = [2.0e-16, 2.5e-16, 3.0e-16, 3.5e-16, 4.0e-16, 5.0e-16]
    res = run_batch({f"a{a:.2e}": config(a, time=INIT_ONLY) for a in grid})
    batteries = [res[f"a{a:.2e}"]["battery_intact_mV"] for a in grid]
    a_na = float(np.exp(_crossing(list(np.log(grid)), batteries, target)))
    check = run_batch({"a": config(a_na, time=INIT_ONLY)})["a"]["battery_intact_mV"]
    rec.data["phase1"] = {"grid": grid, "battery_mV": batteries, "a_na": a_na, "check_mV": check}

    # Phase 2: A's d and B's density, on the wounded block.
    d_grid = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    g_grid = [1e-16, 3e-16, 1e-15, 3e-15, 1e-14, 3e-14]
    cases = {"A:none": config(a_na)}
    for d in d_grid:
        cases[f"A:d{d:g}"] = config(a_na, interventions("A", d, 0.0)[AMILORIDE])
    for g in g_grid:
        for name in ("none", AMILORIDE):
            cases[f"B:{name}:g{g:.0e}"] = config(a_na, interventions("B", None, g)[name])
    res = run_batch(cases, workers=8)
    base_a = res["A:none"]["field_20_100"]
    red_a = [1 - res[f"A:d{d:g}"]["field_20_100"] / base_a for d in d_grid]
    red_b = [
        1
        - res[f"B:{AMILORIDE}:g{g:.0e}"]["field_20_100"] / res[f"B:none:g{g:.0e}"]["field_20_100"]
        for g in g_grid
    ]
    d_star = _crossing(d_grid, red_a, MEASURED)
    g_star = float(np.exp(_crossing(list(np.log(g_grid)), red_b, MEASURED)))
    rec.data["phase2"] = {
        "d_grid": d_grid,
        "reduction_A": red_a,
        "g_grid": g_grid,
        "reduction_B": red_b,
        "d_star_um": d_star,
        "g_star": g_star,
        "field_A_unblocked": base_a,
    }
    rec.save()

    # Phase 3: frozen predictions (hash first, then run).
    frozen, cases = {}, {}
    for rival, d, g in (("A", d_star, 0.0), ("B", None, g_star)):
        for name, extra in interventions(rival, d, g).items():
            cfg = config(a_na, extra)
            frozen[f"{rival}:{name}"] = sha(cfg)
            cases[f"{rival}:{name}"] = cfg
    rec.data["frozen_config_sha256"] = frozen
    rec.save()  # the hashes are on disk before the predictions are computed
    res = run_batch(cases, workers=8)

    def reduction(rival: str, name: str, band: str = "field_20_100") -> float:
        return 1 - res[f"{rival}:{name}"][band] / res[f"{rival}:none"][band]

    cal = {r: reduction(r, AMILORIDE) for r in ("A", "B")}
    batteries = {r: res[f"{r}:none"]["battery_intact_mV"] for r in ("A", "B")}
    ok = all(abs(v - MEASURED) <= 0.01 for v in cal.values()) and all(
        abs(b - target) <= 0.01 * target for b in batteries.values()
    )
    exposed_g = res["B:none"].get("wound_cl_S_per_m2", 0.0)
    rec.check(
        "R1",
        ok,
        {
            "a_na_m2_per_s": a_na,
            "d_star_um": d_star,
            "g_star_m2_per_s": g_star,
            "amiloride_reduction": cal,
            "battery_intact_mV": batteries,
            "target_mV": target,
            "B_wound_cl_S_per_m2": exposed_g,
            "B_ano1_per_um2": exposed_g / (ANO1_PS * 1e-12) * 1e-12,
            "B_exposed_faces": res["B:none"].get("wound_cl_faces", 0),
            "field_unblocked_20_100": {r: res[f"{r}:none"]["field_20_100"] for r in ("A", "B")},
        },
        "battery within 1% of T; amiloride reduction within 64 +- 1 points",
        "calibration of both rivals; plausibility: A's d against amiloride diffusion, B's "
        "density as ANO1 channels per um2 of exposed membrane (2.63 pS each)",
    )
    predictions = {}
    for name in ("I1_cl_blocker", "I2_amiloride_plus_cl_blocker", "I4_pge2"):
        predictions[name] = {r: reduction(r, name) for r in ("A", "B")}
    predictions["I3_amiloride_far_band"] = {
        r: reduction(r, AMILORIDE, "field_200_400") for r in ("A", "B")
    }
    rec.report(
        "R2",
        predictions,
        "predicted change of the wound field (reduction: 1 - with / without), 20-100 um "
        "band unless named; I4 negative means an increase",
    )
    design = {}
    for name, p in predictions.items():
        delta = abs(p["A"] - p["B"])
        n = int(np.ceil(2 * (1.96 + 0.84) ** 2 * SIGMA**2 / delta**2)) if delta > 0 else None
        design[name] = {"separation_in_sd": delta / SIGMA, "wounds_per_arm": n}
    best = max(design, key=lambda k: design[k]["separation_in_sd"])
    rec.report(
        "R3",
        {"per_intervention": design, "recommended": best, "sigma": SIGMA},
        "separation of the rivals' predictions in per-wound SDs (0.23) and wounds per "
        "arm for a two-sided 5% test at 80% power",
    )
    rec.report(
        "R4",
        {
            "A": -predictions["I4_pge2"]["A"],
            "B": -predictions["I4_pge2"]["B"],
            "measured_change": 0.82,
            "vehicle_change": 0.22,
        },
        "predicted fractional change of the field under PGE2 (positive: increase); "
        "weak evidence (vehicle +22 +- 20%, cAMP-insensitive keratinocyte Cl- transport)",
    )
    rec.data["conservation"] = {
        k: {"kcl": v["kcl"], "charge": v["charge"]} for k, v in res.items()
    }
    return rec.save()


# ---------------------------------------------------------------------------- rival B2
VM_TARGET = -32.0  # cultured NHEK rest at -24 to -40 mV (Yamanoi 2023); the midpoint
B2_NOTE = (
    "B2 (and A2) were fixed in docs/plan.md after B failed R1 and before any B2 run. "
    "Machine-checked, not human-confirmed."
)


def _depolarize(factor: float) -> tuple[Modifier, ...]:
    return (Modifier(target="base_na", side="any", factor=factor, label="non-selective Na leak"),)


def _secant(
    evaluate,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
    tol: float,
    log: bool = False,
    steps: int = 4,
) -> tuple[float, float, list]:
    """Secant iterations toward y(x) = target from two bracketing samples; returns the last x,
    its y, and the history. ``evaluate`` runs the model at x."""
    history = []
    tr = (lambda v: np.log(v)) if log else (lambda v: v)
    inv = (lambda v: np.exp(v)) if log else (lambda v: v)
    for _ in range(steps):
        if abs(y1 - target) <= tol:
            break
        if y1 == y0:
            break
        x2 = inv(tr(x1) + (target - y1) * (tr(x1) - tr(x0)) / (y1 - y0))
        y2 = evaluate(x2)
        history.append((float(x2), float(y2)))
        x0, y0, x1, y1 = x1, y1, x2, y2
    return float(x1), float(y1), history


def _bracket(xs: list[float], ys: list[float], target: float):
    """The two samples on either side of the target (or the closest two)."""
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0:
            return xs[i], ys[i], xs[i + 1], ys[i + 1]
    i = int(np.argmin(np.abs(ys - target)))
    j = i - 1 if i == len(xs) - 1 else i + 1
    return xs[j], ys[j], xs[i], ys[i]


def run_rivals_b2(runs_dir: Path = RUNS) -> Path:
    rec = Record("skin-rivals-b2", runs_dir)
    rec.data["note"] = B2_NOTE
    target = target_battery_mv()
    rec.data["target_battery_mV"], rec.data["vm_target_mV"] = target, VM_TARGET
    a0 = 2.9785365223142775e-16  # A's apical Na density (runs/skin-rivals)

    # 1. Depolarize: the factor on the base Na permeability that puts basal and spinous cells at
    # the target voltage.
    f_grid = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    res = run_batch(
        {f"f{f:g}": config(a0, time=INIT_ONLY, base_modifiers=_depolarize(f)) for f in f_grid}
    )
    vms = [res[f"f{f:g}"]["vmem_basal_spinous_mV"] for f in f_grid]

    def vm_at(f: float) -> float:
        cfg = config(a0, time=INIT_ONLY, base_modifiers=_depolarize(f))
        return run_batch({"x": cfg})["x"]["vmem_basal_spinous_mV"]

    f_star, vm_star, f_hist = _secant(
        vm_at, *_bracket(f_grid, vms, VM_TARGET), VM_TARGET, 0.5, log=True
    )
    depol = _depolarize(f_star)

    # 2. Recalibrate the apical Na density to T on the depolarized cells.
    a_grid = [1.0e-16, 2.0e-16, 3.0e-16, 4.5e-16, 7.0e-16, 1.0e-15, 1.5e-15]
    res = run_batch({f"a{a:.2e}": config(a, time=INIT_ONLY, base_modifiers=depol) for a in a_grid})
    bats = [res[f"a{a:.2e}"]["battery_intact_mV"] for a in a_grid]

    def battery_at(a: float) -> float:
        return run_batch({"x": config(a, time=INIT_ONLY, base_modifiers=depol)})["x"][
            "battery_intact_mV"
        ]

    a_star, bat_star, a_hist = _secant(
        battery_at, *_bracket(a_grid, bats, target), target, 0.005 * target, log=True
    )
    vm_check = run_batch({"x": config(a_star, time=INIT_ONLY, base_modifiers=depol)})["x"]
    rec.data["calibration"] = {
        "depolarize": {
            "grid": f_grid,
            "vm_grid": vms,
            "factor": f_star,
            "vm": vm_star,
            "history": f_hist,
        },
        "apical_na": {
            "grid": a_grid,
            "battery_grid": bats,
            "a_na": a_star,
            "battery": bat_star,
            "history": a_hist,
        },
        "check": vm_check,
    }
    rec.save()

    # 3. A2's d and B2's wound Cl- density to 64%.
    d_grid = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    g_grid = [1e-16, 1e-15, 1e-14, 3e-14, 1e-13, 3e-13]
    cases = {"A:none": config(a_star, base_modifiers=depol)}
    for d in d_grid:
        cases[f"A:d{d:g}"] = config(
            a_star, interventions("A", d, 0.0)[AMILORIDE], base_modifiers=depol
        )
    for gg in g_grid:
        for name in ("none", AMILORIDE):
            cases[f"B:{name}:g{gg:.0e}"] = config(
                a_star, interventions("B", None, gg)[name], base_modifiers=depol
            )
    res = run_batch(cases, workers=8)
    base_a = res["A:none"]["field_20_100"]
    red_a = [1 - res[f"A:d{d:g}"]["field_20_100"] / base_a for d in d_grid]
    red_b = [
        1
        - res[f"B:{AMILORIDE}:g{gg:.0e}"]["field_20_100"]
        / res[f"B:none:g{gg:.0e}"]["field_20_100"]
        for gg in g_grid
    ]

    def reduction_a(d: float) -> float:
        cfg = config(a_star, interventions("A", d, 0.0)[AMILORIDE], base_modifiers=depol)
        return 1 - run_batch({"x": cfg})["x"]["field_20_100"] / base_a

    def reduction_b(gg: float) -> float:
        out = run_batch(
            {
                n: config(a_star, interventions("B", None, gg)[n], base_modifiers=depol)
                for n in ("none", AMILORIDE)
            }
        )
        return 1 - out[AMILORIDE]["field_20_100"] / out["none"]["field_20_100"]

    d_star, red_d, d_hist = _secant(
        reduction_a, *_bracket(d_grid, red_a, MEASURED), MEASURED, 0.005
    )
    fits_b = min(red_b) <= MEASURED + 0.01
    g_star, red_g, g_hist = (
        _secant(reduction_b, *_bracket(g_grid, red_b, MEASURED), MEASURED, 0.005, log=True)
        if fits_b
        else (float("nan"), float(min(red_b)), [])
    )
    rec.data["phase2"] = {
        "d_grid": d_grid,
        "reduction_A": red_a,
        "d_hist": d_hist,
        "g_grid": g_grid,
        "reduction_B": red_b,
        "g_hist": g_hist,
        "field_unblocked_B": [res[f"B:none:g{gg:.0e}"]["field_20_100"] for gg in g_grid],
    }
    rec.save()

    # 4. Frozen predictions for the rivals that fit.
    rivals = [("A", d_star, 0.0)] + ([("B", None, g_star)] if fits_b else [])
    frozen, cases = {}, {}
    for rival, d, gg in rivals:
        for name, extra in interventions(rival, d, gg).items():
            cfg = config(a_star, extra, base_modifiers=depol)
            frozen[f"{rival}:{name}"] = sha(cfg)
            cases[f"{rival}:{name}"] = cfg
    rec.data["frozen_config_sha256"] = frozen
    rec.save()
    res = run_batch(cases, workers=8)

    def reduction(rival: str, name: str, band: str = "field_20_100") -> float:
        return 1 - res[f"{rival}:{name}"][band] / res[f"{rival}:none"][band]

    names = [r for r, *_ in rivals]
    cal = {r: reduction(r, AMILORIDE) for r in names}
    batteries = {r: res[f"{r}:none"]["battery_intact_mV"] for r in names}
    ok = (
        fits_b
        and all(abs(v - MEASURED) <= 0.01 for v in cal.values())
        and all(abs(b - target) <= 0.01 * target for b in batteries.values())
    )
    exposed_g = res["B:none"].get("wound_cl_S_per_m2", 0.0) if fits_b else 0.0
    rec.check(
        "R1b",
        ok,
        {
            "depolarize_factor": f_star,
            "vm_basal_spinous_mV": vm_check["vmem_basal_spinous_mV"],
            "vm_granular_mV": vm_check["vmem_granular_mV"],
            "a_na_m2_per_s": a_star,
            "d_star_um": d_star,
            "g_star_m2_per_s": g_star,
            "B2_fits": fits_b,
            "B2_best_reduction": float(min(red_b)),
            "amiloride_reduction": cal,
            "battery_intact_mV": batteries,
            "target_mV": target,
            "B_wound_cl_S_per_m2": exposed_g,
            "B_ano1_per_um2": exposed_g / (ANO1_PS * 1e-12) * 1e-12,
            "field_unblocked_20_100": {r: res[f"{r}:none"]["field_20_100"] for r in names},
        },
        "Vm within 0.5 mV of -32 mV; battery within 1% of T; reduction within 64 +- 1 points",
        "calibration of A2 and B2 on depolarized keratinocytes",
    )
    predictions = {}
    for name in ("I1_cl_blocker", "I2_amiloride_plus_cl_blocker", "I4_pge2"):
        predictions[name] = {r: reduction(r, name) for r in names}
    predictions["I3_amiloride_far_band"] = {
        r: reduction(r, AMILORIDE, "field_200_400") for r in names
    }
    rec.report("R2b", predictions, "as R2, on depolarized keratinocytes")
    if fits_b:
        design = {}
        for name, p in predictions.items():
            delta = abs(p["A"] - p["B"])
            n = int(np.ceil(2 * (1.96 + 0.84) ** 2 * SIGMA**2 / delta**2)) if delta > 0 else None
            design[name] = {"separation_in_sd": delta / SIGMA, "wounds_per_arm": n}
        best = max(design, key=lambda k: design[k]["separation_in_sd"])
        rec.report(
            "R3b", {"per_intervention": design, "recommended": best, "sigma": SIGMA}, "as R3"
        )
        rec.report(
            "R4b",
            {
                "A": -predictions["I4_pge2"]["A"],
                "B": -predictions["I4_pge2"]["B"],
                "measured_change": 0.82,
                "vehicle_change": 0.22,
            },
            "as R4",
        )
    rec.data["conservation"] = {
        k: {"kcl": v["kcl"], "charge": v["charge"]} for k, v in res.items()
    }
    return rec.save()
