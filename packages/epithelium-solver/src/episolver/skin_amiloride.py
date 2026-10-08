"""Held-out prediction H1 (docs/plan.md): the reduction of the mouse wound field under amiloride.

The model is calibrated only on the intact battery (forearm membranes, unchanged); the block
fraction comes from the dose and the potency (1 mM against K_i = 42 nM), fixed before the run.
The resolved configuration of the primary run is hashed into the record before the prediction
is compared with the measurement."""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from episolver.audit import Record
from episolver.skin_audit import FULL, _mean, _simulate, battery_mv, node_volume
from episolver.skin_physics import EPIDERMIS, Frame, SkinModel, SkinRunConfig

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parents[2] / "configs" / "skin-mouse-wide.yaml"
K_I_M, DOSE_M = 42e-9, 1e-3  # amiloride on rat alpha-beta-gamma ENaC (Schild 1997); the dose
BLOCK = 1.0 - K_I_M / (K_I_M + DOSE_M)
MEASURED, SEM = 0.64, 0.07  # Nuccitelli 2008, Table 1: -64 +- 7% (SEM, 11 wounds)
NOTE = (
    "H1 was specified in docs/plan.md before this run: block fraction, primary and reported "
    "variants, measure and verdict rule. Machine-checked, not human-confirmed."
)
VARIANTS = {
    "unblocked": {},
    "primary": {"na_block_fraction": BLOCK},
    "within_50": {"na_block_fraction": BLOCK, "na_block_within_um": 50.0},
    "within_100": {"na_block_fraction": BLOCK, "na_block_within_um": 100.0},
    "within_200": {"na_block_fraction": BLOCK, "na_block_within_um": 200.0},
    "uniform_25": {"na_block_fraction": 0.25},
    "uniform_50": {"na_block_fraction": 0.50},
    "uniform_75": {"na_block_fraction": 0.75},
    "uniform_90": {"na_block_fraction": 0.90},
}


def config(variant: str) -> SkinRunConfig:
    base = SkinRunConfig.model_validate(yaml.safe_load(CONFIG.read_text()))
    return base.model_copy(
        update={"physics": base.physics.model_copy(update={"time": FULL} | VARIANTS[variant])}
    )


def field_beneath_corneum(model: SkinModel, frame: Frame, lo: float = 20.0, hi: float = 100.0):
    """Band mean of E_x beneath the stratum corneum, positive toward the wound (mV/mm)."""
    z, _, x = model.grid.coords()
    offset = x - model.config.skin.wound.x_um
    toward = -frame.data["efield_x"] * (offset >= 0) + frame.data["efield_x"] * (offset < 0)
    band = (frame.data["region"] == EPIDERMIS) & (z > model.tj_z + 0.1)
    band &= (abs(offset) >= lo) & (abs(offset) < hi)
    return _mean(toward, band, node_volume(model))


def run_variant(variant: str) -> tuple[str, dict]:
    model, frames = _simulate(config(variant))
    return variant, {
        "field_20_100_um": field_beneath_corneum(model, frames[-1]),
        "battery_intact_mV": battery_mv(model, frames[0], far_um=300.0),
        "battery_far_after_mV": battery_mv(model, frames[-1], far_um=800.0),
        "kcl": model.diag.max_kcl_residual,
        "charge": model.diag.max_charge_error,
    }


def run_amiloride(runs_dir: Path) -> Path:
    rec = Record("skin-amiloride", runs_dir)
    rec.data["note"] = NOTE
    resolved = config("primary").model_dump_json().encode()
    rec.data["primary_config_sha256"] = hashlib.sha256(resolved).hexdigest()
    rec.data["config_file_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    rec.data["block_fraction"] = BLOCK
    rec.save()  # the hash and the block are on disk before any comparison
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with ProcessPoolExecutor(len(VARIANTS)) as pool:
        results = dict(pool.map(run_variant, VARIANTS))
    base = results["unblocked"]["field_20_100_um"]
    reductions = {
        k: 1.0 - v["field_20_100_um"] / base for k, v in results.items() if k != "unblocked"
    }
    predicted = reductions["primary"]
    lo, hi = MEASURED - 2 * SEM, MEASURED + 2 * SEM
    rec.check(
        "H1",
        lo <= predicted <= hi,
        {
            "predicted_reduction": predicted,
            "measured": MEASURED,
            "sem": SEM,
            "field_unblocked_mV_per_mm": base,
            "field_blocked_mV_per_mm": results["primary"]["field_20_100_um"],
        },
        f"within {lo:.2f}-{hi:.2f} (64 +- 2 x 7%)",
        "reduction of the band mean beneath the stratum corneum 20-100 um from the wound "
        "centre, 2 s after the cut, block on every apical Na channel",
    )
    rec.report(
        "H1-variants",
        {"reductions": reductions, "runs": results},
        "block confined near the wound, and partial uniform blocks; reported, not judged",
    )
    rec.data["interpretation_fixed_in_advance"] = (
        "a predicted reduction of 90% or more at the dose-derived block means the Na-only "
        "battery over-predicts: another current (for example Cl- secretion) carries part of "
        "the field, or the block is partial in vivo"
    )
    return rec.save()
