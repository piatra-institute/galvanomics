"""The Phase 1 audit: each BETSE run in its own subprocess, then the checks on the bundles.

Tolerances below are copied from docs/plan.md, where they were fixed before any run.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from tissuebundle.reader import Bundle
from tissuebundle.validate import validate_bundle

from betse_adapter import checks
from betse_adapter.config import CONFIG_NAME
from betse_adapter.results import AuditRun
from betse_adapter.run import VMEM_CSV_DIR

logger = logging.getLogger(__name__)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _run(work_root: Path, name: str, *args: str) -> Path:
    work_dir = work_root / name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    command = [
        sys.executable,
        "-m",
        "betse_adapter.cli",
        "run",
        "--name",
        name,
        "--work-root",
        str(work_root),
        *args,
    ]
    logger.info("running %s", " ".join(command))
    subprocess.run(command, check=True)
    return work_dir


def _bundle(work_dir: Path) -> Bundle:
    return Bundle(work_dir / f"{work_dir.name}.tbundle")


def _export_report(work_dir: Path) -> dict:
    return json.loads((work_dir / "export.json").read_text())


def _structural(audit: AuditRun, work_dir: Path, bundle: Bundle) -> None:
    """C1, C2, C3, C5 and contract validation, common to every run."""
    report = _export_report(work_dir)
    mapping = report["checks"]
    c1 = {k: v for k, v in mapping.items() if k != "grid_c_order"}
    audit.check(
        "C1",
        all(c1.values()),
        c1,
        "bitwise",
        f"post-wound geometry equals kept pre-wound geometry; "
        f"{len(report['removed_cells'])} cells removed",
    )
    c2 = checks.c2_mean_consistency(bundle)
    audit.check(
        "C2",
        c2 <= 1e-12,
        c2,
        "<= 1e-12 V",
        "exported cell V_mem vs plain mean of exported membrane V_mem",
    )
    c3 = checks.c3_polygons(bundle)
    audit.check(
        "C3",
        c3 == 0,
        c3,
        "0 bad cells",
        "cells not counter-clockwise with positive area (vertex count equals "
        "membrane count is enforced by the exporter)",
    )
    audit.check(
        "C5",
        mapping["grid_c_order"],
        mapping["grid_c_order"],
        "exact",
        "grid points equal (X, Y) flattened in C order",
    )
    errors = validate_bundle(bundle.path)
    audit.check("contract", not errors, errors, "no violations", "tissuebundle validate")
    audit.set(
        timings_s=report["timings_s"],
        skipped_quantities=report["skipped"],
        cells=report["n_cells_pre"],
        cells_removed=len(report["removed_cells"]),
        frames=bundle.n_frames,
        grid_shape=bundle.grid_shape,
    )


def audit_wound_default(seed: int, work_root: Path, runs_dir: Path) -> Path:
    spec = str(CONFIGS / "wound-default.yaml")
    audit = AuditRun("wound-default", runs_dir)
    a = _run(work_root, "wound-default-a", spec, "--seed", str(seed), "--plot")
    b = _run(work_root, "wound-default-b", spec, "--seed", str(seed))
    mid = _run(work_root, "wound-default-dt5e-5", "--from-init", str(a), "--sim-dt", "5e-5")
    fine = _run(work_root, "wound-default-dt2.5e-5", "--from-init", str(a), "--sim-dt", "2.5e-5")
    bundle_a = _bundle(a)
    audit.set(
        seed=seed,
        spec="configs/wound-default.yaml",
        config_sha256=bundle_a.manifest["provenance"]["config_sha256"],
    )

    _structural(audit, a, bundle_a)
    c4 = checks.c4_csv(bundle_a, a / VMEM_CSV_DIR)
    audit.check(
        "C4",
        c4["frames"] == bundle_a.n_frames - 1
        and c4["max_vmem_mV"] <= 1e-9
        and c4["max_centre_um"] <= 1e-9,
        c4,
        "<= 1e-9 mV, <= 1e-9 um, every frame",
        "I/O check only: BETSE's own Vmem2D CSVs read the same array",
    )
    differing = checks.d1_differences(bundle_a, _bundle(b))
    audit.check(
        "D1",
        not differing,
        differing,
        "no differing arrays",
        "two full runs with the same seed, separate processes",
    )
    p3 = checks.p3_convergence(bundle_a, _bundle(mid), _bundle(fine))
    audit.check(
        "P3",
        p3["max_diff_coarse_mid_mV"] <= 1.0,
        p3,
        "<= 1 mV (provisional)",
        "sim dt 1e-4 vs 5e-5 s from the same init, matched on sample index",
    )
    audit.report(
        "R1",
        checks.r1_offset(bundle_a),
        "V_mem minus BETSE's GHK estimate (includes molecule-network channels)",
    )
    audit.report(
        "R2",
        checks.r2_field(bundle_a),
        "-grad(venv) by centred differences vs BETSE's efield_ecm, last frame",
    )
    return audit.save()


def audit_ghk(seed: int, work_root: Path, runs_dir: Path) -> Path:
    audit = AuditRun("ghk-audit", runs_dir)
    work = _run(work_root, "ghk-audit", str(CONFIGS / "ghk-audit.yaml"), "--seed", str(seed))
    bundle = _bundle(work)
    audit.set(
        seed=seed,
        spec="configs/ghk-audit.yaml",
        config_sha256=bundle.manifest["provenance"]["config_sha256"],
    )
    _structural(audit, work, bundle)
    p1 = checks.p1_ghk_reading(bundle)
    audit.check(
        "P1",
        p1 <= 1e-6,
        p1,
        "<= 1e-6 V",
        "our GHK from the bundle's concentrations and diffusion constants vs BETSE's "
        "vm_GHK, every post-wound frame and live cell",
    )
    audit.report("R1", checks.r1_offset(bundle), "V_mem minus BETSE's GHK, molecules off")
    return audit.save()


def audit_pump_blocked(seed: int, work_root: Path, runs_dir: Path) -> Path:
    audit = AuditRun("pump-blocked", runs_dir)
    work = _run(work_root, "pump-blocked", str(CONFIGS / "pump-blocked.yaml"), "--seed", str(seed))
    bundle = _bundle(work)
    audit.set(
        seed=seed,
        spec="configs/pump-blocked.yaml",
        config_sha256=bundle.manifest["provenance"]["config_sha256"],
    )
    _structural(audit, work, bundle)
    p2 = checks.p2_passive(bundle)
    audit.check(
        "P2",
        p2["max_abs_vs_our_ghk_mV"] <= 0.5,
        p2,
        "<= 0.5 mV at the last frame",
        "pump blocked, molecules off, no wound: V_mem vs GHK per cell",
    )
    audit.report(
        "R4",
        checks.r4_relaxation(bundle),
        "Added after the first run, reported only: the time course of mean V_mem and of "
        "max |V_mem - GHK|, showing V_mem relaxing from the pump-held baseline onto GHK",
    )
    return audit.save()


REFERENCE_ENV = "BETSE13_REFERENCE_DIR"


def reference_dir() -> Path | None:
    """A BETSE 1.3.0 wound run (its sim_config, geo, extra_configs, INITS and per-frame V_mem
    CSVs), given by the environment variable; it is not distributed with this repository."""
    value = os.environ.get(REFERENCE_ENV)
    return Path(value).expanduser() if value else None


def audit_cross_version(seed: int, work_root: Path, runs_dir: Path) -> Path | None:
    """R3: BETSE 1.5.0's init and sim on a BETSE 1.3.0 run's seeded world (copied, untouched),
    against the V_mem CSVs 1.3.0 wrote for its sim. Resuming 1.3.0's init directly fails in
    1.5.0's molecule-network setup, so the comparison starts from the world. Skipped unless
    BETSE13_REFERENCE_DIR names such a run."""
    reference = reference_dir()
    if reference is None:
        logger.warning("R3 skipped: set %s to a BETSE 1.3.0 run directory", REFERENCE_ENV)
        return None
    audit = AuditRun("betse-1-3-cross-version", runs_dir)
    staging = work_root / "betse-1-3-reference-copy"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(reference / "sample_sim.yaml", staging / CONFIG_NAME)
    for folder in ("geo", "extra_configs", "INITS"):
        shutil.copytree(reference / folder, staging / folder)
    work = _run(
        work_root,
        "betse-1-3-cross-version",
        "--from-init",
        str(staging),
        "--from-world",
        "--seed",
        str(seed),
    )
    bundle = _bundle(work)
    audit.set(
        source="a BETSE 1.3.0 wound run with per-frame V_mem exports (not distributed)", seed=seed
    )
    _structural(audit, work, bundle)
    audit.report(
        "R3",
        checks.c4_csv(bundle, reference / VMEM_CSV_DIR),
        "BETSE 1.5.0 init and sim on the BETSE 1.3.0 world vs 1.3.0's own sim Vmem2D CSVs: "
        "max differences over every frame (reported, not gated)",
    )
    return audit.save()


AUDITS: dict[str, Callable[..., Path | None]] = {
    "wound-default": audit_wound_default,
    "ghk-audit": audit_ghk,
    "pump-blocked": audit_pump_blocked,
    "betse-1-3-cross-version": audit_cross_version,
}
