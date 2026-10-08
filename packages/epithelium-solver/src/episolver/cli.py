"""``episolver``: run a 3D epithelium and export a bundle; run the audit."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer
from tissuebundle.writer import sha256_file

from episolver.export import export_bundle
from episolver.geometry import build_tissue
from episolver.params import load_config
from episolver.solver import Epithelium

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = PACKAGE_ROOT / "data" / "work"
DEFAULT_RUNS = PACKAGE_ROOT / "runs"

app = typer.Typer(no_args_is_help=True, add_completion=False)
logger = logging.getLogger("episolver")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def run_config(
    config_path: Path, out: Path, overwrite: bool = True, progress: bool = False
) -> tuple[Path, dict]:
    config = load_config(config_path)
    tissue = build_tissue(config)
    model = Epithelium(config, tissue)
    start = time.perf_counter()

    def report(done: int, total: int) -> None:
        if progress:
            typer.echo(f"PROGRESS {done}/{total}", err=True)

    frames, _ = model.run(on_frame=report)
    elapsed = time.perf_counter() - start
    path = export_bundle(
        model, frames, out, str(config_path), sha256_file(config_path), overwrite=overwrite
    )
    summary = {
        "seconds": elapsed,
        "steps": model.diag.steps,
        "cells": tissue.n_cells,
        "wounded": int(tissue.wounded.sum()),
        "grid": [model.grid.nx, model.grid.ny],
        "max_kcl_residual": model.diag.max_kcl_residual,
        "max_charge_error": model.diag.max_charge_error,
    }
    (Path(out).parent / f"{Path(out).stem}.run.json").write_text(json.dumps(summary, indent=2))
    return path, summary


@app.command()
def run(
    config: Path,
    out: Path = typer.Option(
        None, help="bundle directory; default data/work/<name>/<name>.tbundle"
    ),
    progress: bool = typer.Option(False, help="print PROGRESS lines to stderr"),
) -> None:
    """Simulate a configuration (init, wound, sim) and export a bundle."""
    name = config.stem
    out = out or DEFAULT_WORK / name / f"{name}.tbundle"
    path, summary = run_config(config, out, progress=progress)
    typer.echo(f"wrote {path} ({summary['steps']} steps in {summary['seconds']:.1f} s)")


@app.command()
def skin(
    spec: Path = typer.Option(None, help="YAML with SkinSpec fields; defaults otherwise"),
    out: Path = typer.Option(None, help="bundle directory"),
) -> None:
    """Generate the procedural skin anatomy (geometry only) and export it as a bundle."""
    import yaml

    from episolver.skin import SkinSpec, generate
    from episolver.skin_export import export_skin_bundle

    data = yaml.safe_load(spec.read_text()) if spec else {}
    model = generate(SkinSpec.model_validate(data or {}))
    out = out or DEFAULT_WORK / "skin-anatomy" / "skin-anatomy.tbundle"
    typer.echo(f"wrote {export_skin_bundle(model, out)} ({model.n_cells} cells)")


@app.command("skin-sim")
def skin_sim(
    config: Path = typer.Argument(None, help="YAML with SkinRunConfig fields"),
    out: Path = typer.Option(None, help="bundle directory"),
    progress: bool = typer.Option(False, help="print PROGRESS lines to stderr"),
) -> None:
    """Simulate the skin (init, wound, sim) and export a bundle with the 3D extracellular space."""
    import yaml

    from episolver.skin import generate
    from episolver.skin_physics import SkinModel, SkinRunConfig
    from episolver.skin_sim_export import export_skin_sim

    data = yaml.safe_load(config.read_text()) if config else {"name": "skin-wound"}
    run = SkinRunConfig.model_validate(data)
    model = SkinModel(run, generate(run.skin))
    start = time.perf_counter()

    def report(done: int, total: int) -> None:
        if progress:
            typer.echo(f"PROGRESS {done}/{total}", err=True)

    frames = model.run(on_frame=report)
    out = out or DEFAULT_WORK / run.name / f"{run.name}.tbundle"
    path = export_skin_sim(
        model,
        frames,
        out,
        str(config) if config else None,
        sha256_file(config) if config else None,
    )
    summary = {
        "seconds": time.perf_counter() - start,
        "steps": model.diag.steps,
        "max_kcl_residual": model.diag.max_kcl_residual,
        "max_charge_error": model.diag.max_charge_error,
    }
    (Path(out).parent / f"{Path(out).stem}.run.json").write_text(json.dumps(summary, indent=2))
    typer.echo(f"wrote {path} ({model.diag.steps} steps in {summary['seconds']:.0f} s)")


@app.command()
def audit(
    only: list[str] = typer.Option(None, help="check ids, e.g. E1 E6"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS),
    work_root: Path = typer.Option(DEFAULT_WORK),
) -> None:
    """Run the Phase 1b checks and write runs/<name>/result.json."""
    from episolver.audit import run_audit

    for path in run_audit(only=only, runs_dir=runs_dir, work_root=work_root):
        typer.echo(f"wrote {path}")


@app.command("skin-audit")
def skin_audit(
    runs_dir: Path = typer.Option(DEFAULT_RUNS),
    wide: bool = typer.Option(False, help="the wide-block checks (L1-L5, S4, S5) instead"),
    followup: bool = typer.Option(False, help="the wide-block follow-ups (L6, S6) instead"),
) -> None:
    """Run the skin checks (K1-K5, S1-S3, or with --wide L1-L5, S4, S5) and write
    runs/skin/result.json or runs/skin-wide/result.json."""
    if followup:
        from episolver.skin_audit_wide import run_followups

        typer.echo(f"wrote {run_followups(runs_dir)}")
        return
    if wide:
        from episolver.skin_audit_wide import run_wide_audit

        typer.echo(f"wrote {run_wide_audit(runs_dir)}")
        return
    from episolver.skin_audit import run_skin_audit

    typer.echo(f"wrote {run_skin_audit(runs_dir)}")


@app.command("skin-amiloride")
def skin_amiloride(runs_dir: Path = typer.Option(DEFAULT_RUNS)) -> None:
    """Held-out prediction H1: the mouse wound field under amiloride; writes
    runs/skin-amiloride/result.json."""
    from episolver.skin_amiloride import run_amiloride

    typer.echo(f"wrote {run_amiloride(runs_dir)}")


@app.command("skin-rivals")
def skin_rivals(
    runs_dir: Path = typer.Option(DEFAULT_RUNS),
    b2: bool = typer.Option(False, help="rival B2 and A2 on depolarized keratinocytes"),
) -> None:
    """Rival mechanisms after H1 (R1-R4, or with --b2 R1b-R4b): calibrate, freeze predictions,
    name the most discriminating experiment; writes runs/skin-rivals[-b2]/result.json."""
    from episolver.skin_rivals import run_rivals, run_rivals_b2

    typer.echo(f"wrote {(run_rivals_b2 if b2 else run_rivals)(runs_dir)}")


if __name__ == "__main__":
    app()
