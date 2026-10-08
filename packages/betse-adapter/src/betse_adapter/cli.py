"""``betse-adapter``: run BETSE in-process, export bundles, run the audit."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import typer

from betse_adapter.config import Override, load_spec, materialize
from betse_adapter.export import export_run
from betse_adapter.run import INIT_PICKLE, PHASES, clone_for_sim, run_phases

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = PACKAGE_ROOT / "data" / "work"
DEFAULT_RUNS = PACKAGE_ROOT / "runs"
DEFAULT_SEED = 20261007

app = typer.Typer(no_args_is_help=True, add_completion=False)
logger = logging.getLogger("betse_adapter")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def bundle_path(work_dir: Path) -> Path:
    return work_dir / f"{work_dir.name}.tbundle"


@app.command()
def run(
    spec: Path = typer.Argument(None, help="configs/<name>.yaml; omit with --from-init"),
    name: str = typer.Option(None, help="work directory name; defaults to the spec name"),
    seed: int = typer.Option(DEFAULT_SEED),
    work_root: Path = typer.Option(DEFAULT_WORK),
    from_init: Path = typer.Option(None, help="reuse this run's config and init; sim only"),
    from_world: bool = typer.Option(
        False, help="with --from-init, keep only the seeded world and rerun init and sim"
    ),
    sim_dt: float = typer.Option(None, help="override the sim time step (s)"),
    plot: bool = typer.Option(False, help="also run BETSE's plot phase (writes its CSVs)"),
) -> None:
    """Run seed, init and sim (or sim only, from an existing init) and export a bundle."""
    overrides = [Override(("sim time settings", "time step"), sim_dt)] if sim_dt else []
    if from_init is not None:
        work_dir = work_root / (name or f"{from_init.name}-sim")
        conf = clone_for_sim(from_init, work_dir, overrides)
        phases: tuple[str, ...] = ("sim",)
        description = f"Sim-only rerun of {from_init.name} with overrides {overrides}"
        if from_world:
            (work_dir / INIT_PICKLE).unlink()
            phases = ("init", "sim")
            description = f"Init and sim rerun on the world of {from_init.name}"
    elif spec is not None:
        loaded = load_spec(spec)
        loaded = type(loaded)(loaded.name, loaded.description, (*loaded.overrides, *overrides))
        work_dir = work_root / (name or loaded.name)
        conf = materialize(loaded, work_dir)
        phases = PHASES
        description = loaded.description
    else:
        raise typer.BadParameter("give a spec or --from-init")
    timings = run_phases(conf, seed, phases, plot_sim=plot)
    path, report = export_run(
        work_dir,
        bundle_path(work_dir),
        name=work_dir.name.replace("-", "_").replace(".", "_"),
        description=description,
        seed=seed,
        command=tuple(sys.argv),
    )
    (work_dir / "export.json").write_text(
        json.dumps({"timings_s": timings, "seed": seed, **asdict(report)}, indent=2) + "\n"
    )
    typer.echo(f"wrote {path}")


@app.command()
def export(work_dir: Path, seed: int = typer.Option(None), overwrite: bool = False) -> None:
    """Re-export an existing BETSE work directory as a bundle."""
    path, report = export_run(
        work_dir,
        bundle_path(work_dir),
        name=work_dir.name.replace("-", "_"),
        description=f"Export of {work_dir.name}",
        seed=seed,
        command=tuple(sys.argv),
        overwrite=overwrite,
    )
    typer.echo(f"wrote {path}; skipped {report.skipped}")


@app.command()
def audit(
    only: list[str] = typer.Option(None, help="wound-default, ghk-audit, pump-blocked"),
    seed: int = typer.Option(DEFAULT_SEED),
    work_root: Path = typer.Option(DEFAULT_WORK),
    runs_dir: Path = typer.Option(DEFAULT_RUNS),
) -> None:
    """Run the Phase 1 audit runs and write runs/<name>/result.json for each."""
    from betse_adapter.audit import AUDITS

    for key in only or list(AUDITS):
        path = AUDITS[key](seed=seed, work_root=work_root, runs_dir=runs_dir)
        if path is not None:
            typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
