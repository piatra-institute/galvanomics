"""``tissuebundle``: validate, inspect and generate run bundles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import typer

from tissuebundle.model import Manifest
from tissuebundle.reader import open_bundle
from tissuebundle.synthetic import make_synthetic
from tissuebundle.validate import validate_bundle

app = typer.Typer(no_args_is_help=True, add_completion=False)


def schema_json() -> str:
    return json.dumps(Manifest.model_json_schema(), indent=2) + "\n"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)


@app.command()
def validate(path: Path) -> None:
    """Exit non-zero and list the violations if the bundle breaks its contract."""
    errors = validate_bundle(path)
    for error in errors:
        typer.echo(error, err=True)
    if errors:
        raise typer.Exit(1)
    typer.echo(f"valid: {path}")


@app.command()
def info(path: Path) -> None:
    """Summarize a bundle: geometry, frames and quantities with their ranges."""
    bundle = open_bundle(path)
    removed = int(bundle.removed().sum())
    typer.echo(
        f"{bundle.name}: {bundle.n_cells} cells ({removed} removed), "
        f"{bundle.n_membranes} membranes, grid {bundle.grid_shape}, "
        f"{bundle.n_frames} frames (baseline {list(bundle.baseline_frames)})"
    )
    prov = bundle.manifest["provenance"]
    typer.echo(f"solver {prov['solver']} {prov['solver_version']}, seed {prov['seed']}")
    t = bundle.array("time_integrated_s")
    typer.echo(f"time {t[0]:.6g} .. {t[-1]:.6g} s")
    for key, record in bundle.quantities.items():
        data = bundle.array(key)
        finite = data[np.isfinite(data)]
        span = f"{finite.min():.4g} .. {finite.max():.4g}" if finite.size else "no finite values"
        typer.echo(
            f"  {key:<24} {record['location']:<13} {record['unit']:<7} "
            f"{tuple(record['shape'])}  {span}"
        )


@app.command()
def schema() -> None:
    """Print the manifest's JSON schema."""
    typer.echo(schema_json(), nl=False)


@app.command("make-synthetic")
def make_synthetic_cmd(
    out: Path,
    cols: int = 14,
    rows: int = 14,
    frames: int = 12,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Write a synthetic wounded hexagonal sheet (no physics) for testing."""
    path = make_synthetic(out, cols=cols, rows=rows, frames=frames, seed=seed, overwrite=overwrite)
    typer.echo(f"wrote {path}")
