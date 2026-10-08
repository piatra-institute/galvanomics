"""A real BETSE run of the default wound configuration, exported and validated."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tissuebundle.reader import Bundle
from tissuebundle.validate import validate_bundle

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.slow
def test_wound_default_runs_and_exports(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "betse_adapter.cli",
            "run",
            str(CONFIGS / "wound-default.yaml"),
            "--name",
            "it",
            "--work-root",
            str(tmp_path),
            "--seed",
            "1",
        ],
        check=True,
    )
    path = tmp_path / "it" / "it.tbundle"
    assert validate_bundle(path) == []
    bundle = Bundle(path)
    assert bundle.removed().sum() > 0
    assert bundle.baseline_frames == (0,)
    assert {"vmem", "venv", "conc_cell", "diff_membrane"} <= set(bundle.quantities)
