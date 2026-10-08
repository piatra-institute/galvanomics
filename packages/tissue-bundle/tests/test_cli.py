from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tissuebundle.cli import app

runner = CliRunner()


def test_validate_and_info(synthetic: Path) -> None:
    result = runner.invoke(app, ["validate", str(synthetic)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["info", str(synthetic)])
    assert result.exit_code == 0, result.output
    assert "vmem" in result.output and "removed" in result.output


def test_validate_fails_on_missing_bundle(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope")])
    assert result.exit_code == 1


def test_make_synthetic(tmp_path: Path) -> None:
    out = tmp_path / "s.tbundle"
    result = runner.invoke(app, ["make-synthetic", str(out), "--cols", "4", "--rows", "4"])
    assert result.exit_code == 0, result.output
    assert (out / "bundle.json").is_file()
