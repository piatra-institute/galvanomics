"""Build, install into an isolated user profile, and run the headless smoke script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from blender_bin import blender_bin  # noqa: E402

BLENDER = blender_bin()
pytestmark = [
    pytest.mark.blender,
    pytest.mark.skipif(BLENDER is None, reason="Blender binary not found"),
]


def _smoke(tmp_path: Path, bundle: Path) -> dict:
    """Build and install the extension in an isolated profile, run the smoke script."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_extension.py")], check=True)
    (zip_path,) = (ROOT / "dist").glob("*.zip")
    env = {**os.environ, "BLENDER_USER_RESOURCES": str(tmp_path / "user")}
    subprocess.run(
        [
            BLENDER,
            "--command",
            "extension",
            "install-file",
            "-r",
            "user_default",
            "-e",
            str(zip_path),
        ],
        check=True,
        env=env,
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            BLENDER,
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "-P",
            str(ROOT / "scripts" / "smoke_blender.py"),
            "--",
            str(bundle),
            str(out),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "SMOKE OK" in proc.stdout
    result = json.loads((out / "smoke.json").read_text())
    result["out"] = out
    return result


def _solver(*args: str) -> None:
    solver = ROOT.parent / "epithelium-solver"
    subprocess.run(["uv", "run", "--project", str(solver), "episolver", *args], check=True)


def test_smoke_on_synthetic_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "synthetic.tbundle"
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT.parent / "tissue-bundle"),
            "tissuebundle",
            "make-synthetic",
            str(bundle),
            "--cols",
            "8",
            "--rows",
            "6",
            "--frames",
            "5",
        ],
        check=True,
    )
    result = _smoke(tmp_path, bundle)
    assert len(list(result["out"].glob("frame_*.png"))) == 3


def test_smoke_on_3d_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "tiny.tbundle"
    solver = ROOT.parent / "epithelium-solver"
    _solver("run", str(solver / "configs" / "tiny-test.yaml"), "--out", str(bundle))
    objects = _smoke(tmp_path, bundle)["objects"]
    assert any(name.endswith("_field") for name in objects)
    assert any("open_kir2p1_basolateral" in name for name in objects)


def test_smoke_on_skin_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "skin.tbundle"
    solver = ROOT.parent / "epithelium-solver"
    _solver("skin-sim", str(solver / "configs" / "skin-tiny.yaml"), "--out", str(bundle))
    objects = _smoke(tmp_path, bundle)["objects"]
    for part in ("Na channels", "K channels", "Na/K pumps", "Na+ ions", "K+ ions", "section"):
        assert any(part in name for name in objects), (part, objects)


def test_nanoscope_on_skin_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "skin.tbundle"
    solver = ROOT.parent / "epithelium-solver"
    _solver("skin-sim", str(solver / "configs" / "skin-tiny.yaml"), "--out", str(bundle))
    _smoke(tmp_path, bundle)  # builds and installs the extension in the isolated profile
    env = {**os.environ, "BLENDER_USER_RESOURCES": str(tmp_path / "user")}
    png = tmp_path / "nanoscope.png"
    proc = subprocess.run(
        [
            BLENDER,
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "-P",
            str(ROOT / "scripts" / "smoke_nanoscope.py"),
            "--",
            str(bundle),
            str(png),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "NANOSCOPE OK" in proc.stdout and png.exists()
