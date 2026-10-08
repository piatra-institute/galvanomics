"""Sync the vendored reader, then validate and build the extension zip into ``dist/``.

Run with ``uv run python scripts/build_extension.py``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from blender_bin import blender_bin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bioelectric_playback"
READER = ROOT.parent / "tissue-bundle" / "src" / "tissuebundle" / "reader.py"
DIST = ROOT / "dist"

logger = logging.getLogger("build_extension")


def build() -> Path:
    shutil.copyfile(READER, SOURCE / "reader.py")
    logger.info("synced reader.py from %s", READER)
    binary = blender_bin()
    if binary is None:
        raise SystemExit("Blender not found; set BLENDER_BIN")
    subprocess.run([binary, "--command", "extension", "validate", str(SOURCE)], check=True)
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob("*.zip"):
        old.unlink()
    subprocess.run(
        [
            binary,
            "--command",
            "extension",
            "build",
            "--source-dir",
            str(SOURCE),
            "--output-dir",
            str(DIST),
        ],
        check=True,
    )
    (zip_path,) = DIST.glob("*.zip")
    logger.info("built %s", zip_path)
    return zip_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
