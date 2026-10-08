"""Locate the Blender binary: ``BLENDER_BIN`` if set, else the macOS application bundle."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MACOS_DEFAULT = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def blender_bin() -> Path | None:
    env = os.environ.get("BLENDER_BIN")
    if env:
        return Path(env)
    found = shutil.which("blender")
    if found:
        return Path(found)
    return MACOS_DEFAULT if MACOS_DEFAULT.exists() else None
