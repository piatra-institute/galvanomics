from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE = Path(__file__).parents[1] / "bioelectric_playback"


def load(name: str) -> ModuleType:
    """Load one bpy-free module by path, without importing the bpy-bound package."""
    spec = importlib.util.spec_from_file_location(f"_bp_{name}", PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def core() -> ModuleType:
    return load("core")
