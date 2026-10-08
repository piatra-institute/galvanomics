"""Import BETSE headless. Matplotlib's backend must be chosen before BETSE imports it."""

from __future__ import annotations

import os
from types import ModuleType


def import_betse() -> ModuleType:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import betse
    import betse.metadata

    return betse


def betse_version() -> str:
    import_betse()
    from betse import metadata

    return str(metadata.VERSION)
