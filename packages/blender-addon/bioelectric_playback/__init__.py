"""Bioelectric Playback: import tissue run bundles and play them back frame by frame.

The simulator never runs inside Blender. A bundle written by a solver adapter is read with
NumPy alone (``reader.py``, vendored from the ``tissue-bundle`` package) and drawn as one
polygon per cell, coloured by the selected quantity at the current frame.
"""

from __future__ import annotations


def register() -> None:
    from . import playback, ui

    playback.register()
    ui.register()


def unregister() -> None:
    from . import playback, ui

    ui.unregister()
    playback.unregister()
