"""Read a run bundle with nothing but the standard library and NumPy.

This module is vendored byte-for-byte into the Blender extension, which may not install
packages, so it must not import pydantic, typer or anything else from this package. A test
enforces that by parsing its imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST = "bundle.json"
SUPPORTED_MAJOR = "0"

TRAILING = {
    "cell": ("n_cells",),
    "membrane": ("n_membranes",),
    "grid": ("ny", "nx"),
    "cell_ion": ("n_ions", "n_cells"),
    "membrane_ion": ("n_ions", "n_membranes"),
    "grid_ion": ("n_ions", "ny", "nx"),
    "volume": ("vz", "vy", "vx"),
}


class BundleError(Exception):
    """The bundle is missing, unreadable, or of an unsupported version."""


class Bundle:
    """A read-only view of a bundle directory. Arrays are memory-mapped on request."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST
        if not manifest_path.is_file():
            raise BundleError(f"no {MANIFEST} in {self.path}")
        with manifest_path.open(encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        version = str(self.manifest.get("schema_version", ""))
        if version.split(".")[0] != SUPPORTED_MAJOR:
            raise BundleError(f"unsupported bundle schema version {version!r}")
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def n_cells(self) -> int:
        return int(self.manifest["geometry"]["n_cells"])

    @property
    def n_membranes(self) -> int:
        return int(self.manifest["geometry"]["n_membranes"])

    @property
    def n_frames(self) -> int:
        return int(self.manifest["frames"]["count"])

    @property
    def baseline_frames(self) -> tuple[int, ...]:
        return tuple(int(i) for i in self.manifest["frames"]["baseline"])

    @property
    def grid_shape(self) -> tuple[int, int] | None:
        shape = self.manifest["geometry"]["grid_shape"]
        return None if shape is None else (int(shape[0]), int(shape[1]))

    @property
    def extrusion_height_um(self) -> float | None:
        height = self.manifest["geometry"].get("extrusion_height_um")
        return None if height is None else float(height)

    @property
    def volume_shape(self) -> tuple[int, int, int] | None:
        shape = self.manifest["geometry"].get("volume_shape")
        return None if shape is None else tuple(int(n) for n in shape)

    def volume_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Node coordinates (µm) along x, y and z of the 3D extracellular grid."""
        g = self.manifest["geometry"]
        if g.get("volume_x") is not None:
            return tuple(self.array(f"volume_{a}", mmap=False) for a in "xyz")
        nz, ny, nx = g["volume_shape"]
        x0, y0, z0 = g["volume_origin_um"]
        h = float(g["volume_spacing_um"])
        return x0 + h * np.arange(nx), y0 + h * np.arange(ny), z0 + h * np.arange(nz)

    @property
    def kind(self) -> str:
        return str(self.manifest["geometry"].get("kind", "polygons"))

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.manifest["geometry"].get("layers", ()))

    @property
    def layer_colors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(c) for c in self.manifest["geometry"].get("layer_colors", ()))

    def context_meshes(self) -> dict[str, dict[str, Any]]:
        """Non-cell anatomy: name -> {verts [V, 3], faces (list of index arrays), color}."""
        out = {}
        for key, rec in self.manifest["geometry"].get("context_meshes", {}).items():
            offsets = np.load(self.path / rec["face_offsets"]["file"], allow_pickle=False)
            index = np.load(self.path / rec["face_index"]["file"], allow_pickle=False)
            out[key] = {
                "verts": np.load(self.path / rec["verts"]["file"], allow_pickle=False),
                "faces": [index[a:b] for a, b in zip(offsets[:-1], offsets[1:])],
                "color": tuple(rec["color"]),
                "description": rec["description"],
            }
        return out

    @property
    def ions(self) -> tuple[str, ...]:
        return tuple(self.manifest["ions"])

    @property
    def ion_charges(self) -> tuple[int, ...]:
        return tuple(int(z) for z in self.manifest["ion_charges"])

    @property
    def constants(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.manifest["constants"].items()}

    @property
    def quantities(self) -> dict[str, dict[str, Any]]:
        return dict(self.manifest["quantities"])

    def record(self, name: str) -> dict[str, Any]:
        """The manifest entry for a geometry array, a time array or a quantity."""
        for section in (
            self.manifest["quantities"],
            self.manifest["geometry"],
            self.manifest["frames"],
        ):
            entry = section.get(name)
            if isinstance(entry, dict) and "file" in entry:
                return entry
        raise KeyError(name)

    def array(self, name: str, mmap: bool = True) -> np.ndarray:
        """Load one array by its manifest name, memory-mapped by default."""
        key = f"{name}:{mmap}"
        if key not in self._cache:
            path = self.path / self.record(name)["file"]
            self._cache[key] = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        return self._cache[key]

    def frame(self, name: str, index: int) -> np.ndarray:
        """One frame of a quantity, copied out of the memory map."""
        return np.array(self.array(name)[index])

    def unit(self, name: str) -> str:
        return str(self.record(name)["unit"])

    def location(self, name: str) -> str:
        return str(self.manifest["quantities"][name]["location"])

    def domain(self, name: str) -> str | None:
        return self.manifest["quantities"][name].get("domain")

    def polygons(self) -> tuple[np.ndarray, np.ndarray]:
        """``(verts [V, 2] in µm, offsets [N + 1])`` for building a mesh."""
        return self.array("cell_verts", mmap=False), self.array("cell_offsets", mmap=False)

    def removed(self) -> np.ndarray:
        """Boolean mask of cells removed by the wound."""
        return self.array("cell_removed", mmap=False)


def open_bundle(path: str | Path) -> Bundle:
    return Bundle(path)
