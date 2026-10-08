from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tissuebundle.validate import validate_bundle


def test_detects_hash_mismatch(synthetic: Path) -> None:
    vmem = np.load(synthetic / "vmem.npy")
    vmem[0, 0] += 1.0
    np.save(synthetic / "vmem.npy", vmem)
    assert any("sha256" in e for e in validate_bundle(synthetic))


def test_detects_values_on_removed_cells(synthetic: Path) -> None:
    removed = np.load(synthetic / "cell_removed.npy")
    vmem = np.load(synthetic / "vmem.npy")
    vmem[2, np.flatnonzero(removed)[0]] = -50.0
    np.save(synthetic / "vmem.npy", vmem)
    _rehash(synthetic, "vmem")
    assert any("removed entities" in e for e in validate_bundle(synthetic))


def test_detects_nan_on_live_cells(synthetic: Path) -> None:
    removed = np.load(synthetic / "cell_removed.npy")
    vmem = np.load(synthetic / "vmem.npy")
    vmem[1, np.flatnonzero(~removed)[0]] = np.nan
    np.save(synthetic / "vmem.npy", vmem)
    _rehash(synthetic, "vmem")
    assert any("non-finite values on live" in e for e in validate_bundle(synthetic))


def test_allows_a_wholly_unavailable_frame(synthetic: Path) -> None:
    vmem = np.load(synthetic / "vmem.npy")
    vmem[0] = np.nan
    np.save(synthetic / "vmem.npy", vmem)
    _rehash(synthetic, "vmem")
    assert validate_bundle(synthetic) == []


def test_detects_clockwise_polygon(synthetic: Path) -> None:
    verts = np.load(synthetic / "cell_verts.npy")
    verts[0:6] = verts[0:6][::-1]
    np.save(synthetic / "cell_verts.npy", verts)
    _rehash(synthetic, "cell_verts", section="geometry")
    assert any("counter-clockwise" in e for e in validate_bundle(synthetic))


def test_detects_wrong_location_shape(synthetic: Path) -> None:
    manifest = json.loads((synthetic / "bundle.json").read_text())
    manifest["quantities"]["vmem"]["location"] = "membrane"
    (synthetic / "bundle.json").write_text(json.dumps(manifest))
    assert any("expected" in e for e in validate_bundle(synthetic))


def _rehash(root: Path, name: str, section: str = "quantities") -> None:
    from tissuebundle.writer import sha256_file

    manifest = json.loads((root / "bundle.json").read_text())
    manifest[section][name]["sha256"] = sha256_file(root / f"{name}.npy")
    (root / "bundle.json").write_text(json.dumps(manifest))


def test_detects_asymmetric_neighbours(synthetic: Path) -> None:
    nb = np.load(synthetic / "mem_neighbour.npy")
    k = int(np.flatnonzero(nb >= 0)[0])
    nb[k] = (nb[k] + 1) % len(nb)
    np.save(synthetic / "mem_neighbour.npy", nb)
    _rehash(synthetic, "mem_neighbour", section="geometry")
    assert any("mem_neighbour" in e for e in validate_bundle(synthetic))
