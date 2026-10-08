from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tissuebundle.reader import Bundle, BundleError
from tissuebundle.synthetic import hex_sheet, make_synthetic
from tissuebundle.validate import polygon_areas, validate_bundle
from tissuebundle.writer import QuantityInput, SourceInfo, write_bundle


def test_synthetic_is_valid(synthetic: Path) -> None:
    assert validate_bundle(synthetic) == []


def test_reader_matches_written_arrays(tmp_path: Path) -> None:
    verts, offsets, centres = hex_sheet(3, 2, 5.0)
    n = len(centres)
    removed = np.zeros(n, dtype=bool)
    removed[1] = True
    vmem = np.full((3, n), -40.0)
    vmem[1:, removed] = np.nan
    path = write_bundle(
        tmp_path / "b.tbundle",
        name="tiny",
        description="",
        source=SourceInfo(solver="test", solver_version="1", seed=7, command=("x",)),
        ions=(),
        constants={"temperature_k": 310.0},
        cell_verts=verts,
        cell_offsets=offsets,
        cell_centres=centres,
        cell_removed=removed,
        time_integrated_s=np.array([0.0, 1.0, 2.0]),
        time_reported_s=np.array([np.nan, 0.9, 1.9]),
        baseline_frames=(0,),
        quantities={"vmem": QuantityInput(vmem, "mV", "cell", "")},
    )
    assert validate_bundle(path) == []
    bundle = Bundle(path)
    assert bundle.n_cells == n and bundle.n_frames == 3 and bundle.baseline_frames == (0,)
    np.testing.assert_array_equal(bundle.array("cell_verts"), verts)
    np.testing.assert_array_equal(bundle.array("cell_offsets"), offsets)
    read = bundle.array("vmem")
    assert read.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(read), np.isnan(vmem))
    np.testing.assert_array_equal(read[np.isfinite(read)], vmem[np.isfinite(vmem)])
    np.testing.assert_array_equal(np.isnan(bundle.array("time_reported_s")), [True, False, False])
    assert bundle.unit("vmem") == "mV" and bundle.location("vmem") == "cell"
    assert bundle.manifest["provenance"]["seed"] == 7


def test_refuses_to_overwrite(synthetic: Path) -> None:
    with pytest.raises(FileExistsError):
        make_synthetic(synthetic)


def test_hex_sheet_is_counter_clockwise() -> None:
    verts, offsets, _ = hex_sheet(4, 3, 5.0)
    assert np.all(polygon_areas(verts, offsets) > 0)


def test_reader_rejects_unknown_major_version(synthetic: Path) -> None:
    manifest = synthetic / "bundle.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = "1.0.0"
    manifest.write_text(json.dumps(data))
    with pytest.raises(BundleError):
        Bundle(synthetic)


def test_hex_neighbours_are_symmetric_and_face_each_other() -> None:
    from tissuebundle.synthetic import hex_neighbours

    verts, offsets, _ = hex_sheet(5, 4, 5.0)
    nb = hex_neighbours(5, 4)
    paired = np.flatnonzero(nb >= 0)
    assert np.all(nb[nb[paired]] == paired)
    mids = 0.5 * (verts + np.roll(verts.reshape(-1, 6, 2), 1, axis=1).reshape(-1, 2))
    # Facing midpoints differ only by the gap left by shrinking each hexagon (0.69 um here).
    np.testing.assert_allclose(mids[paired], mids[nb[paired]], atol=1.0)
