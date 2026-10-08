from __future__ import annotations

from pathlib import Path

import numpy as np
from tissuebundle.reader import Bundle
from tissuebundle.validate import validate_bundle

from episolver.cli import run_config
from episolver.geometry import build_tissue, point_in_polygon
from episolver.params import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_hex_tissue_geometry() -> None:
    tissue = build_tissue(load_config(CONFIGS / "tiny-test.yaml"))
    assert tissue.n_cells == 25 and tissue.wounded.sum() > 0
    np.testing.assert_allclose(tissue.area_m2, tissue.area_m2[0])
    assert np.all(tissue.lateral_m2 > tissue.area_m2)
    assert len(tissue.pair_len_m) == (tissue.neighbour >= 0).sum() // 2


def test_point_in_polygon() -> None:
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    inside = point_in_polygon(np.array([[1, 1], [3, 1], [1.9, 0.1]], float), square)
    assert inside.tolist() == [True, False, True]


def test_tiny_run_conserves_and_exports(tmp_path: Path) -> None:
    path, summary = run_config(CONFIGS / "tiny-test.yaml", tmp_path / "tiny.tbundle")
    assert summary["max_kcl_residual"] <= 1e-9
    assert summary["max_charge_error"] <= 1e-6
    assert validate_bundle(path) == []
    bundle = Bundle(path)
    assert bundle.extrusion_height_um == 10.0
    assert {
        "vmem_apical",
        "vmem_basolateral",
        "tep",
        "efield_apical_x",
        "open_kir2p1_basolateral",
    } <= set(bundle.quantities)
    assert bundle.domain("vmem_apical") == "apical"
    assert np.isnan(bundle.frame("vmem_apical", 1)[bundle.removed()]).all()
