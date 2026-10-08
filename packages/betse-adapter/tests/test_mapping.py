from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from betse_adapter.export import ExportError, _scatter, mapping


def _cells(n: int, per_cell: int = 3) -> SimpleNamespace:
    centres = np.arange(2 * n, dtype=float).reshape(n, 2)
    verts = np.empty(n, dtype=object)
    for i in range(n):
        verts[i] = centres[i] + np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mids = np.arange(2 * n * per_cell, dtype=float).reshape(-1, 2)
    return SimpleNamespace(
        cell_i=list(range(n)),
        mem_i=list(range(n * per_cell)),
        cell_centres=centres,
        cell_verts=verts,
        mem_mids_flat=mids,
        map_mem2ecm=np.arange(n * per_cell),
    )


def _cut(cells: SimpleNamespace, removed: list[int]) -> SimpleNamespace:
    kept = np.delete(np.arange(len(cells.cell_i)), removed)
    kept_m = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in kept])
    return SimpleNamespace(
        cell_i=list(range(len(kept))),
        mem_i=list(range(len(kept_m))),
        cell_centres=cells.cell_centres[kept],
        cell_verts=cells.cell_verts[kept],
        mem_mids_flat=cells.mem_mids_flat[kept_m],
        map_mem2ecm=cells.map_mem2ecm[kept_m],
    )


def test_exact_mapping_after_cut() -> None:
    init = _cells(5)
    sim = SimpleNamespace(target_inds_cell_o=[1, 3], target_inds_mem_o=[3, 4, 5, 9, 10, 11])
    kept_c, kept_m, checks = mapping(init, _cut(init, [1, 3]), sim)
    np.testing.assert_array_equal(kept_c, [0, 2, 4])
    assert len(kept_m) == 9 and all(checks.values())


def test_mapping_detects_reordering() -> None:
    init = _cells(4)
    post = _cut(init, [2])
    post.cell_centres = post.cell_centres[::-1].copy()
    sim = SimpleNamespace(target_inds_cell_o=[2], target_inds_mem_o=[6, 7, 8])
    _, _, checks = mapping(init, post, sim)
    assert not checks["centres_identical"]


def test_cell_count_change_without_indices_is_an_error() -> None:
    init = _cells(4)
    with pytest.raises(ExportError):
        mapping(init, _cut(init, [0]), SimpleNamespace())


def test_no_cut_keeps_everything() -> None:
    init = _cells(3)
    kept_c, kept_m, checks = mapping(init, init, SimpleNamespace())
    assert len(kept_c) == 3 and len(kept_m) == 9 and all(checks.values())


def test_scatter_fills_nan() -> None:
    out = _scatter(np.array([[1.0, 2.0]]), np.array([0, 2]), 3)
    np.testing.assert_array_equal(np.isnan(out), [[False, True, False]])
