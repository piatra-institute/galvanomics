from __future__ import annotations

import numpy as np
from tissuebundle.validate import validate_bundle

from episolver.skin import BASEMENT, SURFACE, LayerSpec, SkinSpec, facing_faces, generate
from episolver.skin_export import export_skin_bundle

SMALL = SkinSpec(
    width_x_um=60.0,
    width_y_um=40.0,
    papilla_wavelength_um=40.0,
    papilla_amplitude_um=4.0,
    layers=(
        LayerSpec(name="basale", rows=1, thickness_um=10.0, lateral_spacing_um=9.0),
        LayerSpec(name="spinosum", rows=2, thickness_um=20.0, lateral_spacing_um=12.0),
        LayerSpec(name="corneum", rows=3, thickness_um=3.0, lateral_spacing_um=24.0),
    ),
    wound={"x_um": 30.0, "top_width_um": 16.0, "bottom_width_um": 8.0},
)


def test_cells_tile_the_epidermis_exactly() -> None:
    skin = generate(SMALL)
    d = skin.derived
    assert np.all(d["volume_um3"] > 0)
    # No gaps or overlaps: shared faces cancel exactly, so the cells sum to the volume that the
    # epidermis boundary encloses.
    signed = np.einsum("ij,ij->i", d["face_normal"], d["face_centre_um"]) * d["face_area_um2"] / 3
    enclosed = signed[skin.face_boundary != 0].sum()
    assert abs(d["volume_um3"].sum() / enclosed - 1) < 1e-9
    # The papillae have zero mean, so the epidermis keeps the volume of the flat slab, up to
    # their piecewise-linear sampling at cell vertices (0.11% on this coarse fixture, 0.03% on
    # the default skin).
    box = 60.0 * 40.0 * 33.0
    assert abs(d["volume_um3"].sum() / box - 1) < 2e-3


def test_shared_faces_pair_up_with_equal_area() -> None:
    skin = generate(SMALL)
    facing = facing_faces(skin)
    shared = skin.face_neighbour_cell >= 0
    assert np.all(facing[shared] >= 0)
    area = skin.derived["face_area_um2"]
    np.testing.assert_allclose(area[shared], area[facing[shared]], rtol=1e-9)


def test_basal_cells_sit_on_the_basement_membrane_and_corneum_forms_the_surface() -> None:
    skin = generate(SMALL)
    face_cell = skin.derived["face_cell"]
    on_basement = np.unique(skin.cell_layer[face_cell[skin.face_boundary == BASEMENT]])
    on_surface = np.unique(skin.cell_layer[face_cell[skin.face_boundary == SURFACE]])
    assert on_basement.tolist() == [0]
    assert on_surface.tolist() == [2]


def test_incision_removes_cells_and_bundle_validates(tmp_path) -> None:
    skin = generate(SMALL)
    assert 0 < skin.wounded.sum() < skin.n_cells
    path = export_skin_bundle(skin, tmp_path / "skin.tbundle")
    assert validate_bundle(path) == []


def test_kelvin_layer_makes_tetrakaidecahedra() -> None:
    # Without jitter, a body-centred lattice gives Kelvin cells: 6 square and 8 hexagonal faces
    # (flattened here, so the faces are affine images of squares and regular hexagons).
    spec = SkinSpec(
        width_x_um=4 * 20.0,
        width_y_um=4 * 20.0,
        papilla_amplitude_um=0.0,
        layers=(
            LayerSpec(name="basale", rows=1, thickness_um=6.0, lateral_spacing_um=10.0),
            LayerSpec(
                name="granulosum",
                rows=5,
                thickness_um=15.0,
                lateral_spacing_um=20.0 / np.sqrt(np.sqrt(3) / 2),
                jitter=0.0,
                packing="kelvin",
            ),
            LayerSpec(
                name="corneum", rows=1, thickness_um=2.0, lateral_spacing_um=30.0, row_sheets=True
            ),
        ),
        wound={"kind": "none"},
    )
    skin = generate(spec)
    d = skin.derived
    sides = np.diff(skin.face_offsets)
    face_cell = d["face_cell"]
    centroid = d["centroid"]
    middle = (
        (skin.cell_layer == 1)
        & (np.abs(centroid[:, 2] - (6.0 + 7.5)) < 0.5)
        & np.all((centroid[:, :2] > 20.0) & (centroid[:, :2] < 60.0), axis=1)
    )
    assert middle.sum() >= 4
    for c in np.flatnonzero(middle):
        own = sides[face_cell == c]
        assert len(own) == 14 and (own == 4).sum() == 6 and (own == 6).sum() == 8, own
    signed = np.einsum("ij,ij->i", d["face_normal"], d["face_centre_um"]) * d["face_area_um2"] / 3
    assert abs(d["volume_um3"].sum() / signed[skin.face_boundary != 0].sum() - 1) < 1e-9
