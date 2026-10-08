from __future__ import annotations

import numpy as np


def test_faces_follow_offsets(core) -> None:
    assert core.faces_from_offsets(np.array([0, 3, 7])) == [[0, 1, 2], [3, 4, 5, 6]]


def test_grid_faces_cover_grid(core) -> None:
    faces = core.grid_faces(3, 4)
    assert len(faces) == 2 * 3
    assert faces[0] == [0, 1, 5, 4]


def test_colormap_endpoints_nan_and_clipping(core) -> None:
    rgba = core.colormap(np.array([-50.0, 50.0, np.nan, 1e6]), -50.0, 50.0)
    np.testing.assert_allclose(rgba[0, :3], core._ANCHORS[0, 1:], atol=1e-6)
    np.testing.assert_allclose(rgba[1, :3], core._ANCHORS[-1, 1:], atol=1e-6)
    np.testing.assert_allclose(rgba[2], core.REMOVED_RGBA)
    np.testing.assert_allclose(rgba[3], rgba[1])


def test_corner_colors_repeat_per_cell(core) -> None:
    out = core.corner_colors(np.array([[1, 0, 0, 1], [0, 1, 0, 1]], float), np.array([0, 3, 5]))
    assert out.shape == (5, 4)
    assert (out[:3, 0] == 1).all() and (out[3:, 1] == 1).all()


def test_default_range_rounds_outward(core) -> None:
    assert core.default_range(np.array([-44.2, np.nan, -3.1])) == (-45.0, 0.0)
    assert core.default_range(np.array([-0.3, 1.2]), step=0.5, symmetric=True) == (-1.5, 1.5)


def test_frame_index_clamps(core) -> None:
    assert core.frame_index(-3, 0, 10) == 0
    assert core.frame_index(25, 0, 10) == 9
    assert core.frame_index(5, 2, 10) == 3


def test_blender_units(core) -> None:
    xyz = core.to_blender_xyz(np.array([[150.0, 20.0]]), z=-0.05)
    np.testing.assert_allclose(xyz, [[15.0, 2.0, -0.05]], rtol=1e-6)


def _hexagon(cx: float = 0.0) -> np.ndarray:
    a = np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack((cx + 5 * np.cos(a), 5 * np.sin(a)))


def test_prism_faces_point_outward_and_close(core) -> None:
    verts = np.concatenate((_hexagon(), _hexagon(20.0)))
    offsets = np.array([0, 6, 12])
    xyz, faces, cell, domain = core.prism_mesh(verts, offsets, 10.0)
    assert xyz.shape == (24, 3) and len(faces) == 2 * (2 + 6)
    assert (domain == core.APICAL).sum() == 2 and (domain == core.LATERAL).sum() == 12
    for f, owner in zip(faces, cell, strict=True):
        ring = xyz[f]
        normal = sum(np.cross(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring)))
        centre = xyz[[i for i, c in enumerate(np.repeat([0, 1], 6).tolist() * 2) if c == owner]]
        outward = ring.mean(axis=0) - centre.mean(axis=0)
        assert np.dot(normal, outward) > 0
    edges = {}
    for f in faces:
        for k in range(len(f)):
            e = (f[k], f[(k + 1) % len(f)])
            edges[e] = edges.get(e, 0) + 1
    assert all(edges.get((b, a), 0) == 1 for (a, b) in edges), "every edge has one opposite"


def test_domain_pair(core) -> None:
    names = {"vmem_apical", "vmem_basolateral", "tep"}
    assert core.domain_pair("vmem_apical", names) == ("vmem_apical", "vmem_basolateral")
    assert core.domain_pair("vmem_basolateral", names) == ("vmem_apical", "vmem_basolateral")
    assert core.domain_pair("tep", names) == ("tep", "tep")


def test_arrows_point_along_field(core) -> None:
    tri = core.arrow_triangles(
        np.array([[0.0, 0.0]]), np.array([2.0]), np.array([0.0]), 5.0, 1.0, 1.0
    )
    assert tri.shape == (3, 3)
    np.testing.assert_allclose(tri[2], [1.0, 0.0, 1.0])  # tip 10 um along +x, in Blender units


def test_glyph_colors(core) -> None:
    rgba = core.glyph_colors(np.array([0.0, 1.0, np.nan]), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(rgba[1, :3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rgba[2], core.REMOVED_RGBA)


def test_trilinear_sample_reproduces_linear_field_on_nonuniform_grid(core) -> None:
    xs, ys, zs = np.linspace(0, 10, 6), np.linspace(0, 4, 3), np.array([0.0, 0.5, 1.0, 3.0])
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    data = 2 * X - Y + 3 * Z
    pts = np.array([[1.3, 2.2, 0.7], [9.9, 0.1, 2.5], [20.0, 1.0, 1.0]])
    got = core.trilinear_sample((xs, ys, zs), data, pts)
    np.testing.assert_allclose(got[:2], 2 * pts[:2, 0] - pts[:2, 1] + 3 * pts[:2, 2])
    assert np.isnan(got[2])


def test_assign_particles_follows_weights(core) -> None:
    idx = core.assign_particles(np.array([0.0, 3.0, 1.0, np.nan]), 400)
    counts = np.bincount(idx, minlength=4)
    assert counts[0] == 0 and counts[3] == 0 and counts[1] == 300 and counts[2] == 100
    assert (core.assign_particles(np.zeros(3), 5) == -1).all()


def test_expected_counts_follow_the_derivation(core) -> None:
    counts = core.expected_counts(
        {"naleak": 8.0, "kleak": 0.55}, {"naleak": 4.7, "kleak": 34.0}, 1e-7, 200.0, 1.0
    )
    assert abs(counts["naleak"] - 8.0 / 4.7) < 1e-12
    assert abs(counts["kleak"] - 0.55 / 34.0) < 1e-12
    assert abs(counts["pump"] - 1e-7 * core.AVOGADRO * 1e-12 / 200.0) < 1e-9


def test_positions_have_the_expected_density_in_both_placements(core) -> None:
    rng = np.random.default_rng(0)
    for placement in ("uniform", "clustered"):
        n = [len(core.sample_positions(400.0, 1000.0, rng, placement)) for _ in range(200)]
        assert abs(np.mean(n) - 400.0) < 3 * np.sqrt(
            400.0 * (1 if placement == "uniform" else 6) / 200
        )
        pts = core.sample_positions(400.0, 1000.0, rng, placement)
        assert pts.min() >= 0 and pts.max() < 1000.0


def test_sampled_open_fraction_tracks_the_solver_within_3_over_sqrt_n(core) -> None:
    # Check N1 (docs/plan.md): at every frame, the sampled open fraction is within 3/sqrt(N) of
    # the exported one.
    fractions = np.array([1e-4, 0.01, 0.11, 0.5, 0.93, 1.0])
    n = 2000
    for frame, f in enumerate(fractions):
        drawn = core.sample_open(float(f), n, seed=3, frame=frame).mean()
        assert abs(drawn - f) <= 3 / np.sqrt(n)
    a = core.sample_open(0.5, 50, seed=3, frame=7)
    assert np.array_equal(a, core.sample_open(0.5, 50, seed=3, frame=7))  # reproducible
