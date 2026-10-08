from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from tissuebundle.reader import Bundle
from tissuebundle.validate import validate_bundle

from episolver.params import ChannelSpec, Domain, TimeSpec
from episolver.skin import generate
from episolver.skin_audit import battery_mv, passive_config, passive_ghk_gap_mv
from episolver.skin_physics import SkinModel, SkinRunConfig
from episolver.skin_sim_export import export_skin_sim

TINY = SkinRunConfig.model_validate(
    yaml.safe_load((Path(__file__).parents[1] / "configs" / "skin-tiny.yaml").read_text())
)


def test_skin_run_conserves_charge_and_exports(tmp_path: Path) -> None:
    model = SkinModel(TINY, generate(TINY.skin))
    frames = model.run()
    assert model.diag.max_kcl_residual <= 1e-9
    assert model.diag.max_charge_error <= 1e-6
    assert len(frames) == 3 and model.wounded
    # The apical Na entry above the tight junctions charges the space beneath them positive.
    assert battery_mv(model, frames[0], far_um=12.0) > 0
    path = export_skin_sim(model, frames, tmp_path / "skin.tbundle", None, None)
    assert validate_bundle(path) == []
    bundle = Bundle(path)
    assert {"vmem", "phi_ecs", "flux_na_face", "flux_k_face", "pump_face"} <= set(
        bundle.quantities
    )
    vmem = np.asarray(bundle.array("vmem"))
    inert = np.asarray(bundle.array("cell_inert"))
    assert inert.any() and np.isnan(vmem[:, inert]).all()  # corneocytes
    assert np.isnan(vmem[-1][bundle.removed()]).all()


def test_uncoupled_passive_faces_settle_at_ghk() -> None:
    # Gap junctions off: coupled cells that drift apart in concentration hold each other away
    # from their own GHK voltage (audit K3/K3b), which is physics, not a solver error.
    time = TimeSpec(init_dt_s=0.2, init_s=40.0, dt_s=0.2, sim_s=0.0)
    cfg = passive_config(TINY, time, gap_junctions=False)
    model = SkinModel(cfg, generate(cfg.skin))
    model.run()
    assert passive_ghk_gap_mv(model).max() <= 0.01


def test_graded_grid_with_uniform_spacing_matches_the_uniform_grid() -> None:
    # TINY's wound is at x = 24 um, a multiple of the 4 um spacing, so a graded axis whose fine
    # zone covers the whole block has exactly the uniform grid's nodes.
    quick = TimeSpec(init_dt_s=0.02, init_s=0.1, dt_s=0.01, sim_s=0.03, sample_s=0.01)
    uniform = TINY.model_copy(update={"physics": TINY.physics.model_copy(update={"time": quick})})
    graded = uniform.model_copy(
        update={"physics": uniform.physics.model_copy(update={"x_fine_halfwidth_um": 100.0})}
    )
    runs = []
    for cfg in (uniform, graded):
        model = SkinModel(cfg, generate(cfg.skin))
        runs.append((model.grid.xs, np.stack([f.data["vmem"] for f in model.run()])))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_allclose(runs[0][1], runs[1][1], rtol=0, atol=1e-12, equal_nan=True)


def test_graded_half_block_conserves_charge() -> None:
    cfg = TINY.model_copy(
        update={
            "skin": TINY.skin.model_copy(
                update={
                    "width_x_um": 96.0,
                    "papilla_offset_x_um": 24.0,
                    "wound": TINY.skin.wound.model_copy(update={"x_um": 0.0}),
                }
            ),
            "physics": TINY.physics.model_copy(
                update={"x_fine_halfwidth_um": 20.0, "x_max_spacing_um": 12.0}
            ),
        }
    )
    model = SkinModel(cfg, generate(cfg.skin))
    frames = model.run()
    assert np.diff(model.grid.xs).max() > np.diff(model.grid.xs).min()  # graded
    assert model.diag.max_kcl_residual <= 1e-9 and model.diag.max_charge_error <= 1e-6
    assert model.skin.wounded.any() and np.all(
        model.skin.derived["centroid"][model.skin.wounded, 0] < 10
    )
    assert np.all(frames[-1].data["efield_x"][:, :, 0] == 0.0)  # mirror plane at the wound


def test_na_block_scales_only_apical_na_channels_in_its_zone() -> None:
    cfg = TINY.model_copy(
        update={
            "physics": TINY.physics.model_copy(
                update={"na_block_fraction": 0.75, "na_block_within_um": 10.0}
            )
        }
    )
    model = SkinModel(cfg, generate(cfg.skin))
    x = model.skin.derived["centroid"][:, 0]
    for (channel, *_rest, side), scale in zip(model.channels, model.channel_scale, strict=True):
        if channel.name == "NaLeak" and side == "apical":
            near = np.abs(x[model.f_cell[_rest[2]]] - cfg.skin.wound.x_um) < 10.0
            assert near.any() and (~near).any()
            np.testing.assert_allclose(scale[near], 0.25)
            np.testing.assert_allclose(scale[~near], 1.0)
        else:
            np.testing.assert_allclose(scale, 1.0)


def test_nkcc_is_zero_at_equilibrium_signed_and_bounded() -> None:
    from episolver.flux import nkcc_flux

    one = np.ones(3)
    # Equal ion products: no free energy, no flux (here Na K Cl^2 = 145 * 5 * 115^2 both sides).
    j = nkcc_flux(145 * one, 5 * one, 115 * one, 145 * one, 5 * one, 115 * one, 1e-7 * one)
    np.testing.assert_allclose(j, 0.0, atol=0)
    j = nkcc_flux(12 * one, 139 * one, 4 * one, 145 * one, 5 * one, 115 * one, 1e-7 * one)
    assert np.all(j > 0) and np.all(j <= 1e-7)  # loads the cell, at most alpha


def _chloride_tiny(nkcc: float = 1e-6, apical_cl: float = 1e-16) -> SkinRunConfig:
    sides = {}
    for layer in (0, 1, 2):
        baso = Domain(
            pump_alpha=1e-7, nkcc_alpha=nkcc, channels=(ChannelSpec(type="KLeak", max_d=5e-17),)
        )
        apical = baso
        if layer == 2:
            apical = Domain(
                pump_alpha=0.0,
                channels=(
                    ChannelSpec(type="NaLeak", max_d=3e-16),
                    ChannelSpec(type="KLeak", max_d=5e-17),
                    ChannelSpec(type="ClLeak", max_d=apical_cl),
                ),
            )
        sides[layer] = {"apical": apical, "basolateral": baso}
    return TINY.model_copy(
        update={
            "physics": TINY.physics.model_copy(
                update={"ion_profile": "mammal_no_ca", "membranes": sides}
            )
        }
    )


def test_nkcc_carries_no_current_and_chloride_skin_conserves(tmp_path: Path) -> None:
    cfg = _chloride_tiny()
    model = SkinModel(cfg, generate(cfg.skin))
    assert model.ions == ["na", "k", "cl", "m", "p"]
    v = model.face_vmem()
    d = model.face_diffusion()
    with_nkcc = model.face_fluxes(v, d)
    saved, model.f_nkcc = model.f_nkcc, np.zeros_like(model.f_nkcc)
    without = model.face_fluxes(v, d)
    model.f_nkcc = saved
    current = lambda f: (model.z[:, None] * f).sum(axis=0)  # noqa: E731
    np.testing.assert_allclose(current(with_nkcc), current(without), rtol=0, atol=1e-18)
    assert not np.allclose(with_nkcc, without)  # it does move ions
    frames = model.run()
    assert model.diag.max_kcl_residual <= 1e-9 and model.diag.max_charge_error <= 1e-6
    granular = model.skin.cell_layer == 2
    cl_in = model.conc[model.i_cl][granular & model.alive].mean()
    assert cl_in > 4.0  # NKCC loads chloride above BETSE's 4 mM starting value
    path = export_skin_sim(model, frames, tmp_path / "cl.tbundle", None, None)
    assert validate_bundle(path) == []
    bundle = Bundle(path)
    assert {"flux_cl_face", "nkcc_face"} <= set(bundle.quantities)
    assert "nkcc_alpha_l0_basolateral" in bundle.constants


def test_modifiers_scale_their_pathway_in_their_zone() -> None:
    from episolver.skin_physics import Modifier

    base = _chloride_tiny()
    mods = (
        Modifier(target="ClLeak", side="apical", factor=0.0),
        Modifier(target="nkcc", side="basolateral", factor=0.5, within_um=10.0),
        Modifier(target="base_cl", side="apical", factor=2.0),
    )
    cfg = base.model_copy(update={"physics": base.physics.model_copy(update={"modifiers": mods})})
    plain = SkinModel(base, generate(base.skin))
    model = SkinModel(cfg, generate(cfg.skin))
    for (channel, *_rest, side), scale in zip(model.channels, model.channel_scale, strict=True):
        expected = 0.0 if (channel.name == "ClLeak" and side == "apical") else 1.0
        np.testing.assert_allclose(scale, expected)
    baso = model.f_domain == 1
    near = np.abs(model.skin.derived["centroid"][model.f_cell, 0] - cfg.skin.wound.x_um) < 10
    np.testing.assert_allclose(model.f_nkcc[baso & near], 0.5 * plain.f_nkcc[baso & near])
    np.testing.assert_allclose(model.f_nkcc[baso & ~near], plain.f_nkcc[baso & ~near])
    apical = model.f_domain == 0
    np.testing.assert_allclose(
        model.f_base_d[model.i_cl, apical], 2.0 * plain.f_base_d[plain.i_cl, apical]
    )


def test_wound_chloride_channels_switch_on_only_at_exposed_faces() -> None:
    from episolver.skin_physics import WOUND

    cfg = TINY.model_copy(
        update={
            "physics": TINY.physics.model_copy(
                update={
                    "ion_profile": "mammal_no_ca",
                    "cell_conc_override": {"cl": 6.8, "m": 9.2},
                    "wound_cl_d": 1e-15,
                }
            )
        }
    )
    model = SkinModel(cfg, generate(cfg.skin))
    k = next(i for i, ch in enumerate(model.channels) if ch[7] == "wound")
    assert np.all(model.channel_scale[k] == 0.0)  # off before the wound
    assert np.allclose(model.conc[model.i_cl], 6.8)
    frames = model.run()
    on = model.channel_scale[k] > 0
    idx, w = model.f_nodes
    touching = np.any((model.region[idx] == WOUND) & (w > 1e-12), axis=1)
    assert on.any() and np.all(touching[on]) and np.all(model.alive[model.f_cell[on]])
    assert model.diag.max_kcl_residual <= 1e-9 and model.diag.max_charge_error <= 1e-6
    assert "chan_g_clleak" in frames[-1].data
