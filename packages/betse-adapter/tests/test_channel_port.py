"""Check E5 (docs/plan.md, Phase 1b): episolver's ported channels and gap-junction gate against
BETSE 1.5.0's own classes, on a voltage grid that avoids removable singularities."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
from betse.science.channels import gap_junction, vg_cl, vg_funny, vg_k, vg_na  # noqa: E402
from episolver.channels import CHANNELS, GapJunctionGate  # noqa: E402

V = np.arange(-120.5, 60.0, 1.0)
BETSE = {
    "KLeak": vg_k.KLeak,
    "Kir2p1": vg_k.Kir2p1,
    "Kv1p5": vg_k.Kv1p5,
    "Kv3p4": vg_k.Kv3p4,
    "NaLeak": vg_na.NaLeak,
    "Nav1p3": vg_na.Nav1p3,
    "HCN2": vg_funny.HCN2,
    "ClLeak": vg_cl.ClLeak,
}


@pytest.mark.parametrize("name", sorted(BETSE))
def test_gating_functions_match_betse(name: str) -> None:
    theirs = object.__new__(BETSE[name])  # bypass BETSE's simulation-bound constructor
    theirs._calculate_state(V)
    ours = CHANNELS[name]
    for attr, fn in (
        ("_mInf", ours.m_inf),
        ("_mTau", ours.m_tau),
        ("_hInf", ours.h_inf),
        ("_hTau", ours.h_tau),
    ):
        expected = np.broadcast_to(np.asarray(getattr(theirs, attr), float), V.shape)
        np.testing.assert_allclose(fn(V), expected, rtol=1e-12, atol=0, err_msg=f"{name}{attr}")


@pytest.mark.parametrize("name", sorted(BETSE))
def test_powers_and_time_units_match_betse(name: str) -> None:
    theirs = object.__new__(BETSE[name])
    theirs.data_length = len(V)
    theirs._init_state(V)
    ours = CHANNELS[name]
    assert (ours.m_power, ours.h_power, ours.time_unit) == (
        theirs._mpower,
        theirs._hpower,
        theirs.time_unit,
    )


def test_gap_junction_rates_and_step_match_betse() -> None:
    vgj = V * 1e-3
    p = SimpleNamespace(gj_vthresh=15.0, gj_min=0.1, dt=1e-3)
    sim = SimpleNamespace(vgj=vgj, gj_block=1.0)
    theirs = gap_junction.Gap_Junction(sim, None, p)
    ours = GapJunctionGate(threshold_mv=15.0, minimum=0.1)
    alpha, beta = ours.rates(V)
    np.testing.assert_allclose(alpha, theirs.alpha, rtol=1e-12, atol=0)
    np.testing.assert_allclose(beta, theirs.beta, rtol=1e-12, atol=0)
    g0 = np.asarray(theirs.gjopen, float).copy()  # BETSE keeps the initial state on the gate
    sim.gjopen = g0.copy()  # BETSE's simulator copies the gate state before stepping it
    theirs.run(sim, None, p)
    np.testing.assert_allclose(ours.step(g0, V, p.dt), sim.gjopen, rtol=1e-12, atol=0)
