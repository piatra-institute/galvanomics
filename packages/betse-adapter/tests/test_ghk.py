from __future__ import annotations

import numpy as np
import pytest

from betse_adapter.ghk import ghk_voltage, membrane_mean

R, F, T = 8.314462618, 96485.33212, 310.0


def test_single_cation_is_nernst() -> None:
    v = ghk_voltage(
        np.array([[140.0], [12.0]]),
        np.array([[5.0], [145.0]]),
        np.array([[1.0], [0.0]]),
        np.array([1, 1]),
        T,
    )
    assert v[0] == pytest.approx(R * T / F * np.log(5.0 / 140.0), rel=1e-12)
    assert v[0] * 1e3 == pytest.approx(-89.0, abs=0.1)


def test_single_anion_has_inverted_ratio() -> None:
    v = ghk_voltage(np.array([[10.0]]), np.array([[110.0]]), np.array([[1.0]]), np.array([-1]), T)
    assert v[0] == pytest.approx(R * T / F * np.log(10.0 / 110.0), rel=1e-12)


def test_equal_concentrations_give_zero() -> None:
    c = np.array([[50.0], [50.0], [50.0]])
    v = ghk_voltage(c, c, np.array([[1.0], [0.3], [2.0]]), np.array([1, 1, -1]), T)
    assert v[0] == pytest.approx(0.0, abs=1e-15)


def test_classic_three_ion_value() -> None:
    # K:Na:Cl permeabilities 1:0.04:0.45 (Hodgkin and Katz), concentrations in mM.
    inside = np.array([[400.0], [50.0], [52.0]])
    outside = np.array([[20.0], [440.0], [560.0]])
    perm = np.array([[1.0], [0.04], [0.45]])
    v = ghk_voltage(inside, outside, perm, np.array([1, 1, -1]), 279.45)
    expected = (
        R * 279.45 / F * np.log((20 + 0.04 * 440 + 0.45 * 52) / (400 + 0.04 * 50 + 0.45 * 560))
    )
    assert v[0] == pytest.approx(expected, rel=1e-12)
    assert -0.065 < v[0] < -0.055


def test_divalent_ions_are_refused() -> None:
    with pytest.raises(ValueError):
        ghk_voltage(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), np.array([2]), T)


def test_membrane_mean() -> None:
    values = np.array([[1.0, 3.0, 5.0, 2.0, 4.0]])
    np.testing.assert_allclose(membrane_mean(values, np.array([0, 3, 5])), [[3.0, 3.0]])
