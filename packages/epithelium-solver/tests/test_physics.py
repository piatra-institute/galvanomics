from __future__ import annotations

import numpy as np
import pytest

from episolver.flux import Constants, PumpParameters, electroflux, ghk_voltage, pump_nak

C = Constants()


def test_electroflux_reverses_at_nernst() -> None:
    e_k = C.gas_constant * C.temperature_k / C.faraday * np.log(5.0 / 140.0)
    flux = electroflux(
        np.array(5.0), np.array(140.0), 1e-18, C.membrane_thickness_m, 1, np.array(e_k), C
    )
    assert abs(flux) < 1e-22


def test_electroflux_zero_voltage_is_fick() -> None:
    flux = electroflux(np.array(10.0), np.array(4.0), 1e-18, 1e-8, 1, np.array(0.0), C)
    assert flux == pytest.approx(1e-18 * (10.0 - 4.0) / 1e-8, rel=1e-6)


def test_pump_moves_three_na_out_two_k_in() -> None:
    f_na, f_k = pump_nak(
        np.array(12.0),
        np.array(145.0),
        np.array(139.0),
        np.array(5.0),
        np.array(-0.05),
        1e-7,
        C,
        PumpParameters(),
    )
    assert f_na < 0 < f_k
    assert f_k == pytest.approx(-2 / 3 * f_na)


def test_ghk_voltage_matches_flux_balance() -> None:
    cin = np.array([[12.0], [139.0], [16.0]])
    cout = np.array([[145.0], [5.0], [140.0]])
    d = np.array([[2e-18], [1e-18], [1e-18]])
    z = np.array([1, 1, -1])
    v = ghk_voltage(cin, cout, d / C.membrane_thickness_m, z, C)
    flux = electroflux(cout, cin, d, C.membrane_thickness_m, z[:, None], v[None, :], C)
    assert abs((z[:, None] * flux).sum()) < 1e-20
