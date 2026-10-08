"""Transmembrane fluxes, ported from BETSE 1.5.0 so that the non-polarized limit matches it.

BETSE is Copyright 2014-2025 Alexis Pietak & Cecil Curry, BSD 2-clause licence; see
THIRD_PARTY_NOTICES.md.

Sign convention (BETSE's): fluxes are positive from the outside (or the neighbour) into the
cell, in mol/m²/s; voltages are inside minus outside, in volts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOAT_NONCE = 1.0e-25
"""BETSE's guard against 0/0 at zero voltage or zero valence (``sim_toolbox.electroflux``)."""


@dataclass(frozen=True)
class Constants:
    """BETSE's rounded physical constants, kept for comparability with BETSE runs."""

    temperature_k: float = 310.0
    gas_constant: float = 8.314
    faraday: float = 96485.0
    membrane_thickness_m: float = 7.5e-9
    capacitance_f_per_m2: float = 0.05


@dataclass(frozen=True)
class PumpParameters:
    """BETSE's Na/K-ATPase parameters (``Parameters``: ``KmNK_*``, ``cATP``, ``deltaGATP``)."""

    km_na: float = 12.0
    km_k: float = 0.2
    km_atp: float = 0.5
    atp: float = 1.5
    adp: float = 0.1
    pi: float = 0.1
    delta_g_atp: float = -37000.0


def electroflux(
    c_out: np.ndarray,
    c_in: np.ndarray,
    diffusion: np.ndarray,
    distance: float | np.ndarray,
    z: float | np.ndarray,
    v_in_out: np.ndarray,
    const: Constants,
) -> np.ndarray:
    """Goldman flux from outside (A) to inside (B); BETSE's ``electroflux`` verbatim."""
    v = v_in_out + FLOAT_NONCE
    zc = z + FLOAT_NONCE
    alpha = (zc * v * const.faraday) / (const.gas_constant * const.temperature_k)
    exp_alpha = np.exp(-alpha)
    denominator = -np.expm1(-alpha)
    return -((diffusion * alpha) / distance) * ((c_in - c_out * exp_alpha) / denominator)


def pump_nak(
    na_in: np.ndarray,
    na_out: np.ndarray,
    k_in: np.ndarray,
    k_out: np.ndarray,
    v_in_out: np.ndarray,
    alpha: float | np.ndarray,
    const: Constants,
    pump: PumpParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Na and K fluxes into the cell (mol/m²/s); BETSE's ``pumpNaKATP`` verbatim.

    ``alpha`` is BETSE's ``alpha_NaK`` (its ``block`` factor folded in); zero disables it.
    """
    q_num = (pump.adp * 1e-3) * (pump.pi * 1e-3) * ((na_out * 1e-3) ** 3) * ((k_in * 1e-3) ** 2)
    q_den = (pump.atp * 1e-3) * ((na_in * 1e-3) ** 3) * ((k_out * 1e-3) ** 2)
    q_den = np.where(q_den == 0.0, 1.0e-15, q_den)
    q = q_num / q_den
    rt = const.gas_constant * const.temperature_k
    k_eq = np.exp(-(pump.delta_g_atp / rt - (const.faraday * v_in_out) / rt))
    forward = (
        ((na_in / pump.km_na) ** 3) * ((k_out / pump.km_k) ** 2) * (pump.atp / pump.km_atp)
    ) / (
        (1 + (na_in / pump.km_na) ** 3)
        * (1 + (k_out / pump.km_k) ** 2)
        * (1 + pump.atp / pump.km_atp)
    )
    f_na = -3 * alpha * forward * (1 - q / k_eq)
    f_k = -(2 / 3) * f_na
    return f_na, f_k


def ghk_voltage(
    c_in: np.ndarray,
    c_out: np.ndarray,
    permeability: np.ndarray,
    z: np.ndarray,
    const: Constants,
) -> np.ndarray:
    """Closed-form GHK voltage for monovalent ions; ``[n_ions, ...]`` arrays."""
    z = np.asarray(z)
    if np.any(np.abs(z) > 1):
        raise ValueError("GHK voltage is closed-form only for monovalent ions")
    cat, an = z > 0, z < 0
    num = (permeability[cat] * c_out[cat]).sum(0) + (permeability[an] * c_in[an]).sum(0)
    den = (permeability[cat] * c_in[cat]).sum(0) + (permeability[an] * c_out[an]).sum(0)
    return const.gas_constant * const.temperature_k / const.faraday * np.log(num / den)


def nkcc_flux(
    na_in: np.ndarray,
    k_in: np.ndarray,
    cl_in: np.ndarray,
    na_out: np.ndarray,
    k_out: np.ndarray,
    cl_out: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Cycles per membrane area and second of the Na-K-2Cl cotransporter, positive inward
    (each cycle moves 1 Na+, 1 K+ and 2 Cl- into the cell, so it carries no net charge).

    Not a BETSE port (BETSE has no cotransporter). The form is thermodynamically consistent and
    bounded: zero when the ion products Na K Cl^2 are equal on both sides (no free energy to
    drive it), signed by the free energy, and at most ``alpha`` in magnitude."""
    p_out = na_out * k_out * cl_out**2
    p_in = na_in * k_in * cl_in**2
    return alpha * (p_out - p_in) / (p_out + p_in)
