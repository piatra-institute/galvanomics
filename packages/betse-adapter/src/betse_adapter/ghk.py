"""The Goldman-Hodgkin-Katz voltage equation, written independently of BETSE.

For monovalent ions only: with divalent ions the GHK voltage has no closed form, and this
module refuses rather than approximating.
"""

from __future__ import annotations

import numpy as np
from tissuebundle.reader import Bundle


def ghk_voltage(
    conc_in: np.ndarray,
    conc_out: np.ndarray,
    permeability: np.ndarray,
    charges: np.ndarray,
    temperature_k: float,
    gas_constant: float = 8.314462618,
    faraday: float = 96485.33212,
) -> np.ndarray:
    """V_in - V_out in volts. Arrays are ``[n_ions, ...]``; permeability in any consistent
    unit (only ratios matter). Ions with zero charge are ignored."""
    charges = np.asarray(charges)
    if np.any(np.abs(charges) > 1):
        raise ValueError("GHK voltage equation is closed-form only for monovalent ions")
    cation, anion = charges > 0, charges < 0
    numerator = (permeability[cation] * conc_out[cation]).sum(axis=0) + (
        permeability[anion] * conc_in[anion]
    ).sum(axis=0)
    denominator = (permeability[cation] * conc_in[cation]).sum(axis=0) + (
        permeability[anion] * conc_out[anion]
    ).sum(axis=0)
    return gas_constant * temperature_k / faraday * np.log(numerator / denominator)


def membrane_mean(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Plain mean over each cell's membranes along the last axis: ``[..., M] -> [..., N]``."""
    sums = np.add.reduceat(values, offsets[:-1], axis=-1)
    return sums / np.diff(offsets)


def ghk_from_bundle(bundle: Bundle, frame: int) -> np.ndarray:
    """Per-cell GHK voltage (V) at one frame, from the bundle's concentrations and membrane
    diffusion constants, averaged to cells the way BETSE's calculator does."""
    const = bundle.constants
    _, offsets = bundle.polygons()
    diff = bundle.frame("diff_membrane", frame)
    conc_in = bundle.frame("conc_cell", frame)
    conc_env = bundle.frame("conc_env", frame).reshape(len(bundle.ions), -1)
    at_membranes = conc_env[:, bundle.array("mem_grid_index", mmap=False)]
    permeability = membrane_mean(diff, offsets) / const["membrane_thickness_m"]
    conc_out = membrane_mean(at_membranes, offsets)
    return ghk_voltage(
        conc_in,
        conc_out,
        permeability,
        np.asarray(bundle.ion_charges),
        const["temperature_k"],
        gas_constant=const["gas_constant_j_per_k_mol"],
        faraday=const["faraday_c_per_mol"],
    )
