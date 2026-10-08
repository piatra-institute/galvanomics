"""Ion channels and gap-junction gating, ported verbatim from BETSE 1.5.0.

BETSE is Copyright 2014-2025 Alexis Pietak & Cecil Curry, BSD 2-clause licence; see
THIRD_PARTY_NOTICES.md.

Each channel is Hodgkin-Huxley style: open fraction m^p h^q, gates relaxing toward m∞(V),
h∞(V) with time constants τ(V), voltage in mV, time in the channel's own unit (ms for
voltage-gated channels). The source class is named in every entry; a test in
``betse-adapter`` compares every function with BETSE's on a voltage grid (check E5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Fn = Callable[[np.ndarray], np.ndarray]


def _const(value: float) -> Fn:
    return lambda v: np.full_like(np.asarray(v, dtype=np.float64), value)


@dataclass(frozen=True)
class ChannelModel:
    name: str
    source: str
    ions: dict[str, float]
    m_power: int
    h_power: int
    time_unit: float
    m_inf: Fn
    m_tau: Fn
    h_inf: Fn = field(default_factory=lambda: _const(1.0))
    h_tau: Fn = field(default_factory=lambda: _const(1.0))

    def open_fraction(self, m: np.ndarray, h: np.ndarray) -> np.ndarray:
        return (m**self.m_power) * (h**self.h_power)

    def step(self, m: np.ndarray, h: np.ndarray, v_mv: np.ndarray, dt_s: float) -> tuple:
        """BETSE's semi-implicit gate update (``ChannelsABC.update_mh``)."""
        dt = dt_s * self.time_unit
        m_tau, h_tau = self.m_tau(v_mv), self.h_tau(v_mv)
        m = (m_tau * m + dt * self.m_inf(v_mv)) / (m_tau + dt)
        h = (h_tau * h + dt * self.h_inf(v_mv)) / (h_tau + dt)
        return m, h


def _nav1p3_rates(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = (0.182 * (v - -26)) / (1 - np.exp(-(v - -26) / 9))
    beta = (0.124 * (-v - 26)) / (1 - np.exp(-(-v - 26) / 9))
    return alpha, beta


def _leak(name: str, source: str, ion: str) -> ChannelModel:
    return ChannelModel(name, source, {ion: 1.0}, 0, 0, 1.0, _const(1.0), _const(1.0))


CHANNELS: dict[str, ChannelModel] = {
    "KLeak": _leak("KLeak", "betse.science.channels.vg_k.KLeak", "k"),
    "NaLeak": _leak("NaLeak", "betse.science.channels.vg_na.NaLeak", "na"),
    "ClLeak": _leak("ClLeak", "betse.science.channels.vg_cl.ClLeak", "cl"),
    "Kir2p1": ChannelModel(
        "Kir2p1",
        "betse.science.channels.vg_k.Kir2p1",
        {"k": 1.0},
        1,
        2,
        1.0e3,
        m_inf=lambda v: 1 / (1 + np.exp((v - (-96.48)) / 23.26)),
        m_tau=lambda v: 3.7 + (-3.37 / (1 + np.exp((v - -32.9) / 27.93))),
        h_inf=lambda v: 1 / (1 + np.exp((v - (-168.28)) / -44.13)),
        h_tau=lambda v: 0.85 + (306.3 / (1 + np.exp((v - -118.29) / -27.23))),
    ),
    "Kv1p5": ChannelModel(
        "Kv1p5",
        "betse.science.channels.vg_k.Kv1p5",
        {"k": 1.0},
        1,
        1,
        1.0e3,
        m_inf=lambda v: 1.0000 / (1 + np.exp((v - -6.0000) / -6.4000)),
        m_tau=lambda v: (-0.1163 * v) + 8.3300,
        h_inf=lambda v: 1.0000 / (1 + np.exp((v - -25.3000) / 3.5000)),
        h_tau=lambda v: (-15.5000 * v) + 1620.0000,
    ),
    "Kv3p4": ChannelModel(
        "Kv3p4",
        "betse.science.channels.vg_k.Kv3p4",
        {"k": 1.0},
        1,
        1,
        1.0e3,
        m_inf=lambda v: 1 / (1 + np.exp((v - (-3.400)) / (-8.400))),
        m_tau=lambda v: 10.000 / (1 + np.exp((v - (4.440)) / (38.140))),
        h_inf=lambda v: 1 / (1 + np.exp((v - (-53.320)) / (7.400))),
        h_tau=lambda v: 20000.000 / (1 + np.exp((v - (-46.560)) / (-44.140))),
    ),
    "Nav1p3": ChannelModel(
        "Nav1p3",
        "betse.science.channels.vg_na.Nav1p3",
        {"na": 1.0},
        3,
        1,
        1.0e3,
        m_inf=lambda v: _nav1p3_rates(v)[0] / np.add(*_nav1p3_rates(v)),
        m_tau=lambda v: 1 / np.add(*_nav1p3_rates(v)),
        h_inf=lambda v: 1 / (1 + np.exp((v - (-65.0)) / 8.1)),
        h_tau=lambda v: 0.40 + (0.265 * np.exp(-v / 9.47)),
    ),
    "HCN2": ChannelModel(
        "HCN2",
        "betse.science.channels.vg_funny.HCN2 (Ca term dropped)",
        {"na": 0.2, "k": 1.0},
        1,
        0,
        1.0e3,
        m_inf=lambda v: 1.0000 / (1 + np.exp(((v - 10) - -99) / 6.2)),  # BETSE: V = V - 10
        m_tau=_const(184.0000),
    ),
}


@dataclass(frozen=True)
class GapJunctionGate:
    """BETSE's kinetic gap-junction gate (``channels.gap_junction.Gap_Junction``)."""

    threshold_mv: float = 15.0
    minimum: float = 0.1
    lam: float = 0.0013
    a1: float = 0.077
    a2: float = 0.14

    def rates(self, v_mv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v1 = np.abs(v_mv)
        alpha = self.lam * np.exp(-self.a1 * (v1 - self.threshold_mv))
        beta = self.lam * np.exp(self.a2 * (v1 - self.threshold_mv))
        return alpha, beta / (1 + 50 * beta)

    def steady(self, v_mv: np.ndarray) -> np.ndarray:
        """Fixed point of :meth:`step`. (BETSE initializes to alpha/(alpha+beta) + minimum,
        which can exceed 1; the fixed point is used here instead.)"""
        alpha, beta = self.rates(v_mv)
        return (alpha + beta * self.minimum) / (alpha + beta)

    def step(self, g: np.ndarray, v_mv: np.ndarray, dt_s: float) -> np.ndarray:
        alpha, beta = self.rates(v_mv)
        dt = dt_s * 1e3
        return (g + dt * (alpha + beta * self.minimum)) / (1 + alpha * dt + beta * dt)
