"""Time stepping: one sparse Kirchhoff solve per step, then the concentration update.

Unknowns: intracellular potential of each cell (index i), apical bath nodes (N + k) and basal
bath nodes (N + G + k). Every connection is a *branch* between two weighted node sets, with
current ``G * (u - v) + s`` from the first set to the second; membranes add backward-Euler
capacitance and a chord linearization of their ionic current about the previous step.
See ``docs/epithelium.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from episolver.channels import CHANNELS, ChannelModel, GapJunctionGate
from episolver.flux import Constants, PumpParameters, electroflux, pump_nak
from episolver.geometry import UM, BathGrid, Tissue, point_in_polygon
from episolver.params import RunConfig

logger = logging.getLogger(__name__)

DOMAINS = ("apical", "basolateral")
DV = 1.0e-6  # centred-difference step for chord conductances [V]


@dataclass
class ChannelState:
    model: ChannelModel
    max_d: float
    weights: np.ndarray  # [n_ions] relative permeability per ion
    m: np.ndarray
    h: np.ndarray

    def open_fraction(self) -> np.ndarray:
        return self.model.open_fraction(self.m, self.h)


@dataclass
class Diagnostics:
    max_kcl_residual: float = 0.0
    max_charge_error: float = 0.0  # relative to the charge of 1 mV on the cell's membranes
    steps: int = 0


@dataclass
class Frame:
    time_s: float
    data: dict[str, np.ndarray] = field(default_factory=dict)


class Epithelium:
    def __init__(self, config: RunConfig, tissue: Tissue) -> None:
        self.config = config
        self.tissue = tissue
        c = config.constants
        self.const = Constants(
            c.temperature_k,
            c.gas_constant,
            c.faraday,
            c.membrane_thickness_m,
            c.capacitance_f_per_m2,
        )
        self.pump = PumpParameters(**config.pump.model_dump())
        self.ions = [ion.name for ion in config.ions]
        self.z = np.array([ion.z for ion in config.ions], float)
        self.d_free = np.array([ion.d_free for ion in config.ions])
        self.bath = {
            "apical": np.array([ion.bath_apical for ion in config.ions]),
            "basolateral": np.array([ion.bath_basal for ion in config.ions]),
        }
        n = tissue.n_cells
        self.n = n
        self.conc = np.array(
            [
                tissue.initial_conc[ion.name]
                if config.initial_conc_from_bundle
                and tissue.initial_conc
                and ion.name in tissue.initial_conc
                else np.full(n, ion.cell)
                for ion in config.ions
            ],
            dtype=float,
        )
        self.area = {"apical": tissue.area_m2, "basolateral": tissue.area_m2 + tissue.lateral_m2}
        self.base_d = {
            d: np.array([getattr(config, d).base_d.get(i, 0.0) for i in self.ions])
            for d in DOMAINS
        }
        self.alpha = {d: getattr(config, d).pump_alpha for d in DOMAINS}
        self.has_pump = "na" in self.ions and "k" in self.ions
        self.i_na = self.ions.index("na") if self.has_pump else -1
        self.i_k = self.ions.index("k") if self.has_pump else -1
        v0 = config.initial_vmem_mv * 1e-3
        self.channels: dict[str, list[ChannelState]] = {}
        for d in DOMAINS:
            states = []
            for spec in getattr(config, d).channels:
                model = CHANNELS[spec.type]
                weights = np.array([model.ions.get(i, 0.0) for i in self.ions])
                v = np.full(n, v0 * 1e3)
                states.append(
                    ChannelState(
                        model,
                        spec.max_d,
                        weights,
                        model.m_inf(v) * np.ones(n),
                        model.h_inf(v) * np.ones(n),
                    )
                )
            self.channels[d] = states

        gj = config.gap_junctions
        self.gj_gate = GapJunctionGate(gj.threshold_mv, gj.minimum)
        self.gj_open = self.gj_gate.steady(np.zeros(len(tissue.pair_len_m)))

        self.grid = BathGrid.around(tissue, config.baths.spacing_um, config.baths.margin_um)
        g = self.grid.size
        self.g = g
        self.cell_nodes = self.grid.bilinear(tissue.centres_um)
        self.pair_nodes = self.grid.bilinear(tissue.pair_mid_um)
        self.grid_edges = self.grid.edges()
        self.dirichlet = np.concatenate(
            (np.zeros(n, bool), self.grid.boundary(), np.zeros(g, bool))
        )
        rt = self.const.gas_constant * self.const.temperature_k
        self.sigma = {
            d: self.const.faraday**2 / rt * float(np.sum(self.z**2 * self.d_free * self.bath[d]))
            for d in DOMAINS
        }
        self.sheet = {
            "apical": self.sigma["apical"] * config.baths.apical_thickness_um * UM,
            "basolateral": self.sigma["basolateral"] * config.baths.basal_thickness_um * UM,
        }
        junction_per_area = tissue.pair_len_m.sum() / tissue.area_m2.sum()
        r_t = config.tight_junctions.resistance_ohm_cm2 * 1e-4
        self.tj_per_length = 1.0 / (r_t * junction_per_area) if junction_per_area > 0 else 0.0

        self.alive = np.ones(n, bool)
        self.wounded = False
        self.x = np.zeros(n + 2 * g)
        self.x[:n] = v0
        self.time_s = 0.0
        self.diag = Diagnostics()
        logger.info(
            "grid %dx%d (%.1f um), sigma %.3f S/m, tj %.3g S/m per junction length",
            self.grid.nx,
            self.grid.ny,
            self.grid.spacing_um,
            self.sigma["apical"],
            self.tj_per_length,
        )

    # ------------------------------------------------------------------ state views
    def phi_cell(self) -> np.ndarray:
        return self.x[: self.n]

    def phi_bath(self, domain: str) -> np.ndarray:
        start = self.n if domain == "apical" else self.n + self.g
        return self.x[start : start + self.g]

    def bath_at_cells(self, domain: str) -> np.ndarray:
        idx, w = self.cell_nodes
        return (self.phi_bath(domain)[idx] * w).sum(axis=1)

    def vmem(self, domain: str) -> np.ndarray:
        return self.phi_cell() - self.bath_at_cells(domain)

    # ------------------------------------------------------------------ physics
    def diffusion(self, domain: str) -> np.ndarray:
        d = np.repeat(self.base_d[domain][:, None], self.n, axis=1)
        for ch in self.channels[domain]:
            d = d + ch.max_d * ch.weights[:, None] * ch.open_fraction()[None, :]
        return d

    def membrane_fluxes(self, domain: str, v: np.ndarray) -> np.ndarray:
        """Inward flux per ion [n_ions, N] (mol/m²/s), pump included."""
        out = self.bath[domain][:, None] * np.ones((1, self.n))
        flux = electroflux(
            out,
            self.conc,
            self.diffusion(domain),
            self.const.membrane_thickness_m,
            self.z[:, None],
            v[None, :],
            self.const,
        )
        if self.has_pump and self.alpha[domain] > 0:
            f_na, f_k = pump_nak(
                self.conc[self.i_na],
                out[self.i_na],
                self.conc[self.i_k],
                out[self.i_k],
                v,
                self.alpha[domain],
                self.const,
                self.pump,
            )
            flux[self.i_na] += f_na
            flux[self.i_k] += f_k
        return flux

    def pump_rate(self, domain: str, v: np.ndarray) -> np.ndarray:
        if not (self.has_pump and self.alpha[domain] > 0):
            return np.zeros(self.n)
        out = self.bath[domain]
        f_na, _ = pump_nak(
            self.conc[self.i_na],
            out[self.i_na],
            self.conc[self.i_k],
            out[self.i_k],
            v,
            self.alpha[domain],
            self.const,
            self.pump,
        )
        return -f_na / 3.0

    def gj_fluxes(self, dv: np.ndarray) -> np.ndarray:
        """Flux per ion from cell b into cell a for every pair [n_ions, P], per face area."""
        a, b = self.tissue.pair_cells.T
        gj = self.config.gap_junctions
        d = self.d_free[:, None] * gj.surface * self.gj_open[None, :]
        return electroflux(
            self.conc[:, b],
            self.conc[:, a],
            d,
            gj.pore_m,
            self.z[:, None],
            dv[None, :],
            self.const,
        )

    # ------------------------------------------------------------------ assembly
    def _stamp(self, acc: list, a_idx, a_w, b_idx, b_w, conductance, source) -> None:
        idx = np.concatenate((a_idx, b_idx), axis=1)
        coef = np.concatenate((a_w, -b_w), axis=1)
        rows = np.repeat(idx[:, :, None], idx.shape[1], axis=2)
        cols = np.repeat(idx[:, None, :], idx.shape[1], axis=1)
        vals = coef[:, :, None] * conductance[:, None, None] * coef[:, None, :]
        acc[0].append(rows.ravel())
        acc[1].append(cols.ravel())
        acc[2].append(vals.ravel())
        rhs_rows = idx.ravel()
        rhs_vals = (-coef * source[:, None]).ravel()
        acc[3].append((rhs_rows, rhs_vals))

    def step(self, dt: float) -> None:
        n, g, const, tissue = self.n, self.g, self.const, self.tissue
        live = np.flatnonzero(self.alive)
        v_old = {d: self.vmem(d) for d in DOMAINS}

        for d in DOMAINS:  # gates first, from the previous step's voltages (BETSE's order)
            for ch in self.channels[d]:
                ch.m, ch.h = ch.model.step(ch.m, ch.h, v_old[d] * 1e3, dt)
        pa, pb = tissue.pair_cells.T
        phi = self.phi_cell()
        dv_pair = phi[pa] - phi[pb]
        if self.config.gap_junctions.voltage_gated:
            self.gj_open = self.gj_gate.step(self.gj_open, -dv_pair * 1e3, dt)

        acc: list = [[], [], [], []]
        cm = const.capacitance_f_per_m2
        lin = {}
        for d in DOMAINS:
            v = v_old[d]
            f0 = self.membrane_fluxes(d, v)
            slope = (self.membrane_fluxes(d, v + DV) - self.membrane_fluxes(d, v - DV)) / (2 * DV)
            lin[d] = (f0, slope)
            i_out = -const.faraday * (self.z[:, None] * f0).sum(axis=0)
            g_out = -const.faraday * (self.z[:, None] * slope).sum(axis=0)
            area = self.area[d]
            conductance = area * (cm / dt + g_out)
            source = area * (i_out - (cm / dt + g_out) * v)
            offset = n if d == "apical" else n + g
            idx, w = self.cell_nodes
            self._stamp(
                acc,
                live[:, None],
                np.ones((len(live), 1)),
                idx[live] + offset,
                w[live],
                conductance[live],
                source[live],
            )

        pairs_live = np.flatnonzero(self.alive[pa] & self.alive[pb])
        gj_lin = None
        if self.config.gap_junctions.enabled and len(pairs_live):
            j0 = self.gj_fluxes(dv_pair)
            js = (self.gj_fluxes(dv_pair + DV) - self.gj_fluxes(dv_pair - DV)) / (2 * DV)
            gj_lin = (j0, js)
            area = tissue.pair_area_m2
            i_ab = -const.faraday * (self.z[:, None] * j0).sum(axis=0) * area
            g_ab = -const.faraday * (self.z[:, None] * js).sum(axis=0) * area
            k = pairs_live
            one = np.ones((len(k), 1))
            self._stamp(
                acc, pa[k, None], one, pb[k, None], one, g_ab[k], (i_ab - g_ab * dv_pair)[k]
            )

        idx, w = self.pair_nodes
        k = pairs_live
        if self.tj_per_length > 0 and len(k):
            self._stamp(
                acc,
                idx[k] + n,
                w[k],
                idx[k] + n + g,
                w[k],
                self.tj_per_length * tissue.pair_len_m[k],
                np.zeros(len(k)),
            )
        if self.wounded and len(self.shunt_nodes):
            one = np.ones((len(self.shunt_nodes), 1))
            self._stamp(
                acc,
                self.shunt_nodes[:, None] + n,
                one,
                self.shunt_nodes[:, None] + n + g,
                one,
                self.shunt_g,
                np.zeros(len(self.shunt_nodes)),
            )
        if self.wounded and len(self.shunt_fallback):
            cidx, cw = self.cell_nodes
            dead = self.shunt_fallback
            self._stamp(
                acc,
                cidx[dead] + n,
                cw[dead],
                cidx[dead] + n + g,
                cw[dead],
                self.shunt_sigma * tissue.area_m2[dead] / tissue.height_m,
                np.zeros(len(dead)),
            )
        e = self.grid_edges
        one = np.ones((len(e), 1))
        for d, offset in (("apical", n), ("basolateral", n + g)):
            self._stamp(
                acc,
                e[:, :1] + offset,
                one,
                e[:, 1:] + offset,
                one,
                np.full(len(e), self.sheet[d]),
                np.zeros(len(e)),
            )

        size = n + 2 * g
        rows, cols, vals = (np.concatenate(a) for a in acc[:3])
        rhs = np.zeros(size)
        for r, v in acc[3]:
            np.add.at(rhs, r, v)
        fixed = self.dirichlet.copy()
        fixed[:n] = ~self.alive
        keep = ~fixed[rows]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        fixed_ids = np.flatnonzero(fixed)
        rows = np.concatenate((rows, fixed_ids))
        cols = np.concatenate((cols, fixed_ids))
        vals = np.concatenate((vals, np.ones(len(fixed_ids))))
        rhs[fixed] = 0.0
        matrix = coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
        x_new = spsolve(matrix, rhs)
        residual = np.abs(matrix @ x_new - rhs).max() / max(np.abs(rhs).max(), 1e-300)
        self.diag.max_kcl_residual = max(self.diag.max_kcl_residual, float(residual))

        self.x = x_new
        dconc = np.zeros_like(self.conc)
        charge_mem = np.zeros(n)
        for d in DOMAINS:
            f0, slope = lin[d]
            dv = self.vmem(d) - v_old[d]
            dconc += self.area[d][None, :] * (f0 + slope * dv[None, :])
            charge_mem += self.area[d] * cm * dv
        if gj_lin is not None:
            j0, js = gj_lin
            phi = self.phi_cell()
            ddv = (phi[pa] - phi[pb]) - dv_pair
            flux = (j0 + js * ddv[None, :]) * tissue.pair_area_m2[None, :]
            flux[:, ~(self.alive[pa] & self.alive[pb])] = 0.0
            np.add.at(dconc.T, pa, flux.T)
            np.add.at(dconc.T, pb, -flux.T)
        dconc *= dt / tissue.volume_m3[None, :]
        dconc[:, ~self.alive] = 0.0
        self.conc += dconc
        charge_ions = const.faraday * tissue.volume_m3 * (self.z[:, None] * dconc).sum(axis=0)
        scale = cm * (self.area["apical"] + self.area["basolateral"]) * 1e-3
        err = np.abs(charge_ions - charge_mem)[self.alive] / scale[self.alive]
        if err.size:
            self.diag.max_charge_error = max(self.diag.max_charge_error, float(err.max()))
        self.time_s += dt
        self.diag.steps += 1

    # ------------------------------------------------------------------ phases
    def apply_wound(self) -> None:
        """Remove the wounded cells and short the baths through their footprint area.

        Each removed cell's vertical conductance σ·A/h is spread evenly over the bath nodes
        inside its polygon (a point shunt at the centre makes the near-wound field grow without
        bound as the grid is refined, which check E8 caught). A cell smaller than the grid
        spacing, with no node inside, keeps the centre shunt.
        """
        self.alive = ~self.tissue.wounded
        self.x[: self.n][~self.alive] = 0.0
        self.wounded = True
        t = self.tissue
        self.shunt_sigma = 0.5 * (self.sigma["apical"] + self.sigma["basolateral"])
        gx, gy = self.grid.coordinates()
        points = np.column_stack((gx.ravel(), gy.ravel()))
        nodes, conductance, fallback = [], [], []
        for cell in np.flatnonzero(t.wounded):
            poly = t.verts_um[t.offsets[cell] : t.offsets[cell + 1]]
            inside = np.flatnonzero(point_in_polygon(points, poly))
            total = self.shunt_sigma * t.area_m2[cell] / t.height_m
            if len(inside):
                nodes.append(inside)
                conductance.append(np.full(len(inside), total / len(inside)))
            else:
                fallback.append(cell)
        self.shunt_nodes = np.concatenate(nodes) if nodes else np.zeros(0, int)
        self.shunt_g = np.concatenate(conductance) if conductance else np.zeros(0)
        self.shunt_fallback = np.array(fallback, dtype=int)

    def snapshot(self) -> Frame:
        frame = Frame(self.time_s)
        v_ap, v_bl = self.vmem("apical"), self.vmem("basolateral")
        frame.data["phi_cell"] = self.phi_cell().copy()
        frame.data["vmem_apical"] = v_ap
        frame.data["vmem_basolateral"] = v_bl
        frame.data["tep"] = self.bath_at_cells("apical") - self.bath_at_cells("basolateral")
        for d, v in (("apical", v_ap), ("basolateral", v_bl)):
            f = self.membrane_fluxes(d, v)
            frame.data[f"current_{d}"] = -self.const.faraday * (self.z[:, None] * f).sum(axis=0)
            frame.data[f"pump_{d}"] = self.pump_rate(d, v)
            for ch in self.channels[d]:
                frame.data[f"open_{ch.model.name.lower()}_{d}"] = ch.open_fraction().copy()
        frame.data["conc_cell"] = self.conc.copy()
        for d in DOMAINS:
            frame.data[f"phi_bath_{d}"] = (
                self.phi_bath(d).reshape(self.grid.ny, self.grid.nx).copy()
            )
        frame.data["gj_open"] = self.gj_open.copy()
        return frame

    def run(self, on_frame=None) -> tuple[list[Frame], list[float]]:
        """Init (intact), baseline frame, wound, sim. Returns frames and their times."""
        t = self.config.time
        for _ in range(int(round(t.init_s / t.init_dt_s))):
            self.step(t.init_dt_s)
        self.time_s = 0.0
        frames = [self.snapshot()]
        if self.tissue.wounded.any():
            self.apply_wound()
        steps = int(round(t.sim_s / t.dt_s))
        every = max(int(round(t.sample_s / t.dt_s)), 1)
        for s in range(1, steps + 1):
            self.step(t.dt_s)
            if s % every == 0:
                frames.append(self.snapshot())
                if on_frame is not None:
                    on_frame(len(frames), steps // every + 1)
        return frames, [f.time_s for f in frames]
