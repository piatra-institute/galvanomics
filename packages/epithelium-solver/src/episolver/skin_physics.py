"""Bioelectric physics on the procedural skin: living keratinocytes in a 3D conductive space.

Every living cell is one compartment (intracellular potential and ion concentrations). Every one
of its faces is a patch of membrane facing the extracellular space (ECS) at that face, with the
channel set of its layer and side: in the granular layer, faces above the tight-junction line
are apical (Na entry, ENaC-like), the rest basolateral (K leak, Na/K-ATPase); basal and spinous
cells are basolateral all round. Shared faces between living cells also carry gap junctions.
Corneocytes are dead and electrically inert. The ECS is a 3D grid whose conductivity depends on
the region: dermis, intercellular fluid of the viable epidermis, the tight-junction barrier band,
the stratum corneum, and, after the cut, the wound fluid. The bottom of the dermis is the
electrical reference.

The fluxes, pump, channels and gap junctions are the same BETSE ports as in ``solver.py``, and the
time step is the same: one sparse Kirchhoff solve with backward-Euler membrane capacitance and
ionic currents linearized about the previous step, then a charge-consistent concentration update.
All parameter values are provisional until sourced (``docs/skin.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyamg
from pydantic import BaseModel, ConfigDict, Field
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve

from episolver.channels import CHANNELS, GapJunctionGate
from episolver.flux import Constants, PumpParameters, electroflux, nkcc_flux, pump_nak
from episolver.params import (
    BETSE_BASIC_IONS,
    BETSE_MAMMAL_D,
    BETSE_MAMMAL_NO_CA_IONS,
    ChannelSpec,
    Domain,
    GapJunctions,
    Ion,
    PhysicalConstants,
    Pump,
    TimeSpec,
)
from episolver.skin import SIDE, Skin, SkinSpec, facing_faces, in_wound, papilla_height

logger = logging.getLogger(__name__)

UM = 1e-6
DV = 1.0e-6
APICAL, BASOLATERAL = 0, 1
REGIONS = ("dermis", "epidermis", "tight junction", "corneum", "wound")
DERMIS, EPIDERMIS, TIGHT, CORNEUM, WOUND = range(5)


def _keratinocyte(apical_na: float = 0.0, apical_k: float = 5.0e-17) -> dict[str, Domain]:
    """K leak and Na/K-ATPase on every face; with ``apical_na``, the faces facing the space
    beneath the stratum corneum (above the tight junctions) instead carry Na channels
    (ENaC-like) and K channels and no pump: a transporting epithelium. Densities are
    assumptions; ``apical_na`` and the barrier are calibrated to the reported battery."""
    basolateral = Domain(pump_alpha=1.0e-7, channels=(ChannelSpec(type="KLeak", max_d=5.0e-17),))
    if apical_na <= 0:
        return {"apical": basolateral, "basolateral": basolateral}
    apical = Domain(
        pump_alpha=0.0,
        channels=(
            ChannelSpec(type="NaLeak", max_d=apical_na),
            ChannelSpec(type="KLeak", max_d=apical_k),
        ),
    )
    return {"apical": apical, "basolateral": basolateral}


class Modifier(BaseModel):
    """A drug or other intervention: multiply one transport pathway by ``factor`` (0 blocks it,
    2 doubles it) on the faces of one side, everywhere or within ``within_um`` of the wound
    centre (in x, by cell centroid)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(
        description="a channel type (NaLeak, ClLeak, KLeak, ...), nkcc, base_cl or base_na"
    )
    side: str = Field("any", description="apical, basolateral or any")
    factor: float = Field(ge=0.0)
    within_um: float | None = None
    label: str = ""


class SkinPhysics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spacing_um: float = Field(
        4.0, description="ECS grid spacing in y, in x near the wound, and (by default) coarse z"
    )
    coarse_z_um: float | None = Field(
        None, description="ECS z spacing in the dermis and lower epidermis; default spacing_um"
    )
    x_fine_halfwidth_um: float | None = Field(
        None,
        description="If set, the x spacing is spacing_um only within this distance of the "
        "wound centre and grows geometrically beyond it (for wide blocks); otherwise uniform",
    )
    x_growth: float = Field(1.15, gt=1.0, description="Ratio of successive x spacings, graded")
    x_max_spacing_um: float = Field(20.0, description="Largest x spacing, graded")
    fine_z_um: float = Field(0.75, description="ECS z spacing through the granular layer")
    sigma_fluid: float = Field(
        1.79, description="Wound / interstitial fluid [S/m]; CSF at 37 C as a proxy (Baumann 1997)"
    )
    epidermal_ecs_factor: float = Field(
        0.0145,
        description="Viable-epidermis ECS conductivity over the fluid's: 0.026 S/m, the "
        "modelling convention of Sun 2017 (not a measurement)",
    )
    sigma_dermis: float = Field(0.222, description="[S/m], modelling convention (Sun 2017)")
    sigma_corneum: float = Field(2.0e-6, description="[S/m], modelling convention (Sun 2017)")
    tight_junction_ohm_cm2: float = Field(
        2000.0,
        description="Barrier band resistance. Calibrated (with the apical Na channels) "
        "so that the intact battery lies inside the 10-60 mV reported for human skin; not "
        "measured",
    )
    living_layers: tuple[int, ...] = (0, 1, 2)
    granular_layer: int = 2
    tight_junction_row: int = Field(1, description="Row of the granular layer whose top is sealed")
    membranes: dict[int, dict[str, Domain]] = Field(
        default_factory=lambda: {0: _keratinocyte(), 1: _keratinocyte(), 2: _keratinocyte(3e-16)}
    )
    apical_na_override: float | None = Field(
        None, description="If set, the apical Na channel density of the granular layer [m2/s]"
    )
    na_block_fraction: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of apical Na channels blocked (a drug such as amiloride)",
    )
    na_block_within_um: float | None = Field(
        None,
        description="Block only cells whose centroid lies within this distance of the wound "
        "centre (x); everywhere if unset",
    )
    modifiers: tuple[Modifier, ...] = Field(
        (), description="Interventions applied to transport pathways (see Modifier)"
    )
    cell_conc_override: dict[str, float] = Field(
        default_factory=dict,
        description="Initial cytoplasmic concentrations [mol/m3] replacing the profile's",
    )
    wound_cl_d: float = Field(
        0.0,
        description="Ca-activated Cl channels (ANO1-like, BETSE's ClLeak) switched on at "
        "wounding on every living membrane face that touches the wound fluid [m2/s]",
    )
    ions: tuple[Ion, ...] = BETSE_BASIC_IONS
    ion_profile: str = Field(
        "basic",
        description="basic (BETSE's: Na, K, anion, protein) or mammal_no_ca (BETSE's mammal "
        "profile without Ca: adds Cl-); sets the ions and default membrane permeabilities",
    )
    gap_junctions: GapJunctions = GapJunctions()
    time: TimeSpec = TimeSpec(init_dt_s=0.02, init_s=2.0, dt_s=0.01, sim_s=2.0, sample_s=0.05)
    constants: PhysicalConstants = PhysicalConstants()
    pump: Pump = Pump()
    initial_vmem_mv: float = -40.0


class SkinRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    skin: SkinSpec = SkinSpec()
    physics: SkinPhysics = SkinPhysics()


def stamp(acc: list, a_idx, a_w, b_idx, b_w, conductance, source) -> None:
    """Add branches (current G * (a - b) + s from weighted node set a to b) to a KCL system."""
    idx = np.concatenate((a_idx, b_idx), axis=1)
    coef = np.concatenate((a_w, -b_w), axis=1)
    k = idx.shape[1]
    acc[0].append(np.repeat(idx[:, :, None], k, axis=2).ravel())
    acc[1].append(np.repeat(idx[:, None, :], k, axis=1).ravel())
    acc[2].append((coef[:, :, None] * conductance[:, None, None] * coef[:, None, :]).ravel())
    acc[3].append((idx.ravel(), (-coef * source[:, None]).ravel()))


def x_axis(spec: SkinSpec, ph: SkinPhysics) -> np.ndarray:
    """ECS node positions in x (µm): uniform, or graded outward from the wound centre."""
    h, width = ph.spacing_um, spec.width_x_um
    if ph.x_fine_halfwidth_um is None:
        return np.linspace(0.0, width, int(round(width / h)) + 1)
    centre = min(max(spec.wound.x_um, 0.0), width)

    def side(length: float) -> list[float]:
        out, x, step = [], 0.0, h
        while length - x > 1e-9:
            if x >= ph.x_fine_halfwidth_um - 1e-9:
                step = min(step * ph.x_growth, ph.x_max_spacing_um)
            if length - (x + step) < 0.5 * step:  # absorb a short last interval
                x = length
            else:
                x += step
            out.append(x)
        return out

    right = [centre + d for d in side(width - centre)]
    left = [centre - d for d in side(centre)][::-1]
    return np.array(left + [centre] + right)


@dataclass
class EcsGrid:
    """A tensor-product grid (µm): uniform in x and y, non-uniform in z so that thin layers
    (the space under the stratum corneum, the tight-junction band) are resolved."""

    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.zs), len(self.ys), len(self.xs)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.zs, self.ys, self.xs, indexing="ij")

    def trilinear(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nz, ny, nx = self.shape
        cells, fracs = [], []
        for axis, nodes in ((0, self.xs), (1, self.ys), (2, self.zs)):
            i = np.clip(np.searchsorted(nodes, points[:, axis]) - 1, 0, len(nodes) - 2)
            t = (points[:, axis] - nodes[i]) / (nodes[i + 1] - nodes[i])
            cells.append(i)
            fracs.append(np.clip(t, 0.0, 1.0))
        (ix, iy, iz), (tx, ty, tz) = cells, fracs
        idx, w = [], []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    idx.append(((iz + dz) * ny + (iy + dy)) * nx + (ix + dx))
                    w.append(
                        (tx if dx else 1 - tx) * (ty if dy else 1 - ty) * (tz if dz else 1 - tz)
                    )
        return np.stack(idx, axis=1), np.stack(w, axis=1)

    @staticmethod
    def _dual(nodes: np.ndarray) -> np.ndarray:
        mids = 0.5 * (nodes[1:] + nodes[:-1])
        return np.diff(np.concatenate(([nodes[0]], mids, [nodes[-1]])))

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Adjacent node pairs and their geometric factor area/length (m)."""
        nz, ny, nx = self.shape
        ids = np.arange(self.size).reshape(nz, ny, nx)
        wx, wy, wz = self._dual(self.xs), self._dual(self.ys), self._dual(self.zs)
        dx, dy, dz = np.diff(self.xs), np.diff(self.ys), np.diff(self.zs)
        Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        pairs, factor = [], []
        ex = (slice(None), slice(None), slice(None, -1))
        pairs.append(np.column_stack((ids[ex].ravel(), ids[:, :, 1:].ravel())))
        factor.append((wy[Y[ex]] * wz[Z[ex]] / dx[X[ex]]).ravel())
        ey = (slice(None), slice(None, -1), slice(None))
        pairs.append(np.column_stack((ids[ey].ravel(), ids[:, 1:, :].ravel())))
        factor.append((wx[X[ey]] * wz[Z[ey]] / dy[Y[ey]]).ravel())
        ez = (slice(None, -1), slice(None), slice(None))
        pairs.append(np.column_stack((ids[ez].ravel(), ids[1:, :, :].ravel())))
        factor.append((wx[X[ez]] * wy[Y[ez]] / dz[Z[ez]]).ravel())
        return np.concatenate(pairs), np.concatenate(factor) * UM


@dataclass
class Diagnostics:
    max_kcl_residual: float = 0.0
    max_charge_error: float = 0.0
    steps: int = 0
    max_cg_iterations: int = 0
    cg_iterations: int = 0
    solver_retries: int = 0


@dataclass
class Frame:
    time_s: float
    data: dict[str, np.ndarray] = field(default_factory=dict)


class SkinModel:
    def __init__(self, config: SkinRunConfig, skin: Skin) -> None:
        self.config, self.skin = config, skin
        ph = config.physics
        c = ph.constants
        self.const = Constants(
            c.temperature_k,
            c.gas_constant,
            c.faraday,
            c.membrane_thickness_m,
            c.capacitance_f_per_m2,
        )
        self.pump = PumpParameters(**ph.pump.model_dump())
        ions = BETSE_MAMMAL_NO_CA_IONS if ph.ion_profile == "mammal_no_ca" else ph.ions
        self.default_d = BETSE_MAMMAL_D if ph.ion_profile == "mammal_no_ca" else {}
        self.ions = [i.name for i in ions]
        self.z = np.array([i.z for i in ions], float)
        self.d_free = np.array([i.d_free for i in ions])
        self.c_out = np.array([i.bath_basal for i in ions])
        self.i_cl = self.ions.index("cl") if "cl" in self.ions else None
        self.i_na, self.i_k = self.ions.index("na"), self.ions.index("k")
        d = skin.derived
        n = skin.n_cells
        self.n = n
        self.living = np.isin(skin.cell_layer, ph.living_layers)
        self.alive = self.living.copy()
        self.volume = d["volume_um3"] * UM**3
        self.conc = np.array([np.full(n, i.cell) for i in ions])
        for name, value in ph.cell_conc_override.items():
            self.conc[self.ions.index(name)] = value

        # The tight-junction line: top of the sealed granular row.
        spec = config.skin
        bounds = np.concatenate(([0.0], np.cumsum([lyr.thickness_um for lyr in spec.layers])))
        g = ph.granular_layer
        row_h = spec.layers[g].thickness_um / spec.layers[g].rows
        self.tj_z = bounds[g] + (ph.tight_junction_row + 1) * row_h
        self.granular_top = bounds[g + 1]
        self.top = bounds[-1]

        # Membrane faces: every face of a living cell except those on the block's side walls. The
        # walls are mirror planes, so a cell clipped by one is half of a doubled cell and its wall
        # face lies inside that cell.
        face_cell = d["face_cell"]
        faces = np.flatnonzero(self.living[face_cell] & (skin.face_boundary != SIDE))
        self.f_face = faces
        self.f_cell = face_cell[faces]
        self.f_area = d["face_area_um2"][faces] * UM**2
        self.f_centre = d["face_centre_um"][faces]
        layer = skin.cell_layer[self.f_cell]
        self.f_domain = np.where(
            (layer == g) & (self.f_centre[:, 2] >= self.tj_z - 1e-6), APICAL, BASOLATERAL
        )

        # Per (layer, domain) membrane properties, broadcast to faces.
        self.f_base_d = np.zeros((len(self.ions), len(faces)))
        self.f_alpha = np.zeros(len(faces))
        self.f_nkcc = np.zeros(len(faces))
        self.channels = []  # (model, max_d, weights, face mask, m, h)
        membranes = dict(ph.membranes)
        if ph.apical_na_override is not None:
            membranes[ph.granular_layer] = _keratinocyte(ph.apical_na_override)
        self.membranes = membranes
        for lyr, sides in membranes.items():
            for name, dom_id in (("apical", APICAL), ("basolateral", BASOLATERAL)):
                mask = (layer == lyr) & (self.f_domain == dom_id)
                if not mask.any():
                    continue
                dom = sides[name]
                base_d = {**self.default_d, **dom.base_d}
                self.f_base_d[:, mask] = np.array([base_d.get(i, 0.0) for i in self.ions])[:, None]
                self.f_alpha[mask] = dom.pump_alpha
                self.f_nkcc[mask] = dom.nkcc_alpha
                for spec_c in dom.channels:
                    model = CHANNELS[spec_c.type]
                    weights = np.array([model.ions.get(i, 0.0) for i in self.ions])
                    v = np.full(int(mask.sum()), ph.initial_vmem_mv)
                    self.channels.append(
                        [
                            model,
                            spec_c.max_d,
                            weights,
                            mask,
                            model.m_inf(v) * np.ones_like(v),
                            model.h_inf(v) * np.ones_like(v),
                            lyr,
                            name,
                        ]
                    )

        if ph.wound_cl_d > 0:  # inactive until the wound exposes faces (apply_wound)
            model = CHANNELS["ClLeak"]
            every = np.ones(len(faces), bool)
            v0 = np.full(len(faces), ph.initial_vmem_mv)
            weights = np.array([model.ions.get(i, 0.0) for i in self.ions])
            self.channels.append(
                [
                    model,
                    ph.wound_cl_d,
                    weights,
                    every,
                    model.m_inf(v0),
                    model.h_inf(v0),
                    -1,
                    "wound",
                ]
            )
        self.channel_scale = self._channel_scales()
        self._apply_transporter_modifiers()

        # Gap junctions on faces shared by two living cells.
        facing = facing_faces(skin)
        a = np.flatnonzero(
            (facing > np.arange(len(facing)))
            & self.living[face_cell]
            & self.living[face_cell[np.maximum(facing, 0)]]
        )
        self.gj_faces = a
        self.gj_cells = np.column_stack((face_cell[a], face_cell[facing[a]]))
        self.gj_area = d["face_area_um2"][a] * UM**2
        gj = ph.gap_junctions
        self.gj_gate = GapJunctionGate(gj.threshold_mv, gj.minimum)
        self.gj_open = self.gj_gate.steady(np.zeros(len(a)))

        # Extracellular grid: 4 µm laterally; in z, coarse in dermis and viable epidermis, fine
        # through the granular layer and the space beneath the stratum corneum.
        h = ph.spacing_um
        ny = int(round(spec.width_y_um / h)) + 1
        hz = ph.coarse_z_um or h
        # The stratum corneum (about 2e-6 S/m) is the insulating top boundary: for steady currents
        # nothing crosses it, and keeping it in the grid only adds a conductivity contrast of six
        # orders of magnitude that slows the solver. The grid ends just above the granular layer,
        # in the space beneath the stratum corneum.
        fine_lo, fine_hi = self.tj_z - 6.0, self.granular_top + 1.0
        coarse = np.linspace(
            -spec.dermis_depth_um,
            fine_lo,
            max(2, int(np.ceil((fine_lo + spec.dermis_depth_um) / hz))) + 1,
        )
        fine = np.linspace(fine_lo, fine_hi, int(np.ceil((fine_hi - fine_lo) / ph.fine_z_um)) + 1)
        zs = np.concatenate((coarse[:-1], fine))
        self.grid = EcsGrid(x_axis(spec, ph), np.linspace(0, spec.width_y_um, ny), zs)
        self.tj_node_z = zs[np.argmin(np.abs(zs - self.tj_z))]
        self.band_dz = EcsGrid._dual(zs)[np.argmin(np.abs(zs - self.tj_z))]
        self.region = self._regions(wounded=False)
        self.f_nodes = self.grid.trilinear(self.f_centre)
        self.edges, self.edge_factor = self.grid.edges()
        self.ground = np.zeros(self.grid.size, bool)
        self.ground[: self.grid.shape[1] * self.grid.shape[2]] = True  # bottom of the dermis
        self.wounded = False
        self._amg = None
        self._amg_age = 0

        # Unknowns: all cells (dead/inert ones pinned at 0), then ECS nodes.
        self.x = np.zeros(n + self.grid.size)
        self.x[:n][self.living] = ph.initial_vmem_mv * 1e-3
        self.time_s = 0.0
        self.diag = Diagnostics()
        logger.info(
            "skin physics: %d living cells, %d membrane faces (%d apical), %d gap-junction "
            "faces, ECS grid %s (%.2f um)",
            int(self.living.sum()),
            len(faces),
            int((self.f_domain == APICAL).sum()),
            len(a),
            self.grid.shape,
            ph.spacing_um,
        )

    # ------------------------------------------------------------------ geometry helpers
    def _regions(self, wounded: bool) -> np.ndarray:
        z, y, x = self.grid.coords()
        ph = self.config.physics
        region = np.full(z.shape, EPIDERMIS)
        region[z < papilla_height(self.config.skin, x, y)] = DERMIS
        region[(z == self.tj_node_z) & (region == EPIDERMIS)] = TIGHT
        region[z > self.granular_top + 1e-9] = CORNEUM
        if wounded:
            pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            region.ravel()[in_wound(self.config.skin, pts)] = WOUND
        del ph
        return region.ravel()

    def sigma(self) -> np.ndarray:
        ph = self.config.physics
        # The band is one node layer of dual thickness band_dz: its two half-links in series
        # give the specified area resistance.
        values = np.array(
            [
                ph.sigma_dermis,
                ph.sigma_fluid * ph.epidermal_ecs_factor,
                self.band_dz * UM / (ph.tight_junction_ohm_cm2 * 1e-4),
                ph.sigma_corneum,
                ph.sigma_fluid,
            ]
        )
        return values[self.region]

    def phi_ecs(self) -> np.ndarray:
        return self.x[self.n :]

    def face_vmem(self) -> np.ndarray:
        idx, w = self.f_nodes
        return self.x[: self.n][self.f_cell] - (self.phi_ecs()[idx] * w).sum(axis=1)

    # ------------------------------------------------------------------ physics
    def face_diffusion(self) -> np.ndarray:
        d = self.f_base_d.copy()
        for k, (model, max_d, weights, mask, m, hh, *_) in enumerate(self.channels):
            open_ = model.open_fraction(m, hh) * self.channel_scale[k]
            d[:, mask] += max_d * weights[:, None] * open_[None, :]
        return d

    def _zone(self, faces_mask: np.ndarray, within_um: float | None) -> np.ndarray:
        cells = self.f_cell[faces_mask]
        if within_um is None:
            return np.ones(len(cells), bool)
        x = self.skin.derived["centroid"][cells, 0]
        return np.abs(x - self.config.skin.wound.x_um) < within_um

    def _modifier_factor(self, target: str, faces_mask: np.ndarray, side: str) -> np.ndarray:
        """Product of every modifier's factor that applies to these faces (1 if none)."""
        factor = np.ones(int(faces_mask.sum()))
        for mod in self.config.physics.modifiers:
            if mod.target != target or mod.side not in ("any", side):
                continue
            factor[self._zone(faces_mask, mod.within_um)] *= mod.factor
        return factor

    def _channel_scales(self) -> list:
        """Per channel, the factor on each of its faces: 1 unless a block or modifier is set."""
        ph = self.config.physics
        scales = []
        for model, _, _, mask, *_rest, side in self.channels:
            scale = self._modifier_factor(model.name, mask, side)
            if side == "wound":
                scale *= self._wound_exposed(mask) if getattr(self, "wounded", False) else 0.0
            if model.name == "NaLeak" and side == "apical" and ph.na_block_fraction > 0:
                scale[self._zone(mask, ph.na_block_within_um)] *= 1.0 - ph.na_block_fraction
            scales.append(scale)
        return scales

    def _wound_exposed(self, faces_mask: np.ndarray) -> np.ndarray:
        """Living membrane faces whose extracellular nodes include wound fluid."""
        idx, w = self.f_nodes
        touching = np.any((self.region[idx] == WOUND) & (w > 1e-12), axis=1)
        live = self.alive[self.f_cell]
        return (touching & live)[faces_mask].astype(float)

    def _apply_transporter_modifiers(self) -> None:
        """NKCC and base Cl permeability modifiers act on the per-face arrays directly."""
        for side_name, dom_id in (("apical", APICAL), ("basolateral", BASOLATERAL)):
            mask = self.f_domain == dom_id
            if not mask.any():
                continue
            self.f_nkcc[mask] *= self._modifier_factor("nkcc", mask, side_name)
            if self.i_cl is not None:
                self.f_base_d[self.i_cl, mask] *= self._modifier_factor("base_cl", mask, side_name)
            self.f_base_d[self.i_na, mask] *= self._modifier_factor("base_na", mask, side_name)

    def face_fluxes(self, v: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
        cin = self.conc[:, self.f_cell]
        cout = np.repeat(self.c_out[:, None], len(v), axis=1)
        flux = electroflux(
            cout,
            cin,
            diffusion,
            self.const.membrane_thickness_m,
            self.z[:, None],
            v[None, :],
            self.const,
        )
        pumped = self.f_alpha > 0
        if pumped.any():
            f_na, f_k = pump_nak(
                cin[self.i_na, pumped],
                cout[self.i_na, pumped],
                cin[self.i_k, pumped],
                cout[self.i_k, pumped],
                v[pumped],
                self.f_alpha[pumped],
                self.const,
                self.pump,
            )
            flux[self.i_na, pumped] += f_na
            flux[self.i_k, pumped] += f_k
        loaded = (self.f_nkcc > 0) if self.i_cl is not None else np.zeros(0, bool)
        if loaded.any():
            j = nkcc_flux(
                cin[self.i_na, loaded],
                cin[self.i_k, loaded],
                cin[self.i_cl, loaded],
                cout[self.i_na, loaded],
                cout[self.i_k, loaded],
                cout[self.i_cl, loaded],
                self.f_nkcc[loaded],
            )
            flux[self.i_na, loaded] += j
            flux[self.i_k, loaded] += j
            flux[self.i_cl, loaded] += 2.0 * j
        return flux

    def gj_fluxes(self, dv: np.ndarray) -> np.ndarray:
        a, b = self.gj_cells.T
        gj = self.config.physics.gap_junctions
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

    # ------------------------------------------------------------------ stepping
    def step(self, dt: float) -> None:
        n, const = self.n, self.const
        cm = const.capacitance_f_per_m2
        live_face = self.alive[self.f_cell]
        v_old = self.face_vmem()
        for ch in self.channels:
            model, _, _, mask, m, hh = ch[:6]
            cells = self.f_cell[mask]
            v_cell = np.bincount(cells, weights=v_old[mask] * self.f_area[mask], minlength=n)
            area = np.bincount(cells, weights=self.f_area[mask], minlength=n)
            v_mean = (v_cell / np.maximum(area, 1e-30))[cells] * 1e3
            ch[4], ch[5] = model.step(m, hh, v_mean, dt)
        phi = self.x[:n]
        pa, pb = self.gj_cells.T
        gj_live = self.alive[pa] & self.alive[pb]
        dv_gj = phi[pa] - phi[pb]
        if self.config.physics.gap_junctions.voltage_gated:
            self.gj_open = self.gj_gate.step(self.gj_open, -dv_gj * 1e3, dt)

        acc: list = [[], [], [], []]
        diffusion = self.face_diffusion()
        f0 = self.face_fluxes(v_old, diffusion)
        slope = (
            self.face_fluxes(v_old + DV, diffusion) - self.face_fluxes(v_old - DV, diffusion)
        ) / (2 * DV)
        i_out = -const.faraday * (self.z[:, None] * f0).sum(axis=0)
        g_out = -const.faraday * (self.z[:, None] * slope).sum(axis=0)
        k = np.flatnonzero(live_face)
        conductance = self.f_area * (cm / dt + g_out)
        source = self.f_area * (i_out - (cm / dt + g_out) * v_old)
        idx, w = self.f_nodes
        stamp(
            acc,
            self.f_cell[k, None],
            np.ones((len(k), 1)),
            idx[k] + n,
            w[k],
            conductance[k],
            source[k],
        )

        gj_lin = None
        kk = np.flatnonzero(gj_live)
        if self.config.physics.gap_junctions.enabled and len(kk):
            j0 = self.gj_fluxes(dv_gj)
            js = (self.gj_fluxes(dv_gj + DV) - self.gj_fluxes(dv_gj - DV)) / (2 * DV)
            gj_lin = (j0, js)
            i_ab = -const.faraday * (self.z[:, None] * j0).sum(axis=0) * self.gj_area
            g_ab = -const.faraday * (self.z[:, None] * js).sum(axis=0) * self.gj_area
            one = np.ones((len(kk), 1))
            stamp(acc, pa[kk, None], one, pb[kk, None], one, g_ab[kk], (i_ab - g_ab * dv_gj)[kk])

        sig = self.sigma()
        e = self.edges
        g_edge = self.edge_factor * 2 * sig[e[:, 0]] * sig[e[:, 1]] / (sig[e[:, 0]] + sig[e[:, 1]])
        one = np.ones((len(e), 1))
        stamp(acc, e[:, :1] + n, one, e[:, 1:] + n, one, g_edge, np.zeros(len(e)))

        size = n + self.grid.size
        rows, cols, vals = (np.concatenate(a) for a in acc[:3])
        rhs = np.zeros(size)
        for r, v in acc[3]:
            rhs += np.bincount(r, weights=v, minlength=size)
        fixed = np.concatenate((~self.alive, self.ground))
        keep = ~fixed[rows] & ~fixed[cols]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        fixed_ids = np.flatnonzero(fixed)
        rows = np.concatenate((rows, fixed_ids))
        cols = np.concatenate((cols, fixed_ids))
        vals = np.concatenate((vals, np.ones(len(fixed_ids))))
        rhs[fixed] = 0.0
        matrix = coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
        x_new = self._solve(matrix, rhs)
        residual = np.abs(matrix @ x_new - rhs).max() / max(np.abs(rhs).max(), 1e-300)
        self.diag.max_kcl_residual = max(self.diag.max_kcl_residual, float(residual))
        self.x = x_new

        dv = self.face_vmem() - v_old
        per_face = (f0 + slope * dv[None, :]) * self.f_area[None, :]
        per_face[:, ~live_face] = 0.0
        dconc = np.stack([np.bincount(self.f_cell, weights=pf, minlength=n) for pf in per_face])
        charge_mem = np.bincount(
            self.f_cell, weights=np.where(live_face, self.f_area * cm * dv, 0), minlength=n
        )
        if gj_lin is not None:
            j0, js = gj_lin
            phi = self.x[:n]
            flux = (j0 + js * ((phi[pa] - phi[pb]) - dv_gj)[None, :]) * self.gj_area[None, :]
            flux[:, ~gj_live] = 0.0
            for k in range(len(flux)):
                dconc[k] += np.bincount(pa, weights=flux[k], minlength=n)
                dconc[k] -= np.bincount(pb, weights=flux[k], minlength=n)
        dconc *= dt / self.volume[None, :]
        dconc[:, ~self.alive] = 0.0
        self.conc += dconc
        charge_ions = const.faraday * self.volume * (self.z[:, None] * dconc).sum(axis=0)
        area_cell = np.bincount(self.f_cell, weights=self.f_area, minlength=n)
        scale = cm * np.maximum(area_cell, 1e-30) * 1e-3
        err = np.abs(charge_ions - charge_mem)[self.alive] / scale[self.alive]
        if err.size:
            self.diag.max_charge_error = max(self.diag.max_charge_error, float(err.max()))
        self.time_s += dt
        self.diag.steps += 1
        self._last = (f0, slope, dv, v_old)

    def _solve(self, matrix, rhs: np.ndarray) -> np.ndarray:
        """Conjugate gradients preconditioned by classical (Ruge-Stuben) algebraic multigrid; the
        hierarchy is reused while it converges fast, since the matrix changes slowly between
        steps. Classical AMG handles the strongly anisotropic links of a graded grid (wide
        block: 19 iterations, against about 100 for smoothed aggregation)."""
        if self._amg is None or self._amg_age > 200:
            self._amg = pyamg.ruge_stuben_solver(matrix, max_coarse=500)
            self._amg_age = 0
        self._amg_age += 1
        iterations = []
        x, info = cg(
            matrix,
            rhs,
            x0=self.x,
            rtol=1e-12,
            atol=0.0,
            maxiter=400,
            M=self._amg.aspreconditioner(),
            callback=lambda _: iterations.append(1),
        )
        self.diag.max_cg_iterations = max(self.diag.max_cg_iterations, len(iterations))
        self.diag.cg_iterations += len(iterations)
        if info != 0 or len(iterations) > 40:
            self._amg = None
        if info != 0:
            # A fresh hierarchy and a longer budget first; a direct solve only for small systems,
            # since its fill-in on a large 3D grid needs gigabytes.
            self._amg = pyamg.ruge_stuben_solver(matrix, max_coarse=500)
            self._amg_age = 0
            x, info = cg(
                matrix,
                rhs,
                x0=self.x,
                rtol=1e-12,
                atol=0.0,
                maxiter=4000,
                M=self._amg.aspreconditioner(),
            )
            self.diag.solver_retries += 1
            if info != 0:
                if matrix.shape[0] > 60_000:
                    raise RuntimeError(f"conjugate gradients did not converge (info {info})")
                x = spsolve(matrix, rhs)
        return x

    def apply_wound(self) -> None:
        self.alive = self.living & ~self.skin.wounded
        self.x[: self.n][~self.alive] = 0.0
        self.region = self._regions(wounded=True)
        self.wounded = True
        self._amg = None
        if any(ch[7] == "wound" for ch in self.channels):
            self.channel_scale = self._channel_scales()  # wound Cl channels switch on

    # ------------------------------------------------------------------ recording
    def snapshot(self) -> Frame:
        n = self.n
        frame = Frame(self.time_s)
        v = self.face_vmem()
        live_face = self.alive[self.f_cell]
        area = np.bincount(self.f_cell, weights=np.where(live_face, self.f_area, 0), minlength=n)
        vmem = np.bincount(
            self.f_cell, weights=np.where(live_face, v * self.f_area, 0), minlength=n
        ) / np.maximum(area, 1e-30)
        frame.data["vmem"] = np.where(self.alive, vmem, np.nan)
        frame.data["phi_cell"] = np.where(self.alive, self.x[:n], np.nan)
        frame.data["conc_cell"] = self.conc.copy()
        diffusion = self.face_diffusion()
        total = self.face_fluxes(v, diffusion)
        alpha, self.f_alpha = self.f_alpha, np.zeros_like(self.f_alpha)
        nkcc, self.f_nkcc = self.f_nkcc, np.zeros_like(self.f_nkcc)
        channel = self.face_fluxes(v, diffusion)  # electrodiffusion only (transporters off)
        self.f_alpha, self.f_nkcc = alpha, nkcc
        cycles = np.zeros(len(v))  # NKCC cycles, inward positive
        loaded = (self.f_nkcc > 0) if self.i_cl is not None else np.zeros(len(v), bool)
        if loaded.any():
            cin = self.conc[:, self.f_cell[loaded]]
            cycles[loaded] = nkcc_flux(
                cin[self.i_na],
                cin[self.i_k],
                cin[self.i_cl],
                self.c_out[self.i_na],
                self.c_out[self.i_k],
                self.c_out[self.i_cl],
                self.f_nkcc[loaded],
            )
        # Pump turnover (3 Na out per cycle): the Na flux that is neither channel nor NKCC.
        pump = (channel[self.i_na] - (total[self.i_na] - cycles)) / 3.0
        n_faces = len(self.skin.face_offsets) - 1
        exported = [
            ("vmem_face", v * 1e3),
            ("current_face", -self.const.faraday * (self.z[:, None] * total).sum(0)),
            ("flux_na_face", channel[self.i_na]),
            ("flux_k_face", channel[self.i_k]),
            ("pump_face", pump),
        ]
        if self.i_cl is not None:
            exported += [("flux_cl_face", channel[self.i_cl]), ("nkcc_face", cycles)]
        for key, values in exported:
            full = np.zeros(n_faces)
            full[self.f_face] = np.where(live_face, values, 0.0)
            frame.data[key] = full
        for model, _, _, mask, m, hh, lyr, side in self.channels:
            cells = self.f_cell[mask]
            open_ = np.bincount(cells, weights=model.open_fraction(m, hh), minlength=n)
            count = np.bincount(cells, minlength=n)
            key = f"open_{model.name.lower()}_{side}"
            prev = frame.data.get(key, np.zeros(n))
            frame.data[key] = np.where(count > 0, open_ / np.maximum(count, 1), prev)
        # Per face and channel type: the conductance the face's channels would have if all were
        # open (S/m², the electrodiffusion slope at the present voltage and concentrations), from
        # which the nanoscope derives channel counts; and, for gated channels, the open fraction.
        n_faces_all = len(self.skin.face_offsets) - 1
        cin_all = self.conc[:, self.f_cell]
        for model, max_d, weights, mask, m, hh, *_ in self.channels:
            key = model.name.lower()
            g_key, o_key = f"chan_g_{key}", f"chan_open_{key}"
            g_full = frame.data.setdefault(g_key, np.zeros(n_faces_all))
            vm = v[mask]
            cin = cin_all[:, mask]
            cout = np.repeat(self.c_out[:, None], len(vm), axis=1)
            dd = max_d * weights[:, None] * np.ones((1, len(vm)))

            def current(volts, cin=cin, cout=cout, dd=dd):
                flux = electroflux(
                    cout,
                    cin,
                    dd,
                    self.const.membrane_thickness_m,
                    self.z[:, None],
                    volts[None, :],
                    self.const,
                )
                return -self.const.faraday * (self.z[:, None] * flux).sum(axis=0)

            slope = (current(vm + DV) - current(vm - DV)) / (2 * DV)
            g_full[self.f_face[mask]] += np.where(live_face[mask], slope, 0.0)
            if model.m_power or model.h_power:
                opened = frame.data.setdefault(o_key, np.zeros(n_faces_all))
                opened[self.f_face[mask]] = model.open_fraction(m, hh)
        phi = self.phi_ecs().reshape(self.grid.shape)
        frame.data["phi_ecs"] = phi.copy()
        gz, gy, gx = np.gradient(phi, self.grid.zs * UM, self.grid.ys * UM, self.grid.xs * UM)
        # The side walls are mirror planes: the field has no normal component there (a one-sided
        # difference would invent one).
        gx[:, :, [0, -1]] = 0.0
        gy[:, [0, -1], :] = 0.0
        frame.data["efield_x"], frame.data["efield_y"], frame.data["efield_z"] = -gx, -gy, -gz
        frame.data["region"] = self.region.reshape(self.grid.shape).astype(float)
        return frame

    def run(self, on_frame=None) -> list[Frame]:
        t = self.config.physics.time
        for _ in range(int(round(t.init_s / t.init_dt_s))):
            self.step(t.init_dt_s)
        self.time_s = 0.0
        frames = [self.snapshot()]
        if self.skin.wounded.any():
            self.apply_wound()
        steps = int(round(t.sim_s / t.dt_s))
        every = max(int(round(t.sample_s / t.dt_s)), 1)
        for s in range(1, steps + 1):
            self.step(t.dt_s)
            if s % every == 0:
                frames.append(self.snapshot())
                if on_frame is not None:
                    on_frame(len(frames), steps // every + 1)
        return frames
