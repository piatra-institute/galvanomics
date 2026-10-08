"""Run configuration. Defaults are BETSE 1.5.0's values unless marked as an assumption; see
``docs/epithelium.md`` for the table of sources."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from episolver.channels import CHANNELS


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Ion(Model):
    name: str
    z: int
    d_free: float = Field(description="Free diffusion constant [m2/s] (BETSE Do_*)")
    cell: float = Field(description="Initial cytoplasmic concentration [mol/m3]")
    bath_apical: float
    bath_basal: float


BETSE_BASIC_IONS = (
    Ion(name="na", z=1, d_free=1.33e-9, cell=12.0, bath_apical=145.0, bath_basal=145.0),
    Ion(name="k", z=1, d_free=1.96e-9, cell=139.0, bath_apical=5.0, bath_basal=5.0),
    Ion(name="m", z=-1, d_free=1.0e-9, cell=16.0, bath_apical=140.0, bath_basal=140.0),
    Ion(name="p", z=-1, d_free=0.0, cell=135.0, bath_apical=10.0, bath_basal=10.0),
)
BETSE_BASIC_D = {"na": 2.0e-18, "k": 1.0e-18, "m": 1.0e-18, "p": 0.0}
# BETSE's "mammal" ion profile (parameters.py, IonProfileType.MAMMAL) without Ca2+, since no Ca
# pump is modelled: Na, K, Cl and protein as BETSE sets them, the anion M rebalancing charge as
# BETSE's _balance_charge does (environment 145 + 5 - 115 - 10 = 25; cell 12 + 139 - 4 - 135 =
# 12). Free diffusion and membrane diffusion constants are BETSE's defaults (sim_config.yaml).
BETSE_MAMMAL_NO_CA_IONS = (
    Ion(name="na", z=1, d_free=1.33e-9, cell=12.0, bath_apical=145.0, bath_basal=145.0),
    Ion(name="k", z=1, d_free=1.96e-9, cell=139.0, bath_apical=5.0, bath_basal=5.0),
    Ion(name="cl", z=-1, d_free=2.03e-9, cell=4.0, bath_apical=115.0, bath_basal=115.0),
    Ion(name="m", z=-1, d_free=1.0e-9, cell=12.0, bath_apical=25.0, bath_basal=25.0),
    Ion(name="p", z=-1, d_free=0.0, cell=135.0, bath_apical=10.0, bath_basal=10.0),
)
BETSE_MAMMAL_D = {"na": 2.0e-18, "k": 1.0e-18, "cl": 1.0e-18, "m": 1.0e-18, "p": 0.0}
BETSE_ION_NAMES = {
    "sodium": "na",
    "potassium": "k",
    "anion": "m",
    "proteins": "p",
    "chloride": "cl",
    "calcium": "ca",
}


class ChannelSpec(Model):
    type: str
    max_d: float = Field(description="Diffusion constant when fully open [m2/s] (BETSE max Dm)")

    def model_post_init(self, _context: object) -> None:
        if self.type not in CHANNELS:
            raise ValueError(f"unknown channel {self.type!r}; known: {sorted(CHANNELS)}")


class Domain(Model):
    base_d: dict[str, float] = Field(default_factory=lambda: dict(BETSE_BASIC_D))
    pump_alpha: float = Field(1.0e-7, description="Na/K-ATPase rate [mol/m2/s] (BETSE alpha_NaK)")
    nkcc_alpha: float = Field(
        0.0,
        description="Na-K-2Cl cotransporter (NKCC1) maximal rate [mol/m2/s]; not in BETSE: "
        "a thermodynamic form, flux = alpha (P_out - P_in) / (P_out + P_in), P = Na K Cl^2",
    )
    channels: tuple[ChannelSpec, ...] = ()


class Footprint(Model):
    kind: Literal["hex", "betse_bundle"] = "hex"
    cols: int = 12
    rows: int = 12
    radius_um: float = 5.0
    bundle: str | None = None


class Wound(Model):
    kind: Literal["none", "from_bundle", "circle"] = "none"
    x_um: float = 0.0
    y_um: float = 0.0
    r_um: float = 0.0


class GapJunctions(Model):
    enabled: bool = True
    surface: float = Field(5.0e-8, description="BETSE gj_surface")
    pore_m: float = Field(2.6e-8, description="BETSE cell_space, the gap-junction pipe length")
    voltage_gated: bool = True
    threshold_mv: float = 15.0
    minimum: float = 0.1


class TightJunctions(Model):
    resistance_ohm_cm2: float = Field(500.0, description="Assumption: paracellular resistance")


class Baths(Model):
    apical_thickness_um: float = Field(50.0, description="Assumption")
    basal_thickness_um: float = Field(10.0, description="Assumption")
    spacing_um: float = 5.0
    margin_um: float = 30.0


class TimeSpec(Model):
    init_dt_s: float = 5.0e-3
    init_s: float = 2.0
    dt_s: float = 1.0e-3
    sim_s: float = 2.0
    sample_s: float = 0.05


class PhysicalConstants(Model):
    temperature_k: float = 310.0
    gas_constant: float = 8.314
    faraday: float = 96485.0
    membrane_thickness_m: float = 7.5e-9
    capacitance_f_per_m2: float = 0.05


class Pump(Model):
    km_na: float = 12.0
    km_k: float = 0.2
    km_atp: float = 0.5
    atp: float = 1.5
    adp: float = 0.1
    pi: float = 0.1
    delta_g_atp: float = -37000.0


class RunConfig(Model):
    name: str
    description: str = ""
    footprint: Footprint = Footprint()
    height_um: float = Field(10.0, description="BETSE cell_height")
    wound: Wound = Wound()
    ions: tuple[Ion, ...] = BETSE_BASIC_IONS
    initial_conc_from_bundle: bool = True
    apical: Domain = Domain()
    basolateral: Domain = Domain()
    gap_junctions: GapJunctions = GapJunctions()
    tight_junctions: TightJunctions = TightJunctions()
    baths: Baths = Baths()
    time: TimeSpec = TimeSpec()
    constants: PhysicalConstants = PhysicalConstants()
    pump: Pump = Pump()
    initial_vmem_mv: float = -20.0

    def resolve(self, base: Path) -> RunConfig:
        """Make a relative footprint bundle path absolute against ``base``."""
        bundle = self.footprint.bundle
        if bundle is None or Path(bundle).is_absolute():
            return self
        footprint = self.footprint.model_copy(update={"bundle": str((base / bundle).resolve())})
        return self.model_copy(update={"footprint": footprint})


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(data).resolve(Path(path).resolve().parent)
