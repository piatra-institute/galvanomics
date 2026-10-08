"""Write a run as a tissue bundle (schema 0.2): prisms, two bath layers, per-domain values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tissuebundle.writer import QuantityInput, SourceInfo, write_bundle

from episolver import __version__
from episolver.geometry import UM
from episolver.solver import DOMAINS, Epithelium, Frame

MV = 1e3


def _cell_series(frames: list[Frame], key: str, wounded: np.ndarray, scale: float = 1.0):
    data = np.stack([f.data[key] for f in frames]).astype(float) * scale
    data[1:, ..., wounded] = np.nan
    return data


def export_bundle(
    model: Epithelium,
    frames: list[Frame],
    out: Path,
    config_path: str | None,
    config_sha256: str | None,
    overwrite: bool = False,
) -> Path:
    config, tissue, grid = model.config, model.tissue, model.grid
    wounded = tissue.wounded if model.wounded else np.zeros(tissue.n_cells, bool)
    q: dict[str, QuantityInput] = {}

    def cell(key: str, unit: str, desc: str, domain: str | None = None, scale: float = 1.0):
        q[key] = QuantityInput(
            _cell_series(frames, key, wounded, scale), unit, "cell", desc, domain=domain
        )

    cell("vmem_apical", "mV", "Apical membrane voltage, cell minus apical bath", "apical", MV)
    cell(
        "vmem_basolateral",
        "mV",
        "Basolateral membrane voltage, cell minus basal bath",
        "basolateral",
        MV,
    )
    cell("phi_cell", "mV", "Intracellular potential (apical bath edge grounded)", "cell", MV)
    cell(
        "tep",
        "mV",
        "Transepithelial potential under the cell, apical minus basal bath",
        "cell",
        MV,
    )
    for d in DOMAINS:
        cell(f"current_{d}", "A/m2", f"Outward ionic current density, {d} membrane", d)
        if model.alpha[d] > 0 and model.has_pump:
            cell(f"pump_{d}", "mol/m2/s", f"Na/K-ATPase turnover, {d} membrane", d)
        for ch in model.channels[d]:
            key = f"open_{ch.model.name.lower()}_{d}"
            cell(key, "1", f"Open fraction of {ch.model.name} ({ch.model.source}), {d}", d)
    conc = _cell_series(frames, "conc_cell", wounded)
    q["conc_cell"] = QuantityInput(
        conc, "mol/m3", "cell_ion", "Cytoplasmic concentration per ion", domain="cell"
    )

    h_um = config.height_um
    spacing = grid.spacing_um * UM
    for d, name, z in (
        ("apical", "apical", h_um + config.baths.apical_thickness_um / 2),
        ("basolateral", "basal", -config.baths.basal_thickness_um / 2),
    ):
        phi = np.stack([f.data[f"phi_bath_{d}"] for f in frames])
        q[f"phi_bath_{name}"] = QuantityInput(
            phi * MV, "mV", "grid", f"Potential of the {name} bath", domain=f"bath_{name}", z_um=z
        )
        gy, gx = np.gradient(phi, spacing, spacing, axis=(1, 2))
        for comp, values in (("x", -gx), ("y", -gy)):
            q[f"efield_{name}_{comp}"] = QuantityInput(
                values,
                "V/m",
                "grid",
                f"Electric field in the {name} bath, {comp} component",
                domain=f"bath_{name}",
                z_um=z,
            )

    pa, pb = tissue.pair_cells.T
    gates = np.zeros((len(frames), len(tissue.neighbour)))
    for t, frame in enumerate(frames):
        gates[t, tissue.pair_mem[:, 0]] = frame.data["gj_open"]
        gates[t, tissue.pair_mem[:, 1]] = frame.data["gj_open"]
    dead_pair = wounded[pa] | wounded[pb]
    gates[1:, tissue.pair_mem[dead_pair, 0]] = 0.0
    gates[1:, tissue.pair_mem[dead_pair, 1]] = 0.0
    gates[1:, wounded[tissue.mem_cell]] = np.nan
    q["gj_open"] = QuantityInput(
        gates, "1", "membrane", "Gap-junction open fraction (BETSE gate)", domain="lateral"
    )

    times = np.array([f.time_s for f in frames])
    gx_um, gy_um = grid.coordinates()
    return write_bundle(
        out,
        name=config.name.replace("-", "_"),
        description=config.description or "3D polarized epithelium",
        source=SourceInfo(
            solver="episolver",
            solver_version=__version__,
            config_file=config_path,
            config_sha256=config_sha256,
            seed=None,
            command=tuple(sys.argv),
            producer="epithelium-solver",
            producer_version=__version__,
        ),
        ions=model.ions,
        ion_charges=[int(z) for z in model.z],
        constants={
            "temperature_k": model.const.temperature_k,
            "membrane_thickness_m": model.const.membrane_thickness_m,
            "gas_constant_j_per_k_mol": model.const.gas_constant,
            "faraday_c_per_mol": model.const.faraday,
            "capacitance_f_per_m2": model.const.capacitance_f_per_m2,
            "sim_time_step_s": config.time.dt_s,
            "init_time_step_s": config.time.init_dt_s,
            "init_total_time_s": config.time.init_s,
            "bath_conductivity_s_per_m": model.sigma["apical"],
            "tight_junction_resistance_ohm_cm2": config.tight_junctions.resistance_ohm_cm2,
            "apical_bath_thickness_um": config.baths.apical_thickness_um,
            "basal_bath_thickness_um": config.baths.basal_thickness_um,
        },
        cell_verts=tissue.verts_um,
        cell_offsets=tissue.offsets,
        cell_centres=tissue.centres_um,
        cell_removed=wounded,
        time_integrated_s=times,
        time_reported_s=times,
        baseline_frames=(0,),
        quantities=q,
        grid_x=gx_um,
        grid_y=gy_um,
        mem_neighbour=tissue.neighbour,
        extrusion_height_um=h_um,
        overwrite=overwrite,
    )
