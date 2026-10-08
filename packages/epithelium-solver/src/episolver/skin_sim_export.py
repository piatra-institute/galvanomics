"""Write a skin simulation as a polyhedral bundle with a 3D extracellular volume."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tissuebundle.writer import QuantityInput, SourceInfo, write_bundle

from episolver import __version__
from episolver.skin import facing_faces
from episolver.skin_export import dermis_mesh
from episolver.skin_physics import REGIONS, Frame, SkinModel


def membrane_constants(model: SkinModel) -> dict[str, float]:
    """The membrane specification as flat constants: per layer and side, the base diffusion
    constant of each ion, the pump rate, and each channel's maximum diffusion constant
    (m2/s), e.g. ``max_d_l2_apical_naleak``. Enough to rebuild what each face carries."""
    out: dict[str, float] = {}
    for layer, sides in sorted(model.membranes.items()):
        for side in ("apical", "basolateral"):
            dom = sides[side]
            out[f"pump_alpha_l{layer}_{side}"] = float(dom.pump_alpha)
            if dom.nkcc_alpha:
                out[f"nkcc_alpha_l{layer}_{side}"] = float(dom.nkcc_alpha)
            for ion, value in sorted(dom.base_d.items()):
                out[f"base_d_l{layer}_{side}_{ion}"] = float(value)
            for channel in dom.channels:
                out[f"max_d_l{layer}_{side}_{channel.type.lower()}"] = float(channel.max_d)
    return out


def export_skin_sim(
    model: SkinModel,
    frames: list[Frame],
    out: Path,
    config_path: str | None,
    config_sha256: str | None,
    overwrite: bool = True,
) -> Path:
    skin, ph = model.skin, model.config.physics
    d = skin.derived
    wounded = skin.wounded if model.wounded else np.zeros(skin.n_cells, bool)
    alive_any = model.living

    inert_face = ~alive_any[d["face_cell"]]

    def faces(key: str) -> np.ndarray:
        data = np.stack([f.data[key] for f in frames]).astype(float)
        data[:, inert_face] = np.nan
        data[1:, wounded[d["face_cell"]]] = np.nan
        return data

    q: dict[str, QuantityInput] = {}
    vmem = np.stack([f.data["vmem"] for f in frames]) * 1e3
    vmem[1:, wounded] = np.nan
    # Corneocytes are inert: they carry no voltage (NaN), only their layer index.
    vmem[:, ~alive_any] = np.nan
    layer = np.stack([skin.cell_layer.astype(float)] * len(frames))
    layer[1:, wounded] = np.nan
    q["layer"] = QuantityInput(layer, "1", "cell", "Epidermal layer index (see layers)")
    q["vmem"] = QuantityInput(
        vmem, "mV", "cell", "Membrane voltage, area-weighted over the cell's faces (living cells)"
    )
    conc = np.stack([f.data["conc_cell"] for f in frames]).astype(float)
    conc[:, :, ~alive_any] = np.nan
    conc[1:, :, wounded] = np.nan
    q["conc_cell"] = QuantityInput(conc, "mol/m3", "cell_ion", "Cytoplasmic concentration per ion")
    for key in [k for k in frames[0].data if k.startswith("open_")]:
        data = np.stack([f.data[key] for f in frames]).astype(float)
        data[:, ~alive_any] = np.nan
        data[1:, wounded] = np.nan
        side = "apical" if key.endswith("_apical") else "basolateral"
        q[key] = QuantityInput(data, "1", "cell", f"Open fraction, {key[5:]}", domain=side)
    for key, unit, desc in (
        ("vmem_face", "mV", "Membrane voltage of each face (cell minus extracellular space)"),
        ("current_face", "A/m2", "Outward ionic current density through each face"),
        ("flux_na_face", "mol/m2/s", "Na+ flux into the cell through each face's channels"),
        ("flux_k_face", "mol/m2/s", "K+ flux into the cell through each face's channels"),
        ("pump_face", "mol/m2/s", "Na/K-ATPase turnover on each face (3 Na out, 2 K in)"),
        ("flux_cl_face", "mol/m2/s", "Cl- flux into the cell through each face's channels"),
        ("nkcc_face", "mol/m2/s", "Na-K-2Cl cotransporter cycles on each face, inward"),
    ):
        if key in frames[0].data:
            q[key] = QuantityInput(faces(key), unit, "membrane", desc)
    for key in [k for k in frames[0].data if k.startswith("chan_")]:
        name = key.split("_", 2)[2]
        if key.startswith("chan_g_"):
            q[key] = QuantityInput(
                faces(key).astype(np.float32),
                "S/m2",
                "membrane",
                f"Conductance of each face's {name} channels if all were open (for counts)",
            )
        else:
            q[key] = QuantityInput(
                faces(key).astype(np.float32), "1", "membrane", f"Open fraction of {name} channels"
            )
    domain = np.zeros(len(skin.face_offsets) - 1)
    domain[model.f_face] = np.where(model.f_domain == 0, 1.0, 2.0)  # 1 apical, 2 basolateral
    domain_frames = np.stack([domain] * len(frames))
    domain_frames[:, inert_face] = np.nan
    domain_frames[1:, wounded[d["face_cell"]]] = np.nan
    q["face_domain"] = QuantityInput(
        domain_frames,
        "1",
        "membrane",
        "0 none (inert cell), 1 apical (Na channels), 2 basolateral (K channels, Na/K pumps)",
    )
    for key, unit, scale, desc in (
        ("phi_ecs", "mV", 1e3, "Extracellular potential; the bottom of the dermis is 0"),
        ("efield_x", "V/m", 1.0, "Extracellular field, x"),
        ("efield_y", "V/m", 1.0, "Extracellular field, y"),
        ("efield_z", "V/m", 1.0, "Extracellular field, z"),
        ("region", "1", 1.0, "Region: " + ", ".join(f"{i} {r}" for i, r in enumerate(REGIONS))),
    ):
        q[key] = QuantityInput(
            np.stack([f.data[key] for f in frames]) * scale, unit, "volume", desc
        )

    verts, mesh_faces = dermis_mesh(model.config.skin)
    times = np.array([f.time_s for f in frames])
    spec = model.config.skin
    return write_bundle(
        out,
        name=model.config.name.replace("-", "_"),
        description=model.config.description or "Skin wound simulation",
        source=SourceInfo(
            solver="episolver.skin_physics",
            solver_version=__version__,
            config_file=config_path,
            config_sha256=config_sha256,
            seed=spec.seed,
            command=tuple(sys.argv),
            producer="epithelium-solver",
            producer_version=__version__,
        ),
        ions=model.ions,
        ion_charges=[int(z) for z in model.z],
        constants={
            "temperature_k": model.const.temperature_k,
            "faraday_c_per_mol": model.const.faraday,
            "tight_junction_z_um": float(model.tj_z),
            "granular_top_um": float(model.granular_top),
            "tight_junction_resistance_ohm_cm2": ph.tight_junction_ohm_cm2,
            "sim_time_step_s": ph.time.dt_s,
            "init_total_time_s": ph.time.init_s,
            "wound_x_um": float(spec.wound.x_um),
            **membrane_constants(model),
        },
        cell_verts=skin.face_verts,
        cell_offsets=skin.cell_offsets,
        cell_centres=d["centroid"],
        cell_removed=wounded,
        time_integrated_s=times,
        time_reported_s=times,
        baseline_frames=(0,),
        quantities=q,
        mem_neighbour=facing_faces(skin),
        kind="polyhedra",
        face_offsets=skin.face_offsets,
        face_boundary=skin.face_boundary,
        cell_layer=skin.cell_layer,
        cell_inert=~alive_any,
        layers=[layer.name for layer in spec.layers],
        layer_colors=[layer.color for layer in spec.layers],
        context_meshes={
            "dermis": (
                verts,
                mesh_faces,
                (0.93, 0.62, 0.58, 1.0),
                "Dermis; its top is the dermal-epidermal junction",
            )
        },
        volume_shape=model.grid.shape,
        volume_axes=(model.grid.xs, model.grid.ys, model.grid.zs),
        overwrite=overwrite,
    )
