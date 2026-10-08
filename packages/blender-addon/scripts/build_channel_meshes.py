"""Build the channel and pump meshes the nanoscope instances, from structures in the Protein Data
Bank. Run with the add-on's asset dependencies:

    uv run --group assets python scripts/build_channel_meshes.py [key ...]

For each structure: download the mmCIF from RCSB (cached under data/raw/pdb/, gitignored), keep
the chains of the channel itself, orient it with the pore along +z and the membrane midplane at
z = 0, build a Gaussian density of the heavy atoms and take the isosurface that encloses the
volume expected from the protein's mass (1.21 Å³ per dalton), then decimate to two levels of
detail. Coordinates are written in nanometres to
bioelectric_playback/assets/channels/<key>.npz with a JSON of provenance next to it.

Orientation: the pore axis is the principal axis that differs from the other two (channels are
rotationally symmetric about it); the membrane is the 30 Å slab along it with the largest
fraction of hydrophobic residues; the cytoplasmic side is the flank with more lysine and
arginine (the positive-inside rule). This is a heuristic, recorded as such in the JSON.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import fast_simplification
import gemmi
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

ROOT = Path(__file__).parents[1]
CACHE = ROOT / "data" / "raw" / "pdb"
OUT = ROOT / "bioelectric_playback" / "assets" / "channels"
HYDROPHOBIC = {"LEU", "ILE", "VAL", "PHE", "MET", "TRP", "ALA"}
logger = logging.getLogger("build_channel_meshes")


@dataclass(frozen=True)
class Structure:
    key: str  # the name the nanoscope uses
    pdb_id: str
    stands_for: str  # which model channel or pump it depicts
    exclude: tuple[str, ...] = ("antibody", "fab", "nanobody", "heavy chain", "light chain")
    keep_only: tuple[str, ...] = ()  # entity description keywords to keep; all if empty
    hemichannel: bool = False  # gap junction: keep the half on one side of the docking plane
    # Which side is extracellular when the positive-inside rule cannot be trusted: "larger" (a
    # large extracellular domain, e.g. ENaC) or "docking" (a connexon faces its partner).
    extracellular: str = "auto"
    note: str = ""
    extra: dict = field(default_factory=dict)


# From docs/nanoscope.md (sources and verdicts there).
STRUCTURES: dict[str, Structure] = {
    s.key: s
    for s in (
        Structure(
            "enac",
            "6BQN",
            "NaLeak, apical (ENaC)",
            extracellular="larger",
            note="human alpha-beta-gamma ENaC; transmembrane domain only partly ordered; its "
            "large extracellular domain sets the orientation",
        ),
        Structure("twik1", "3UKM", "KLeak (two-pore-domain K channel TWIK-1 as stand-in)"),
        Structure("kv3_open", "7PQT", "Kv3.4, open (human Kv3.1 as stand-in)"),
        Structure(
            "kv_closed",
            "9OIC",
            "Kv3.4, closed (Drosophila Shaker I384R as stand-in)",
            note="closed pore with activated voltage sensors; no fully resting Kv "
            "structure exists",
        ),
        Structure("nak_atpase", "7E20", "Na/K-ATPase (human alpha1 beta1 FXYD2, E2-2K)"),
        Structure(
            "ano1",
            "5OYB",
            "ClLeak (Ca2+-activated Cl- channel ANO1/TMEM16A, mouse, Ca2+-bound)",
        ),
        Structure(
            "cx43",
            "7Z1T",
            "gap junction (human Cx43); one hemichannel drawn",
            hemichannel=True,
            extracellular="docking",
        ),
    )
}
OPM = "https://opm-assets.storage.googleapis.com/pdb/{}.pdb"


def fetch_opm(pdb_id: str) -> Path | None:
    """OPM's coordinates of the entry, oriented with the membrane normal along z and the
    hydrocarbon core centred at z = 0; None if OPM has no entry."""
    path = CACHE / f"{pdb_id.lower()}_opm.pdb"
    if not path.exists():
        try:
            with urllib.request.urlopen(OPM.format(pdb_id.lower()), timeout=60) as response:
                path.write_bytes(response.read())
        except Exception:  # no OPM entry
            return None
    return path


def fetch(pdb_id: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{pdb_id.lower()}.cif"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        logger.info("downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _value(block, *tags: str):
    """The first tag with a real value (mmCIF writes '?' or '.' for missing)."""
    for tag in tags:
        v = block.find_value(tag)
        if v not in (None, "?", "."):
            return gemmi.cif.as_string(v)
    return None


def load_atoms(spec: Structure) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Heavy-atom coordinates (Å), residue names, chain index per atom, and metadata, for the
    polymer entities of the channel itself within the first biological assembly."""
    doc = gemmi.cif.read(str(fetch(spec.pdb_id)))
    block = doc.sole_block()
    st = gemmi.make_structure_from_block(block)
    st.setup_entities()
    ent_ids = [gemmi.cif.as_string(v) for v in block.find_values("_entity.id")]
    ent_desc = [gemmi.cif.as_string(v) for v in block.find_values("_entity.pdbx_description")]
    describe = dict(zip(ent_ids, ent_desc, strict=True))
    keep, dropped = set(), []
    for ent in st.entities:
        text = describe.get(ent.name, "").lower()
        if ent.entity_type != gemmi.EntityType.Polymer:
            continue
        if any(w in text for w in spec.exclude) or (
            spec.keep_only and not any(w in text for w in spec.keep_only)
        ):
            dropped.append(describe.get(ent.name, ent.name))
            continue
        keep.update(ent.subchains)
    if st.assemblies:  # the biological unit: subchains its identity operators take
        in_assembly = set()
        for gen in st.assemblies[0].generators:
            if any(op.transform.is_identity() for op in gen.operators):
                in_assembly.update(gen.subchains)
        keep &= in_assembly or keep
    coords, residues, chain_of, chains_kept = [], [], [], []
    for ci, chain in enumerate(st[0]):
        polymer = chain.get_polymer()
        if not len(polymer) or polymer.subchain_id() not in keep:
            continue
        chains_kept.append(chain.name)
        for res in polymer:
            for atom in res:
                if atom.element.name != "H":
                    coords.append(atom.pos.tolist())
                    residues.append(res.name)
                    chain_of.append(ci)
    meta = {
        "title": _value(block, "_struct.title"),
        "method": _value(block, "_exptl.method"),
        "resolution_A": _value(
            block,
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ),
        "organism": _value(
            block,
            "_entity_src_gen.pdbx_gene_src_scientific_name",
            "_entity_src_nat.pdbx_organism_scientific",
            "_pdbx_entity_src_syn.organism_scientific",
        ),
        "entities_kept": sorted(
            {describe.get(e.name, "") for e in st.entities if set(e.subchains) & keep}
        ),
        "entities_dropped": dropped,
        "chains_kept": chains_kept,
    }
    return np.array(coords), residues, np.array(chain_of), meta


def orient(xyz: np.ndarray, residues: list[str]) -> tuple[np.ndarray, dict]:
    """Rotate so that the pore axis is z, the membrane midplane is z = 0 and the cytoplasm is
    at negative z."""
    centred = xyz - xyz.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    gaps = [
        abs(values[0] - values[1]) + abs(values[0] - values[2]),
        abs(values[1] - values[0]) + abs(values[1] - values[2]),
        abs(values[2] - values[0]) + abs(values[2] - values[1]),
    ]
    axis = vectors[:, int(np.argmax(gaps))]
    x = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(axis, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(axis, x), axis])
    pts = centred @ rot.T
    hydro = np.array([r in HYDROPHOBIC for r in residues])
    z = pts[:, 2]
    centres = np.arange(z.min() + 15, z.max() - 15, 1.0)
    frac = [hydro[(z > c - 15) & (z < c + 15)].mean() for c in centres]
    mid = float(centres[int(np.argmax(frac))])
    pts[:, 2] -= mid
    positive = np.array([r in ("LYS", "ARG") for r in residues])
    above = positive[(pts[:, 2] > 15) & (pts[:, 2] < 30)].sum()
    below = positive[(pts[:, 2] < -15) & (pts[:, 2] > -30)].sum()
    if above > below:  # more K/R above: flip so the cytoplasm is below
        pts[:, 1:] *= -1
    radial = np.linalg.norm(pts[:, :2], axis=1)
    return pts, {
        "method": "heuristic: symmetry axis, 30 A hydrophobic slab, positive-inside rule",
        "hydrophobic_fraction_in_slab": float(max(frac)),
        "lys_arg_cytoplasmic_flank": int(min(above, below)),
        "lys_arg_extracellular_flank": int(max(above, below)),
        "radius_nm": float(np.percentile(radial, 99) / 10),
        "z_range_nm": [float(pts[:, 2].min() / 10), float(pts[:, 2].max() / 10)],
    }


def surface(pts: np.ndarray, mass_da: float, sigma: float = 2.4, spacing: float = 1.0):
    """Isosurface of a Gaussian density enclosing 1.21 Å³/Da (vertices in Å)."""
    lo = pts.min(axis=0) - 4 * sigma
    shape = np.ceil((pts.max(axis=0) + 4 * sigma - lo) / spacing).astype(int) + 1
    grid = np.zeros(shape)
    idx = np.round((pts - lo) / spacing).astype(int)
    np.add.at(grid, tuple(idx.T), 1.0)
    density = ndimage.gaussian_filter(grid, sigma / spacing)
    target = 1.21 * mass_da / spacing**3
    lo_t, hi_t = 1e-4, density.max()
    for _ in range(50):
        mid = 0.5 * (lo_t + hi_t)
        if (density > mid).sum() > target:
            lo_t = mid
        else:
            hi_t = mid
    verts, faces, _, _ = marching_cubes(density, level=0.5 * (lo_t + hi_t))
    return verts * spacing + lo, faces, float(0.5 * (lo_t + hi_t))


def opm_atoms(spec: Structure, chains: list[str]):
    """Heavy atoms of the kept chains in OPM's frame, with OPM's half bilayer thickness."""
    path = fetch_opm(spec.pdb_id)
    if path is None:
        return None
    half = None
    for line in path.read_text().splitlines()[:5]:
        if "1/2 of bilayer thickness" in line:
            half = float(line.split(":")[1])
    st = gemmi.read_structure(str(path))
    coords, residues = [], []
    for chain in st[0]:
        if chain.name not in chains:
            continue
        for res in chain:
            if res.het_flag == "H":
                continue
            for atom in res:
                if atom.element.name != "H":
                    coords.append(atom.pos.tolist())
                    residues.append(res.name)
    if not coords:
        return None
    return np.array(coords), residues, half


def build(spec: Structure) -> dict:
    xyz, residues, chain_of, meta = load_atoms(spec)
    if spec.hemichannel:
        # A gap junction spans two membranes: split at the docking plane (through the centroid,
        # normal to the channel axis) and keep one connexon.
        centred = xyz - xyz.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(centred.T))
        axis = vectors[:, int(np.argmax(values))]  # the long axis of the dodecamer
        side = {c: float((centred[chain_of == c] @ axis).mean()) for c in np.unique(chain_of)}
        keep = np.isin(chain_of, [c for c, v in side.items() if v < 0])
        docking_point = xyz.mean(axis=0)
        xyz, residues = xyz[keep], [r for r, k in zip(residues, keep, strict=True) if k]
        meta["chains_kept"] = meta["chains_kept"][: len(side) // 2]
        meta["hemichannel"] = "one connexon of the docked pair, split at the docking plane"
    opm = None if spec.hemichannel else opm_atoms(spec, meta["chains_kept"])
    if opm is not None:
        pts, residues, half = opm
        positive = np.array([r in ("LYS", "ARG") for r in residues])
        above = positive[(pts[:, 2] > 15) & (pts[:, 2] < 30)].sum()
        below = positive[(pts[:, 2] < -15) & (pts[:, 2] > -30)].sum()
        if above > below:  # cytoplasm (more K/R) at negative z
            pts[:, 1:] *= -1
        pts[:, :2] -= pts[:, :2].mean(axis=0)
        orientation = {
            "method": "OPM (membrane normal along z, core centred at z = 0); cytoplasmic side "
            "by the positive-inside rule",
            "opm_half_thickness_A": half,
            "radius_nm": float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 99) / 10),
            "z_range_nm": [float(pts[:, 2].min() / 10), float(pts[:, 2].max() / 10)],
        }
    else:
        pts, orientation = orient(xyz, residues)
        flip = False
        if spec.extracellular == "larger":
            flip = -pts[:, 2].min() > pts[:, 2].max()
            orientation["method"] += "; extracellular side: the larger domain (stated rule)"
        elif spec.extracellular == "docking":
            # orient() maps xyz rigidly to pts (centred, rotated, shifted in z, maybe flipped),
            # so the docking point's z follows from the fitted map.
            centred = xyz - xyz.mean(axis=0)
            design = np.column_stack((centred, np.ones(len(centred))))
            coef, *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
            z_dock = float(np.append(docking_point - xyz.mean(axis=0), 1.0) @ coef)
            flip = z_dock < 0
            orientation["method"] += "; extracellular side: toward the docking plane"
        if flip:
            pts[:, 1:] *= -1
            orientation["z_range_nm"] = [float(pts[:, 2].min() / 10), float(pts[:, 2].max() / 10)]
    mass = 14.0 * len(pts)  # about 14 Da per heavy atom (C, N, O averaged with hydrogens)
    verts, faces, level = surface(pts, mass)
    lods = {}
    for name, target in (("high", 5000), ("low", 500)):
        reduction = max(0.0, 1.0 - target / len(faces))
        v, f = fast_simplification.simplify(
            verts.astype(np.float32), faces.astype(np.int64), reduction
        )
        lods[name] = (np.asarray(v, np.float32) / 10.0, np.asarray(f, np.int32))  # nm
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{spec.key}.npz",
        high_verts=lods["high"][0],
        high_faces=lods["high"][1],
        low_verts=lods["low"][0],
        low_faces=lods["low"][1],
    )
    info = {
        "key": spec.key,
        "pdb_id": spec.pdb_id.upper(),
        "stands_for": spec.stands_for,
        "source": f"https://www.rcsb.org/structure/{spec.pdb_id.upper()}",
        "licence": "PDB data are CC0 1.0 (wwPDB)",
        **meta,
        "heavy_atoms": len(pts),
        "isosurface": {"sigma_A": 2.4, "grid_A": 1.0, "level": level, "volume_rule": "1.21 A3/Da"},
        "triangles": {k: int(len(v[1])) for k, v in lods.items()},
        "orientation": orientation,
        "note": spec.note,
        **spec.extra,
    }
    (OUT / f"{spec.key}.json").write_text(json.dumps(info, indent=2) + "\n")
    logger.info(
        "%s: %s, %d heavy atoms, %s triangles", spec.key, spec.pdb_id, len(pts), info["triangles"]
    )
    return info


def main(keys: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for key in keys or list(STRUCTURES):
        build(STRUCTURES[key])


if __name__ == "__main__":
    main(sys.argv[1:])
