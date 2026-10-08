# galvanomics: Architecture

Written 2026-10-07, when the first three packages were started; updated 2026-10-08 for the epithelium solver, the skin model, bundle schema 0.3, the wide block and the nanoscope. This document fixes the boundaries; `docs/plan.md` fixes what each one is for.

## The boundary, stated first

The simulator never imports Blender, and Blender never imports the simulator. They meet at one place: a **run bundle** on disk. Anything that produces a bundle (BETSE through the adapter today, a minimal solver later) can be inspected in Blender; anything that reads a bundle (Blender today, another viewer or a notebook later) needs only NumPy.

## The packages

| Package | Runtime | What it owns |
| --- | --- | --- |
| `packages/tissue-bundle` (`tissuebundle`) | Python ≥ 3.12, uv | The bundle contract: pydantic models, writer, validator, JSON schema, a stdlib-plus-NumPy reader, a synthetic hexagonal-sheet fixture, CLI `tissuebundle` |
| `packages/betse-adapter` (`betse_adapter`) | Python 3.12 pinned by uv, `betse==1.5.0` | Runs BETSE in-process with a recorded seed, maps pre/post-wound indices, exports bundles, computes the audit checks, CLI `betse-adapter` |
| `packages/epithelium-solver` (`episolver`) | Python ≥ 3.12, uv; NumPy, SciPy, pyamg | Two models on the same BETSE ports (flux law, pump, channels, gap junctions). The 3D polarized epithelium (`docs/epithelium.md`): prisms between two baths, schema-0.2 bundles, audit E1–E8. The skin (`docs/skin.md`): procedural polyhedral anatomy, keratinocyte membranes, tight junctions and a 3D extracellular grid, schema-0.3 bundles, audit K1–K5 |
| `packages/blender-addon` (`bioelectric_playback`) | Blender 5.2's bundled Python 3.13 | Imports a bundle as a mesh (polygons, prisms, or polyhedral cells), plays it back frame by frame, shows the selected cell's values. Draws bath planes, field arrows and channel glyphs. For the skin: channel and pump populations and ion particles as Geometry Nodes instances, a movable section plane, a probe with a time plot, a scale bar, per-face quantities, and the nanoscope (`docs/nanoscope.md`): a membrane patch at nanometre scale with molecules from Protein Data Bank structures, sampled from the solver's per-face outputs. Simulate buttons run either solver through uv. uv is used only to lint and test its pure-Python core |

One uv project per package, each with its own lock. The adapter depends on the bundle package through a path source:

```toml
[tool.uv.sources]
tissue-bundle = { path = "../tissue-bundle", editable = true }
```

## Dependency direction

```
betse  ←  betse-adapter  →  tissue-bundle  ←(vendored reader)  blender-addon
                                  ↑                                  │ runs, via uv
                          epithelium-solver  ←───────────────────────┘
```

`betse-adapter` also depends on `epithelium-solver` for tests only (check E5 compares the channel port with BETSE), and the solver reads BETSE bundles as footprints without importing BETSE.

The add-on carries a byte-identical copy of `tissuebundle/reader.py`, because Blender extensions may not install packages or modify `sys.path`. A test in the add-on fails if the copy drifts; `scripts/build_extension.py` refreshes it. The reader imports only the standard library and NumPy, which a test enforces by parsing its imports.

## The run bundle (v0.3.0)

A bundle is a directory:

```
<name>.tbundle/
  bundle.json          manifest: schema version, provenance, frames, quantities
  cell_verts.npy       float64 [V, 2]   polygon vertices, µm, counter-clockwise per cell
  cell_offsets.npy     int64   [N+1]    CSR offsets; cell i owns vertices and membranes offsets[i]:offsets[i+1]
  cell_centres.npy     float64 [N, 2]   µm
  cell_removed.npy     bool    [N]      cells removed by the wound (present in the baseline frame only)
  mem_grid_index.npy   int64   [M]      extracellular grid point each membrane faces (if a grid exists)
  grid_x.npy, grid_y.npy float64 [ny, nx]  grid coordinates, µm
  time_integrated_s.npy float64 [T]     integrated time of each frame
  time_reported_s.npy   float64 [T]     time as the solver labels it
  <quantity>.npy       float  [T, ...]  one file per time series
```

- **Cells are indexed in the pre-wound numbering.** N counts every cell that existed in the baseline; removed cells keep their polygons and are NaN in every post-wound frame. One mesh therefore serves every frame.
- **Frame 0 is the baseline**: the init phase's final state, which is exactly the sim phase's initial condition before the wound fires. `bundle.json` says which frames are baseline and which are post-wound. Quantities BETSE does not keep for the final state (its GHK estimate) are NaN in that frame.
- **Every quantity declares its location** (`cell`, `membrane`, `grid`, `cell_ion`, `membrane_ion`, `grid_ion`), its unit, shape, dtype and SHA-256. Location is never inferred from length: a 15 × 15 grid and a 225-cell sheet have the same length.
- **Units** are SI except lengths (µm) and voltages (mV), which are the units people read; each quantity states its own.
- **Provenance** in `bundle.json`: solver name and version, configuration file SHA-256, random seed, command line, Python and NumPy versions, creation time, and the hash of every array. The writer builds the bundle in a temporary directory and renames it into place.
- **Ions and constants**: ion names with their valences, and the solver constants needed to recompute derived quantities from the bundle alone (temperature, membrane thickness, the gas and Faraday constants as the solver rounds them, time steps). The audit's GHK check is computed from these, not from BETSE.
- **3D tissue** (schema 0.2): `extrusion_height_um` makes every cell a prism; `mem_neighbour` gives each membrane's facing membrane (-1 on the sheet edge); each quantity may name a `domain` (`apical`, `basolateral`, `lateral`, `bath_apical`, `bath_basal`) and grid quantities a height `z_um`. Versions 0.1 and 0.2 remain readable.
- **Polyhedral tissue** (schema 0.3): with `kind: polyhedra`, `cell_verts` is [V, 3] and every cell is a closed set of faces. `face_offsets` [F+1] indexes vertices per face, `cell_offsets` [N+1] indexes faces per cell, and membranes are faces, so membrane quantities have one value per face. `face_boundary` marks each face shared, basement, surface or side; `cell_layer` with `layers` and `layer_colors` names the strata; `cell_inert` marks cells that carry no electrical state (corneocytes), NaN in every frame without failing validation.
- **Volume quantities** (schema 0.3): location `volume`, shape [T, nz, ny, nx], on a tensor grid whose axes `volume_x`, `volume_y`, `volume_z` may be non-uniform (the skin grid is 0.75 µm in z through the granular layer, 4 µm elsewhere).
- **Per-face channel data and membrane constants** (skin, 2026-10-08, no schema change): `chan_g_<channel>` (S/m², membrane location, float32) is each face's conductance if all its channels of that type were open; `chan_open_<channel>` is the open fraction of gated channels; the constants carry `wound_x_um` and the membrane specification (`max_d_l<layer>_<side>_<channel>`, `pump_alpha_l<layer>_<side>`, `base_d_...`).
- **Context meshes** (schema 0.3): named static meshes that are not cells (the dermis block, the wound floor), each with a colour and a description.
- `.npy` files are uncompressed and memory-mappable, so Blender reads one frame without loading the run. Conversion to HDF5 or Zarr later is lossless; the founding notes' "HDF5 or Zarr" is deferred because Blender's Python ships neither.

## Inputs and outputs

| Package | Reads | Writes |
| --- | --- | --- |
| betse-adapter | a run spec under `configs/`: BETSE's default configuration plus named overrides | the resolved `sim_config.yaml` and BETSE's own INITS/SIMS/RESULTS in a work directory; a bundle; `export.json`; `runs/<name>/result.json` |
| epithelium-solver | a run config (YAML) under `configs/` | a bundle under `data/work/<name>/`; `runs/epithelium/`, `runs/skin/`, `runs/skin-wide/`, `runs/skin-wide-followup/`, `runs/skin-amiloride/`, `runs/skin-rivals/`, `runs/skin-rivals-b2/` (each `result.json`) |
| tissue-bundle | a bundle | validation reports, `schemas/tissue-bundle-v0.{1,2,3}.0.schema.json` |
| blender-addon | a bundle; a skin config written from the panel; `assets/channels/*.npz` | a `.blend` with the tissue and its views and any nanoscope scenes, a rendered frame for review |
| blender-addon `scripts/build_channel_meshes.py` | mmCIF from RCSB and OPM orientations, cached in `data/raw/pdb/` (gitignored) | `bioelectric_playback/assets/channels/<key>.npz` and `.json` (versioned, about 55 kB each; PDB data are CC0) |

## Artifact policy

| Versioned | Regenerated (gitignored) |
| --- | --- |
| run specs and configurations under `packages/*/configs/` (BETSE's default configuration and geometry images are regenerated by BETSE, not copied), the JSON schemas, `runs/*/result.json`, tests, the molecule meshes in `packages/blender-addon/bioelectric_playback/assets/channels/` (built from PDB data, CC0), the figures in `docs/figures/` | work directories, bundles and `.blend` files under `packages/*/data/work/`, the PDB download cache under `packages/blender-addon/data/raw/`, built extension zips under `packages/blender-addon/dist/` |

## Build order (as it happened)

1. 2026-10-07: the bundle package and its synthetic fixture; the add-on against it, tested headless; the BETSE adapter; the real bundle in Blender; the BETSE audit.
2. 2026-10-08: the prism epithelium and its audit; the skin anatomy, physics and audit; the wide block; the held-out prediction; the chloride physics and the rival analyses; the nanoscope and its molecule meshes.

Next, in `docs/plan.md` order: an instrument model for the field imager and the wound width as a parameter, then the held-out test restated in instrument terms; inference (Phase 3) and electrotaxis (Phase 4) stay gated.

## Compute

The default wound simulation (227 cells at seed 20261007, 24 × 25 grid, 350 steps) takes 1.6 s to seed, 1.9 s to init and 4.3 s to simulate on an Apple M4 Max; the whole audit, eight BETSE runs, takes under two minutes. The scaling run in `docs/plan.md` decides whether 10,000 cells are feasible.

The skin run (740 cells, 560 living, 36,162 extracellular nodes) takes 0.2 s per step with conjugate gradients preconditioned by classical (Ruge-Stüben) algebraic multigrid, about a minute for 2 s of init and 2 s after the cut. The wide block (4,492 cells, about 107,000 extracellular nodes on a graded grid) takes 0.5 s per step, about two minutes per run; smoothed aggregation needed about 100 iterations and a rebuild per step there (16 s per step). The audits run their simulations in parallel processes: the small block's in about 5 minutes, the wide block's in about 10.

## Running it

```
cd packages/tissue-bundle  && uv sync && uv run pytest
cd packages/betse-adapter  && uv sync && uv run pytest            # includes one real BETSE run
uv run betse-adapter run configs/wound-default.yaml               # -> data/work/wound-default/
uv run betse-adapter audit                                        # -> runs/*/result.json
cd packages/blender-addon  && uv sync && uv run pytest            # includes a headless Blender run
uv run python scripts/build_extension.py                          # -> dist/bioelectric_playback-0.1.0.zip
cd packages/epithelium-solver && uv sync && uv run pytest
uv run episolver skin-sim configs/skin-forearm.yaml               # -> data/work/skin-forearm/
uv run episolver skin-audit                                       # -> runs/skin/result.json
uv run episolver skin-audit --wide                                # -> runs/skin-wide/result.json
uv run episolver skin-amiloride                                   # -> runs/skin-amiloride/result.json (H1)
uv run episolver skin-audit --followup                            # -> runs/skin-wide-followup/ (L6, S6, S6b)
uv run episolver skin-rivals [--b2]                               # -> runs/skin-rivals[-b2]/ (R1-R4)
cd packages/blender-addon && uv run --group assets python scripts/build_channel_meshes.py
```

Blender is invoked by absolute path (`/Applications/Blender.app/Contents/MacOS/Blender`) or through `BLENDER_BIN`. Tests install the extension into a temporary `BLENDER_USER_RESOURCES`, never into the user's own Blender profile.
