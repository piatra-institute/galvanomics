# galvanomics: Plan

Phases with gates, written 2026-10-07. Only the computational phases are the institute's to run. Everything experimental is a gate on a partner laboratory.

Sequencing honesty: there is no laboratory and no electrophysiologist. The founding notes' twelve-week schedule assumed a full-time developer plus regular expert advice; it is not adopted. Phases are ordered, not dated.

## Phase 0: verify

Nothing is built on a claim until it is read at source. Checked 2026-10-07 (machine-checked, not human-confirmed).

| Claim from the founding notes | Verdict |
| --- | --- |
| BETSE 1.4.0 was released on 2026-09-24 with modernized Python dependencies | **Fails.** 1.4.0 and 1.4.1 were released 2024-09-24 (1.4.1 moved to NumPy ≥ 2, SciPy ≥ 1.14, Matplotlib ≥ 3.9). The latest release is 1.5.0, 2025-04-15; the repository (github.com/betsee/betse) has had no commits since 2025-04-16 |
| BETSE models ion channels, pumps, gap junctions, electrodiffusion and wound signals; its principal spatial model is 2D | **Holds** (Pietak and Levin 2016; the default configuration includes a wedge-shaped cut with tight-junction breakage) |
| BETSE's 2016 publication validates its wound simulation against experiment | **Partly.** Simulation 8 removes cells and breaks tight junctions and yields an outward wound current; the comparison is a range check (simulated 1 to 200 µA/cm² against 1 to 500 µA/cm² observed), not a fit (Front. Bioeng. Biotechnol. 4:55, doi:10.3389/fbioe.2016.00055) |
| CellBlender 4.2.0, released May 2026, supports Blender 5.1 and has a Python API | **Holds** (mcell.org download page, released 2026-05-08, bundled with Blender 5.1.0). Support for Blender 5.2 is not stated |
| Morpheus supports 2D/3D cell models with ODE/PDE coupling and has data-fitting tools | **Holds, with a caveat.** Morpheus 2.4.1, 2026-05-19; fitting is FitMultiCell over pyABC, whose PyPI release was last updated 2022-11 and was not checked against MorpheusML5 |
| Durant et al. 2017 showed persistent two-headed regeneration after transient bioelectric perturbation | **Holds** (Biophys. J. 112:2231, doi:10.1016/j.bpj.2017.04.011) |
| A paper of 2026-08-31 combines bistable V_mem, adaptive gap junctions, a synthetic GRN and slow tissue memory | **Holds** (Cortés-Poza, J. Math. Biol. 93:43, doi:10.1007/s00285-026-02459-2); theory only, no BETSE, no experimental validation |
| A 2008 study measured about 177 ± 14 mV/mm at mouse skin wound margins | **Holds** (Nuccitelli et al., Wound Repair Regen 16:432, doi:10.1111/j.1524-475X.2008.00389.x; "(61)" in the abstract is probably the sample count, not stated) |
| The first milestone is a sheet of about 10,000 coupled cells | **Contradicted by the notes themselves**, which later says to begin with a few hundred cells and scale only after benchmarking. Resolved in favour of a few hundred |
| Twelve weeks for the prototype | **Not adopted**; it assumed staff the project does not have |
| No Blender add-on for BETSE or bioelectric tissue visualisation exists | **Not found** (GitHub and extensions.blender.org searched; nearest are CellBlender and Goo). Not proven absent |

BETSE internals the exporter depends on, read in the installed 1.5.0 source and confirmed on a run of its default configuration:

| Fact | Where | Consequence |
| --- | --- | --- |
| The cutting event fires at simulation time zero (`event_cut_time = 0.0` hard-coded; the config key is commented out) | `science/parameters.py:687` | Every sim frame is post-wound. The pre-wound tissue exists only in the init phase, so the bundle carries the last init frame as a baseline |
| Removed cells and membranes are stored as `sim.target_inds_cell_o` / `target_inds_mem_o` in pre-cut numbering, and removal preserves order | `science/tissue/tishandler.py:1200` | Exact pre/post mapping: post-cut centres, membrane midpoints and polygons are bitwise equal to the kept pre-cut ones (checked) |
| Sampled frames are labelled from `np.linspace(0, n·dt, n)`, and stored after that step's update | `science/sim.py:2455`, `write2storage` | BETSE's reported time (1.00287 ms for the first default frame) differs from the integrated time ((i+1)·dt = 1.1 ms). The bundle stores both |
| Polygons are per cell (no shared vertices), counter-clockwise, in metres; membrane i of a cell is the edge from vertex i-1 to vertex i; membranes are numbered contiguously cell by cell | `science/cells.py` | One offsets array indexes both polygon vertices and membranes |
| Cell V_mem (`vm_ave`) is the plain mean of membrane V_mem | `science/sim.py` | Recomputable from membrane values as a consistency check (max difference 1.4e-17 V on the default run) |
| With extracellular spaces on, the environment is a regular grid of shape `cells.X.shape` (24 × 25 in the default run, not square), flattened in C order with rows along y | `science/cells.py` | Every quantity's location is declared in the bundle, never inferred from array length |
| BETSE never seeds NumPy's global random generator; lattice disorder uses it | whole package | Runs are made in-process after `np.random.seed(seed)`, and the seed is recorded |
| BETSE's GHK estimate (`vm_GHK_time`) includes channels defined in the molecule network, whose per-frame permeabilities are not stored | `science/sim_toolbox.py:526` | The exact GHK reading check runs on a configuration with the molecule network off |
| Licence: BSD 2-clause text in the distribution; the PyPI classifier says MIT | `betse-1.5.0.dist-info` | Permissive either way; recorded as an open item |

**Gate: no claim enters a document or a paper until it has a source and a date, and a claim that fails is recorded as failed rather than dropped.**

## Phase 1: reproduce and audit (in progress)

The founding notes' "first action": run a BETSE wound simulation, export geometry and the accessible time-dependent electrical quantities, and play them back in Blender; then audit the physics. **Done 2026-10-08** for BETSE's default wound configuration: the run is reproducible from a seed, exported as a validated bundle, played back and rendered in Blender 5.2, and audited below. The scaling run and the week-2 decision remain.

### Audit checks, fixed before running

Fixed 2026-10-07, before any audit run. Each run writes `packages/betse-adapter/runs/<name>/result.json` with the verdicts. A failed check is reported as failed, with its number.

| Id | Check | Configuration | Tolerance |
| --- | --- | --- | --- |
| C1 | Post-cut geometry identity: centres, membrane midpoints and polygons equal the kept pre-cut ones | wound-default | bitwise |
| C2 | Exported cell V_mem equals the plain mean of exported membrane V_mem | wound-default | ≤ 1e-12 V |
| C3 | Every polygon is counter-clockwise with positive area and as many vertices as membranes | wound-default | exact |
| C4 | Exported cell V_mem and centres equal BETSE's own `Vmem2D_*.csv` export, frame by frame. **An I/O check only**: both read the same array | wound-default | ≤ 1e-9 mV, ≤ 1e-9 µm |
| C5 | Grid orientation: grid points equal (X, Y) flattened in C order | wound-default | exact |
| D1 | Same seed twice gives bit-identical exported arrays | wound-default | exact |
| P1 | Our GHK voltage, computed from the bundle's concentrations and permeabilities, reproduces BETSE's `vm_GHK_time` per cell and frame. Proves only that the variables are read correctly | ghk-audit (molecule network off, GHK calculator on) | ≤ 1e-6 V |
| P2 | Passive limit: with the Na/K-ATPase blocked from the start of the sim and the molecule network off, each cell's V_mem at the last frame equals its GHK voltage | pump-blocked (sim dt 1e-3 s, 10 s) | ≤ 0.5 mV |
| P3 | Time-step convergence: same init, sim dt 1e-4, 5e-5 and 2.5e-5 s; max over surviving cells and frames of the V_mem difference between the 1e-4 and 5e-5 runs at matched sample index. Matched frames are offset by 5e-5 s of integrated time, so max |dV/dt| × 5e-5 s is reported beside it; the observed order is reported from the three runs | wound-default | ≤ 1 mV (provisional) |
| R1 | Electrogenic offset: mean and range of V_mem − V_GHK with the pump on, BETSE's own GHK | wound-default | reported, not gated |
| R2 | −∇(extracellular potential) by centred differences against BETSE's `efield_ecm` | wound-default | reported, not gated |
| R3 | Cross-version: a BETSE 1.3.0 wound run's V_mem exports (a 2024 reference run, not distributed; `BETSE13_REFERENCE_DIR`) against a 1.5.0 run of the same configuration | BETSE 1.3.0 reference run | reported, not gated |
| R4 | *Added after the first pump-blocked run, reported only:* the time course of mean V_mem and max \|V_mem − GHK\| | pump-blocked | reported, not gated |

### Results, 2026-10-08

Run with `uv run betse-adapter audit` (seed 20261007); records in `packages/betse-adapter/runs/*/result.json`. Every gated check passed. The numbers matter more than the verdicts, and two of them change what the next phase has to answer.

| Id | Verdict | Measured |
| --- | --- | --- |
| C1 | pass | centres, membrane midpoints, polygons and the membrane-to-grid map bitwise identical after the cut; 14 of 227 cells removed |
| C2 | pass | 1.4e-17 V |
| C3 | pass | 0 bad cells |
| C4 | pass | 0.0 mV and 0.0 µm over all 34 frames (I/O check only) |
| C5 | pass | exact |
| D1 | pass | no differing array across two full runs in separate processes |
| P1 | pass | 2.3e-17 V: the bundle carries everything BETSE's GHK calculator uses |
| P2 | pass | 0.0009 mV at 9.9 s |
| P3 | pass, with a caveat | 0.66 mV between dt 1e-4 and 5e-5 s; 0.34 mV between 5e-5 and 2.5e-5 s; observed order 0.97 (first order). The 5e-5 s offset between matched frames alone bounds the difference at 0.73 mV, so this check cannot separate time-step error from sampling offset. It passes its tolerance; it does not demonstrate convergence below about 1 mV |
| R1 | reported | default wound run: V_mem − BETSE's GHK ranges −74 to +47 mV (mean −7 mV), because BETSE's GHK includes the molecule network's Nav/Kv/K_Leak channels. With the network off: −24.0 mV in every cell, the electrogenic Na/K-ATPase offset |
| R2 | reported | −∇(v_env) correlates 0.9998 with BETSE's extracellular field but is about 9,000 times larger (42 V/m against 0.005 V/m at the last frame). See finding 1 |
| R3 | reported | BETSE 1.5.0 (Python 3.12, NumPy 2.5) on BETSE 1.3.0's seeded world reproduces 1.3.0's own sim V_mem (Python 3.8, NumPy 1.24) to 6.2e-9 mV over all 34 frames |
| R4 | reported | with the pump blocked, mean V_mem relaxes from −23.3 mV (pump-held) to +0.69 mV, reaching within 0.07 mV of GHK by 2 s and 0.001 mV by 9.9 s |

**Finding 1: BETSE's extracellular field is not the gradient of its extracellular potential.** In `science/physics/ion_current.py`, BETSE computes `E_env = −∇(screen · v_env)` with a Debye-screening factor `screen = 2/(ko_env·δ) · (cell_radius / true_cell_size)`, which is 1.10e-4 in the default run (`ko_env` = 1.24e9 m⁻¹, grid spacing δ = 7.35 µm). The relation was confirmed exactly with BETSE's own gradient routine. Its source comments describe `v_env` as "the voltage measured with electrodes". So the two extracellular observables BETSE exports disagree by four orders of magnitude about the field: 0.005 V/m (0.005 mV/mm) from `E_env`, 42 V/m (42 mV/mm) from −∇`v_env`. The reference measurement in Phase 0 is about 177 mV/mm at mouse skin wound margins in vivo. Which of BETSE's quantities a vibrating probe or field imager would report is now the first question of the representation gate. It is not settled here, and neither number is compared with the measurement until it is.

**Finding 2: in BETSE's default configuration, the V_mem pattern is not the wound's.** The visible depolarized patch is one of the default tissue profiles (a "spot" with its own channel set), present before the wound. Over the 35 ms simulated after the cut, the 11 live cells within 12 µm of the wound shift by 1.1 to 1.3 mV, while cells of the spot 34 to 51 µm away depolarize by up to 12.7 mV (to −25.8 mV). The default run demonstrates the machinery; it is not a wound-signal model, and a wound study needs its own configuration with the spot profiles removed and a time span long enough for the wound response.

**Finding 3: where the resting potential comes from.** Without the molecule network, BETSE's base membrane has a GHK voltage of about +0.7 mV and its −23 mV rest is held entirely by the electrogenic pump (R1, R4). The −45 mV rest of the default run comes from the molecule network's K_Leak and Kv channels. Any fitted model inherits this split, so it has to be stated in the model, not discovered in a residual.

**Also found:** BETSE 1.5.0 cannot resume a BETSE 1.3.0 init file when the molecule network is on (an `AttributeError` in the network's `tissue_init`), so R3 starts from the 1.3.0 world instead; and the reference run's 1.3.0 sim pickle does not decompress (truncated). Neither file was modified.

**Compute:** the default wound run takes 1.6 s to seed, 1.9 s to init and 4.3 s to simulate 35 ms (350 steps) for 227 cells; the 10 s pump-blocked run (10,000 steps) takes 21 s.

### Then, before the Phase 1 gate

- **Scaling run.** World sizes 150, 300 and 600 µm; wall time and peak memory per simulated second; projection to 10,000 cells. Not run yet.
- **A wound configuration that studies the wound**: the default's spot profiles removed, a simulated span long enough for the wound response, and the same audit rerun on it.
- **Finding 1 resolved at source**: what `v_env` and `E_env` each represent physically in BETSE, which one a probe measurement corresponds to, and whether the screening factor is a modelling choice to keep or a scaling to undo.
- **A tighter convergence check** that matches frames on integrated time, so P3 can say something below 1 mV.
- **The week-2 decision**, written here with its numbers: adapt BETSE directly, or build the smallest additional solver the planned experiments need.

**Gate: every check above has a verdict; BETSE counts as sufficient only if it runs on 1.5.0 under Python 3.12 on this machine, C1 to C5, D1 and P1 to P3 pass, the scaling projection reaches 10,000 cells in usable time, and it can represent the wound-edge extracellular field as a measurable quantity.**

**Kill criteria:** if P1 or P2 fails with no identifiable cause in BETSE's source, BETSE is not used as the reference engine. If BETSE's in-plane extracellular space cannot represent a transepithelial-driven wound field, the next artifact is a minimal solver with apical and basolateral compartments, not a fitted BETSE.

## Phase 1b: the 3D epithelium solver (in progress)

Started 2026-10-08, ahead of the week-2 decision: three-dimensional cells with apical and basolateral membranes between two baths, built from BETSE's flux law, pump, gap junctions and channel library, specified in `docs/epithelium.md`. It is the "minimal additional solver" the Phase 1 kill criterion names, built early; it does not yet replace BETSE as the reference engine, and BETSE is its first verifier.

### Checks, fixed before running

Fixed 2026-10-08, before the solver's first run. Records in `packages/epithelium-solver/runs/<name>/result.json`.

| Id | Check | Tolerance |
| --- | --- | --- |
| E1 | Passive limit: one cell, pump off, no channels, identical domains and baths; steady-state V_ap and V_bl equal the GHK voltage of the base permeabilities computed in closed form | ≤ 0.01 mV |
| E2 | Charge conservation, every step and cell: F·volume·Δ(Σ z c) equals the change of charge on the cell's membrane capacitors | ≤ 1e-6 of the charge that 1 mV puts on that cell's membranes |
| E3 | Kirchhoff residual of every linear solve, ‖Ax − b‖∞ / ‖b‖∞ | ≤ 1e-9 |
| E4 | BETSE cross-check, non-polarized limit: for every cell of the `ghk-audit` BETSE bundle's baseline frame, the zero-net-current voltage computed with episolver's flux and pump functions from BETSE's exported concentrations and diffusion constants equals BETSE's V_mem | ≤ 0.5 mV |
| E5 | Channel port: m∞, τ_m, h∞, τ_h of every ported channel, and the gap-junction rates, equal BETSE 1.5.0's at V = −120.5 to 59.5 mV in 1 mV steps (test in `betse-adapter`) | ≤ 1e-12 relative |
| E6 | Intact polarized hexagonal sheet: interior cells' TEP, V_ap and V_bl after init equal the root of the single-column zero-current equations (apical membrane, basolateral membrane, tight-junction shunt), solved independently of the time stepper | ≤ 0.05 mV |
| E7 | Time step: wound run at dt, dt/2 and dt/4, frames at identical integrated times; max difference of V_ap and V_bl between dt/2 and dt/4 | ≤ 0.1 mV; order reported |
| E8 | Bath grid: spacing Δ and Δ/2; peak apical-bath field near the wound, and the far-field TEP | ≤ 10% and ≤ 0.1 mV (provisional) |
| W1 | TEP of the intact sheet, polarized and non-polarized (BETSE-uniform membranes) | reported |
| W2 | Wound field in the apical bath: peak magnitude (mV/mm), direction relative to the wound, decay length | reported |
| W3 | The same wound with BETSE-uniform membranes | reported |

**Gate: E1 to E8 pass. Only then are W1 to W3 read as properties of the model, and even then they are model outputs with assumed tight-junction and bath parameters, not predictions of any tissue.**

### Results, 2026-10-08

Run with `uv run episolver audit`; record in `packages/epithelium-solver/runs/epithelium/result.json` (v2). The v1 record, before the wound-shunt change below, is kept in `runs/epithelium-v1-point-shunt/`. **The gate is not passed: E6 and E8 fail.** Denominator: one dry run of E1, E4 and E6 into a scratch folder before the first official run (E6 failed there too, which started the diagnosis below), two official runs (v1, v2), and two runs of E5.

| Id | Verdict | Measured |
| --- | --- | --- |
| E1 | pass | 0.0007 mV from the closed-form GHK voltage (0.653 mV) |
| E2 | pass | 3.3e-13 of 1 mV of membrane charge |
| E3 | pass | 1.2e-12 |
| E4 | pass | 0.19 mV: from BETSE's own exported baseline state, episolver's flux and pump functions give a zero-current voltage of −23.30 mV against BETSE's −23.32 mV, every cell within 0.19 mV |
| E5 | pass on the second run | the first run caught a porting error: BETSE's HCN2 shifts the voltage by −10 mV before its gating function, and the port had missed it. Fixed; all seven channels and the gap-junction gate now match BETSE to 1e-12. No audited run used HCN2 |
| E6 | **fail** | 0.22 mV against 0.05 mV. Cause found (E6b): the sheet (140 µm) is smaller than its electrical length constant (about 0.7 mm), so its TEP is set by the sheet-average junction density, and interior cells, which have more junctions than the average, pass current sideways to the edges. The premise of the check, a cell at its own column solution, does not hold for this sheet |
| E6b | reported, added after E6 failed | against the same column with the sheet-average junction density: 0.001 mV. Kirchhoff's law at every free node, written independently of the solver's assembly and evaluated at its final state: residual 1e-4 of a membrane current, which is the capacitive current of the slow concentration drift. The same residual holds with baths 0.001 µm thick, where any column picture breaks down |
| E7 | pass | 0.00016 mV between dt/2 and dt/4; observed order 0.99 (backward Euler is first order) |
| E8 | **fail**, improved | v1: the peak near-wound field changed by 34% between 5 and 2.5 µm grids because each removed cell shorted the baths at one point. v2 spreads the short over the wound's footprint: 11.3% against 10%. The far-field TEP changes by 0.0001 mV. Field values below carry about 12% grid uncertainty at the 5 µm default |
| W1 | reported | intact TEP −2.69 mV (apical negative) for the polarized sheet; 0.00001 mV with BETSE's uniform membranes |
| W2 | reported | the wound short-circuits the whole sheet (TEP −2.69 → −0.03 mV), because the sheet is smaller than its length constant. Field near the wound edge: in the basal compartment it points **toward** the wound, peaking at 0.73 mV/mm and falling from 0.36 to 0.12 mV/mm over 70 µm; in the apical bath it points away, peaking at 0.12 mV/mm |
| W3 | reported | with BETSE's uniform membranes: TEP 0.00001 mV, fields below 0.0001 mV/mm. Without apical–basolateral asymmetry there is no wound field, which is the representation question of Phase 1 answered in the model |

What this does and does not show. The solver reproduces BETSE's resting physics where the two should agree (E4, E5) and is internally exact (E1 to E3, E6b, E7). The wound-field numbers are outputs of assumed polarity, tight-junction resistance and bath geometry, chosen to give a TEP of a few millivolts; the reference measurement (about 177 mV/mm in mouse skin, Phase 0) is 250 times the basal peak here, and skin has a resistive dermis instead of a thin saline layer (its TEP is commonly described as tens of millivolts; not yet checked at source, see open items). Nothing is compared with it until the parameters come from a measured tissue.

Next for Phase 1b: pass E8 (finer grid or a converged ring-averaged field), a sheet larger than its length constant so that wounds are local, and parameters taken from a published epithelium with measured TEP and resistance.

## Phase 2: the editor

**Started 2026-10-08** with the anatomy: a procedural skin model (stratified epidermis of real 3D cells over a dermis with papillae, with an incision), viewable in Blender, specified in `docs/skin.md`. Dimensions are from sources (human dorsal forearm) where found, and marked where not. Physics on it ran the same day and was audited (below); in Blender it shows channel, pump and ion populations, a section plane and a probe, and can be re-run from the panel.

### Skin physics checks, fixed before running

Fixed 2026-10-08, before `episolver/skin_physics.py` first ran. Records in `packages/epithelium-solver/runs/skin/result.json`.

| Id | Check | Tolerance |
| --- | --- | --- |
| K1 | Kirchhoff residual of every solve | ≤ 1e-9 |
| K2 | Charge conservation per cell and step (as E2) | ≤ 1e-6 of 1 mV of membrane charge |
| K3 | Passive limit: all channels and pumps off, no wound; every living membrane face settles at the GHK voltage of the base permeabilities | ≤ 0.01 mV |
| K4 | Time step: all steps halved; max difference of cell V_mem at matched times | ≤ 0.1 mV |
| K5 | ECS grid: spacing 4 vs 3 µm; the battery voltage (ECS potential below minus above the tight-junction band, far from the wound) and the peak lateral field in the viable epidermis near the wound | ≤ 5% and ≤ 15% (provisional) |
| S1 | Battery voltage of the intact skin, against the 10–60 mV (inside positive) reported for human skin | reported; the apical Na and barrier parameters are assumptions, so agreement would be calibration, not prediction |
| S2 | Wound field in the viable epidermis and beneath the stratum corneum: peak, direction, decay; against 107 ± 13 mV/mm (young human forearm) and 177 ± 14 mV/mm (mouse) | reported, not comparable while the block (160 µm) is smaller than the field's space constant (0.3–0.4 mm) |
| S3 | The same skin without apical Na channels | reported |

### Skin physics results (2026-10-08)

Run on the default forearm skin (740 cells, 560 living, 6921 membrane faces) after one correction to the anatomy made before the audit: the papilla field had twice the sourced density (`docs/skin.md`). Machine-checked, not human-confirmed. Record: `packages/epithelium-solver/runs/skin/result.json` (`episolver skin-audit`).

| Id | Verdict | Value and reading |
| --- | --- | --- |
| K1 | pass | 7e-13 |
| K2 | pass | 7e-11 |
| K3 | **fail** | 0.036 mV against 0.01 mV, after 30 s with channels and pumps off. The premise does not hold for coupled cells: without pumps, cells of different surface-to-volume ratio drift apart in concentration, and gap junctions then hold each cell away from its own GHK voltage. A design check on the small test skin, run before this audit, showed the gap growing with time with gap junctions on and not with them off; K3 was nevertheless run as written |
| K3b | reported, added before the run (see K3) | the same with gap junctions off as well: 0.0014 mV on every layer, the lag of V_mem behind the slowly drifting GHK voltage (about 2 µV/s times the membrane time constant) |
| K4 | pass | 0.084 mV over 41 frames |
| K5 | **fail** | battery 20.89 vs 20.50 mV (1.9%, within 5%); peak lateral field in the viable epidermis 20.4 vs 29.8 V/m (31%, against 15%). Post hoc: the peak is about seven times the band-mean field and comes from grid-scale structure around individual membrane faces, not from the lateral wound field; the band means 20–50 µm from the wound centre change by 8–14% between grids. The peak was the wrong statistic to pre-register; a ring- or band-averaged field is the candidate for the next round, with its tolerance fixed before it runs |
| S1 | reported | intact battery 20.9 mV, inside positive, inside the 10–60 mV reported for human skin **by calibration** (apical Na density and barrier resistance). 2 s after the cut: 2.9 mV, measured more than 50 µm from the wound: the cut short-circuits the whole 160 µm block, as the space constant predicts |
| S2 | reported | 2 s after the cut. Viable epidermis: the field points **toward** the wound, band means 3.2, 2.4, 1.8, 1.0 mV/mm at 20–30, 30–40, 40–50, 50–60 µm from the wound centre. Beneath the stratum corneum: it points **away** from the wound, band means 77, 71, 55, 40, 25 mV/mm over the same bands and 60–70 µm, peak 92 mV/mm. Opposite directions are what a short between the two sides of the barrier gives. These are of the order of the measured 107 ± 13 mV/mm, but the block is smaller than the space constant and the battery is calibrated, so the agreement means nothing yet; the direction relative to the measurement's sign convention has not been checked at source |
| S3 | reported | without apical Na channels: battery 1e-5 mV, fields below 1e-4 mV/mm. As in W3, no apical–basolateral asymmetry, no battery, no wound field |

Resting voltages (frame 0): basal −73.3 mV, spinous −72.9 mV (range −73.3 to −71.8), granular −8.5 mV mean (range −69.2 to +35.1: the SG1 cells, which carry only apical membrane, depolarize). The wound changes cell voltages by under 2 mV within 2 s; what it changes is the extracellular potential.

What this shows. The solver is internally exact (K1, K2) and converged in time (K4); the battery is grid-converged (K5, first half). Two checks failed for reasons found and recorded (K3: premise; K5: statistic), not for solver errors as far as the added diagnostics show. Nothing here is a prediction: the battery is calibrated and the block is too small for the wound field to be compared.

Next for the skin: a block about 1 mm wide (coarser extracellular grid away from the cut), a band-averaged field check with its tolerance fixed first, and calibrated parameters replaced by measured ones where a source exists.

### Wide skin block and held-out prediction: checks fixed before running

Fixed 2026-10-08, before any wide-block run and before the held-out prediction is computed. The wide block is a half domain, 960 µm from the wound centre (a mirror plane) to the far wall, so it stands for a block about 1.9 mm wide; the extracellular grid is graded in x, fine near the wound. Band means are volume-weighted means of the lateral field E_x (positive toward the wound) over the nodes of a region, in bands of distance from the wound centre. Records in `packages/epithelium-solver/runs/skin-wide/result.json` and `runs/skin-amiloride/result.json`.

| Id | Check | Tolerance |
| --- | --- | --- |
| L1 | Locality: battery more than 800 µm from the wound, 2 s after the cut, against the intact battery | within 5% |
| L2 | Domain size: band means beneath the stratum corneum and in the viable epidermis, bands 20–50, 50–100, 100–200, 200–400 µm, at half-width 960 vs 640 µm | within 5% |
| L3 | Grid: the same band means, fine spacing 4 vs 3 µm | within 10%, the E8 standard. Stated because it was seen: on the small block (K5, post hoc) the 20–50 µm band means changed by 8–14% between these spacings |
| L4 | Conservation and time step on the wide block, as K1, K2 and K4 | as K1, K2, K4 |
| L5 | Mirror: half domain against a full block with the wound centred, at half-width 320 µm | reported |
| S4 | Model space constant, from an exponential fit to the potential beneath the stratum corneum against distance from the wound, against 0.3–0.4 mm (guinea pig, Barker 1982) | reported |
| S5 | Field beneath the stratum corneum near the wound against 107 ± 13 mV/mm (young human forearm, Nuccitelli 2011) | reported; the battery is calibrated |
| H1 | Held out: the reduction of the wound field beneath the stratum corneum (band 20–100 µm) when the apical Na channels are blocked by the fraction that the amiloride dose and potency imply, against 64 ± 7% (mouse, Nuccitelli 2008). The model is calibrated only on the intact battery; the block fraction, the configuration and its SHA-256 are written here before the run | pass if within 64 ± 14% (two of the reported ± 7); the reduction over block fractions 0–100% is reported with it. Interpretation fixed now: a predicted reduction of 90% or more at the dose-derived block means the Na-only battery over-predicts, so either another current (for example Cl⁻ secretion) carries part of the field or the block is partial in vivo |

**H1 specifics, fixed 2026-10-08 before the run** (sources read in Nuccitelli 2008, PMC3086402, and Schild 1997, PMC2217053; verdicts in `docs/skin.md`):
- The measurement: 1 mM amiloride in PBS applied topically to the wound; field change −64 ± 7% (mean ± SEM, 11 wounds in 9 mice, Table 1); a PBS control changed it by +22 ± 20% (3 wounds). The text says 68% on average; Table 1 is used. Timing after wounding and absolute fields before and after are not reported.
- Block fraction: 1 − K_i / (K_i + [A]) with K_i = 42 nM (rat αβγ ENaC, oocytes) and [A] = 1 mM: **0.99996**. At 1 mM amiloride also blocks NHE1, ASICs and NCX, which the model does not contain; this is noted, not modelled.
- Where the block acts is unknown (no penetration data). Primary prediction: the block on every apical Na channel of the block. Reported with it, not judged: the block confined to cells whose centroid lies within 50, 100 and 200 µm of the wound centre, and uniform blocks of 25, 50, 75 and 90%.
- Model: a mouse skin preset (`configs/skin-mouse-wide.yaml`) built from the sourced mouse anatomy, with the forearm membrane parameters unchanged: no recalibration to a mouse battery (the one sourced mouse value, about 6 mV, is ex vivo in hairless skin). Measured: the band mean beneath the stratum corneum 20–100 µm from the wound centre, 2 s after the cut, with and without the block; reduction = 1 − blocked / unblocked.
- Verdict rule as above (pass within 64 ± 14%). Configuration SHA-256 (resolved primary configuration, written to the record before the runs): `75ed0a938892cbedbe95e03c71bd66dd84ad1bbd96341c87ec21cb919e7439e7`.

### Wide block and held-out prediction: results (2026-10-08)

Machine-checked, not human-confirmed. Records: `packages/epithelium-solver/runs/skin-wide/result.json` (`episolver skin-audit --wide`) and `runs/skin-amiloride/result.json` (`episolver skin-amiloride`). Before these runs the model changed in four ways, all recorded in `docs/skin.md`: granular cells became Kelvin cells, wall faces stopped being membrane (mirror walls), the linear solver became classical algebraic multigrid (wide block 16 s → 0.5 s per step), and the far-field x spacing was set to 10 µm (a numerical choice made from the intact battery, before any check ran). The small-block audit was re-run on the changed model (`runs/skin`; the first record is kept in `runs/skin-v1-row-granular`): K1, K2, K4 pass (K4 at 0.095 mV), K3 and K5 fail as before; battery 22.6 mV.

| Id | Verdict | Value and reading |
| --- | --- | --- |
| L1 | pass | battery beyond 800 µm: 24.06 mV intact, 23.80 mV 2 s after the cut (1.1%). The wound's effect is local |
| L2 | **fail** | band means at half-width 640 vs 960 µm differ by 2–17% beneath the stratum corneum and by 5–110% in the viable epidermis (whose means beyond 100 µm are near zero, about 0.1–0.25 mV/mm, so relative changes are large). Post hoc (L2b): the same 960 µm block with two other generator seeds spreads by 5–9% (20–200 µm) and 20% (200–400 µm) beneath the stratum corneum, as much as L2's differences; changing the width re-randomizes every cell, so the check as designed cannot separate domain size from cell arrangement. With a space constant of about 0.1 mm (S4) the wound's effect at 640 µm is under 1%. Next round: compare with identical anatomy in the shared region, or average over seeds |
| L3 | pass | 3 vs 4 µm: every band within 5.3% except the viable-epidermis 50–100 µm band (7.4%), within 10% |
| L4 | pass | Kirchhoff residual 7e-12, charge 1e-9, halved time step 0.086 mV; at most 27 conjugate-gradient iterations per solve |
| L5 | reported | half domain at 320 µm against a full 640 µm block with the wound centred: 0.5–8% beneath the stratum corneum, within the seed spread |
| S4 | reported | model space constant 106 µm (fit 50–350 µm) against 0.3–0.4 mm in guinea pig: the model's field falls off about three times faster than measured |
| S5 | reported | beneath the stratum corneum the field points away from the wound: −172 mV/mm at 20–50 µm, −124 at 50–100, −72 at 100–200, −21 at 200–400 µm; in the viable epidermis it points toward the wound, 2.5 mV/mm at 20–50 µm. The 107 ± 13 mV/mm measured at human forearm wounds is of the same order; the battery is calibrated and the space constant is short, so the agreement is not a prediction, and the sign convention of the measurement is not yet checked |
| **H1** | **fail** | **predicted reduction 97% against the measured 64 ± 7%** (field 20–100 µm beneath the stratum corneum: −144.7 mV/mm without the block, −4.1 mV/mm with it). By the interpretation fixed before the run, the Na-only battery over-predicts. Reported variants: a uniform block must reach about 92% of the apical Na channels to give 64% (25% → 8%, 50% → 20%, 75% → 38%, 90% → 59%), which 1 mM amiloride, 24,000 times its K_i, exceeds wherever it reaches; a block confined to cells within 200 µm of the wound gives 71% (inside the measured range), within 100 µm 25%, within 50 µm 2% |

What H1 shows. Calibrated only on the intact battery, the model's Na-only battery cannot reproduce the measured amiloride effect if the drug reaches the whole field-generating region. Two explanations remain and the model separates them: limited penetration (a block reaching about 200 µm from the wound edge reproduces the measurement) or a second current, such as the Cl⁻ efflux the authors propose, that amiloride does not touch. The discriminating experiments are a measurement of amiloride penetration around the wound, or a Cl⁻ channel blocker alone and with amiloride. In the mouse preset (forearm membranes, not recalibrated) the intact battery is 21.3 mV.

### Follow-ups to the wide audit: checks fixed before running (2026-10-08)

| Id | Check | Tolerance |
| --- | --- | --- |
| L6 | Domain size with the anatomy held fixed (replaces L2): the 960 µm half-block against a 640 µm one whose cells come from the same seeds (the seed lattice is generated for 960 µm and cut at 640 µm, so cells away from the new wall are identical); band means as in L2, 2 s after the cut | each band within 5%, or within 0.1 mV/mm where the band mean is near zero (stated because L2's viable-epidermis bands were seen to be near zero) |
| S6 | Space constant sensitivity: the S4 fit with the barrier resistance halved and doubled, the conductivity of the viable epidermis' extracellular space doubled, and the dermis conductivity doubled; with a one-dimensional cable estimate for comparison | reported |

**Results (2026-10-08).** Record: `packages/epithelium-solver/runs/skin-wide-followup/result.json` (`episolver skin-audit --followup`). Machine-checked, not human-confirmed.

| Id | Verdict | Value and reading |
| --- | --- | --- |
| L6 | pass | with the cells held fixed, the 640 and 960 µm half-blocks agree within 1.7% beneath the stratum corneum and within 0.003 mV/mm in the viable epidermis: the domain is wide enough, and L2 failed because changing the width re-randomized the cells |
| S6 | reported | exponential fit of the potential beneath the stratum corneum: 106 µm (baseline), 271 (barrier halved), 134 (barrier doubled), 148 (extracellular conductivity doubled), 106 (dermis conductivity doubled); one-dimensional cable estimates 125, 88, 176, 176, 125 µm. The halved-barrier value contradicts the cable estimate: its profile is not monotonic, because the static variation of the intact potential with the cell arrangement (about 3%) is comparable to the wound's effect far from the wound when the battery is weak |
| S6b | reported, post hoc | the same fit applied to what the wound changed (after minus before, same nodes): 107, 81, 135, 147, 107 µm. Now every change goes the way the cable estimate does, 15–25% below it. The dermis is not limiting; the space above the tight junctions is. To reach the 0.3–0.4 mm measured in guinea pig, that space would have to conduct about ten times better relative to the barrier than the modelling conventions assume (for example a thicker or more conductive layer beneath the stratum corneum), which would also raise the field |

### Rival mechanisms after H1: fixed before running (2026-10-08)

H1 left two explanations of the partial amiloride effect. Each is built as a model calibrated on the same two numbers: the intact battery T = 21.3 mV (the mouse preset in H1) and the measured 64% amiloride reduction. Both models then fit the existing data equally well, and the analysis asks which intervention would separate them.

Common basis:
- the mouse wide preset;
- BETSE's mammal ion profile without Ca²⁺, so Cl⁻ is present, with the keratinocyte [Cl⁻]i of 6.8 mM (Yamanoi 2023; the generic anion rebalances charge);
- the forearm membranes otherwise, and the H1 measure (band mean beneath the stratum corneum 20–100 µm from the wound centre, 2 s after the cut).

Chosen after the chloride sources (`docs/skin.md`) and before any run of either rival:

| Rival | Mechanism | Calibrated (to T, then to 64%) |
| --- | --- | --- |
| A | Na-only battery; drugs reach only cells within distance d of the wound centre | apical Na density; d |
| B | Na battery plus Ca²⁺-activated Cl⁻ channels (ANO1-like, BETSE's ClLeak) switched on at wounding on every living membrane face that touches the wound fluid; drugs reach everything | apical Na density; wound Cl⁻ channel density |

Not modelled as a rival: apical Cl⁻ secretion, for lack of a documented loader and against the measured [Cl⁻]i.

| Id | What | Rule |
| --- | --- | --- |
| R1 | Calibration of each rival | battery within 1% of T and reduction within 64 ± 1 percentage points; a rival that cannot reach this within bounds (A: d up to 960 µm; B: density up to the equivalent of 1,000 ANO1 channels per µm² of exposed membrane at 2.6 pS) fails to fit and is recorded so. The fitted values are reported with their plausibility: A's d against how far 1 mM amiloride could diffuse in minutes; B's density as ANO1 channels per µm² |
| R2 | Frozen predictions, each configuration hashed before any comparison: I1 a Cl⁻ channel blocker on the wound (ANO1 inhibitor, full block of the wound Cl⁻ channels; in A it has no target); I2 amiloride plus the Cl⁻ blocker; I3 the amiloride reduction of the far band, 200–400 µm, against the near band's 64%; I4 PGE2 as a doubling of the wound Cl⁻ conductance (no target in A) | reported |
| R3 | Discriminating design: for each intervention, the separation of the two predictions divided by the per-wound standard deviation of a percent change, 0.07 × √11 = 0.23 (the amiloride arm of Nuccitelli's Table 1), and the wounds per arm for a two-sided 5% test at 80% power, n = 2 (1.96 + 0.84)² σ² / Δ²; the largest separation is the recommended experiment | reported |
| R4 | I4 against the existing PGE2 measurement (+82 ± 21%, vehicle +22 ± 20%): the sign of each rival's prediction | reported only: the vehicle effect and the cAMP insensitivity of keratinocyte Cl⁻ transport make this weak |

**Results (2026-10-08).** Record: `packages/epithelium-solver/runs/skin-rivals/result.json` (`episolver skin-rivals`). Machine-checked, not human-confirmed.

| Id | Verdict | Value and reading |
| --- | --- | --- |
| R1 | **fail** | Shared apical Na density 2.98e-16 m²/s gives the intact battery T = 21.28 mV for both rivals. **A**: amiloride reaching 177 µm from the wound centre gives a 65.2% reduction, 0.2 points outside the tolerance, because the calibration interpolated once between grid points (150 µm: 55%, 200 µm: 71%) instead of iterating. **B cannot fit**: wound Cl⁻ channels move the amiloride reduction only from 97.1% to 97.3% across a 300-fold range of density (up to about 330 ANO1 channels per µm² of exposed membrane, inside the bound). The cause is physical: the model's basal and spinous keratinocytes rest at −73 mV, almost at the Cl⁻ equilibrium of −75.5 mV with 6.8 mM inside, so a Cl⁻ conductance carries almost no current |
| R2 | reported | A's predictions: a Cl⁻ blocker alone 0% (no target); amiloride with the Cl⁻ blocker 65%; PGE2 0%; and, under amiloride, **the field 200–400 µm from the wound rises by 165%** while the near band falls by 65%: a block confined near the wound moves the field outward rather than removing it. B's predictions were computed at a density that does not fit and mean nothing |
| R3 | reported | The spatial signature (I3) separates A from any uniform block by 11 per-wound SDs, about one wound per arm: a spatial scan of the field before and after amiloride, which the field imager already provides, tests the penetration explanation directly. The other interventions only separate A from a fitted B, and B did not fit |
| R4 | reported | no fitted rival to compare with the PGE2 measurement |

What R1–R4 show. Calibrated on the battery and the amiloride effect, the penetration explanation survives and predicts something measurable: under amiloride the far field rises. The chloride explanation, as modelled, does not survive. That conclusion rests on keratinocytes resting at the Cl⁻ equilibrium, which is the model's BETSE default, not a keratinocyte measurement; cultured keratinocytes rest at −24 to −40 mV (Yamanoi 2023).

### Rival B2 (depolarized keratinocytes): fixed before running (2026-10-08)

Added after B failed R1 for the reason above, before any run of B2. Same as A and B, with the living keratinocytes depolarized to the cultured range:
- **Depolarization.** A non-selective Na⁺ leak (the base Na⁺ permeability on every living face) is raised until the mean intact voltage of basal and spinous cells is −32 mV, the midpoint of −24 to −40 mV in cultured NHEK (Yamanoi 2023). Using a Na⁺ leak as the lever is an assumption; keratinocytes do carry non-selective cation channels such as TRPV3. The apical Na density is then recalibrated to T.
- **A2 and B2.** As A and B on these cells. A2's d and B2's wound Cl⁻ density are calibrated to 64%, now iterating (secant steps from the grid crossing) until within tolerance or for at most four steps.
- **Rules.** R1–R4 as before, recorded as R1b–R4b. Interpretation fixed now: if B2 fits, the chloride explanation is viable only if keratinocytes in situ are depolarized, and their resting voltage near a wound becomes a discriminating measurement in itself; if B2 does not fit either, wound Cl⁻ channels cannot explain the partial amiloride effect in this model.

**B2 results (2026-10-08).** Record: `packages/epithelium-solver/runs/skin-rivals-b2/result.json` (`episolver skin-rivals --b2`). Machine-checked, not human-confirmed.

| Id | Verdict | Value and reading |
| --- | --- | --- |
| R1b | **fail** | A non-selective Na⁺ leak 7.3 times the base permeability puts basal and spinous cells at −32.0 mV; apical Na 4.72e-16 m²/s restores T (21.29 mV). **A2 fits**: amiloride reaching 171 µm gives 63.8% (one secant step). **B2 cannot fit**: the amiloride reduction stays at 97.3% across densities from 1e-16 to 3e-13 m²/s, although the wound Cl⁻ channels do change the field (beneath the stratum corneum, 20–50 µm: −182 → −270 mV/mm at the highest density) |
| R2b | reported | A2: the same signature as A, the field 200–400 µm away rises by 156% under amiloride |
| R3b, R4b | not computed | no fitted B2 |

Why B2 fails (diagnostic, post hoc). Without the drug, granular cells near the wound sit at −1.7 mV, depolarized by their own ENaC entry, and they carry most of the wound Cl⁻ influx. Under amiloride they fall to −36 mV and their Cl⁻ influx drops tenfold. The Cl⁻ influx of basal and spinous cells is unchanged, but that current closes through the viable epidermis, not beneath the stratum corneum. In this model the wound Cl⁻ current depends on ENaC, so it cannot be the amiloride-resistant part of the field.

### The instrument: finding after R1–R4 (2026-10-08, post hoc)

The field imager of Nuccitelli 2008 reads the surface potential through a flat probe 320 × 700 µm, about 150 µm above the skin, and reports a field from the potential difference between scan positions. The model's band means are point fields. Rival A's bundles, with the potential beneath the stratum corneum averaged over a probe width (boxcar in x, mirrored at the wound centre) and then differentiated, give:

| Probe width | 20–100 µm, no drug → amiloride | 200–400 µm |
| --- | --- | --- |
| point | −138 → −48 mV/mm (−65%) | −14 → −38 (+165%) |
| 100 µm | −103 → −40 (−61%) | −16 → −41 (+160%) |
| 320 µm | −16 → −26 (+58%) | −23 → −41 (+79%) |
| 700 µm | −1 → −2 | −19 → −26 (+34%) |

Consequences:
1. **H1 compared unlike quantities**: a point field of the model against a probe-averaged field. With the imager's probe and the model's 40 µm incision, the model would report about 16 mV/mm near the wound, not 177. The experiments' scalpel wounds (1–3 mm long, width not reported) must open far wider than the model's cut. H1's verdict stands as recorded, with this caveat: it is not yet known what the model predicts for the instrument.
2. **Rival A's spatial signature is finer than the imager's probe.** It is detectable as stated only with a point-like probe (a vibrating probe, tens of µm), or as a forward-modelled imager scan.
3. **Next, before any further comparison with these measurements:** an instrument model (probe footprint, standoff, scan spacing) and the wound width as a parameter, with H1 restated in instrument terms and pre-registered again.

### Nanoscope (2026-10-08)

A molecular-scale view of one membrane face, specified in `docs/nanoscope.md`:
- **Molecules:** six structures from the Protein Data Bank, built into meshes in the extension: ENaC, TWIK-1, Kv3.1 open, Shaker closed, Na/K-ATPase and a Cx43 hemichannel.
- **Counts:** derived from the solver's per-face conductances and sourced single-channel conductances.
- **Positions:** assumed.
- **States:** drawn each frame at the solver's open fraction.
- **Check N1** (unit test, fixed with the code): sampled open fractions within 3/√N of the solver's at every frame. Passes.
- **Kv3.4** was ported from BETSE and matches it (E5 extended). It sits in a separate demonstration configuration, because no voltage-gated current has been recorded in human keratinocytes.

### Next for Phase 2

1. A rival mechanism to the Na-only battery, both calibrated on the intact battery only: Cl⁻ secretion. Then the intervention that separates them, with the size of the difference relative to measurement noise: the project's thesis, for the first time on a real measurement.
2. Why the model's space constant (106 µm) is three times shorter than measured.
3. A domain-size check with fixed cells in the shared region (replacing L2).
4. The editor gate below.

Blender as the tissue editor: draw tissue outlines and wounds, paint channel and gap-junction densities, and write them as the image masks BETSE already accepts for tissue and cut profiles; launch runs in the pinned external environment; inspect any cell's time series.

**Gate: a wound drawn in Blender produces a BETSE run whose bundle round-trips back into Blender, and the configuration that produced it is hashed into the bundle.**

## Phase 3: inference on synthetic data

Parameter fitting, sensitivity and identifiability on simulated measurements first: generate data from known parameters, add noise at the precision of a real instrument, and report which parameters are recoverable. Then the discriminating-intervention rule: for two rival mechanisms that fit the baseline equally well, the intervention that maximizes their predicted difference relative to measurement noise.

**Gate: at least one parameter set is shown recoverable and at least one shown unidentifiable from a stated measurement design, with the denominator of designs tried.**

## Phase 4: electrotaxis

A constrained cell-migration response to the simulated extracellular field, against a prespecified comparator with no bioelectric input.

**Gate: the comparator is fixed before the coupled model is fitted.**

## Gated, not planned

| Track | Hard gate |
| --- | --- |
| Epithelial wound assay: geometry, extracellular field, migration, optionally calibrated optical V_mem | a named partner laboratory with field measurement, a signed protocol, independent replicates |
| Held-out intervention predictions (two or more ion-channel or gap-junction interventions) | Phase 3 complete; predictions frozen and hashed before the intervention data are seen |
| Bioelectric pattern memory (bistability, state switching) | a validated tissue model; compared against published planarian work before any new design |
| Dental extension | validated tissue electrophysiology and experimentally supported dental signalling; CBCT gives anatomy, not cell-resolution geometry or V_mem |

## Open items

- Licences: the Blender extension must be GPL-3.0-or-later; the bundle and adapter packages are undecided. BETSE ships BSD 2-clause text with an MIT classifier.
- The name: *galvanomics*, chosen 2026-10-08 (etymology and collision check in `README.md`); the folder is renamed from `blender-bioelectrics`. Check collisions again before any public use.
- The destination paper's working title and slug.
- The Blender add-on's interactive pieces (sidebar panel, quantity menu, probe and selected-cell plots, section plane) are tested headless for import, playback and rendering, and were checked by screenshots of the Blender interface on 2026-10-08; no one has yet used them by hand for an analysis.
- Skin TEP: the human 10–60 mV range is known only through secondary quotes of Foulds & Barker 1983 (per-site values unread); epidermal resistance without the stratum corneum and the epidermal extracellular volume fraction were not found (`docs/skin.md`). Cultured-epithelium TEP and resistance for the prism model are still unsourced.
- BETSE 1.3.0 pickles of the reference run: the world and init files load under 1.5.0 and NumPy 2; the sim file is truncated; resuming the init fails with the molecule network on (see Results).
