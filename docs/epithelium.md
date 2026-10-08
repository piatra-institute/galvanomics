# galvanomics: the 3D epithelium model

Written 2026-10-08, before the solver's first run; the wound shunt was revised the same day after check E8 (see `docs/plan.md`, Phase 1b results). It specifies the minimal additional solver that `docs/plan.md` anticipated: a polarized epithelium of three-dimensional cells between two baths, built from BETSE's physics so that its non-polarized limit can be checked against BETSE. Code: `packages/epithelium-solver` (`episolver`).

## Why it exists

BETSE's tissue is a single layer in the plane, with extracellular space in that plane. A real epithelium separates two compartments: its cells have an apical membrane facing one bath and a basolateral membrane facing the other, and tight junctions seal the paracellular path between them. Asymmetric transport across the two membranes produces a transepithelial potential (TEP); a wound short-circuits it, and the current returning through the bath is the wound field that probes measure. None of that can be represented in BETSE (Phase 1, Finding 1 and the representation gate).

## Geometry

- **Footprint**: a 2D cell layer, either generated (a hexagonal sheet) or taken from a BETSE run bundle (same cells, same numbering, same wound). Each footprint edge is a lateral membrane; its facing membrane in the neighbour cell is recorded as `mem_neighbour` (-1 on the sheet's edge).
- **Cells are prisms** of height h (default 10 µm, BETSE's `cell_height`): apical face (area A = footprint area) at the top, basal face at the bottom, one lateral face per footprint edge (area = edge length × h). Volume = A·h.
- **Membrane domains**: apical (top face) and basolateral (lateral plus basal faces). Each domain has its own permeabilities, channels and pump density.
- **Baths**: two thin conductive layers, apical (above) and basal (below), each discretized on the same regular grid covering the footprint plus a margin. Bath conductivity σ is computed from the bath composition, σ = F²/(RT) Σ z² D c.

## Electrical network

Unknowns at each time step: the intracellular potential φᵢ of every live cell and the potential of every apical and basal bath node. Membrane potentials are differences: V_ap = φᵢ − φ_a(i), V_bl = φᵢ − φ_b(i), where φ_a(i) and φ_b(i) are the bath potentials under the cell centre (bilinear weights).

- **Membrane ion fluxes** follow BETSE's `electroflux` (the Goldman flux equation) with permeability D/t_m per ion and domain, positive inward. Current density is Σ z F J.
- **Channels** add to the base diffusion constant: D_s = D_s,base + Σ_c D_c,max · P_open,c · rel_perm_c,s, with P_open = m^p h^q. The gating functions are ported verbatim from BETSE 1.5.0 (`science/channels/`), with BETSE's semi-implicit gate update in milliseconds. Ported: KLeak, NaLeak, ClLeak, Kir2p1, Kv1p5, Nav1p3, HCN2 (Na:K = 0.2:1; BETSE's Ca term is dropped because the ion set has no Ca).
- **Na/K-ATPase**: BETSE's `pumpNaKATP` verbatim (3 Na⁺ out, 2 K⁺ in, thermodynamic back-reaction, BETSE's Km values and ATP/ADP/Pi), with a rate constant per domain (default: basolateral only, as in real epithelia).
- **Gap junctions** on lateral faces shared by two live cells: BETSE's form, electroflux between the two cytoplasms with D = D_free · gj_surface · gj_open over the pore length `cell_space` (26 nm), and BETSE's kinetic voltage gate (`Gap_Junction`: λ, A₁, A₂, threshold, minimum).
- **Tight junctions**: a paracellular conductance per unit junction length, between the apical and basal bath nodes at each junction, set from a stated tissue resistance. Ohmic; baths are clamped, so no ion bookkeeping is needed there.
- **Wound**: removed cells are deleted at t = 0 (as in BETSE). Each one's footprint becomes a vertical shunt σ·A/h between the baths, spread evenly over the bath nodes inside the footprint (a single point shunt made the near-wound field grid-dependent; check E8); their neighbours' lateral faces that faced them become exposed basolateral membrane.
- **Bath edges**: the apical bath is grounded on the outer edge of the grid; the basal bath is insulated there (an Ussing-chamber arrangement: the baths communicate only through the tissue or a wound).

## Time stepping

Each step solves one sparse linear system (Kirchhoff's current law at every node) with membrane capacitors treated by backward Euler and every ionic current linearized about the previous step (chord conductance by a centred difference). Gap-junction and channel gates are updated with the previous step's voltages, as BETSE does. Cell concentrations are then advanced with the fluxes evaluated at the new voltages through the same linearization, so the charge carried by ions equals the charge moved on the capacitors to rounding.

Bath concentrations are held fixed (large, stirred reservoirs).

## Phases

- **init**: intact tissue relaxes from its initial state (BETSE's per-cell baseline concentrations when the footprint comes from BETSE).
- **sim**: the wound is applied at t = 0, then the run continues.

## Outputs

A run bundle (schema 0.2) with the footprint, `mem_neighbour` and the prism height; per-cell `vmem_apical`, `vmem_basolateral`, `phi_cell`, `tep`, currents and channel open fractions; per-node bath potentials and the apical bath field (`efield_apical_x/y`); concentrations.

## Parameters and their sources

| Parameter | Default | Source |
| --- | --- | --- |
| T, R, F | 310 K, 8.314, 96485 | BETSE (its rounded constants, for comparability) |
| membrane thickness t_m | 7.5 nm | BETSE |
| specific capacitance | 0.05 F/m² | BETSE (`cm`); five times the textbook 0.01 F/m², kept for comparability and noted |
| base D (Na, K, M, P) | 2e-18, 1e-18, 1e-18, 0 m²/s | BETSE "basic" ion profile, measured from a run |
| free D (Na, K, M) | 1.33e-9, 1.96e-9, 1e-9 m²/s | BETSE (`Do_*`) |
| Na/K-ATPase | α = 1e-7 mol/m²/s; Km Na 12, K 0.2, ATP 0.5 mM; ATP 1.5, ADP 0.1, Pi 0.1 mM; ΔG° = −37 kJ/mol | BETSE |
| gap junctions | gj_surface 5e-8, pore 26 nm, threshold 15 mV, minimum 0.1 | BETSE |
| cell height | 10 µm | BETSE (`cell_height`) |
| tight-junction resistance | 500 Ω·cm² | assumption, within the 10² to 10³ Ω·cm² range commonly reported for cultured epithelia; to be replaced by a measured value |
| bath layer thickness | 50 µm apical, 10 µm basal | assumption |

## What it does not do

No fluid flow, no deformation, no lateral intercellular space as its own compartment, no calcium, no gene regulation, no bath concentration changes, no electrostatics beyond membrane capacitance (the baths are resistors). Each of these is a stated assumption, not an oversight; any result that depends on one has to say so.
