# galvanomics: a guide

Written 2026-10-08 for someone meeting the project for the first time. It explains what the models represent, how a result is produced, what is measured and what is assumed, and how to look at a run. Details live in the other documents; this one links to them.

## The physics in five ideas

1. **Every cell is a small battery.** Pumps such as the Na⁺/K⁺-ATPase keep ion concentrations different inside and out. Ion channels let ions leak back down those gradients, and the balance sets the **membrane voltage** (V_mem, inside minus outside), typically −20 to −80 mV. The Goldman-Hodgkin-Katz (GHK) equation gives it when only passive leaks act.
2. **The fluid between cells has its own potential.** Currents through membranes flow on through the narrow spaces between cells. That **extracellular potential** is not V_mem: one cell's V_mem can stay put while the potential around it changes.
3. **A polarized epithelium is a battery in series.** In a sheet of cells joined by **tight junctions**, the outer (apical) and inner (basolateral) membranes carry different channels. In skin, Na⁺ enters through ENaC channels near the surface and is pumped out at the inner side. This builds the **transepithelial potential** (TEP, the "skin battery"): the inside is 10–60 mV positive relative to the outside in human skin.
4. **A wound short-circuits the battery.** Where the sheet is cut, current leaks out through the wound, and a lateral **electric field** appears in the tissue around it. Measured beneath the outer dead layer of skin, it is about 100–180 mV/mm. Cells migrate along it, a behaviour called **electrotaxis**, and that is why wound fields matter for healing.
5. **Instruments see different things.**
   - A voltage dye or a microelectrode reads V_mem in one cell.
   - A vibrating probe reads the current density or field just above the tissue at a point.
   - A field imager reads the surface potential averaged over a probe hundreds of µm wide.

   A model is only tested when it predicts what the instrument reads. That is the **observables principle**.

## The pieces

```
 configuration (YAML)  →  solver  →  run bundle (a folder of arrays)  →  Blender (to look)
                                         ↓
                                   audit / test record (runs/*/result.json)
```

| Piece | What it is | Where |
| --- | --- | --- |
| **BETSE** | The established bioelectric tissue simulator (Pietak & Levin). A 2D sheet of cells with ion channels, pumps, gap junctions and a wound. Our reference and first verifier | run by `packages/betse-adapter` |
| **Prism epithelium** | Our first solver. A sheet of 3D cells with apical and basolateral membranes between two baths, built from BETSE's flux law, pump, channels and gap junctions | `packages/epithelium-solver`, `docs/epithelium.md` |
| **Skin** | Our second solver, on procedural anatomy. A stratified epidermis of polyhedral cells over a dermis, with tight junctions in the granular layer and a 3D grid for the extracellular space, on blocks up to 1.9 mm wide | `packages/epithelium-solver`, `docs/skin.md` |
| **Run bundle** | The data contract. Geometry, time, and every quantity with its unit and location (per cell, per membrane face, per grid node), as plain NumPy arrays with a JSON manifest and checksums. Any solver writes it; Blender and any notebook read it | `packages/tissue-bundle`, `docs/architecture.md` |
| **Blender extension** | `bioelectric_playback`. Imports a bundle and plays it back: colours, channel and ion populations, section, probe and plots, the nanoscope. It launches the solvers but computes nothing | `packages/blender-addon` |
| **Audit records** | One JSON per audit or test, with values, tolerances and verdicts | `packages/*/runs/*/result.json` |

All solvers share one numerical core. Each time step solves Kirchhoff's current law over every membrane and extracellular node at once (conjugate gradients with algebraic multigrid). Membrane capacitors are integrated implicitly, and ion concentrations are updated so that charge is conserved exactly. The checks show Kirchhoff's law holds to about 1e-12 and charge to 1e-9 or better, relative.

## The discipline

- **Tolerances before results.** Every check is written into `docs/plan.md` with its tolerance before the run that judges it. A check that fails stays failed; a diagnostic added afterwards is labelled *post hoc*.
- **Held-out predictions are frozen.** The model is calibrated on some measurements, then predicts one it never saw. The configuration's SHA-256 is recorded before the comparison.
- **Every number carries its kind.** *Sourced*: read in a paper, with a verdict: V read at source, P secondary, U not found. *Calibrated*: tuned to a measurement. *Derived*: our arithmetic. *Assumed*: a modelling choice.
- **Rivals, not stories.** When a prediction fails, each explanation becomes a model calibrated on the same data. The useful output is the experiment that would separate them, with the number of samples it needs.

## What is real and what is not

| Kind | Examples |
| --- | --- |
| Sourced (V) | layer thicknesses, cell sizes and counts of human forearm skin; tight junctions at SG2; keratinocyte [Cl⁻]i 6.8 mM; single-channel conductances; the amiloride measurement; protein structures |
| From BETSE | the ion flux law, the Na⁺/K⁺ pump, the channel models, gap-junction gating, base permeabilities, the ion profiles |
| Calibrated | the apical Na⁺ channel density and the tight-junction resistance (to the battery); in the rival analyses, the drug's reach or the wound Cl⁻ density (to the amiloride effect) |
| Assumed | the conductivity of the extracellular spaces (modelling conventions); the incision shape; where molecules sit within a membrane face; the mouse layer structure |
| Not represented | hair follicles, glands, vessels, immune cells; Ca²⁺ signalling; anything slower than seconds (channel regulation, migration, healing) |

## Looking at a run in Blender

1. Import a bundle (Bioelectrics panel, **Import Run Bundle**, pick its `bundle.json`) or open one of the prepared `.blend` files in `packages/epithelium-solver/data/work/*/`. Restart Blender after the extension is reinstalled.
2. Press Space to play. Frame 0 is the intact tissue, and the wound appears at frame 1. The legend gives the time, the skin battery, and what the colours and glyphs mean.
3. **Quantity** (panel) recolours the tissue. Per-face quantities such as `chan_g_naleak` show channel density at tissue scale.
4. **Section plane** shows either the extracellular potential or the cut cells. Move it with G, Y. Hide the dermis to see it whole.
5. **Probe** (the small sphere) plots the field at its position over time.
6. **Nanoscope**: select a membrane face (edit mode, face select) or place the 3D cursor, then **Open nanoscope here**. A new scene opens with the bilayer and its molecules. **Back to the tissue** returns. Use Material Preview shading; in Solid view the molecules are black.

Commands to build, test and run everything: `docs/architecture.md`, "Running it".

## Glossary

| Term | Meaning |
| --- | --- |
| V_mem | membrane voltage: cytoplasm minus the extracellular fluid at that membrane |
| TEP, skin battery | transepithelial potential: inside minus outside across the tight-junction barrier |
| Apical, basolateral | the outward- and inward-facing membranes of a polarized cell |
| SG1–SG3 | the three rows of the stratum granulosum, SG1 outermost; human and mouse tight junctions sit at SG2 |
| Kelvin cell | a 14-faced polyhedron (tetrakaidecahedron), the shape of granular cells |
| ENaC | epithelial Na⁺ channel, blocked by amiloride (K_i 42 nM) |
| ANO1 | a Ca²⁺-activated Cl⁻ channel (TMEM16A) in keratinocytes |
| NKCC1 | Na-K-2Cl cotransporter; loads cells with Cl⁻ in secretory epithelia, reported absent from epidermis |
| GHK | the Goldman-Hodgkin-Katz equation for the voltage set by passive leaks |
| Space constant | the distance over which a wound's effect on the battery decays (0.3–0.4 mm in guinea pig; 0.1 mm in the model) |
| Half domain, mirror | a block cut at the wound centre whose walls are mirror planes, so 960 µm stands for 1.9 mm |
| Field imager (BFI) | a non-contact probe, 320 × 700 µm, that maps the surface potential of skin beneath the stratum corneum |
| Vibrating probe | a point-like electrode oscillating above tissue, reading current density with tens of µm resolution |
| Run bundle | the folder of arrays and manifest that every solver writes |
| Held-out prediction | a prediction for a measurement not used in calibration, frozen before comparison |
