# galvanomics

**galvanomics** is the institute's program for simulating, and checking against measurement, the electrical state of living tissue:
- the voltage across each cell's membrane;
- the potential in the fluid between cells;
- the electric fields that tissues generate, above all the field that appears when a tissue is wounded and that guides cells into the wound.

It builds tissues cell by cell, from cultured epithelial sheets to a stratified skin, predicts what an instrument would read, tests those predictions against measurements that were not used to make them, and asks which experiment would separate rival explanations. Blender is the lens. It shows a run at every scale, from a millimetre of tissue down to single ion-channel molecules, but it never produces a number.

A note on the name:
- **Etymology.** *Galvanomics* is Galvani plus *-omics*: the study of a tissue's whole electrical state, its *galvanome*.
- **Why Galvani, not Volta.** Galvani's 1791 frog experiments claimed that living tissue makes its own electricity. Volta answered that the electricity came from the two metals of the apparatus, and built the battery in pursuing it. Both were partly right: tissues do generate current, including the injury current at wounds that this program models. Galvani names the object; Volta's objection (is the signal the tissue's or the instrument's?) is the program's method.
- **Why *-omics*.** It was chosen for findability. The suffix usually names large-scale systematic measurement, which this program does not do yet; the name describes the aim, not the data.
- **Chosen 2026-10-08**, over *galvanome*, *kathodos* (Greek "the way down", Faraday's root of *cathode*), *histovolt* and *histotasis*.
- **Collisions.** None found on 2026-10-08: the name is free on PyPI, npm and GitHub, galvanomics.com is unregistered, and a web search found no use. Check again before any public use.
- **The old name.** The working name was `blender-bioelectrics`, which named the tool rather than the program.

## What this is

- **The question.** Can a computational model of a bioelectric tissue predict the measured electrical response to an intervention that was held out of fitting it? And when it cannot, which experiment would tell the remaining explanations apart?
- **Why it is hard.** Three quantities are easily confused and are measured by different instruments: membrane voltage (a dye or microelectrode, one cell), extracellular potential, and the electric field (a vibrating probe or field imager, through tissue and averaged over a probe). A coloured picture of membrane voltage says nothing about a wound field.
- **What is built:**
  - three solvers, with BETSE (the established bioelectric tissue simulator) as the reference and two solvers of our own built from BETSE's physics;
  - a data format any solver writes and Blender reads;
  - a Blender extension;
  - an audit record for every model, with tolerances written down before each run.
- **The tissue is general.** Skin is the first tissue because its wound fields have been measured; the same machinery takes cultured sheets, cornea, airway, or regeneration models.

A longer introduction, with the physics in five ideas, how a result is produced, and a glossary: [`docs/guide.md`](docs/guide.md).

## What it can do today

- **Run BETSE** reproducibly from a recorded seed, and export its wound simulation.
- **Simulate a polarized epithelium** of 3D cells between two baths: apical and basolateral membranes, tight junctions, gap junctions, BETSE's channel library.
- **Simulate skin.**
  - **Anatomy:** human forearm and mouse, built from sourced dimensions. Stratified layers of polyhedral cells, with granular cells as the 14-sided Kelvin cells measured in real skin, dermal papillae, tight junctions at SG2 and an incision.
  - **Physics:** membranes, a 3D extracellular space and a skin battery, on blocks up to 1.9 mm wide.
  - **Interventions:** drugs and other changes as switchable blocks: amiloride, Cl⁻ channel blockers, wound-activated Cl⁻ channels.
- **Show any run in Blender:**
  - cells coloured by voltage;
  - channel, pump and ion populations;
  - a movable section;
  - a probe with a time plot;
  - per-face channel densities;
  - a **nanoscope** that opens one membrane face at nanometre scale, with molecules from Protein Data Bank structures in the numbers the model implies.
- **Re-run the skin from Blender** with a different wound, barrier or channel density.

## What it has found so far

None of this has been tested against a new measurement; everything is machine-checked, not human-confirmed.

- **BETSE's 2D tissue cannot carry the measurable wound field.** It has no apical–basolateral polarity, so it has no battery to short-circuit; that is why the other two solvers exist.
- **The skin's battery is an assumption that can be tuned, not a prediction.** Its sodium channel density and barrier resistance were set to put it inside the measured 10–60 mV.
- **The first held-out prediction failed.** Topical amiloride reduced the mouse wound field by 64 ± 7%; the model predicted 97%.
  - Of the explanations, **limited drug penetration** fits. It predicts that the field moves outward rather than vanishing: −65% near the wound, +165% at 200–400 µm.
  - **Wound-edge Cl⁻ currents** do not fit, because they depend on the sodium channels amiloride blocks.
  - **Apical Cl⁻ secretion** is ruled out by the sources.
- **The comparison itself needs an instrument model.** The field imager averages over a 320 µm probe, wider than the structure the model predicts. That is the next step, and the reason the project's own observables principle now applies to it ([`docs/discriminating-experiment.md`](docs/discriminating-experiment.md)).
- **The model's field falls off over about 0.1 mm**, a third of what was measured in guinea pig. The space beneath the stratum corneum conducts too little relative to the barrier.

**Status, 2026-10-08:** three solvers and a viewer built and audited, one held-out prediction made and failed, one discriminating experiment proposed. Seven pre-registered checks have failed (E6, E8, K3, K5, L2, H1, R1), each with its cause recorded in [`docs/plan.md`](docs/plan.md). No experimental partner has been contacted, nothing is published, and nothing here is yet a scientific result.

## The position

The project began (2026-09-26) by asking whether a Blender plugin for membrane voltage could go beyond visuals. It can, but only if the three observables above stay distinct and every prediction is stated in what an instrument reads. An animation of regeneration is not evidence that regeneration has been explained.

What survives is narrower and testable: audit an established simulator, build the minimal solvers it cannot replace, fit to baseline measurements only, and predict interventions held out of fitting. **The distinctive contribution is meant to be the inverse problem and the discriminating experiment: which intervention would make rival mechanisms disagree, and by how much, given the instrument's precision.** The direction (an audited, sourced tool first; a product only after it proves useful) is in [`docs/strategy.md`](docs/strategy.md).

## The fork

| Track | Needs | Status |
| --- | --- | --- |
| **Computational**: solvers, audits, run bundles, Blender viewer and nanoscope, held-out predictions, discriminating-experiment design | public software, a laptop | under way; specified by `docs/plan.md` and `docs/architecture.md` |
| **Experimental**: a wound assay with field and migration measurements, ideally at point resolution (vibrating probe), with preregistered interventions | a laboratory with a vibrating probe or field imager, independent replicates | **blocked**, deliberately: a named experimental partner is the hard gate. `docs/discriminating-experiment.md` is written for that partner |

Everything in the first row stands alone and is worth having if the second never happens.

## Layout

| Path | What it is |
| --- | --- |
| `docs/guide.md` | Start here: the physics in five ideas, the pieces and how a result is produced, what is verified, tuned or assumed, how to look at a run in Blender, a glossary |
| `docs/strategy.md` | The question, the thesis, the direction (tool first), what is already the field's, what the project refuses |
| `docs/plan.md` | The ledger: Phase 0's verified and failed claims, every audit with tolerances fixed before running and its results, held-out predictions, open items |
| `docs/architecture.md` | The packages, the run-bundle contract, dependency direction, artifact policy, compute, commands |
| `docs/epithelium.md` | The polarized-epithelium model: geometry, network, BETSE ports, parameters and their sources |
| `docs/skin.md` | The skin model: forearm and mouse anatomy with sourced dimensions and verdicts, membranes, tight junctions, the 3D extracellular space, chloride, the amiloride measurement, what the model cannot say yet |
| `docs/nanoscope.md` | The molecular view: which voltage-gated channel and why, single-channel conductances, the structures drawn, how counts are derived |
| `docs/discriminating-experiment.md` | The amiloride puzzle: explanations examined, the prediction that would test limited drug access, what the field imager can see, recommended experiments |
| `packages/tissue-bundle/` | The on-disk run-bundle format any solver writes and Blender reads |
| `packages/betse-adapter/` | Runs BETSE in-process with a recorded seed, exports bundles, audits it |
| `packages/epithelium-solver/` | The polarized-epithelium and skin solvers, their audits, held-out tests and rival analyses; `runs/*/result.json` are the records |
| `packages/blender-addon/` | The Blender 5.2 extension `bioelectric_playback`: playback of every bundle kind, skin views, the nanoscope, and the solvers launched from the panel |

Commands for building, testing and running everything: `docs/architecture.md`, "Running it".

## Destination

Tentative:
1. **An audit paper** on what BETSE's two-dimensional wound model can and cannot say about wound fields: the reproduction, the numerical checks and the observables distinction.
2. **A paper on the amiloride puzzle**, if an experimental partner runs the discriminating experiment: a held-out prediction, its failure, and the experiment that decided between the explanations, either way.

## Licence

- **Python packages** (`tissue-bundle`, `betse-adapter`, `epithelium-solver`): MIT, `LICENSE`.
- **Blender extension** (`packages/blender-addon`): GPL-3.0-or-later, as Blender requires of extensions; `packages/blender-addon/LICENSE`.
- **Documentation and figures** (`docs/`): CC BY 4.0, `docs/LICENSE`.
- **Third-party material:** code ported from BETSE (BSD 2-clause) and meshes built from Protein Data Bank structures (CC0), listed in `THIRD_PARTY_NOTICES.md`.

Copyright 2026 Piatra Institute.

## Related work

- **BETSE**, the BioElectric Tissue Simulation Engine (Pietak & Levin, Front. Bioeng. Biotechnol. 4:55, 2016; https://gitlab.com/betse/betse): the reference simulator; our solvers port its physics.
- **Wound-field measurements**: Nuccitelli et al., Wound Repair Regen 16:432 (2008) and 19:645 (2011); Barker, Jaffe & Vanable, Am J Physiol 242:R358 (1982).
- **Corneal wound currents**: Reid et al., FASEB J 19:379 (2005); Vieira et al., PLoS One 6:e17411 (2011).
- **Blender front ends to simulators**: CellBlender with MCell (molecular reaction-diffusion); Molecular Nodes (protein structures in Blender).
