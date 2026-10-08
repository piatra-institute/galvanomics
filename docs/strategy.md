# galvanomics: Strategy

Written 2026-10-07 under the working name blender-bioelectrics (the project is now *galvanomics*), after the founding notes (2026-09-26) were read in full, its load-bearing claims were checked at source (`docs/plan.md`, Phase 0), and BETSE 1.5.0 was installed and run on this machine.

## The question

Can a computational model of a bioelectric epithelium predict the measured electrical response of a wounded tissue to an intervention that was excluded from fitting it?

The question has four parts that the founding notes run together and that need separate tests:

1. **Physics.** Does an established simulator (BETSE) conserve what it should, converge as its time step shrinks, and reproduce the textbook limits it claims to (a resting potential at the Goldman-Hodgkin-Katz value when only passive fluxes act)?
2. **Observables.** Which of the simulator's outputs correspond to something an experiment measures? V_mem, extracellular potential and the extracellular field are different quantities; a vibrating probe or a field imager measures the last two, a voltage dye reports the first only after calibration.
3. **Representation.** Can a two-dimensional cell sheet with in-plane extracellular spaces represent a wound field that, in real epithelia, is driven by transepithelial potential across apical and basolateral membranes? If not, no amount of fitting fixes it.
4. **Prediction.** Fitted to baseline only, does the model predict the direction and size of the change in wound field under a held-out ion-channel or gap-junction intervention better than a model without bioelectric inputs?

## The thesis

**Own the audit and the discriminating experiment, not a new physics engine.** The field has simulators (BETSE, MCell, Morpheus), measurements of wound fields, and decades of electrotaxis work. What it lacks for this question is a reproducible, versioned path from a tissue geometry to an observable that an instrument actually reports, with every numerical check stated before it runs, and a design rule for the intervention whose outcome would separate rival mechanisms.

Blender earns its place as the editor and inspector: tissue geometry, wound shapes and interventions are spatial objects, and a researcher should be able to select a cell and read its time series. It is never in the loop that produces a number.

## Direction (2026-10-08)

The direction was set after the first skin model ran: an actual scientific tool, and perhaps later a product of the Mathematica or Wolfram|Alpha kind, but useful first. That changes the thesis above in one respect and keeps it in the rest.

- **What changes.** The project now builds and maintains its own solvers (the prism epithelium and the skin, both from BETSE's physics), because BETSE's 2D tissue cannot carry the measurable wound field (Phase 1 audit). "Not a new physics engine" is replaced by what is refused now: no model without a recorded audit, and no parameter tuned to the test that judges it.
- **What a useful tool is here.** Three things, each of which the project already practises at small scale: computation that can be trusted because every model ships with its audit record; curated data with provenance, every parameter carrying its source and verdict (`docs/skin.md`), eventually as a queryable base rather than prose; and questions answered in one step, for example the field at a forearm wound with 60% of the apical Na channels blocked, returned with its uncertainty and the nearest measurement.
- **The usefulness gate, before any product question.** One question a bioelectricity laboratory actually has, answered better than BETSE, VCell or COMSOL would answer it, and one prediction tested against a measurement the model was not tuned on, reported whichever way it goes. The first candidates are the wound field of a skin block wide enough to be compared with measurement, and the amiloride reduction of the mouse wound field (Nuccitelli 2008), both specified in `docs/plan.md`.
- **Blender's role widens, not its authority.** Besides playback, Blender becomes the place to read the tissue at every scale, down to individual channel molecules (`docs/nanoscope.md`). It still never produces a number: what it draws at molecular scale is sampled from solver outputs, with derived counts and assumed placements labelled.
- **What stays.** The audit discipline, the observables distinction, frozen predictions before held-out data, and the experimental half gated on a partner.

## What is already the field's

- **BETSE**, the BioElectric Tissue Simulation Engine: ion transport, pumps, gap junctions, extracellular spaces, and a wounding event (Pietak and Levin, Front. Bioeng. Biotechnol. 4:55, 2016, doi:10.3389/fbioe.2016.00055). Its wound simulation is checked against experiment at the level of a plausible current range, not fitted to a dataset.
- **Wound fields are measured.** About 177 ± 14 mV/mm at mouse skin wound margins immediately after wounding (Nuccitelli et al., Wound Repair Regen 16:432, 2008, doi:10.1111/j.1524-475X.2008.00389.x). This is a reference observation for skin in vivo, not a target for a cultured sheet.
- **Transient bioelectric perturbation can have persistent anatomical effects** in planaria (Durant et al., Biophys. J. 112:2231, 2017, doi:10.1016/j.bpj.2017.04.011).
- **Theory of bistable tissue states** with adaptive gap junctions and slow memory (Cortés-Poza, J. Math. Biol. 93:43, 2026, doi:10.1007/s00285-026-02459-2); synthetic, not organism-specific.
- **Blender as a front end to a simulator**: CellBlender with MCell (v4.2.0, May 2026, bundled with Blender 5.1), molecular reaction-diffusion rather than tissue bioelectricity; Goo (cell and tissue mechanics in Blender).
- **Multicellular behaviour with fitting**: Morpheus 2.4.1 (May 2026) with FitMultiCell and pyABC; FitMultiCell's last release was 2022.
- **General cell-biology and multiphysics simulators** that a tool here would be compared with: VCell (Virtual Cell, reaction-diffusion and membrane electrophysiology on cell geometries) and COMSOL Multiphysics (commercial; used for electrical models of skin). Neither was examined in depth yet.
- **Molecular-scale scenes**: Molecular Nodes (a Blender add-on that imports structures from the Protein Data Bank) and cellPACK (packing of molecular structures into mesoscale cell models). Relevant to drawing channels; neither simulates bioelectricity.

What the sources examined did not contain: a Blender front end for bioelectric tissue simulation, or a published audit of BETSE's wound physics against its own textbook limits. Not proven absent.

## What the project refuses

- **No claim from a picture.** A coloured mesh is an inspection tool. No regeneration, morphogenesis or therapeutic claim is made from an animation.
- **No unverified citation.** The founding notes' dated claims are carried in `docs/plan.md` with verdicts; the one that failed (BETSE's release history) is recorded as failed, not dropped.
- **No observable conflation.** Every exported quantity carries its location (cell, membrane, extracellular grid) and unit; V_mem is never presented as a field.
- **No tolerance chosen after the result.** Audit tolerances are written in `docs/plan.md` before the run that tests them.
- **No parameter tuned to the test that judges it.** Calibrated parameters are named as calibrated; a held-out prediction is frozen, with its configuration hash, before it is compared.
- **No product claim before the usefulness gate** (Direction, above).
- **No 10,000-cell claim before a benchmark.** The founding notes' first milestone said 10,000 cells; its own plan said a few hundred first. The project starts at a few hundred.
- **No laboratory work here.** Wound assays, vibrating probes and voltage imaging belong with groups that already run them.
- **No budget or schedule as a fact.** The founding notes' twelve weeks assumed a full-time developer and an electrophysiologist; the project has neither.

## Division of labour

The computational half needs careful numerics, reproducible packaging and an interface, and no laboratory. The experimental half needs a partner, and the institute's contribution to it is the frozen predictions, the preregistered comparison and the kill criteria that make the partner's time well spent.

## Success

A BETSE wound simulation that is reproducible from a seed, audited against stated tolerances, and inspectable cell by cell in Blender; then a written decision on whether BETSE can represent the measurable wound field at all. A negative answer counts: it decides whether the next artifact is a minimal solver or a fitted model.

Reached 2026-10-08. The answer was negative: BETSE's 2D tissue has no battery to short-circuit, so minimal solvers were built (`docs/plan.md`). Success from here is the usefulness gate in "Direction" above. The first candidate is a held-out prediction that an experimental partner can test with the discriminating experiment in `docs/discriminating-experiment.md`, once the field imager has been modelled.
