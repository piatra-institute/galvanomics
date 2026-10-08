# galvanomics: the skin model

Written 2026-10-08. This is the anatomically structured tissue that the 3D physics runs on. It is a stratified epidermis of real 3D cells over a dermis with papillae, cut by an incision, with keratinocyte membranes, tight junctions, gap junctions and a 3D extracellular space. It is Phase 2 of `docs/plan.md`, started early.

- Anatomy code: `packages/epithelium-solver/src/episolver/skin.py` (`episolver skin`).
- Physics code: `skin_physics.py` (`episolver skin-sim`).
- Audit code: `skin_audit.py` (`episolver skin-audit`).

**Status.**
- The anatomy uses sourced human dorsal-forearm dimensions where a source was found. Verdicts are in the table below.
- The physics runs and conserves charge.
- Two parameters are calibrated, not measured: the apical Na channel density and the tight-junction resistance. They were chosen so that the intact skin battery lands inside the reported range.
- Audit of the 160 µm block (pre-registered, `docs/plan.md`): Kirchhoff and charge conservation exact (K1, K2), converged in time (K4), battery converged in grid (K5, first half). K3 failed on its premise (gap-junction coupling), and the field half of K5 failed on its statistic (a node-wise peak). Record: `packages/epithelium-solver/runs/skin/result.json`.
- A wide block (half of a 1.9 mm block, mirrored at the wound) makes the wound local: the far battery is within 1.1% of intact 2 s after the cut. Grid, time step and conservation pass; the domain-size check failed because changing the width re-randomizes the cells (`runs/skin-wide`).
- **The held-out test failed.** Calibrated only on the intact battery, the model predicts that 1 mM amiloride cuts the mouse wound field by 97%; the measurement is 64 ± 7% (`runs/skin-amiloride`). Either the drug reaches only about 200 µm from the wound, or a second current, which amiloride does not block, carries part of the field.

**Source verdicts:**
- **V**: the number was read at source.
- **P**: the number comes from a secondary source, or only partly.
- **U**: no source was found.
- **Derived**: our own arithmetic.

## How the cells are made

- **Seeds.** There is one seed per cell, placed row by row with layer-specific lateral spacing and row height. The seeds sit on a jittered hexagonal lattice that fits the domain exactly.
- **Interdigitating layers** (basal and spinous) are tessellated as 3D Voronoi regions within the layer. This gives columnar basal cells and polyhedral spinous cells.
- **Granular cells are Kelvin cells** (since 2026-10-08): flattened tetrakaidecahedra, 6 square and 8 hexagonal faces, as reconstructed by Yokouchi 2016 (eLife, 10.7554/eLife.19593). Seeds sit on a body-centred lattice, one lattice layer per granular row, tessellated in coordinates where the lattice is regular and mapped back; an affine map keeps faces planar and the tiling exact (tested). Before, granular cells were flat row sheets.
- **Corneocytes** are tessellated one row at a time as flat sheets, so they stack like bricks: thin, wide squames.
- **Side walls are mirror planes.** A cell clipped by a wall is half of a doubled cell, so its wall face is not membrane; the extracellular field has no component through a wall.
- **Strata stay distinct.** No cell crosses into another layer, only basal cells touch the basement membrane, and only corneocytes form the surface (tested).
- **Contacts are explicit faces.** Where two layers or rows meet, every overlap of a lower cell's top face with an upper cell's bottom face becomes one shared face. So every cell-to-cell contact is an explicit face pair of equal area, which is what the gap junctions use.
- **Papillae.** A smooth vertical deformation lifts the dermal-epidermal junction into a square lattice of dermal papillae, one per wavelength square. The lift fades to zero at the top of the spinous layer. It is monotonic, so no cell inverts. It has zero mean, so total volume is preserved.
- **Incision.** The cut is V-shaped, along y, from the surface into the dermis. Cells whose centroid lies inside it are removed at the moment of wounding.

The anatomy is checked in `tests/test_skin.py`:
- Cell volumes are positive.
- Shared faces cancel exactly: the cells sum to the volume enclosed by the epidermis boundary (relative difference 1e-16 on the default skin).
- The epidermis keeps the volume of the flat slab to within 0.03%. The residual comes from sampling the papillae at cell vertices.
- Every shared face has a partner of equal area.
- Boundary faces belong to the right layers.
- The exported bundle validates.

## Dimensions (human dorsal forearm)

The default skin has 740 cells: 170 basal, 336 spinous, 54 granular and 180 corneocytes.

| Parameter | Value in the model | Basis | Verdict |
| --- | --- | --- | --- |
| Viable epidermis | 56.6 µm (10 + 37.6 + 9) | Dorsal forearm, 56.6 ± 11.5 µm, biopsy histology, n = 71 (Sandby-Møller 2003, 10.1080/00015550310015419) | V |
| Stratum corneum thickness | 18.3 µm | Dorsal forearm, 18.3 ± 4.9 µm (Sandby-Møller 2003) | V |
| Stratum corneum layers | 15 rows | 15 ± 4 on the extremities (Ya-Xian 1999, 10.1007/s004030050453) | V |
| Corneocyte size | 36.4 µm lattice spacing, 1147 µm² per cell | Forearm 1148 ± 231 µm² by atomic force microscopy (Évora 2023, PMC10591027); spacing from area (derived) | V, derived |
| Granular layer | 3 rows, 9 µm, Kelvin cells | 3 layers SG1–SG3 (Yoshida 2013, 10.1016/j.jdermsci.2013.04.021) **V**; flattened tetrakaidecahedra (Yokouchi 2016) **V**; the 9 µm thickness is a modelling choice **U** | V / U |
| Granular cell size | 28.6 µm spacing (30 µm equivalent diameter) | 25–35 µm en face by reflectance confocal microscopy (Menzinger 2019, PMC7154283) | P |
| Spinous layer | 4 rows, 37.6 µm | Thickness is the viable epidermis minus basal and granular (derived); the number of rows was not found | derived / U |
| Spinous cell size | 13.5 µm spacing (14.2 µm equivalent diameter) | Set so that nucleated-cell density matches the next row; slightly below the 15–25 µm by reflectance confocal (Menzinger 2019) | P, fitted |
| Nucleated-cell density | 438 per 10⁴ µm² | 452 per 10⁴ µm² on the forearm, whole-mount histology (Bergstresser 1978, 10.1111/1523-1747.ep12541516) | V (target) |
| Basal cells | 1 row, 10 µm high, 9.5 µm spacing (10 µm equivalent diameter) | 7–12 µm en face (Menzinger 2019) **P**; columnar height not found **U** | P / U |
| Papilla density | 39 per mm² (one per 160 µm square) | Young volar forearm, 41 per mm² by confocal (Sauermann 2002, quoted in Nuccitelli 2011) | P |
| Papilla depth | 36 µm peak to trough (amplitude 18 µm) | Forearm rete ridge length 37.6 ± 6.9 µm, age 18–30 (Newton 2017, via Roig-Rosello 2020, PMC7760980) | P |
| Domain | 160 × 80 µm, dermis 60 µm deep | Modelling choice. It is smaller than the wound field's space constant (below) | — |
| Incision | 40 µm wide at the surface, 16 µm at the bottom, 20 µm into the dermis | Modelling choice | — |

**Correction, 2026-10-08.** The first papilla field was an egg-crate, cos(kx)·cos(ky). It puts two papillae in every wavelength square: 78 per mm² at 160 µm, which is twice the forearm value. It was replaced by (cos kx + cos ky)/2, which puts one papilla per square, before the skin audit was run.

**Not found at source:** columnar basal cell height, the number of spinous layers, granular cell thickness, per-site skin-battery values (Foulds & Barker 1983 full text), measured epidermal resistance without the stratum corneum, measured interstitial-fluid conductivity, and the epidermal extracellular volume fraction.

## Mouse skin preset (for the held-out amiloride test)

`configs/skin-mouse-wide.yaml`, added 2026-10-08. The reference measurement for the held-out test is in mouse, so the anatomy is mouse where a source exists; the membranes are the forearm ones, unchanged.

| Parameter | Value in the model | Basis | Verdict |
| --- | --- | --- | --- |
| Viable epidermis | 17.5 µm | 17.50 ± 4.98 µm, CD1 female flank, 10 weeks, frozen sections (Wei 2017, Sci Rep, PMC5698453) | V |
| Stratum corneum | 4.2 µm | 4.19 ± 1.79 µm, same | V |
| Dermis | 182 µm | 182.4 ± 46.7 µm, same | V |
| Dermal-epidermal junction | flat (no papillae) | "typically smooth" in hairy animals (Rittié 2016, PMC4882309) | V |
| Granular layer | 3 rows (SG1–SG3), Kelvin cells, tight junctions at SG2 | Tight junctions exclusively in SG2 in mouse ear and trunk (Kubo 2009) **V**; row thickness **U** | V / U |
| Basal and spinous layers | 1 row each, 5 µm each | Two to four nucleated layers are reported (Rittié 2016; Allen & Potten 1976 via Nuccitelli 2008); with three granular rows the model has five, more than reported | U |
| Cell sizes, corneocyte rows | human forearm values, 6 corneocyte rows | not found for mouse | U |
| Battery | not calibrated | the one sourced mouse value is about 6 mV (inside positive), ex vivo, hairless HR-1 dorsal skin (Abe 2024, Cosmetology 32:76) **V**; 20–50 mV is quoted for mammals generally (Nuccitelli 2008, citing Barker 1982) **P** | — |

## The amiloride measurement (held-out test H1)

Read in the full text of Nuccitelli et al. 2008 (Wound Repair Regen 16:432, PMC3086402):

| Item | Finding | Verdict |
| --- | --- | --- |
| Drug | 1 mM amiloride in phosphate-buffered saline, applied topically to the wound | V |
| Result | wound field changed by −64 ± 7% (mean ± SEM, 11 wounds in 9 mice; Table 1); the text says 68% on average. PBS alone: +22 ± 20% (3 wounds) | V |
| Measurement | non-contact probe reading the surface potential of the epidermis beneath the stratum corneum (bioelectric field imager); field from the potential difference between scan points | V |
| Baseline | 177 ± 14 mV/mm over the wound (SEM, 61); 115 ± 64 mV/mm in a band about 1 mm wide outward from the edge; the authors estimate about 40 mV/mm inside the epidermis | V |
| Animals and wounds | CF-1 (depilated) and hairless SKH-1 mice, back, scalpel or scissors cuts 1–3 mm long | V |
| Not reported | time of application relative to wounding and scanning; absolute fields before and after amiloride | U |
| The authors' reading | "both Na⁺ influx and Cl⁻ efflux are carrying the wound current" (PGE2, a Cl⁻ channel activator, raised the field by 82 ± 21%) | V |

Amiloride potency: K_i 42 nM on rat αβγ ENaC (Xenopus oocytes, −100 mV; Schild 1997, PMC2217053) **V**; so 1 mM is about 24,000 times K_i and blocks ENaC almost completely where it reaches. At 1 mM it also blocks the Na⁺/H⁺ exchanger NHE1 (about 3 µM, **P**), ASICs (ASIC3 IC50 9.5–18.6 µM, **V**) and probably the Na⁺/Ca²⁺ exchanger (about 1 mM, inferred, **P**), none of which the model contains. No study of amiloride penetration into mouse epidermis or wound beds was found.

## Chloride in keratinocytes (for the rival to the Na-only battery)

Sources read 2026-10-08. They decide which chloride mechanism can be tested; they rule out the one first proposed.

| Fact | Source | Verdict |
| --- | --- | --- |
| Keratinocyte [Cl⁻]i 6.8 ± 1.3 mM (MQAE fluorescence, normal human epidermal keratinocytes), E_Cl −75.7 mV; at the resting −24 to −40 mV, Cl⁻ flows **into** the cells through ANO1 | Yamanoi 2023, Commun Biol, PMC9870996 | V |
| ANO1 (TMEM16A), a Ca²⁺-activated Cl⁻ channel, in human and mouse keratinocytes (mRNA, protein, currents gated by TRPV3, blocked by T16Ainh-A01 and Ani9); a low-Cl⁻ medium slowed wound healing | Yamanoi 2023 | V |
| NKCC1 absent from rat, mouse and human epidermis (present in the sweat-gland secretory coil) | Nejsum 2005, Am J Physiol Cell Physiol | P |
| CFTR protein not detected in human keratinocytes (3 antibodies, Human Protein Atlas); present in several layers of mouse epidermis, with delayed wound healing in CFTR-mutant mice | HPA; Dong 2015, J Cell Physiol | V / V (abstract) |
| In cultured keratinocytes Cl⁻ transport is activated by Ca²⁺ but not by cAMP (forskolin: no change in short-circuit current or ³⁶Cl efflux) | Kansen 1992, Biochim Biophys Acta | V (abstract) |
| No skin wound field or transepidermal potential has been measured under a Cl⁻ channel blocker | — | not found |
| Rat cornea: the wound current is largely Cl⁻ influx at the wound edge; DIDS almost halves it; Cl⁻-free solution raises the edge current by 237% | Vieira 2011, PMC3045448; Reid 2005, PMC1459277 | V |
| PGE2 raised the mouse wound field by 82 ± 21% (7 wounds); PBS alone raised it by 22 ± 20% (3 wounds); concentration not stated | Nuccitelli 2008 | V |
| Single-channel conductances: ANO1 about 1–3 pS (2.63 pS by noise analysis); CFTR 10.4 pS | Hartzell & Whitlock 2016; Berger 1991 | V |

**Consequences for the model:**
- **Apical Cl⁻ secretion is not supported.** Cl⁻ secretion would need a basolateral loader holding [Cl⁻]i above equilibrium, and none is documented in the granular layer. The measured [Cl⁻]i drives Cl⁻ inward. So Nuccitelli's "Cl⁻ efflux" is not adopted.
- **A supported rival exists.** Ca²⁺-activated Cl⁻ channels (ANO1-like) in keratinocytes exposed at the wound edge would let Cl⁻ in from the wound fluid. That current is driven by each cell's own membrane voltage, not by the battery, and amiloride does not touch it. It is the corneal picture.
- **The PGE2 result is weak evidence.** The vehicle alone changed the field by +22 ± 20%, and keratinocyte Cl⁻ transport is cAMP-insensitive.

## Physics on the anatomy

**Cells.** Every living cell (basal, spinous, granular) is one compartment with an intracellular potential and ion concentrations. Every face of a living cell is a patch of membrane facing the extracellular space at that face. Corneocytes are dead and electrically inert. The fluxes, pump, channels, gap junctions and time step are the BETSE ports used by the prism solver (`docs/epithelium.md`):
- one sparse Kirchhoff solve per step;
- backward-Euler membrane capacitance;
- ionic currents linearized about the previous step;
- a charge-consistent concentration update.

**Membranes.**

| Faces | Channels and pump | Basis |
| --- | --- | --- |
| Basal, spinous, and granular faces below the tight-junction line (basolateral) | K leak (5e-17 m²/s) and Na/K-ATPase (BETSE rate 1e-7) | Na/K-ATPase falls from the basal layer upward and blocking it lowers the battery (Moulin 2012, 10.1089/wound.2011.0318) **V**; densities assumed |
| Granular faces above the tight-junction line (apical) | Na channel (ENaC-like, 3e-16 m²/s) and K leak (5e-17 m²/s), no pump | ENaC is present in all nucleated layers (Oda 1999; Hanukoglu 2017) **V**, and amiloride lowers wound fields (Barker 1982; Nuccitelli 2008) **V**. Confining Na entry to the apical side is a simplification. The Na density is **calibrated**. The apical K leak was added after the first runs put SG1 at +65 mV (post hoc). |
| Shared faces between living cells | Gap junctions with BETSE's voltage gate, uniform | The real connexin pattern is layered: Cx43 highest in the spinous layer, reduced in the granular (Salomon 1994) **V**. This layering is not modelled. |

**Tight junctions.** The barrier sits at the top of the second granular row from the bottom, SG2 in the usual numbering, where SG1 is outermost. The SG2–SG2 junctions are the epidermal barrier in human skin (Yoshida 2013) and are found only at SG2 in mouse (Kubo 2009, 10.1084/jem.20091527) **V**. So SG1 cells lie wholly above the barrier and carry only apical membrane.

**Ions and interventions** (2026-10-08):
- **Ion sets.** The default is BETSE's basic set (Na⁺, K⁺, a generic anion, protein). `ion_profile: mammal_no_ca` uses BETSE's mammal profile without Ca²⁺ (no Ca pump is modelled), adding Cl⁻ at BETSE's default permeability. Initial cytoplasmic concentrations can be set per ion; the rivals use the keratinocyte [Cl⁻]i of 6.8 mM.
- **Modifiers** stand for drugs and other interventions. Each multiplies one pathway (a channel type, the Na-K-2Cl cotransporter, the base Cl⁻ or Na⁺ permeability) on one side, everywhere or within a distance of the wound.
- **Wound Cl⁻ channels.** `wound_cl_d` switches Ca²⁺-activated Cl⁻ channels (ANO1-like, BETSE's ClLeak) on, at wounding, on every living face touching the wound fluid.
- **The cotransporter.** A Na-K-2Cl cotransporter is implemented, with a thermodynamic form that is not ported from BETSE, which has none. It is tested to carry no current and to vanish at equilibrium. It is not used by any rival, because the sources found it absent from epidermis.

**Extracellular space.** A 3D tensor grid, 4 µm laterally. In z it is coarse in the dermis and the lower epidermis, and 0.75 µm through the granular layer and the space beneath the stratum corneum. On a wide block the x spacing is 4 µm within 150 µm of the wound and grows by 15% per node to 10 µm. It is solved by conjugate gradients preconditioned with classical algebraic multigrid: 0.5 s per step for the wide block, 0.2 s for the 160 µm one. Conductivity by region:

| Region | Value | Basis | Verdict |
| --- | --- | --- | --- |
| Wound fluid | 1.79 S/m | Cerebrospinal fluid at 37 °C, as a proxy for interstitial fluid (Baumann 1997, 10.1109/10.554770) | V (proxy) |
| Intercellular space of the viable epidermis | 0.026 S/m (0.0145 × fluid) | Modelling convention of Sun 2017 (10.1155/2017/5289041); original measurement not traceable | P |
| Dermis | 0.222 S/m | Sun 2017 convention | P |
| Stratum corneum | 2e-6 S/m, used as the insulating top boundary (the grid ends just above the granular layer) | Sun 2017 convention; 10³–10⁶ Ω·m DC (Abe & Nishizawa 2021) | P |
| Tight-junction band | 2000 Ω·cm² across one node layer | **Calibrated** with the apical Na density so that the intact battery lies in 10–60 mV | — |

The bottom of the dermis is the electrical reference. The wound replaces cells and tissue in the cut with wound fluid.

**Skin battery.** This is the mean extracellular potential just below the tight-junction band, minus that just above it, more than 50 µm from the wound centre. The Blender legend and the audit use the same definition. Human skin is reported at 10–60 mV, inside positive, depending on site (Foulds & Barker 1983 as quoted by Moulin 2012 and Abe & Nishizawa 2021) **P**. Guinea pig is 30–100 mV on glabrous skin (Barker 1982, 10.1152/ajpregu.1982.242.3.R358) **V**.

## What the model cannot say yet

- **The battery is calibrated, not predicted.** Two free parameters were set to put it inside a reported range.
- **The wound field is of the measured order, but that is not yet a prediction.** On the wide block the field beneath the stratum corneum is 172 mV/mm 20–50 µm from the wound centre, pointing away from the wound, against 107 ± 13 mV/mm at young human forearm wounds (Nuccitelli 2011) and 177 ± 14 mV/mm in mouse (Nuccitelli 2008). The battery is calibrated, and the model's space constant is 106 µm, about a third of the 0.3–0.4 mm measured in guinea pig (Barker 1982): the field falls off too fast. The sign convention of the measurements has not been checked.
- **The battery is Na-only, and the held-out test says that is incomplete** unless amiloride reaches only about 200 µm from the wound (`docs/plan.md`, H1).
- **SG1 cells depolarize** to as much as about +35 mV, because they carry only apical membrane with no pump. Whether real SG1 cells behave like this is unknown to us.
- **Not represented:** a Cl⁻ pathway (Nuccitelli 2008 describes Cl⁻ moving basal to apical), layered connexins, active responses after wounding (channel or pump regulation), cell migration, and any time beyond seconds.

## What the anatomy leaves out

Hair follicles, sweat ducts, melanocytes, Langerhans cells, blood vessels, nerves, nuclei and organelles, desmosomes as structures, and the irregularity of real strata beyond what the jitter gives. The dermis is one homogeneous block. Hairy skin keeps a lower battery than glabrous skin (Barker 1982), so follicles matter for the battery. Any of these can be added as cells or context meshes when a question needs it.

## Next

1. A Cl⁻ pathway as a rival mechanism to the Na-only battery, both calibrated on the intact battery only, then the experiment that separates them: a Cl⁻ channel blocker alone and with amiloride, or a measurement of amiloride penetration around the wound.
2. The space constant: which of the barrier resistance, the conductivity of the space beneath the stratum corneum and the dermis makes the model's 106 µm three times shorter than measured, and whether a measured epidermal resistance closes the gap.
3. A domain-size check that keeps the cells fixed in the shared region (or averages over seeds), replacing L2.
4. Layered connexins, tested by whether the battery and the field change.
