# galvanomics: the nanoscope

Written 2026-10-08. The nanoscope is a Blender view of a patch of one membrane face at nanometre scale: the lipid bilayer, the channel and pump molecules it carries, and their states over the simulated time. It reads the solver's outputs and draws a sample from them. It never computes physics.

What is sourced and what is assumed:
- **Sourced:** the molecules' shapes (structures in the Protein Data Bank) and the single-channel conductances and pump turnover used to turn conductance into counts.
- **Derived:** the counts.
- **Assumed:** the positions of the molecules within a face.
- **Sampled:** the open and closed states, drawn from the solver's open fractions.

**Source verdicts:**
- **V**: the number was read at source, often in the abstract only.
- **P**: the number comes from a secondary source, or only partly.
- **U**: no source was found.

## Which voltage-gated channel, and how much

No voltage-gated current has been recorded natively in human keratinocytes. Manaves et al. 2004 recorded 41 primary human keratinocytes whole-cell and found "no evidence of voltage-gated currents ... from −40 to 80 mV" (BMC Dermatol, PMC446203) **V**.

Expression evidence, ranked:

| Channel | Evidence | Verdict |
| --- | --- | --- |
| Kv7.2 | Antibody staining and a small outward current in neonatal rat keratinocytes, 1.06 ± 0.23 pA/pF at +50 mV, about a quarter of it Kv7 (Reilly 2013, PMC3745621) | V |
| Nav1.5–1.7 | Protein and mRNA in human epidermis; no Na current recorded (Zhao 2008) | V |
| Cav1.2 | Protein in epidermis (Denda 2006) | V |
| Kv3.4 (KCNC4) | mRNA in primary keratinocytes, basal and suprabasal: 9.7 and 7.0 nCPM in the Human Protein Atlas single-cell data, 5.2 nTPM in HaCaT. The highest of the voltage-gated K channels BETSE can port | V (mRNA only) |
| Kir2.1 | mRNA, mostly basal | V (mRNA only) |
| Kv1.x | Below 0.6 nCPM; Kv1.3 in psoriatic skin is on T cells, not keratinocytes | V |

**Choice:** Kv3.4, ported from BETSE, in a separate configuration (`configs/skin-forearm-vgic.yaml`) at a density low enough to stay under Manaves' detection. It is a **demonstration of a voltage-gated channel in the tissue, not an established keratinocyte current**. The calibrated default skin has none.

## Single-channel conductances and pump turnover

| Molecule | Value | Conditions | Source | Verdict |
| --- | --- | --- | --- | --- |
| ENaC (αβγ, rat) | 4.7 ± 0.6 pS | Na⁺, outside-out patches, MDCK cells, 20–23 °C | Ishikawa 1998, J Gen Physiol | V |
| ENaC (αβγ, mouse) | 4.7 pS | oocytes | Ahn 1999, Am J Physiol Renal | V |
| TWIK-1 (K leak stand-in) | 34 pS | conditions not read | Lesage 1996, EMBO J | V (value) |
| Kv3.1 + Kv3.4 | 32–38 pS | on-cell patches, oocytes; no value for homomeric Kv3.4 | Fernandez 2003, J Biol Chem | P |
| Cx43 gap junction | about 120 pS (main state) | HeLa-Cx43, dual whole-cell | Bukauskas & Peracchia 1997 | V |
| Cx26 gap junction | 135 pS | 120 mM KCl, N2A cells | Suchyna 1999, Biophys J | V |
| ANO1 (TMEM16A) | 2.63 pS (1–3 pS) | nonstationary noise analysis | Hartzell & Whitlock 2016, J Gen Physiol (citing Lim 2016) | V / P |
| Na⁺/K⁺-ATPase turnover | about 200 s⁻¹ at 36 °C (about 60 s⁻¹ at 24 °C) | saturating substrates, cardiac giant patches | Friedrich 1996, Biophys J | V |
| Na⁺/K⁺-ATPase site density | about 2,400 per µm² (basolateral) | dog tracheal epithelium; none found for keratinocytes | Widdicombe 1985, Am J Physiol Cell | V (other tissue) |

**Why TWIK-1 for the K leak:** HaCaT cells express several two-pore-domain K channels (TASK, TREK, TRAAK mRNA; TREK and TRAAK protein; Kang 2007) **V**. In the Human Protein Atlas, TWIK-1 (KCNK1) is by far the most abundant in keratinocytes (149 and 130 nCPM), while TREK-1 and TASK-1 are near zero **V**.

## Structures drawn

All from RCSB PDB; PDB data are CC0. Orientation is from OPM where the entry has one; otherwise from a heuristic recorded in each asset's JSON (symmetry axis, hydrophobic belt, positive-inside rule).

| Draws | Model channel | PDB | Organism | Method, resolution | State | In OPM |
| --- | --- | --- | --- | --- | --- | --- |
| ENaC | apical Na channel (NaLeak) | 6BQN | human αβγ (Fab chains removed) | cryo-EM 3.9 Å | not assigned; transmembrane domain partly ordered | no |
| TWIK-1 | K leak (KLeak) | 3UKM | human | X-ray 3.4 Å | not stated | yes |
| Kv3.1 (stands in for Kv3.4) | Kv3.4 (vgic configuration) | 7PQT | human | cryo-EM 2.65 Å | open | yes |
| Shaker (closed counterpart) | Kv3.4 closed | 9OIC | Drosophila, I384R | cryo-EM 3.48 Å | closed pore, sensors activated | no |
| Na⁺/K⁺-ATPase | the pump | 7E20 | human α1β1 + FXYD2 | cryo-EM 2.7 Å | E2·[2K] | yes |
| Cx43 | gap junctions | 7Z1T | human | cryo-EM 2.26 Å | putative closed | yes |
| ANO1 (TMEM16A) | wound-activated Cl⁻ channels (ClLeak, rival B) | 5OYB | mouse | cryo-EM 3.75 Å | Ca²⁺-bound | yes |

Caveats:
- **No fully resting Kv structure exists** (sensors down and pore closed). The closed counterpart, 9OIC, has activated sensors.
- **ENaC's transmembrane part is the least defined** of the set.
- **The structures stand in for the model's generic channels.** The model's "K leak" is a BETSE leak conductance, not TWIK-1 kinetics.

## Counts: how a conductance becomes molecules

For each face and channel type, the solver exports the channel's linearized conductance per membrane area and its open fraction. Then:
- **Channels in a patch** = conductance per area × patch area ÷ (single-channel conductance × open fraction).
- **Pumps** = pump rate (mol m⁻² s⁻¹) × Avogadro's number × patch area ÷ turnover.

Both counts are labelled *derived*. Positions are uniform (Poisson) by default, or clustered (Thomas process with a stated cluster radius); both are labelled *assumed*.

Gating takes milliseconds and frames are 50 ms apart, so each frame's open or closed states are an independent draw at that face's open fraction. Leak channels are always open.

## In Blender

- **Opening it.** Import a skin simulation, then in the Bioelectrics panel use **Open nanoscope here**. It opens the active face (edit mode, face select), or else the living face nearest the 3D cursor. A new scene opens with:
  - the bilayer, with lipid head groups drawn for patches under 200 nm;
  - the molecules as Geometry Nodes instances (never merged into one mesh);
  - ion beads crossing open channels in the direction of the solver's net flux;
  - 10 nm and 100 nm scale bars;
  - a second camera, 40 nm across, on the molecule nearest the patch centre;
  - a legend with the face, its voltage, the counts (expected and drawn), the open channels, and the solver's ion rates per patch and per open channel.
- **Navigation.** The nanoscope follows the tissue scene's timeline. **Back to the tissue** returns to the tissue scene, where a marker shows the patch.
- **Levels of detail.** At tissue scale, the quantity menu now includes the per-face quantities. `chan_g_naleak`, for example, colours each face by its Na channel conductance, which is proportional to channel density. At cell scale, the per-face glyphs remain. At 1 µm and below, the nanoscope takes over.
- **Rendering cost.** 100,000 instanced molecules (64 million triangles) render at 1080p in under a second with about 1 GB of memory; merged into one mesh they take 4 GB. At a 100-cell view a 10 nm molecule is 0.08 pixel wide, which is why individual molecules appear only in the nanoscope.

## What a realistic patch looks like

With the forearm parameters, a keratinocyte membrane is mostly lipid:

| Molecule | Expected density | Notes |
| --- | --- | --- |
| Na/K-ATPase | about 300 per µm² | from BETSE's pump rate and a turnover of 200 s⁻¹; dog tracheal epithelium has about 2,400 per µm² |
| ENaC | about 2 per µm², apical granular faces only | |
| TWIK-1 (K leak) | about 1 per 60 µm² | |
| Kv3.4 (demonstration configuration) | about 1 per 1,600 µm², roughly one per cell | |

A 500 nm patch of a basolateral granular face shows about 70 pumps and no K channel; an apical face shows on average half an ENaC.

Faces with few or no channels are what the parameters imply, and the model should not hide that. "100 cells with 1,000 voltage-gated channels each" would be a density about 1,000 times what the absence of a measurable voltage-gated current in keratinocytes allows. The nanoscope can draw it, but only if the model's density is changed, and the solver then shows what that does to the voltages.

**Consistency check.** The solver's flux through an apical face, divided by the expected number of open ENaC channels, gives 0.22 pA per channel at +1.1 mV. The sourced single channel (4.7 pS) with the Na⁺ driving force at that voltage gives about 0.31 pA. The 30% difference is the curvature of the GHK flux against a linear chord estimate. For the K leak it is 1.6 pA against about 3 pA. Checked on the screen of the demonstration file, not yet in a test.

Check N1, a unit test: the sampled open fraction stays within 3/√N of the solver's at every frame, and draws are reproducible per seed.
