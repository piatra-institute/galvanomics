# Third-party notices

## BETSE

`packages/epithelium-solver/src/episolver/flux.py` and `channels.py` contain code ported from BETSE 1.5.0 (the BioElectric Tissue Simulation Engine, https://gitlab.com/betse/betse): the transmembrane electrodiffusion flux, the Na⁺/K⁺-ATPase pump, the ion channel gating functions and the gap-junction gate. Parameter defaults in `params.py` (ion profiles, membrane diffusion constants, pump constants) are BETSE's. `packages/betse-adapter` runs BETSE but does not redistribute it. BETSE's licence:

```
Copyright 2014-2025 by Alexis Pietak & Cecil Curry.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Protein Data Bank structures

The molecule meshes in `packages/blender-addon/bioelectric_playback/assets/channels/` are built from Protein Data Bank entries 6BQN, 3UKM, 7PQT, 9OIC, 7E20, 7Z1T and 5OYB (https://www.rcsb.org). Membrane orientations are from OPM (https://opm.phar.umich.edu) where available. PDB data are dedicated to the public domain under CC0 1.0 (wwPDB, https://www.wwpdb.org/about/usage-policies). Each asset's JSON names its entry; please cite the structure's primary publication when using a figure made from it.

## Colour map

The diverging blue–white–red colour ramp in `packages/blender-addon/bioelectric_playback/core.py` uses anchor colours of Kenneth Moreland's cool-warm map ("Diverging Color Maps for Scientific Visualization", 2009).
