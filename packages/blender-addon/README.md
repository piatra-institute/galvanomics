# blender-addon

A Blender 5.2 extension, **Bioelectric Playback**, that imports a tissue run bundle as one polygon per cell and colours it by a chosen quantity at the current frame. The simulator never runs inside Blender. uv is used here only to lint, test and build.

```
uv sync
uv run pytest                              # pure-Python core, reader sync, and a headless Blender run
uv run python scripts/build_extension.py   # syncs the vendored reader, validates, builds dist/*.zip
```

In Blender: Edit > Preferences > Get Extensions > the drop-down > Install from Disk, and pick `dist/bioelectric_playback-0.1.0.zip`. Then in the 3D viewport sidebar (N), tab **Bioelectrics** > Import Run Bundle, and select the bundle's `bundle.json`. Scrubbing the timeline plays the run; frame 0 is the pre-wound baseline. To inspect a cell, enter Edit Mode in face-select mode and click it: the panel shows its value and a small plot of its time series is drawn in the viewport. Removed (wounded) cells are dark grey. A second object under the tissue shows the extracellular potential on the solver's grid.

Not yet tested in the interactive interface: the panel, the quantity menu and the overlay plot. Import, frame playback and rendering are tested headless.

- `bioelectric_playback/core.py`: no `bpy`; mesh topology, colour ramp, frame mapping.
- `bioelectric_playback/reader.py`: a byte-identical copy of `tissue-bundle`'s reader (a test enforces it).
- `scripts/make_demo.py`: run inside Blender with the extension enabled; imports a bundle and saves a ready-to-open `.blend` beside it (top-down camera, flat attribute colours, a wound-edge cell selected).
- `scripts/smoke_blender.py`: run inside Blender; imports a bundle, compares the mesh attribute with the bundle at several frames, renders PNGs.
- License: GPL-3.0-or-later, as Blender requires for add-ons.
