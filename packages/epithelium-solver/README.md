# epithelium-solver

A polarized epithelium of 3D cells (prisms with apical and basolateral membranes) between an apical and a basal bath, built from BETSE 1.5.0's physics: its Goldman flux law, Na/K-ATPase, gap junctions and channel library, ported and checked against BETSE. Model and assumptions: `../../docs/epithelium.md`. Checks: `../../docs/plan.md`, Phase 1b.

```
uv sync
uv run pytest
uv run episolver run configs/betse-wound-polarized.yaml   # -> data/work/<name>/<name>.tbundle
uv run episolver audit                                    # -> runs/epithelium/result.json
```

- `configs/`: `betse-wound-polarized` (BETSE's default wound tissue, extruded and polarized), `betse-wound-uniform` (the same with BETSE's non-polarized membranes), `hex-wound-polarized` (a regular sheet with a circular wound), `tiny-test` (for tests). Footprints from BETSE need its bundle (`../betse-adapter/data/work/wound-default-a/`), made by `uv run betse-adapter audit`.
- `flux.py`, `channels.py`: the BETSE ports, each naming its source; `betse-adapter/tests/test_channel_port.py` compares them with BETSE (check E5).
- `solver.py`: one sparse Kirchhoff solve per step (backward-Euler capacitors, linearized ionic currents), then the concentration update, charge-consistent to rounding.
- `audit.py`: checks E1 to E8 and reported W1 to W3; closed forms and root-finding written independently of the time stepper.
- The Blender extension's **Simulate** button runs this package with `uv run --project <here> --frozen --offline`, so run `uv sync` here once first.
