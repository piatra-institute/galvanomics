# betse-adapter

Runs BETSE 1.5.0 in-process (Python 3.12, pinned by uv) with NumPy's global generator seeded first, exports the init and sim phases as one tissue bundle, and runs the Phase 1 audit in `../../docs/plan.md`.

```
uv sync
uv run pytest                                        # unit tests plus one real BETSE run
uv run betse-adapter run configs/wound-default.yaml  # -> data/work/wound-default/wound-default.tbundle
uv run betse-adapter run --from-init data/work/wound-default --sim-dt 5e-5 --name dt-half
uv run betse-adapter audit                           # all audit runs -> runs/<name>/result.json
```

- `configs/<name>.yaml` are run specs: BETSE's own default configuration plus named overrides, each with its reason. The resolved `sim_config.yaml` is written into the work directory and hashed into the bundle.
- `export.py` maps the wound exactly (BETSE keeps the removed indices; removal preserves order), refuses to write if any bitwise identity check fails, and puts the init phase's final state in frame 0 as the pre-wound baseline.
- `ghk.py` is an independent Goldman-Hodgkin-Katz implementation used by the audit.
- `runs/*/result.json` are the committed audit records; `data/work/` is regenerated and gitignored.
