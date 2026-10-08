# tissue-bundle

The on-disk run-bundle format for bioelectric tissue simulations: one directory with a `bundle.json` manifest and one uncompressed `.npy` per array. Any solver can write it; Blender, notebooks and other viewers read it with NumPy alone. The contract is in `../../docs/architecture.md`.

```
uv sync
uv run pytest
uv run tissuebundle make-synthetic /tmp/s.tbundle   # a made-up wounded hexagonal sheet
uv run tissuebundle validate /tmp/s.tbundle
uv run tissuebundle info /tmp/s.tbundle
uv run tissuebundle schema > schemas/tissue-bundle-v0.1.0.schema.json   # after changing model.py
```

- `model.py`: the manifest models (pydantic); the committed JSON schema is generated from them and a test fails if the two drift.
- `reader.py`: standard library and NumPy only, so it can be vendored into the Blender extension; a test checks its imports.
- `writer.py`: writes atomically and hashes every array.
- `validate.py`: hashes, shapes per declared location, CSR geometry, counter-clockwise polygons, NaN exactly on removed cells after the wound.
