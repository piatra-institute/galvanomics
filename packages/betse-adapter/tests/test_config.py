from __future__ import annotations

from pathlib import Path

import pytest

from betse_adapter.config import Override, apply_overrides, load_spec

CONFIGS = Path(__file__).parents[1] / "configs"


def test_overrides_preserve_comments_and_reject_unknown_keys(tmp_path: Path) -> None:
    conf = tmp_path / "c.yaml"
    conf.write_text("a:\n  b: 1  # keep me\n  c: x\n")
    apply_overrides(conf, [Override(("a", "b"), 2.5)])
    text = conf.read_text()
    assert "b: 2.5" in text and "# keep me" in text
    with pytest.raises(KeyError):
        apply_overrides(conf, [Override(("a", "missing"), 1)])


@pytest.mark.parametrize("name", ["wound-default", "ghk-audit", "pump-blocked"])
def test_committed_specs_load(name: str) -> None:
    spec = load_spec(CONFIGS / f"{name}.yaml")
    assert spec.name == name and spec.description and spec.overrides
