from __future__ import annotations

from pathlib import Path

from tissuebundle.cli import schema_json
from tissuebundle.model import SCHEMA_VERSION

SCHEMA = Path(__file__).parents[1] / "schemas" / f"tissue-bundle-v{SCHEMA_VERSION}.schema.json"


def test_committed_schema_matches_model() -> None:
    hint = f"schema drifted: run `uv run tissuebundle schema > schemas/{SCHEMA.name}`"
    assert SCHEMA.read_text(encoding="utf-8") == schema_json(), hint
