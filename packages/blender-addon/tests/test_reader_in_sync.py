from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
UPSTREAM = ROOT.parent / "tissue-bundle" / "src" / "tissuebundle" / "reader.py"


def test_vendored_reader_is_byte_identical() -> None:
    vendored = ROOT / "bioelectric_playback" / "reader.py"
    assert vendored.read_bytes() == UPSTREAM.read_bytes(), (
        "vendored reader drifted: run `uv run python scripts/build_extension.py`"
    )
