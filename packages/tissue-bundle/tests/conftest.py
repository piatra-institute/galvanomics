from __future__ import annotations

from pathlib import Path

import pytest

from tissuebundle.synthetic import make_synthetic


@pytest.fixture
def synthetic(tmp_path: Path) -> Path:
    return make_synthetic(tmp_path / "synthetic.tbundle", cols=6, rows=5, frames=4)
