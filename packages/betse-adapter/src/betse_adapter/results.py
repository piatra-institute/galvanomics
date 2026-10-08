"""Audit records: ``runs/<name>/result.json`` with each check's verdict and tolerance."""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from betse_adapter import __version__
from betse_adapter.betse_env import betse_version

logger = logging.getLogger(__name__)

TOLERANCES_NOTE = (
    "Tolerances were fixed in docs/plan.md (Phase 1, audit checks) on 2026-10-07, before any "
    "audit run. Machine-checked, not human-confirmed."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class AuditRun:
    def __init__(self, name: str, runs_dir: Path) -> None:
        self.name = name
        self.dir = runs_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "run": name,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": TOLERANCES_NOTE,
            "environment": {
                "betse": betse_version(),
                "betse_adapter": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "machine": platform.machine(),
                "system": platform.system(),
            },
            "checks": {},
            "reported": {},
        }

    def check(self, check_id: str, passed: bool, value: Any, tolerance: str, detail: str) -> bool:
        self.record["checks"][check_id] = {
            "passed": bool(passed),
            "value": value,
            "tolerance": tolerance,
            "detail": detail,
        }
        logger.info("%s %s: %s (%s)", check_id, "PASS" if passed else "FAIL", value, tolerance)
        return bool(passed)

    def report(self, report_id: str, value: Any, detail: str) -> None:
        self.record["reported"][report_id] = {"value": value, "detail": detail}
        logger.info("%s: %s", report_id, value)

    def set(self, **fields: Any) -> None:
        self.record.update(fields)

    def save(self) -> Path:
        checks = self.record["checks"].values()
        self.record["all_passed"] = all(c["passed"] for c in checks) if checks else None
        self.record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        path = self.dir / "result.json"
        path.write_text(json.dumps(_jsonable(self.record), indent=2) + "\n", encoding="utf-8")
        return path
