"""Run specifications: BETSE's own default configuration plus a short list of overrides.

A spec in ``configs/<name>.yaml`` names the keys it changes and why. The resolved
``sim_config.yaml`` is written into the run's work directory and its SHA-256 goes into the
bundle, so a run is reproducible from (BETSE version, spec, seed).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from betse_adapter.betse_env import import_betse

logger = logging.getLogger(__name__)

CONFIG_NAME = "sim_config.yaml"


@dataclass(frozen=True)
class Override:
    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str
    overrides: tuple[Override, ...]


def load_spec(path: Path) -> RunSpec:
    data = YAML(typ="safe").load(Path(path).read_text(encoding="utf-8"))
    overrides = tuple(Override(tuple(item["path"]), item["value"]) for item in data["set"])
    return RunSpec(name=Path(path).stem, description=data["description"], overrides=overrides)


def apply_overrides(conf: Path, overrides: Sequence[Override]) -> None:
    """Set each key in place, preserving BETSE's comments. Unknown keys are an error."""
    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(conf)
    for override in overrides:
        node = data
        for key in override.path[:-1]:
            node = node[key]
        if override.path[-1] not in node:
            raise KeyError(f"{' / '.join(override.path)} is not in {conf.name}")
        node[override.path[-1]] = override.value
        logger.info("set %s = %r", " / ".join(override.path), override.value)
    with conf.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def materialize(spec: RunSpec, work_dir: Path) -> Path:
    """Write BETSE's default configuration (with its geometry images) and apply the spec."""
    work_dir.mkdir(parents=True, exist_ok=False)
    conf = work_dir / CONFIG_NAME
    import_betse()
    from betse.science.parameters import Parameters

    Parameters().copy_default(trg_conf_filename=str(conf))
    apply_overrides(conf, spec.overrides)
    return conf
