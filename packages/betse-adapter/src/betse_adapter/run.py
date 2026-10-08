"""Run BETSE phases in-process with NumPy's global generator seeded first.

BETSE never seeds NumPy itself, and its lattice disorder draws from the global generator, so
the seed has to be set here, before the seed phase, for a run to be reproducible.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from betse_adapter.betse_env import import_betse
from betse_adapter.config import CONFIG_NAME, Override, apply_overrides

logger = logging.getLogger(__name__)

INIT_PICKLE = Path("INITS") / "init_1.betse.gz"
SIM_PICKLE = Path("SIMS") / "sim_1.betse.gz"
VMEM_CSV_DIR = Path("RESULTS") / "sim_1" / "Vmem2D_TextExport"
PHASES = ("seed", "init", "sim")


def run_phases(
    conf: Path, seed: int | None, phases: Sequence[str] = PHASES, plot_sim: bool = False
) -> dict[str, float]:
    """Run the named phases on ``conf``; returns wall time per phase in seconds."""
    import_betse()
    from betse.science.parameters import Parameters
    from betse.science.simrunner import SimRunner

    if seed is not None:
        np.random.seed(seed)
    runner = SimRunner(p=Parameters.make(conf_filename=str(conf)))
    timings: dict[str, float] = {}
    for phase in (*phases, *(("plot_sim",) if plot_sim else ())):
        start = time.perf_counter()
        getattr(runner, phase)()
        timings[phase] = time.perf_counter() - start
        logger.info("%s finished in %.2f s", phase, timings[phase])
    return timings


def clone_for_sim(source: Path, target: Path, overrides: Sequence[Override]) -> Path:
    """A new work directory sharing ``source``'s configuration and init, for sim-only runs."""
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / CONFIG_NAME, target / CONFIG_NAME)
    for folder in ("geo", "extra_configs", "INITS"):
        shutil.copytree(source / folder, target / folder)
    conf = target / CONFIG_NAME
    apply_overrides(conf, overrides)
    return conf
