"""Write a run bundle atomically: build it in a temporary directory, hash it, rename it in."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tissuebundle import __version__
from tissuebundle.model import (
    ArrayRecord,
    ContextMesh,
    Frames,
    Geometry,
    Location,
    Manifest,
    Provenance,
    Quantity,
    Unit,
)
from tissuebundle.reader import MANIFEST

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantityInput:
    """A time series to write; ``data`` has the frame axis first."""

    data: np.ndarray
    unit: Unit
    location: Location
    description: str
    domain: str | None = None
    z_um: float | None = None


@dataclass(frozen=True)
class SourceInfo:
    """Where the run came from. Everything else in the provenance block is filled in here."""

    solver: str
    solver_version: str
    config_file: str | None = None
    config_sha256: str | None = None
    seed: int | None = None
    command: Sequence[str] = ()
    producer: str = "tissue-bundle"
    producer_version: str = __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(directory: Path, name: str, data: np.ndarray) -> ArrayRecord:
    array = np.ascontiguousarray(data)
    path = directory / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    return ArrayRecord(
        file=path.name,
        dtype=str(array.dtype),
        shape=tuple(int(n) for n in array.shape),
        sha256=sha256_file(path),
    )


def write_bundle(
    out: Path,
    *,
    name: str,
    description: str,
    source: SourceInfo,
    ions: Sequence[str],
    ion_charges: Sequence[int] = (),
    constants: Mapping[str, float] | None = None,
    cell_verts: np.ndarray,
    cell_offsets: np.ndarray,
    cell_centres: np.ndarray,
    cell_removed: np.ndarray,
    time_integrated_s: np.ndarray,
    time_reported_s: np.ndarray,
    baseline_frames: Sequence[int],
    quantities: Mapping[str, QuantityInput],
    grid_x: np.ndarray | None = None,
    grid_y: np.ndarray | None = None,
    mem_grid_index: np.ndarray | None = None,
    mem_neighbour: np.ndarray | None = None,
    extrusion_height_um: float | None = None,
    kind: str = "polygons",
    face_offsets: np.ndarray | None = None,
    face_boundary: np.ndarray | None = None,
    cell_layer: np.ndarray | None = None,
    cell_inert: np.ndarray | None = None,
    layers: Sequence[str] = (),
    layer_colors: Sequence[Sequence[float]] = (),
    context_meshes: Mapping[str, tuple] | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    volume_origin_um: tuple[float, float, float] | None = None,
    volume_spacing_um: float | None = None,
    volume_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a bundle directory at ``out`` and return its path.

    Shapes are checked by :func:`tissuebundle.validate.validate_bundle`, which callers should
    run on the result; this function only enforces dtypes and the manifest model.
    """
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    if len(ion_charges) != len(ions):
        raise ValueError("ion_charges must give one valence per ion")
    if (grid_x is None) != (grid_y is None):
        raise ValueError("grid_x and grid_y must be given together")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:

        def optional(name: str, data: np.ndarray | None, dtype: type) -> ArrayRecord | None:
            return None if data is None else _save(staging, name, np.asarray(data, dtype=dtype))

        meshes = {}
        for key, (verts, faces, color, desc) in (context_meshes or {}).items():
            sizes = np.array([len(f) for f in faces], dtype=np.int64)
            meshes[key] = ContextMesh(
                verts=_save(staging, f"ctx_{key}_verts", np.asarray(verts, np.float64)),
                face_offsets=_save(
                    staging, f"ctx_{key}_face_offsets", np.concatenate(([0], np.cumsum(sizes)))
                ),
                face_index=_save(
                    staging,
                    f"ctx_{key}_face_index",
                    np.concatenate([np.asarray(f, np.int64) for f in faces]),
                ),
                color=tuple(float(c) for c in color),
                description=desc,
            )
        polyhedra = kind == "polyhedra"
        geometry = Geometry(
            n_cells=int(len(cell_offsets) - 1),
            n_membranes=int(len(face_offsets) - 1) if polyhedra else int(cell_offsets[-1]),
            grid_shape=None if grid_x is None else (int(grid_x.shape[0]), int(grid_x.shape[1])),
            cell_verts=_save(staging, "cell_verts", np.asarray(cell_verts, dtype=np.float64)),
            cell_offsets=_save(staging, "cell_offsets", np.asarray(cell_offsets, dtype=np.int64)),
            cell_centres=_save(staging, "cell_centres", np.asarray(cell_centres, np.float64)),
            cell_removed=_save(staging, "cell_removed", np.asarray(cell_removed, dtype=bool)),
            mem_grid_index=None
            if mem_grid_index is None
            else _save(staging, "mem_grid_index", np.asarray(mem_grid_index, dtype=np.int64)),
            grid_x=None if grid_x is None else _save(staging, "grid_x", grid_x.astype(np.float64)),
            grid_y=None if grid_y is None else _save(staging, "grid_y", grid_y.astype(np.float64)),
            mem_neighbour=None
            if mem_neighbour is None
            else _save(staging, "mem_neighbour", np.asarray(mem_neighbour, dtype=np.int64)),
            extrusion_height_um=None
            if extrusion_height_um is None
            else float(extrusion_height_um),
            kind=kind,
            face_offsets=optional("face_offsets", face_offsets, np.int64),
            face_boundary=optional("face_boundary", face_boundary, np.int64),
            cell_layer=optional("cell_layer", cell_layer, np.int64),
            cell_inert=optional("cell_inert", cell_inert, bool),
            layers=tuple(layers),
            layer_colors=tuple(tuple(float(c) for c in col) for col in layer_colors),
            context_meshes=meshes,
            volume_shape=None if volume_shape is None else tuple(int(n) for n in volume_shape),
            volume_origin_um=None
            if volume_origin_um is None
            else tuple(float(v) for v in volume_origin_um),
            volume_spacing_um=None if volume_spacing_um is None else float(volume_spacing_um),
            volume_x=None if volume_axes is None else _save(staging, "volume_x", volume_axes[0]),
            volume_y=None if volume_axes is None else _save(staging, "volume_y", volume_axes[1]),
            volume_z=None if volume_axes is None else _save(staging, "volume_z", volume_axes[2]),
        )
        frames = Frames(
            count=int(len(time_integrated_s)),
            baseline=tuple(int(i) for i in baseline_frames),
            time_integrated_s=_save(
                staging, "time_integrated_s", np.asarray(time_integrated_s, np.float64)
            ),
            time_reported_s=_save(
                staging, "time_reported_s", np.asarray(time_reported_s, np.float64)
            ),
        )
        records: dict[str, Quantity] = {}
        for key, quantity in quantities.items():
            if key in Geometry.model_fields or key in Frames.model_fields:
                raise ValueError(f"quantity name {key!r} collides with a reserved array name")
            record = _save(staging, key, quantity.data)
            records[key] = Quantity(
                **record.model_dump(),
                unit=quantity.unit,
                location=quantity.location,
                description=quantity.description,
                domain=quantity.domain,
                z_um=quantity.z_um,
            )
        manifest = Manifest(
            name=name,
            description=description,
            provenance=Provenance(
                solver=source.solver,
                solver_version=source.solver_version,
                config_file=source.config_file,
                config_sha256=source.config_sha256,
                seed=source.seed,
                command=tuple(source.command),
                producer=source.producer,
                producer_version=source.producer_version,
                python=platform.python_version(),
                numpy=np.__version__,
                created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
            ions=tuple(ions),
            ion_charges=tuple(int(z) for z in ion_charges),
            constants={k: float(v) for k, v in (constants or {}).items()},
            geometry=geometry,
            frames=frames,
            quantities=records,
        )
        (staging / MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("wrote bundle %s (%d quantities)", out, len(quantities))
    return out


def command_line() -> tuple[str, ...]:
    return tuple(sys.argv)
