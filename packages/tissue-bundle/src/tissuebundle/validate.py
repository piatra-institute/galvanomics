"""Check a bundle against its contract: manifest, hashes, shapes, geometry and NaN placement."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from tissuebundle.model import ArrayRecord, Manifest
from tissuebundle.reader import MANIFEST, TRAILING
from tissuebundle.writer import sha256_file

logger = logging.getLogger(__name__)


def polygon_areas(verts: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Signed shoelace area of every cell; positive means counter-clockwise."""
    x, y = verts[:, 0], verts[:, 1]
    starts, ends = offsets[:-1], offsets[1:]
    idx = np.arange(len(verts))
    cell = np.repeat(np.arange(len(starts)), ends - starts)
    nxt = np.where(idx + 1 < ends[cell], idx + 1, starts[cell])
    cross = x * y[nxt] - x[nxt] * y
    return 0.5 * np.bincount(cell, weights=cross, minlength=len(starts))


def _check_record(root: Path, label: str, record: ArrayRecord, errors: list[str]) -> np.ndarray:
    path = root / record.file
    if not path.is_file():
        errors.append(f"{label}: missing file {record.file}")
        return np.empty(0)
    if sha256_file(path) != record.sha256:
        errors.append(f"{label}: sha256 mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if str(array.dtype) != record.dtype:
        errors.append(f"{label}: dtype {array.dtype} != manifest {record.dtype}")
    if tuple(array.shape) != tuple(record.shape):
        errors.append(f"{label}: shape {array.shape} != manifest {record.shape}")
    return array


def validate_bundle(path: Path) -> list[str]:
    """Return a list of contract violations; an empty list means the bundle is valid."""
    root = Path(path)
    errors: list[str] = []
    try:
        manifest = Manifest.model_validate(json.loads((root / MANIFEST).read_text("utf-8")))
    except FileNotFoundError:
        return [f"no {MANIFEST} in {root}"]
    except (json.JSONDecodeError, ValidationError) as exc:
        return [f"manifest invalid: {exc}"]

    geom = manifest.geometry
    verts = _check_record(root, "cell_verts", geom.cell_verts, errors)
    offsets = _check_record(root, "cell_offsets", geom.cell_offsets, errors)
    centres = _check_record(root, "cell_centres", geom.cell_centres, errors)
    removed = _check_record(root, "cell_removed", geom.cell_removed, errors)
    if errors:
        return errors

    n, m = geom.n_cells, geom.n_membranes
    if geom.kind == "polyhedra":
        errors.extend(_check_polyhedra(root, geom, verts, offsets, centres, removed))
        if errors:
            return errors
    else:
        if verts.ndim != 2 or verts.shape[1] != 2:
            errors.append("cell_verts must be [V, 2]")
        if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(verts):
            errors.append("cell_offsets must be [N + 1], start at 0 and end at V")
        elif np.any(np.diff(offsets) < 3):
            errors.append("every cell needs at least 3 vertices")
        if m != len(verts):
            errors.append("n_membranes must equal the vertex count (one membrane per edge)")
        if centres.shape != (n, 2):
            errors.append("cell_centres must be [N, 2]")
        if removed.shape != (n,):
            errors.append("cell_removed must be [N]")
        if errors:
            return errors
        if not np.all(np.isfinite(verts)):
            errors.append("cell_verts has non-finite values")
        areas = polygon_areas(np.asarray(verts), np.asarray(offsets))
        if np.any(areas <= 0):
            errors.append(
                f"{int(np.sum(areas <= 0))} cells are not counter-clockwise with area > 0"
            )

    sizes = {"n_cells": n, "n_membranes": m, "n_ions": len(manifest.ions)}
    if geom.grid_shape is not None:
        sizes["ny"], sizes["nx"] = geom.grid_shape
        for label, record in (("grid_x", geom.grid_x), ("grid_y", geom.grid_y)):
            if record is None:
                errors.append(f"{label} is required when grid_shape is set")
                continue
            grid = _check_record(root, label, record, errors)
            if grid.shape != tuple(geom.grid_shape):
                errors.append(f"{label} must have shape grid_shape")
        if geom.mem_grid_index is not None:
            index = _check_record(root, "mem_grid_index", geom.mem_grid_index, errors)
            size = geom.grid_shape[0] * geom.grid_shape[1]
            if index.shape != (m,) or index.min() < 0 or index.max() >= size:
                errors.append("mem_grid_index must be [M] with values inside the grid")

    if geom.mem_neighbour is not None:
        neighbour = np.asarray(_check_record(root, "mem_neighbour", geom.mem_neighbour, errors))
        if neighbour.shape != (m,) or neighbour.min() < -1 or neighbour.max() >= m:
            errors.append("mem_neighbour must be [M] with values in [-1, M)")
        else:
            paired = np.flatnonzero(neighbour >= 0)
            owner = np.repeat(np.arange(n), np.diff(offsets))
            if np.any(neighbour[neighbour[paired]] != paired):
                errors.append("mem_neighbour is not symmetric")
            if np.any(owner[neighbour[paired]] == owner[paired]):
                errors.append("mem_neighbour pairs a membrane with its own cell")
    if geom.extrusion_height_um is not None and geom.extrusion_height_um <= 0:
        errors.append("extrusion_height_um must be positive")

    if geom.volume_shape is not None:
        sizes["vz"], sizes["vy"], sizes["vx"] = geom.volume_shape
        if geom.volume_x is None and (geom.volume_origin_um is None or not geom.volume_spacing_um):
            errors.append("volume_shape needs axes or volume_origin_um and volume_spacing_um")

    frames = manifest.frames
    t_int = _check_record(root, "time_integrated_s", frames.time_integrated_s, errors)
    t_rep = _check_record(root, "time_reported_s", frames.time_reported_s, errors)
    if t_int.shape != (frames.count,) or t_rep.shape != (frames.count,):
        errors.append("time arrays must be [T]")
    elif np.any(np.diff(t_int) < 0):
        errors.append("time_integrated_s must be non-decreasing")
    if any(i < 0 or i >= frames.count for i in frames.baseline):
        errors.append("baseline frame index out of range")
    post = np.setdiff1d(np.arange(frames.count), np.asarray(frames.baseline, dtype=int))

    mem_cell = np.repeat(np.arange(n), np.diff(offsets))
    inert = np.zeros(n, bool)
    if geom.cell_inert is not None:
        inert = np.asarray(_check_record(root, "cell_inert", geom.cell_inert, errors), bool)
        if inert.shape != (n,):
            errors.append("cell_inert must be [N]")
            inert = np.zeros(n, bool)
    for key, quantity in manifest.quantities.items():
        array = _check_record(root, key, quantity, errors)
        if array.size == 0:
            continue
        missing = [s for s in TRAILING[quantity.location] if s not in sizes]
        if missing:
            errors.append(f"{key}: location {quantity.location} needs {missing}")
            continue
        expected = (frames.count, *(sizes[s] for s in TRAILING[quantity.location]))
        if tuple(array.shape) != expected:
            errors.append(f"{key}: shape {array.shape} != expected {expected}")
            continue
        if quantity.location in ("cell", "cell_ion", "membrane", "membrane_ion"):
            per_cell = quantity.location.startswith("cell")
            gone = removed if per_cell else removed[mem_cell]
            dead = inert if per_cell else inert[mem_cell]
            values = np.asarray(array)
            gone = gone & ~dead
            errors.extend(_check_nan(key, values, gone, post, skip=dead))
        else:
            flat = np.asarray(array).reshape(array.shape[0], -1)
            whole = np.isfinite(flat).all(axis=1) | np.isnan(flat).all(axis=1)
            if not np.all(whole):
                errors.append(f"{key}: grid quantity has partially non-finite frames")
    return errors


def _check_nan(
    key: str, array: np.ndarray, gone: np.ndarray, post: np.ndarray, skip: np.ndarray | None = None
) -> list[str]:
    """Removed entities are NaN after the wound; everything else is finite whenever the
    quantity is available at all in that frame (a whole frame may be NaN when unavailable)."""
    errors: list[str] = []
    for t in range(array.shape[0]):
        values = array[t]
        if np.all(np.isnan(values)):
            continue
        alive = ~gone if t in post else np.ones_like(gone)
        if skip is not None:
            alive = alive & ~skip
        live = values[..., alive]
        if not np.all(np.isfinite(live)):
            errors.append(f"{key}: frame {t} has non-finite values on live entities")
        if t in post and np.any(gone) and not np.all(np.isnan(values[..., gone])):
            errors.append(f"{key}: frame {t} has values on removed entities")
    return errors


def polyhedron_volumes(verts: np.ndarray, face_offsets: np.ndarray, cell_offsets: np.ndarray):
    """Signed volume of every cell from its faces (positive when faces are ordered outward)."""
    n_faces = len(face_offsets) - 1
    signed = np.empty(n_faces)
    for f in range(n_faces):
        poly = verts[face_offsets[f] : face_offsets[f + 1]]
        vec = np.cross(poly, np.roll(poly, -1, axis=0)).sum(axis=0)
        signed[f] = np.dot(vec, poly.mean(axis=0)) / 6.0
    cell = np.repeat(np.arange(len(cell_offsets) - 1), np.diff(cell_offsets))
    return np.bincount(cell, weights=signed, minlength=len(cell_offsets) - 1)


def _check_polyhedra(root, geom, verts, cell_offsets, centres, removed) -> list[str]:
    errors: list[str] = []
    n, m = geom.n_cells, geom.n_membranes
    if geom.face_offsets is None:
        return ["polyhedra need face_offsets"]
    face_offsets = np.asarray(_check_record(root, "face_offsets", geom.face_offsets, errors))
    if errors:
        return errors
    if verts.ndim != 2 or verts.shape[1] != 3:
        errors.append("cell_verts must be [V, 3] for polyhedra")
    if face_offsets.shape != (m + 1,) or face_offsets[0] != 0 or face_offsets[-1] != len(verts):
        errors.append(
            "face_offsets must be [F + 1], start at 0 and end at V, with F = n_membranes"
        )
    elif np.any(np.diff(face_offsets) < 3):
        errors.append("every face needs at least 3 vertices")
    if cell_offsets.shape != (n + 1,) or cell_offsets[0] != 0 or cell_offsets[-1] != m:
        errors.append("cell_offsets must be [N + 1] over faces, ending at F")
    elif np.any(np.diff(cell_offsets) < 4):
        errors.append("every cell needs at least 4 faces")
    if centres.shape != (n, 3):
        errors.append("cell_centres must be [N, 3] for polyhedra")
    if removed.shape != (n,):
        errors.append("cell_removed must be [N]")
    for name in ("face_boundary", "cell_layer"):
        record = getattr(geom, name)
        if record is not None:
            arr = _check_record(root, name, record, errors)
            expected = (m,) if name == "face_boundary" else (n,)
            if arr.shape != expected:
                errors.append(f"{name} must have shape {expected}")
    if errors:
        return errors
    volumes = polyhedron_volumes(np.asarray(verts), face_offsets, np.asarray(cell_offsets))
    if np.any(volumes <= 0):
        errors.append(f"{int(np.sum(volumes <= 0))} cells have non-positive volume")
    return errors
