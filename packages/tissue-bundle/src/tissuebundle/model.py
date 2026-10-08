"""The run-bundle manifest: what ``bundle.json`` may contain.

These models are the contract. The JSON schema in ``schemas/`` is generated from them and a
test fails if the two drift. The reader in :mod:`tissuebundle.reader` does not import this
module, so that Blender can read bundles without pydantic.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = "0.3.0"
SUPPORTED_VERSIONS = ("0.1.0", "0.2.0", "0.3.0")

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]

Location = Literal["cell", "membrane", "grid", "cell_ion", "membrane_ion", "grid_ion", "volume"]
"""Where a quantity lives. The trailing shape of each location is fixed:

- ``cell``: ``[N]``; ``membrane``: ``[M]``; ``grid``: ``[ny, nx]``
- ``*_ion``: the same with a leading ion axis, ``[n_ions, ...]``
"""

Unit = Literal["um", "mV", "V", "V/m", "A/m2", "mol/m3", "m2/s", "mol/m2/s", "S/m2", "s", "1"]
"""Allowed units. Lengths are µm and voltages mV where people read them; the rest is SI.
``V/m`` equals mV/mm, the unit wound fields are usually reported in."""

DType = Literal["float64", "float32", "int64", "int32", "bool"]


class BundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ArrayRecord(BundleModel):
    """One ``.npy`` file in the bundle."""

    file: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*\.npy$")]
    dtype: DType
    shape: tuple[int, ...]
    sha256: Sha256


Domain = Literal["cell", "apical", "basolateral", "lateral", "bath_apical", "bath_basal"]
"""For 3D (extruded) tissue: which membrane domain or bath a quantity belongs to."""


class Quantity(ArrayRecord):
    """A time series: the first axis is always the frame axis."""

    unit: Unit
    location: Location
    description: str
    domain: Domain | None = None
    z_um: float | None = Field(
        default=None, description="Height of a grid quantity's layer above the basal plane"
    )


class ContextMesh(BundleModel):
    """Non-cell anatomy shown with the tissue (e.g. the dermis): triangles or polygons."""

    verts: ArrayRecord
    face_offsets: ArrayRecord
    face_index: ArrayRecord
    color: tuple[float, float, float, float]
    description: str


class Geometry(BundleModel):
    """Cells are polygons in the plane, stored CSR-style.

    Cell ``i`` owns vertices and membranes ``offsets[i]:offsets[i + 1]``; membrane ``k`` of a
    cell is the edge from its vertex ``k - 1`` to its vertex ``k``. Cells are indexed in the
    pre-wound numbering, so one mesh serves every frame.
    """

    n_cells: int = Field(ge=1)
    n_membranes: int = Field(ge=3)
    grid_shape: tuple[int, int] | None
    cell_verts: ArrayRecord
    cell_offsets: ArrayRecord
    cell_centres: ArrayRecord
    cell_removed: ArrayRecord
    mem_grid_index: ArrayRecord | None
    grid_x: ArrayRecord | None
    grid_y: ArrayRecord | None
    mem_neighbour: ArrayRecord | None = Field(
        default=None,
        description="For each membrane, the facing membrane of the neighbouring cell, or -1",
    )
    extrusion_height_um: float | None = Field(
        default=None, description="Cells are prisms of this height when set (3D tissue)"
    )
    kind: Literal["polygons", "polyhedra"] = Field(
        default="polygons",
        description="polygons: cell_verts [V, 2] and cell_offsets index vertices (membranes are "
        "edges). polyhedra: cell_verts [V, 3] are face vertices, face_offsets index them, "
        "cell_offsets index faces, and membranes are faces",
    )
    face_offsets: ArrayRecord | None = None
    face_boundary: ArrayRecord | None = Field(
        default=None, description="Per face: 0 shared with a cell, 1 basement, 2 surface, 3 side"
    )
    cell_layer: ArrayRecord | None = None
    cell_inert: ArrayRecord | None = Field(
        default=None,
        description="Cells with no electrical state (e.g. corneocytes): their "
        "values are NaN in every frame",
    )
    layers: tuple[str, ...] = ()
    layer_colors: tuple[tuple[float, float, float], ...] = ()
    context_meshes: dict[Name, ContextMesh] = Field(default_factory=dict)
    volume_shape: tuple[int, int, int] | None = Field(
        default=None, description="3D extracellular grid [nz, ny, nx] for 'volume' quantities"
    )
    volume_origin_um: tuple[float, float, float] | None = None
    volume_spacing_um: float | None = None
    volume_x: ArrayRecord | None = Field(
        default=None, description="Node x (µm), non-uniform grids"
    )
    volume_y: ArrayRecord | None = None
    volume_z: ArrayRecord | None = None


class Frames(BundleModel):
    count: int = Field(ge=1)
    baseline: tuple[int, ...] = Field(
        description="Frames before the wound; removed cells still hold values in these."
    )
    time_integrated_s: ArrayRecord
    time_reported_s: ArrayRecord


class Provenance(BundleModel):
    solver: str
    solver_version: str
    config_file: str | None
    config_sha256: Sha256 | None
    seed: int | None
    command: tuple[str, ...]
    producer: str
    producer_version: str
    python: str
    numpy: str
    created_utc: str


class Manifest(BundleModel):
    schema_version: Literal["0.1.0", "0.2.0", "0.3.0"] = SCHEMA_VERSION
    name: Name
    description: str
    provenance: Provenance
    ions: tuple[str, ...]
    ion_charges: tuple[int, ...] = Field(description="Valence of each ion, in the order of ions")
    constants: dict[Name, float] = Field(
        description="Solver constants needed to recompute derived quantities; the unit is in "
        "the name, e.g. temperature_k, membrane_thickness_m, gas_constant_j_per_k_mol"
    )
    geometry: Geometry
    frames: Frames
    quantities: dict[Name, Quantity]
