"""Operators, the sidebar panel, and the selected-cell time-series overlay."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import blf
import bpy
import gpu
import numpy as np
from bpy.props import EnumProperty, StringProperty
from gpu_extras.batch import batch_for_shader

from . import core, nanoscope, playback, skin_view


class BIOELECTRIC_OT_import_bundle(bpy.types.Operator):
    """Import a tissue run bundle (select its bundle.json)"""

    bl_idname = "bioelectric.import_bundle"
    bl_label = "Import Run Bundle"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="bundle.json", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        path = Path(self.filepath)
        if path.name == "bundle.json":
            path = path.parent
        try:
            created = playback.import_bundle(context, str(path))
        except Exception as exc:  # surfaced to the user, not swallowed
            self.report({"ERROR"}, f"Cannot import {path}: {exc}")
            return {"CANCELLED"}
        for obj in context.selected_objects:
            obj.select_set(False)
        created[0].select_set(True)
        context.view_layer.objects.active = created[0]
        self.report({"INFO"}, f"Imported {created[0].name}")
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, _event: bpy.types.Event) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def active_cell(obj: bpy.types.Object, bundle) -> int:
    """The cell under the active face: the face itself in 2D, its owning cell for prisms."""
    face = obj.data.polygons.active
    if obj.get(playback.PROP_ROLE) in ("tissue3d", "tissue_poly"):
        cells = playback._topology(obj).get(playback.CELL_ATTR)
        return int(cells[face]) if cells is not None and 0 <= face < len(cells) else -1
    return face if 0 <= face < bundle.n_cells else -1


def _quantity_items(_self: object, context: bpy.types.Context) -> list[tuple[str, str, str]]:
    obj = context.active_object
    bundle = playback.bundle_for(obj) if obj else None
    if bundle is None:
        return []
    role = obj.get(playback.PROP_ROLE)
    if role == "grid":
        current = bundle.quantities.get(obj.get(playback.PROP_QUANTITY, ""), {})
        names = playback.grid_quantities(bundle, current.get("domain"))
    elif role in ("tissue", "tissue3d", "tissue_poly"):
        names = [n for n in playback.cell_quantities(bundle) if not n.startswith("open_")]
        if role == "tissue_poly":  # per-face quantities colour each face (channel densities)
            names += [n for n in bundle.quantities if bundle.location(n) == "membrane"]
    else:
        names = []
    _quantity_items.cache = [(n, n, bundle.quantities[n]["description"]) for n in names]
    return _quantity_items.cache


class BIOELECTRIC_OT_set_quantity(bpy.types.Operator):
    """Choose which quantity colours the active tissue object"""

    bl_idname = "bioelectric.set_quantity"
    bl_label = "Quantity"
    bl_options = {"REGISTER", "UNDO"}

    quantity: EnumProperty(items=_quantity_items)

    def execute(self, context: bpy.types.Context) -> set[str]:
        playback.set_quantity(context.active_object, self.quantity)
        return {"FINISHED"}


class BIOELECTRIC_PT_panel(bpy.types.Panel):
    bl_label = "Bioelectrics"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bioelectrics"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        if nanoscope.is_nanoscope(context.scene):
            box = layout.box()
            box.label(text="Nanoscope", icon="VIEWZOOM")
            for line in nanoscope.legend_lines(context.scene):
                for chunk in _wrap(line, 52):
                    box.label(text=chunk)
            box.operator(BIOELECTRIC_OT_back_to_tissue.bl_idname, icon="BACK")
            return
        layout.operator(BIOELECTRIC_OT_import_bundle.bl_idname, icon="IMPORT")
        draw_simulate(layout, context)
        tissue = _tissue_object(context)
        tissue_bundle = playback.bundle_for(tissue) if tissue else None
        if tissue_bundle is not None and any(
            q.startswith("chan_g_") for q in tissue_bundle.quantities
        ):
            box = layout.box()
            box.label(text="Nanoscope (molecular scale)", icon="VIEWZOOM")
            box.prop(context.scene, "bioelectric_nano_size")
            box.prop(context.scene, "bioelectric_nano_placement")
            box.operator(BIOELECTRIC_OT_open_nanoscope.bl_idname, icon="VIEWZOOM")
            box.label(text="Face: the active one (edit mode), else nearest the 3D cursor")
        section = skin_view.section_object(context.scene)
        if section is not None:
            box = layout.box()
            box.label(text="Section plane", icon="MOD_BOOLEAN")
            box.prop(section, "hide_viewport", text="Hide section", invert_checkbox=False)
            for other in context.scene.objects:
                if other.get("tbundle_role") == "context":
                    box.prop(other, "hide_viewport", text=f"Hide {other.name.split('_')[-1]}")
            box.prop(section, '["tb_show_cells"]', text="Show cut cells (else extracellular)")
            box.prop(section, '["tb_cutaway"]', text="Cut away cells in front")
        obj = context.active_object
        bundle = playback.bundle_for(obj) if obj else None
        if bundle is None:
            layout.label(text="Select an imported tissue")
            return
        quantity = obj.get(playback.PROP_QUANTITY, "")
        unit = bundle.unit(quantity) if quantity else ""
        index = core.frame_index(
            context.scene.frame_current, int(obj.get(playback.PROP_OFFSET, 0)), bundle.n_frames
        )
        t_ms = 1e3 * float(bundle.array("time_integrated_s")[index])
        stage = "baseline (pre-wound)" if index in bundle.baseline_frames else "post-wound"

        box = layout.box()
        box.label(text=bundle.name)
        box.label(text=f"Frame {index}/{bundle.n_frames - 1}, t = {t_ms:.3f} ms, {stage}")
        box.operator_menu_enum(BIOELECTRIC_OT_set_quantity.bl_idname, "quantity", text=quantity)
        row = box.row(align=True)
        row.prop(obj, f'["{playback.PROP_VMIN}"]', text=f"Min ({unit})")
        row.prop(obj, f'["{playback.PROP_VMAX}"]', text=f"Max ({unit})")
        box.label(text=f"Location: {bundle.location(quantity)}" if quantity else "")

        if obj.get(playback.PROP_ROLE) not in ("tissue", "tissue3d", "tissue_poly"):
            return
        cell = active_cell(obj, bundle)
        if not 0 <= cell < bundle.n_cells:
            return
        cell_box = layout.box()
        removed = bool(bundle.removed()[cell])
        centre = bundle.array("cell_centres")[cell]
        cell_box.label(text=f"Cell {cell}" + ("  (removed by wound)" if removed else ""))
        cell_box.label(text=f"Centre ({centre[0]:.1f}, {centre[1]:.1f}) µm")
        shown = core.domain_pair(quantity, set(bundle.quantities)) if quantity else ()
        for name in dict.fromkeys(shown):
            value = float(bundle.array(name)[index, cell])
            text = "n/a" if np.isnan(value) else f"{value:.4g} {bundle.unit(name)}"
            cell_box.label(text=f"{name} = {text}")
        if "tep" in bundle.quantities and obj.get(playback.PROP_ROLE) == "tissue3d":
            value = float(bundle.array("tep")[index, cell])
            cell_box.label(text="tep = n/a" if np.isnan(value) else f"tep = {value:.4g} mV")
        cell_box.label(text="Edit mode, face select: pick a cell")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def _living_faces(bundle) -> np.ndarray:
    domain = np.asarray(bundle.array("face_domain"))[0]
    return np.flatnonzero(np.isfinite(domain) & (domain > 0))


class BIOELECTRIC_OT_open_nanoscope(bpy.types.Operator):
    """Open one membrane face at nanometre scale: bilayer, channel and pump molecules and their
    states, sampled from the simulation (counts derived, positions assumed)"""

    bl_idname = "bioelectric.open_nanoscope"
    bl_label = "Open nanoscope here"

    def execute(self, context: bpy.types.Context) -> set[str]:
        tissue = _tissue_object(context)
        bundle = playback.bundle_for(tissue) if tissue else None
        if bundle is None or bundle.kind != "polyhedra":
            self.report({"ERROR"}, "Needs an imported skin simulation (polyhedral tissue)")
            return {"CANCELLED"}
        living = _living_faces(bundle)
        face = int(tissue.data.polygons.active) if tissue.data.polygons.active >= 0 else -1
        if face not in set(living.tolist()):
            geo = skin_view.face_geometry(bundle)
            cursor = np.array(context.scene.cursor.location)
            d = np.linalg.norm(geo["centroid"][living] - cursor, axis=1)
            face = int(living[int(np.argmin(d))])
        try:
            scene = nanoscope.open_nanoscope(
                context,
                tissue,
                bundle,
                face,
                size_nm=float(context.scene.bioelectric_nano_size),
                placement=context.scene.bioelectric_nano_placement,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.window.scene = scene
        self.report({"INFO"}, f"Nanoscope on face {face}")
        return {"FINISHED"}


class BIOELECTRIC_OT_back_to_tissue(bpy.types.Operator):
    """Return to the tissue scene this nanoscope belongs to"""

    bl_idname = "bioelectric.back_to_tissue"
    bl_label = "Back to the tissue"

    def execute(self, context: bpy.types.Context) -> set[str]:
        target = bpy.data.scenes.get(context.scene.get("tb_follow", ""))
        if target is None:
            return {"CANCELLED"}
        context.window.scene = target
        return {"FINISHED"}


# ---------------------------------------------------------------------------- simulate
def default_uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    return str(candidate) if candidate.exists() else "uv"


class BioelectricPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    solver_project: StringProperty(
        name="Solver project",
        subtype="DIR_PATH",
        description="The epithelium-solver uv project (packages/epithelium-solver)",
    )
    uv_path: StringProperty(name="uv", subtype="FILE_PATH", default=default_uv())

    def draw(self, _context: bpy.types.Context) -> None:
        self.layout.prop(self, "solver_project")
        self.layout.prop(self, "uv_path")


def preferences() -> BioelectricPreferences:
    return bpy.context.preferences.addons[__package__].preferences


class BIOELECTRIC_OT_simulate(bpy.types.Operator):
    """Run the 3D epithelium solver on the chosen configuration (in its own uv environment,
    outside Blender) and import the result"""

    bl_idname = "bioelectric.simulate"
    bl_label = "Simulate"

    skin: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    _process = None
    _timer = None
    _lines: queue.Queue | None = None
    _out: Path | None = None
    _log: list[str] = []

    def invoke(self, context: bpy.types.Context, _event) -> set[str]:
        prefs = preferences()
        project = Path(bpy.path.abspath(prefs.solver_project)) if prefs.solver_project else None
        if project is None or not (project / "pyproject.toml").exists():
            self.report({"ERROR"}, "Set the solver project in the add-on preferences")
            return {"CANCELLED"}
        if self.skin:
            config = write_skin_config(context.scene, project)
            subcommand = "skin-sim"
        else:
            config = Path(bpy.path.abspath(context.scene.bioelectric_config))
            subcommand = "run"
        if not config.is_file():
            self.report({"ERROR"}, f"No configuration at {config}")
            return {"CANCELLED"}
        self._out = project / "data" / "work" / config.stem / f"{config.stem}.tbundle"
        command = [
            prefs.uv_path,
            "run",
            "--project",
            str(project),
            "--frozen",
            "--offline",
            "episolver",
            subcommand,
            str(config),
            "--out",
            str(self._out),
            "--progress",
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._process = subprocess.Popen(
            command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
        )
        self._lines, self._log = queue.Queue(), []
        threading.Thread(target=self._pump, args=(self._process.stderr,), daemon=True).start()
        threading.Thread(target=self._pump, args=(self._process.stdout,), daemon=True).start()
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(f"Simulating {config.name}...")
        return {"RUNNING_MODAL"}

    def _pump(self, stream) -> None:
        for line in stream:
            self._lines.put(line.rstrip())

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        while not self._lines.empty():
            line = self._lines.get()
            self._log.append(line)
            if line.startswith("PROGRESS"):
                context.workspace.status_text_set(f"Simulating: frame {line.split()[1]}")
        if self._process.poll() is None:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        if self._process.returncode != 0:
            tail = " | ".join(self._log[-4:])
            self.report({"ERROR"}, f"Solver failed ({self._process.returncode}): {tail}")
            return {"CANCELLED"}
        target = str(self._out.resolve())
        for obj in list(bpy.data.objects):
            path = obj.get(playback.PROP_PATH)
            if path and str(Path(bpy.path.abspath(path)).resolve()) == target:
                bpy.data.objects.remove(obj)
        playback._BUNDLES.pop(target, None)
        created = playback.import_bundle(context, target)
        context.view_layer.objects.active = created[0]
        self.report({"INFO"}, f"Simulated and imported {created[0].name}")
        return {"FINISHED"}


SKIN_PROPS = (
    ("bioelectric_skin_wound_width", "Wound width (µm)", 40.0, 10.0, 120.0),
    ("bioelectric_skin_wound_depth", "Wound depth into dermis (µm)", 20.0, 0.0, 55.0),
    ("bioelectric_skin_enac", "Apical Na channels (x1e-16 m²/s)", 3.0, 0.0, 30.0),
    ("bioelectric_skin_barrier", "Tight-junction barrier (Ω·cm²)", 2000.0, 100.0, 20000.0),
    ("bioelectric_skin_seconds", "Simulated time after wound (s)", 2.0, 0.2, 10.0),
)


def write_skin_config(scene, project: Path) -> Path:
    """The panel's settings as a run configuration (JSON is valid YAML)."""
    import json

    width = float(scene.bioelectric_skin_wound_width)
    config = {
        "name": "blender-skin",
        "description": "Skin wound simulated from the Blender panel",
        "skin": {
            "wound": {
                "top_width_um": width,
                "bottom_width_um": 0.4 * width,
                "depth_into_dermis_um": float(scene.bioelectric_skin_wound_depth),
            }
        },
        "physics": {
            "apical_na_override": float(scene.bioelectric_skin_enac) * 1e-16,
            "tight_junction_ohm_cm2": float(scene.bioelectric_skin_barrier),
            "time": {
                "init_dt_s": 0.02,
                "init_s": 2.0,
                "dt_s": 0.01,
                "sim_s": float(scene.bioelectric_skin_seconds),
                "sample_s": 0.05,
            },
        },
    }
    path = project / "data" / "work" / "blender-skin" / "blender-skin.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def draw_simulate(layout, context: bpy.types.Context) -> None:
    box = layout.box()
    box.label(text="Skin model (3D anatomy)", icon="MESH_ICOSPHERE")
    for name, *_ in SKIN_PROPS:
        box.prop(context.scene, name)
    op = box.operator(BIOELECTRIC_OT_simulate.bl_idname, text="Simulate skin", icon="PLAY")
    op.skin = True
    box.label(text="Runs outside Blender (~3 min), then imports")
    box = layout.box()
    box.label(text="Prism epithelium solver")
    box.prop(context.scene, "bioelectric_config", text="Config")
    box.operator(BIOELECTRIC_OT_simulate.bl_idname, icon="PLAY").skin = False


def _tissue_object(context: bpy.types.Context):
    obj = context.active_object
    if obj is not None and obj.get(playback.PROP_ROLE) in ("tissue", "tissue3d", "tissue_poly"):
        return obj
    for candidate in context.scene.objects:
        if (
            candidate.get(playback.PROP_ROLE) in ("tissue", "tissue3d", "tissue_poly")
            and candidate.visible_get()
        ):
            return candidate
    return None


def _rect(shader, x: float, y: float, w: float, h: float, color) -> None:
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]},
    )
    shader.uniform_float("color", color)
    batch.draw(shader)


def _text(x: float, y: float, text: str, size: int = 13, alpha: float = 0.95) -> None:
    blf.size(0, size)
    blf.color(0, 1.0, 1.0, 1.0, alpha)
    blf.position(0, x, y, 0)
    blf.draw(0, text)


_CACHE: dict = {}


def _cached(key, compute):
    if key not in _CACHE:
        if len(_CACHE) > 500:
            _CACHE.clear()
        _CACHE[key] = compute()
    return _CACHE[key]


def _probe_series(bundle, probe):
    where = tuple(round(v, 3) for v in probe.matrix_world.translation)
    return _cached(
        ("probe", str(bundle.path), where), lambda: skin_view.probe_series(bundle, probe)
    )


def _draw_nanoscope_legend(context) -> None:
    k = context.preferences.system.dpi / 72.0
    lines = nanoscope.legend_lines(context.scene)
    wrapped = [chunk for line in lines for chunk in _wrap(line, 95)]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    h = (len(wrapped) + 1) * 20 * k
    _rect(shader, 12 * k, 12 * k, 760 * k, h, (0.0, 0.0, 0.0, 0.55))
    gpu.state.blend_set("NONE")
    y = 12 * k + h - 26 * k
    for j, line in enumerate(wrapped):
        _text(24 * k, y, line, int((14 if j == 0 else 12) * k))
        y -= 20 * k


def draw_legend() -> None:
    """What is on screen: the time, what the colours and the scale mean, arrows and glyphs."""
    context = bpy.context
    if context.region is not None and nanoscope.is_nanoscope(context.scene):
        _draw_nanoscope_legend(context)
        return
    obj = _tissue_object(context)
    bundle = playback.bundle_for(obj) if obj else None
    quantity = obj.get(playback.PROP_QUANTITY, "") if obj else ""
    if bundle is None or not quantity or context.region is None:
        return
    index = core.frame_index(
        context.scene.frame_current, int(obj.get(playback.PROP_OFFSET, 0)), bundle.n_frames
    )
    t = float(bundle.array("time_integrated_s")[index])
    three_d = obj.get(playback.PROP_ROLE) == "tissue3d"
    apical, basolateral = core.domain_pair(quantity, set(bundle.quantities))
    unit = bundle.unit(quantity)
    vmin, vmax = float(obj[playback.PROP_VMIN]), float(obj[playback.PROP_VMAX])
    roles = {o.get(playback.PROP_ROLE) for o in context.scene.objects if o.visible_get()}

    layered = obj.get(playback.PROP_ROLE) == "tissue_poly" and quantity == "layer"
    if index in bundle.baseline_frames:
        when = "Intact tissue, before the wound"
    elif t <= 0.0:
        when = "Wounded (geometry only: no simulation in this file)"
    elif t < 1.0:
        when = f"{t * 1e3:.0f} ms after the wound"
    else:
        when = f"{t:.2f} s after the wound"
    lines = [(f"{bundle.name}   frame {index}/{bundle.n_frames - 1}", 14), (when, 18)]
    skin_sim = bundle.volume_shape is not None
    if skin_sim and quantity == "vmem":
        what = "Colour: membrane voltage of living cells (corneocytes: layer colour, inert)"
    elif layered:
        what = "Colour: epidermal layer (dermis in pink below)"
    elif three_d and apical != basolateral:
        what = f"Colour: membrane voltage, {apical} on top faces, {basolateral} on the sides"
    else:
        what = f"Colour: {quantity}"
    lines.append((what, 13))
    notes = []
    if bundle.removed().any():
        notes.append("Gap: cells removed by the wound (gone from frame 1 on)")
    if "field" in roles:
        notes.append("Arrows: electric field in the apical bath, longest = run maximum")
    if "channels" in roles:
        notes.append("Glyphs: ion channels, orange Na (apical), blue K (basolateral)")
    if "grid" in roles and three_d:
        notes.append("Plane below: basal bath potential (red +, blue -)")
    if "channels_gn" in roles:
        notes.append(
            "Glyphs: orange Na channels, blue K channels, green Na/K pumps; "
            "bigger and brighter = more flux"
        )
    if "ions" in roles:
        notes.append(
            "Particles: Na+ entering cells (orange), K+ leaving (blue), "
            "density proportional to computed flux"
        )
    if "section" in roles:
        notes.append(
            "Section (panel): extracellular potential or the cut cells; cells in front cut "
            "away; hide the dermis to see it all. Move: select, G, Y"
        )
    if skin_sim:
        battery = _cached(
            ("battery", str(bundle.path), index), lambda: skin_view.battery_mv(bundle, index)
        )
        if battery is not None:
            lines.append(
                (
                    f"Skin battery: {battery:.1f} mV (inside positive across the "
                    "tight-junction layer)",
                    13,
                )
            )
        probe = next((o for o in context.scene.objects if o.get("tbundle_role") == "probe"), None)
        if probe is not None:
            field, _ = _probe_series(bundle, probe)
            lines.append(
                (
                    f"Probe: field {field[index]:.1f} mV/mm at its position "
                    "(select the probe to plot it)",
                    13,
                )
            )

    k = context.preferences.system.dpi / 72.0  # Retina and UI-scale aware sizes
    x = 30.0 * k
    rows = [(t, sz, (22 if sz < 16 else 26)) for t, sz in lines]
    extra = 22 * len(bundle.layers) - 22 if layered else 0
    height = (sum(r[2] for r in rows) + 40 + 20 * len(notes) + 14 + extra) * k
    bottom = 210.0 * k  # above the selected-cell plot
    top = bottom + height
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    _rect(shader, x - 12 * k, bottom - 8 * k, 600 * k, height + 8 * k, (0.0, 0.0, 0.0, 0.6))
    gpu.state.blend_set("NONE")
    y = top - 20 * k
    for text, size, step in rows:
        _text(x, y, text, int(size * k))
        y -= step * k
    if layered:
        palette = playback.layer_colors(bundle)
        for j, name in enumerate(bundle.layers):
            _rect(shader, x, y, 14 * k, 14 * k, tuple(palette[j]))
            _text(x + 22 * k, y + 2 * k, name, int(13 * k))
            y -= 22 * k
        y -= 18 * k
        for note in notes:
            _text(x, y, note, int(12 * k), 0.85)
            y -= 20 * k
        return
    bar_w, bar_h, segments = 300.0 * k, 12.0 * k, 48
    y -= 4 * k
    values = np.linspace(vmin, vmax, segments + 1)
    colors = core.colormap(values, vmin, vmax)
    pos, col = [], []
    for j in range(segments):
        x0 = x + bar_w * j / segments
        x1 = x + bar_w * (j + 1) / segments
        c0, c1 = tuple(colors[j]), tuple(colors[j + 1])
        pos += [(x0, y), (x1, y), (x1, y + bar_h), (x0, y), (x1, y + bar_h), (x0, y + bar_h)]
        col += [c0, c1, c1, c0, c1, c0]
    smooth = gpu.shader.from_builtin("SMOOTH_COLOR")
    batch_for_shader(smooth, "TRIS", {"pos": pos, "color": col}).draw(smooth)
    _text(x, y - 16 * k, f"{vmin:.3g} {unit}", int(12 * k))
    _text(x + bar_w - 64 * k, y - 16 * k, f"{vmax:.3g} {unit}", int(12 * k))
    if unit == "mV" and "vmem" in quantity:
        _text(x + bar_w + 14 * k, y, "blue: more negative inside", int(12 * k), 0.8)
    y -= 40 * k
    for note in notes:
        _text(x, y, note, int(12 * k), 0.85)
        y -= 20 * k


def draw_overlay() -> None:
    """A small plot of the active cell's series (or of the probe's field), with the current
    frame marked."""
    context = bpy.context
    obj = context.active_object
    if obj is not None and obj.get("tbundle_role") == "probe":
        tissue = _tissue_object(context)
        bundle = playback.bundle_for(tissue) if tissue else None
        if bundle is not None:
            field, _ = _probe_series(bundle, obj)
            _plot_series(context, bundle, tissue, field, "probe field |E| [mV/mm]")
        return
    if obj is None or obj.get(playback.PROP_ROLE) not in ("tissue", "tissue3d", "tissue_poly"):
        return
    bundle = playback.bundle_for(obj)
    quantity = obj.get(playback.PROP_QUANTITY, "")
    if bundle is None or not quantity:
        return
    if bundle.location(quantity) == "membrane":  # per face: plot the active face
        face = int(obj.data.polygons.active)
        if not 0 <= face < bundle.array(quantity).shape[1]:
            return
        series = np.asarray(bundle.array(quantity)[:, face], dtype=np.float64)
        label = f"face {face}  {quantity} [{bundle.unit(quantity)}]"
        _plot_series(context, bundle, obj, series, label)
        return
    cell = active_cell(obj, bundle)
    if not 0 <= cell < bundle.n_cells:
        return
    series = np.asarray(bundle.array(quantity)[:, cell], dtype=np.float64)
    _plot_series(
        context, bundle, obj, series, f"cell {cell}  {quantity} [{bundle.unit(quantity)}]"
    )


def _plot_series(context, bundle, tissue, series: np.ndarray, label: str) -> None:
    series = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(series)
    if finite.sum() < 2:
        return
    k = context.preferences.system.dpi / 72.0
    x0, y0, width, height = 30.0 * k, 40.0 * k, 320.0 * k, 140.0 * k
    lo, hi = float(series[finite].min()), float(series[finite].max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    frames = np.flatnonzero(finite)
    xs = x0 + width * frames / max(bundle.n_frames - 1, 1)
    ys = y0 + height * (series[finite] - lo) / (hi - lo)
    index = core.frame_index(
        context.scene.frame_current, int(tissue.get(playback.PROP_OFFSET, 0)), bundle.n_frames
    )
    marker_x = x0 + width * index / max(bundle.n_frames - 1, 1)

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    frame = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
    for coords, color, kind in (
        (frame + [frame[0]], (1.0, 1.0, 1.0, 0.35), "LINE_STRIP"),
        (list(zip(xs.tolist(), ys.tolist(), strict=True)), (0.52, 0.80, 0.09, 1.0), "LINE_STRIP"),
        ([(marker_x, y0), (marker_x, y0 + height)], (1.0, 1.0, 1.0, 0.8), "LINES"),
    ):
        batch = batch_for_shader(shader, kind, {"pos": coords})
        shader.uniform_float("color", color)
        batch.draw(shader)
    gpu.state.blend_set("NONE")
    blf.size(0, int(12 * k))
    blf.color(0, 1.0, 1.0, 1.0, 0.9)
    blf.position(0, x0, y0 + height + 8 * k, 0)
    blf.draw(0, f"{label}  {lo:.3g} .. {hi:.3g}")


_draw_handle = None
_legend_handle = None
CLASSES = (
    BioelectricPreferences,
    BIOELECTRIC_OT_import_bundle,
    BIOELECTRIC_OT_set_quantity,
    BIOELECTRIC_OT_simulate,
    BIOELECTRIC_OT_open_nanoscope,
    BIOELECTRIC_OT_back_to_tissue,
    BIOELECTRIC_PT_panel,
)


def register() -> None:
    global _draw_handle, _legend_handle
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    for name, label, default, lo, hi in SKIN_PROPS:
        setattr(
            bpy.types.Scene,
            name,
            bpy.props.FloatProperty(name=label, default=default, min=lo, max=hi),
        )
    bpy.types.Scene.bioelectric_nano_size = bpy.props.FloatProperty(
        name="Patch (nm)",
        default=500.0,
        min=50.0,
        max=2000.0,
        description="Side of the square membrane patch; lipid head groups drawn below 200 nm",
    )
    bpy.types.Scene.bioelectric_nano_placement = EnumProperty(
        name="Placement",
        items=[
            ("uniform", "Uniform", "Poisson: molecules independent and uniform (assumed)"),
            ("clustered", "Clustered", "Thomas process, 50 nm clusters of about 5 (assumed)"),
        ],
        default="uniform",
    )
    bpy.types.Scene.bioelectric_config = StringProperty(
        name="Solver configuration",
        subtype="FILE_PATH",
        description="An episolver run configuration (configs/*.yaml in the solver project)",
    )
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay, (), "WINDOW", "POST_PIXEL"
        )
        _legend_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_legend, (), "WINDOW", "POST_PIXEL"
        )


def unregister() -> None:
    global _draw_handle, _legend_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    if _legend_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_legend_handle, "WINDOW")
        _legend_handle = None
    del bpy.types.Scene.bioelectric_config
    for name, *_ in SKIN_PROPS:
        delattr(bpy.types.Scene, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
