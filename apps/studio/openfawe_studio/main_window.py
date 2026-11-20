"""The studio window: set-up on the left, the 3D view in the middle, live plots below, run controls on the toolbar."""
import multiprocessing as mp
import queue
import zipfile
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_NAME, COPYRIGHT, __version__
from .runstore import RunFolder, export_csv, export_paraview, new_run_dir
from .scene import MODES, Scene
from .spec import AIRCRAFT, COMPUTE, FLIGHT_AERO, GUSTS, METHODS, MOTIONS, Spec

PLOT_DEFAULTS = ("CL", "CD")


def gpu_available():
    try:
        from .resources import ensure_library
        ensure_library()
        from openfawe.aero.vortex import cuda_backend_available
        return bool(cuda_backend_available())
    except Exception:  # noqa: BLE001
        return False


class SetupPanel(QtWidgets.QScrollArea):
    """A form over the Spec fields; every choice shows what it runs and how far it is checked."""
    changed = QtCore.Signal()

    def __init__(self, gpu):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(330)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        self.layout = QtWidgets.QVBoxLayout(body)
        self.widgets = {}
        self.gpu = gpu
        s = Spec()
        self._group("Aircraft and aerodynamics", [
            ("name", self._line(s.name), "Run name"),
            ("aircraft", self._combo(AIRCRAFT, s.aircraft), "Aircraft"),
            ("method", self._combo(METHODS, s.method), "Vortex model"),
            ("compute", self._combo(COMPUTE, s.compute, disabled={k for k, v in COMPUTE.items() if v["device"] and not gpu}),
             "Compute"),
            ("threads", self._int(s.threads, 0, 256, "all cores"), "CPU threads")])
        self._group("Motion", [
            ("motion", self._combo(MOTIONS, s.motion), "Motion"),
            ("alpha_deg", self._float(s.alpha_deg, -89, 89, 0.5, " deg"), "Incidence"),
            ("beta_deg", self._float(s.beta_deg, -89, 89, 0.5, " deg"), "Sideslip (fixed)"),
            ("height_m", self._float(s.height_m, 0, 2000, 10, " m"), "Height / loop centre"),
            ("loop_radius_m", self._float(s.loop_radius_m, 1, 2000, 10, " m"), "Loop radius"),
            ("loop_speed_m_s", self._float(s.loop_speed_m_s, 1, 200, 1, " m/s"), "Loop speed")])
        self._group("Inflow", [
            ("wind_m_s", self._float(s.wind_m_s, 0, 60, 0.5, " m/s"), "Wind speed"),
            ("shear_exponent", self._float(s.shear_exponent, 0, 1, 0.01, ""), "Shear exponent"),
            ("shear_reference_height_m", self._float(s.shear_reference_height_m, 1, 2000, 10, " m"), "Shear ref. height"),
            ("gust", self._combo({k: dict(label=v) for k, v in GUSTS.items()}, s.gust), "Gust (1-cos)"),
            ("gust_amplitude_m_s", self._float(s.gust_amplitude_m_s, 0, 30, 0.5, " m/s"), "Gust amplitude"),
            ("gust_start_s", self._float(s.gust_start_s, 0, 1e4, 0.1, " s"), "Gust start"),
            ("gust_duration_s", self._float(s.gust_duration_s, 0.01, 1e4, 0.1, " s"), "Gust duration")])
        self._group("Time and wake", [
            ("dt_s", self._float(s.dt_s, 1e-4, 1, 0.005, " s", decimals=4), "Time step"),
            ("steps", self._int(s.steps, 1, 100000), "Steps"),
            ("core_radius_m", self._float(s.core_radius_m, 0.001, 5, 0.01, " m", decimals=3), "Vortex core"),
            ("free_convection_rows", self._int(0, 0, 100000, "all free"), "Free rows (frozen beyond)"),
            ("retained_rows", self._int(0, 0, 100000, "keep all"), "Kept rows (cut beyond)")])
        self._group("Rotors", [
            ("rotors", self._check(s.rotors, "8 actuator discs with ring wakes"), "Rotors"),
            ("rotor_omega_rad_s", self._float(s.rotor_omega_rad_s, 1, 400, 5, " rad/s"), "Rotor speed"),
            ("ring_nodes", self._int(s.ring_nodes, 3, 128), "Nodes per ring"),
            ("ring_surface_core_m", self._float(s.ring_surface_core_m, 0.01, 5, 0.05, " m"), "Ring core at surfaces"),
            ("retained_ring_steps", self._int(0, 0, 100000, "keep all"), "Kept ring steps"),
            ("free_convection_ring_steps", self._int(0, 0, 100000, "all free"), "Free ring steps")])
        self._group("Display and output", [
            ("fuselage", self._check(s.fuselage, "draw the fuselage (not in the aerodynamics)"), "Fuselage"),
            ("tether", self._check(s.tether, "lumped-mass tether computed (fixed aircraft, Makani loop)"), "Tether"),
            ("tether_drawn", self._check(s.tether_drawn, "draw the ground station"), "Ground station"),
            ("frame_every", self._int(s.frame_every, 1, 10000), "Store 3D frame every")])
        self._group("Free flight (TurnKey)", [                     # OF-0010 revision 13
            ("flight_aero", self._combo(FLIGHT_AERO, s.flight_aero), "Aerodynamics"),
            ("power_system", self._check(s.power_system, "Makani power system (motors, stacked power, tether loss)"),
             "Power system"),
            ("blown_wing", self._check(s.blown_wing, "blown wing (with the tables)"), "Rotor slipstream"),
            ("keep_rows", self._int(0, 0, 100000, "1872"), "Kept wake rows (vortex)"),
            ("max_periods", self._int(s.max_periods, 0, 200000, "whole flight"), "Periods (0.01 s)"),
            # review STR-03: the free flight's own frame intervals (frame_every above is the prescribed motions')
            ("flight_frame_every", self._int(s.flight_frame_every, 1, 100000), "Store frame every (periods)"),
            ("wake_frame_every", self._int(s.wake_frame_every, 1, 100000), "Store wake every (periods)")])
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#555; font-size:11px")
        self.layout.addWidget(self.note)
        self.layout.addStretch(1)
        self._update_note()

    # form helpers
    def _group(self, title, rows):
        box = QtWidgets.QGroupBox(title)
        self.groups = getattr(self, "groups", {})
        self.groups[title] = box
        form = QtWidgets.QFormLayout(box)
        for name, widget, label in rows:
            self.widgets[name] = widget
            form.addRow(label, widget)
        self.layout.addWidget(box)

    def _line(self, value):
        w = QtWidgets.QLineEdit(value)
        w.textChanged.connect(self.changed)
        return w

    def _combo(self, catalogue, value, disabled=()):
        w = QtWidgets.QComboBox()
        for key, item in catalogue.items():
            w.addItem(item["label"], key)
            index = w.count()-1
            if item.get("status"):
                w.setItemData(index, item["status"], QtCore.Qt.ToolTipRole)
            if key in disabled:
                w.model().item(index).setEnabled(False)
                w.setItemData(index, "no CUDA device found", QtCore.Qt.ToolTipRole)
        w.setCurrentIndex(w.findData(value))
        w.currentIndexChanged.connect(self._update_note)
        w.currentIndexChanged.connect(self.changed)
        return w

    def _float(self, value, low, high, step, suffix, decimals=2):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(low, high)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setSuffix(suffix)
        w.setValue(value)
        w.valueChanged.connect(self.changed)
        return w

    def _int(self, value, low, high, special=None):
        w = QtWidgets.QSpinBox()
        w.setRange(low, high)
        if special:
            w.setSpecialValueText(special)
        w.setValue(value)
        w.valueChanged.connect(self.changed)
        return w

    def _check(self, value, text):
        w = QtWidgets.QCheckBox(text)
        w.setChecked(value)
        w.toggled.connect(self.changed)
        return w

    def _update_note(self):
        get = lambda name: self.widgets[name].currentData() if name in self.widgets else None  # noqa: E731
        if "blown_wing" in self.widgets:                # the vortex choices set the blown wing themselves (STU-08)
            self.widgets["blown_wing"].setEnabled((get("flight_aero") or "tables") == "tables")
        free = get("motion") == "free_flight"           # review STR-03 (ST-02): what the free flight does not use is grey
        groups = getattr(self, "groups", {})
        if "Inflow" in groups:
            groups["Inflow"].setEnabled(not free)
            groups["Inflow"].setToolTip("The free flight takes the wind of the GS02 parameter file (spec.json, settings.json)"
                                        if free else "")
        for name, used in (("frame_every", not free), ("flight_frame_every", free), ("wake_frame_every", free)):
            if name in self.widgets:
                self.widgets[name].setEnabled(used)
        method, compute = get("method") or "vsm_free_wake", get("compute") or "cpu_tree"
        text = (f"<b>{METHODS[method]['label']}</b>: {METHODS[method]['status']}.<br>"
                f"<b>Compute</b>: {COMPUTE[compute]['status']}.")
        if method in ("vlm", "vlm_free_wake"):
            text += "<br>The lattice methods sum directly; the compute choice does not change them."
        if get("motion") == "free_flight":
            aero = get("flight_aero") or "tables"
            text += (f"<br><br><b>Free flight</b>: Makani's controller flies the whole flight (perch, hover, transition, "
                     f"crosswind loops, landing) with the ground station, winch, tether, rotors and motors. "
                     f"<b>{FLIGHT_AERO[aero]['label']}</b>: {FLIGHT_AERO[aero]['status']}.")
            if aero != "tables":
                text += ("<br>The vortex flight runs on 'CPU - tree code' or 'CPU + GPU' only. The vortex model, method and "
                         "time step above are not used: the hook has its own (VSM + free wake, 0.01 s); the blown wing is "
                         "the hook's (only where the tables act for B, none in hover-on-vortex). On a CPU a whole flight "
                         "takes days; stop it whenever you like - what was computed is kept. A whole vortex flight stores "
                         "a few GB: the records (records.jsonl) and 3D frames with the wake, at most every 200 periods.")
        else:
            text += ("<br><br>Prescribed motion: the aircraft does not fly freely here; the loads are those of the chosen "
                     "model along the given path. Tether, gravity and the fuselage's aerodynamics are not included.")
        self.note.setText(text)

    def spec(self):
        values = {}
        for name, w in self.widgets.items():
            if isinstance(w, QtWidgets.QComboBox):
                values[name] = w.currentData()
            elif isinstance(w, QtWidgets.QCheckBox):
                values[name] = w.isChecked()
            elif isinstance(w, QtWidgets.QLineEdit):
                values[name] = w.text() or "run"
            else:
                values[name] = w.value()
        for name in ("free_convection_rows", "retained_rows", "retained_ring_steps", "free_convection_ring_steps", "keep_rows"):
            values[name] = int(values[name]) or None
        for name in ("steps", "threads", "ring_nodes", "frame_every", "max_periods", "flight_frame_every", "wake_frame_every"):
            values[name] = int(values[name])
        return Spec(**values).validate()

    def load(self, spec):
        for name, w in self.widgets.items():
            value = getattr(spec, name)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(w.findData(value))
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QtWidgets.QLineEdit):
                w.setText(str(value))
            else:
                w.setValue(0 if value is None else value)


class Plots(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        import pyqtgraph as pg
        pg.setConfigOptions(background="w", foreground="#333333", antialias=True)
        layout = QtWidgets.QHBoxLayout(self)
        self.series = {}
        self.rows, self.stride, self.seen = 0, 1, 0
        self.boxes, self.curves, self.views = [], [], []
        for default in PLOT_DEFAULTS:
            column = QtWidgets.QVBoxLayout()
            box = QtWidgets.QComboBox()
            box.addItem(default)
            box.currentTextChanged.connect(self.redraw)
            view = pg.PlotWidget()
            view.showGrid(x=True, y=True, alpha=0.25)
            view.setLabel("bottom", "time [s]")
            view.getAxis("bottom").enableAutoSIPrefix(False)
            view.getAxis("left").enableAutoSIPrefix(False)
            curve = view.plot(pen=pg.mkPen("#1f5f9e", width=2))
            column.addWidget(box)
            column.addWidget(view)
            layout.addLayout(column, 1)
            self.boxes.append(box)
            self.curves.append(curve)
            self.views.append(view)
        column = QtWidgets.QVBoxLayout()
        column.addWidget(QtWidgets.QLabel("Bound circulation along the span"))
        self.span = pg.PlotWidget()
        self.span.showGrid(x=True, y=True, alpha=0.25)
        self.span.setLabel("bottom", "span y (body) [m]")
        self.span.setLabel("left", "Gamma [m2/s]")
        self.span_curve = self.span.plot(pen=None, symbol="o", symbolSize=5, symbolBrush="#d1495b")
        column.addWidget(self.span)
        layout.addLayout(column, 1)

    def defaults(self, names):
        for box, name in zip(self.boxes, names):
            box.blockSignals(True)
            box.clear()
            box.addItem(name)
            box.blockSignals(False)

    MAX_POINTS = 20_000                                # above it every second point is dropped (OF-0010 revision 13, STU-01)

    def clear(self):
        self.series = {}
        self.rows, self.stride, self.seen = 0, 1, 0
        for box in self.boxes:
            box.blockSignals(True)
            current = box.currentText()
            box.clear()
            box.addItem(current)
            box.blockSignals(False)
        self.redraw()

    def add(self, row):
        self.seen += 1
        if (self.seen-1) % self.stride:
            return
        known = set(self.series)
        for key, value in row.items():                 # every series as long as the rows: NaN where a key is absent
            if key not in self.series:
                self.series[key] = [np.nan]*self.rows
            self.series[key].append(value)
        for key in known-set(row):
            self.series[key].append(np.nan)
        self.rows += 1
        if self.rows > self.MAX_POINTS:
            for key in self.series:
                self.series[key] = self.series[key][::2]
            self.rows = len(next(iter(self.series.values())))
            self.stride *= 2
        new = sorted(set(self.series)-known)
        if new:
            for box in self.boxes:
                box.blockSignals(True)
                current = box.currentText()
                box.clear()
                box.addItems(sorted(self.series))
                box.setCurrentText(current)
                box.blockSignals(False)

    def redraw(self):
        t = self.series.get("time_s", [])
        for box, curve, view in zip(self.boxes, self.curves, self.views):
            name = box.currentText()
            values = self.series.get(name, [])
            n = min(len(t), len(values))
            curve.setData(np.asarray(t[:n], float), np.asarray(values[:n], float))
            view.setLabel("left", name)

    def spanwise(self, frame):
        mid = 0.5*(frame["leading_edge"][frame["panel_edges"][:, 0]]+frame["leading_edge"][frame["panel_edges"][:, 1]])
        y = (mid-frame["position"])@frame["rotation"][:, 1]
        self.span_curve.setData(y, frame["bound_gamma"])


class StudioWindow(QtWidgets.QMainWindow):
    def __init__(self, runs_root=None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1500, 950)
        self.runs_root = Path(runs_root or Path.home()/"OpenFAWE Studio runs")
        self.process = None
        self.run = None              # RunFolder of the run shown
        self.live = False
        self.count = 0
        self.ctx = mp.get_context("spawn")
        self.setup = SetupPanel(gpu_available())
        dock = QtWidgets.QDockWidget("Simulation set-up")
        dock.setWidget(self.setup)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        import pyvistaqt
        self.view3d = pyvistaqt.QtInteractor(self)
        self.scene = None
        self.plots = Plots()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.show_stored)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(top)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.view3d.interactor, 1)
        column.addWidget(self.slider)
        split.addWidget(top)
        split.addWidget(self.plots)
        split.setSizes([620, 300])
        self.setCentralWidget(split)
        self._toolbar()
        self.status = QtWidgets.QLabel("Ready.")
        self.statusBar().addWidget(self.status, 1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(60)
        self.replay_timer = QtCore.QTimer(self)          # A1: steps through the stored frames
        self.replay_timer.timeout.connect(self.replay_step)
        self.spec_shown = self.setup.spec()
        self.run_dir = None
        self.new_scene(self.spec_shown)

    def _toolbar(self):
        bar = self.addToolBar("Run")
        bar.setMovable(False)
        style = self.style()
        act = lambda text, icon, slot: bar.addAction(style.standardIcon(icon), text, slot)  # noqa: E731
        self.run_action = act("Run", QtWidgets.QStyle.SP_MediaPlay, self.start)
        self.pause_action = act("Pause", QtWidgets.QStyle.SP_MediaPause, self.toggle_pause)
        self.stop_action = act("Stop", QtWidgets.QStyle.SP_MediaStop, self.stop)
        bar.addSeparator()
        act("Open run", QtWidgets.QStyle.SP_DialogOpenButton, self.open_run)
        act("Import job run", QtWidgets.QStyle.SP_FileDialogNewFolder, self.import_job_run)      # A4
        self.play_action = act("Play replay", QtWidgets.QStyle.SP_MediaSeekForward, self.toggle_replay)   # A1
        act("Export CSV", QtWidgets.QStyle.SP_DialogSaveButton, self.export_csv)
        act("Export ParaView", QtWidgets.QStyle.SP_DriveFDIcon, self.export_paraview)
        act("Screenshot", QtWidgets.QStyle.SP_DesktopIcon, self.screenshot)
        act("Export video", QtWidgets.QStyle.SP_MediaVolume, self.export_video)              # A3
        bar.addSeparator()
        self.mode_box = QtWidgets.QComboBox()
        for key, label in MODES.items():
            self.mode_box.addItem(label, key)
        self.mode_box.currentIndexChanged.connect(self.restyle)
        bar.addWidget(QtWidgets.QLabel(" Colour: "))
        bar.addWidget(self.mode_box)
        self.toggles = {}
        for key, text in (("show_wake", "Wake"), ("show_rings", "Rings"), ("colour_surfaces", "Bound Gamma"),
                          ("show_arrows", "Arrows"), ("follow", "Follow")):
            box = QtWidgets.QCheckBox(text)
            box.setChecked(key != "colour_surfaces")
            box.toggled.connect(self.restyle)
            bar.addWidget(box)
            self.toggles[key] = box
        bar.addSeparator()
        for name in ("near", "far", "iso", "top", "side", "front", "rear"):
            bar.addAction(name.capitalize(), lambda n=name: self.scene and self.scene.view(n))
        bar.addSeparator()
        bar.addAction("About", self.about)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)

    # scene
    def new_scene(self, spec):
        self.view3d.clear()
        from .scene import ground_point
        self.scene = Scene(self.view3d, rotors=spec.rotors or spec.motion == "free_flight", fuselage=spec.fuselage,
                           station=spec.tether_drawn,
                           ground=ground_point(spec) if spec.tether_drawn and spec.motion == "circle" else None)
        self.restyle(redraw=False)
        self.last_frame = None

    def restyle(self, *_, redraw=True):
        if self.scene is None:
            return
        self.scene.mode = self.mode_box.currentData()
        for key, box in self.toggles.items():
            setattr(self.scene, key, box.isChecked())
        if redraw and self.last_frame is not None:
            self.draw(self.last_frame)

    def draw(self, frame):
        self.last_frame = frame
        mode = frame.get("label") or (str(frame["mode"]) if "mode" in frame else "")      # stored frames keep 'mode' (A1)
        label = (f"Free flight (Makani controller): {mode}   " if self.spec_shown.motion == "free_flight"
                 else f"{METHODS[self.spec_shown.method]['label']}   ")
        self.scene.draw(frame, label=label)
        self.plots.spanwise(frame)

    # running
    def _stop_replay(self):
        """Review STR-08: a new run or another folder ends a replay."""
        if hasattr(self, "replay_timer") and self.replay_timer.isActive():
            self.replay_timer.stop()
            self.play_action.setText("Play replay")

    def start(self):
        if self.process is not None:
            return
        self._stop_replay()
        try:
            spec = self.setup.spec()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, APP_NAME, str(error))
            return
        self.runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = new_run_dir(self.runs_root, spec.name)
        self.messages, self.frames = self.ctx.Queue(), self.ctx.Queue(maxsize=1)
        self.pause_event, self.stop_event = self.ctx.Event(), self.ctx.Event()
        from . import turnkey_worker, worker
        target = turnkey_worker.run if spec.motion == "free_flight" else worker.run
        self.process = self.ctx.Process(target=target, args=(spec.to_dict(), str(run_dir), self.messages, self.frames,
                                                                  self.pause_event, self.stop_event, spec.threads),
                                        daemon=True)
        self.process.start()
        self.spec_shown = spec
        self.live = True
        self.run = None
        self.count = int(spec.steps) if spec.motion != "free_flight" else int(spec.max_periods or 144000)
        self.new_scene(spec)
        self.plots.defaults(("CL", "CD") if spec.motion != "free_flight" else ("studio_ground_power_kW", "studio_ground_energy_MJ")
                            if spec.power_system else ("tension_bridle_N", "origin_height_m"))
        self.plots.clear()
        self.slider.setEnabled(False)
        self.run_action.setEnabled(False)
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.status.setText(f"Starting: {spec.name} ({spec.steps} steps) ...")
        self.run_dir = run_dir

    def toggle_pause(self):
        if self.process is None:
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_action.setText("Pause")
        else:
            self.pause_event.set()
            self.pause_action.setText("Resume")

    def stop(self):
        if self.process is not None:
            self.stop_event.set()
            self.pause_event.clear()

    def poll(self):
        if self.process is None:
            return
        from .engine import flat_record
        spec = self.spec_shown
        area, chord = AIRCRAFT[spec.aircraft]["area_m2"], AIRCRAFT[spec.aircraft]["chord_m"]
        done = None
        for _ in range(200):
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "step":
                _, index, record, seconds = message
                row = flat_record(record, area, chord)
                row["seconds"] = seconds
                self.plots.add(row)
                left = (self.count-index)*seconds
                if spec.motion == "free_flight":
                    power = record.get("studio_ground_power_kW")
                    self.status.setText(f"period {index}   t = {record['time_s']:.2f} s   {record.get('mode_name', '')}"
                                        f"{'' if not record.get('whole_model') else ' (' + str(record['whole_model']) + ')'}   "
                                        f"tether {record.get('free_length', 0):.1f} m, {record.get('tension_bridle_N', 0)/1e3:.1f} kN"
                                        + ("" if power is None else f"   ground {power:.0f} kW, {record.get('studio_ground_energy_MJ', 0):.1f} MJ")
                                        + (f"   {seconds:.2f} s/period" if seconds else ""))
                else:
                    self.status.setText(f"step {index} / {self.count}   t = {record['time_s']:.3f} s   {seconds:.2f} s/step   "
                                        f"about {left/60:.1f} min left   {COMPUTE[spec.compute]['label']}")
            elif kind == "started":
                self.status.setText(f"Running in {message[1]}")
            elif kind in ("failed", "error", "finished"):
                done = message
        self.plots.redraw()
        try:
            frame = self.frames.get_nowait()
            self.draw(frame)
        except queue.Empty:
            pass
        if done is not None or not self.process.is_alive():
            self.finish(done)

    def finish(self, message):
        try:
            while True:
                self.draw(self.frames.get_nowait())
        except (queue.Empty, OSError, ValueError):
            pass
        self.process.join(timeout=2)
        self.process = None
        self.live = False
        self.run_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.pause_action.setText("Pause")
        self.stop_action.setEnabled(False)
        text = "The solver process ended." if message is None else {
            "finished": f"Run {message[1] if len(message) > 1 else ''}.",
            "failed": f"Step {message[1]} failed in {message[2]}: {message[3]}" if len(message) > 3 else "Failed.",
            "error": f"Not run: {message[1] if len(message) > 1 else ''}"}[message[0]]
        self.status.setText(text.splitlines()[0])
        if message is not None and message[0] == "error":
            QtWidgets.QMessageBox.warning(self, APP_NAME, message[1])
        if self.run_dir and Path(self.run_dir).exists():
            self.attach(RunFolder(self.run_dir, records=False))  # the plots have the records already (STU-01)

    # stored runs
    def attach(self, run):
        self.run = run
        self.spec_shown = run.spec
        if run.frame_steps:
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(run.frame_steps)-1)
            self.slider.setValue(len(run.frame_steps)-1)
            self.slider.blockSignals(False)
            self.slider.setEnabled(True)

    def open_run(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a studio run", str(self.open_start()))
        if folder:
            self.open_folder(folder)

    def open_start(self):
        """Addendum 22 (REL-06): the folder the Open dialog starts in - the runs folder once it holds a run, before that
        the bundled examples."""
        from .resources import data_root
        runs = Path(self.runs_root)
        if runs.is_dir() and any(p.is_dir() for p in runs.iterdir()):
            return runs
        examples = data_root()/"examples"
        return examples if examples.is_dir() else runs

    def open_folder(self, folder):
        self._stop_replay()
        try:
            run = RunFolder(folder)
        except (ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(self, APP_NAME, str(error))
            return
        from .engine import flat_record
        self.setup.load(run.spec)
        self.new_scene(run.spec)
        self.plots.clear()
        for record in run.records:
            self.plots.add(dict(flat_record(record, run.area_m2, run.chord_m), seconds=record.get("seconds", np.nan)))
        self.plots.redraw()
        self.attach(run)
        if run.frame_steps:
            self.show_stored(len(run.frame_steps)-1)
        self.status.setText(f"{run.path.name}: {len(run.records)} steps ({run.status.get('state')})"
                            + ("" if run.stride == 1 else f", every {run.stride}th record shown"))

    def show_stored(self, index):
        if self.run is None or not self.run.frame_steps:
            return
        self.draw(self.run.frame(self.run.frame_steps[int(index)]))

    # A1 (OF-0010 revision 19 addendum 7): replay of a stored run, and A4: a job run imported
    def toggle_replay(self):
        if self.run is None or not self.run.frame_steps:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Open a stored run first.")
            return
        if self.replay_timer.isActive():
            self.replay_timer.stop()
            self.play_action.setText("Play replay")
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self.replay_timer.start(80)
        self.play_action.setText("Pause replay")

    def replay_step(self):
        if self.slider.value() >= self.slider.maximum():
            self.replay_timer.stop()
            self.play_action.setText("Play replay")
            return
        self.slider.setValue(self.slider.value() + 1)

    def import_job_run(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "A folder of a job run (records.jsonl.gz)", str(self.runs_root))
        if not folder:
            return
        from .freeflight_io import import_job_run
        try:
            self.runs_root.mkdir(parents=True, exist_ok=True)
            path = import_job_run(folder, self.runs_root)
        except (ValueError, OSError, KeyError, EOFError, MemoryError, zipfile.BadZipFile) as error:   # review STR-07
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Import failed: {type(error).__name__}: {error}")
            return
        self.open_folder(path)
        state = self.run.status.get("state") if self.run is not None else None
        if state == "truncated":
            QtWidgets.QMessageBox.warning(self, APP_NAME, "The job run's records end early; imported up to the last whole "
                                          f"record: {self.run.status.get('import_error')}")

    def _need_run(self):
        if self.run is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Run a simulation or open a stored run first.")
        return self.run

    def export_csv(self):
        if not self._need_run():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export records", str(self.run.path/"records.csv"), "CSV (*.csv)")
        if path:
            from .freeflight_io import export_flight_csv, is_flight
            try:                                      # review STR-07
                (export_flight_csv if is_flight(self.run) else export_csv)(self.run, path)        # ST-06
            except (ValueError, OSError, MemoryError) as error:
                QtWidgets.QMessageBox.warning(self, APP_NAME, f"Export failed: {type(error).__name__}: {error}")
                return
            self.status.setText(f"Written {path}")

    def export_video(self):
        if not self._need_run():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the stored frames as a video", str(self.run.path/"run.gif"),
                                                        "Animated GIF (*.gif);;MP4, needs ffmpeg on the PATH or imageio-ffmpeg (*.mp4)")
        if not path:
            return
        from .video import export_video
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            out = export_video(self.run, path, rotors=self.spec_shown.rotors or self.spec_shown.motion == "free_flight",
                               fuselage=self.spec_shown.fuselage, mode=self.mode_box.currentData())
        except (RuntimeError, ValueError, OSError, MemoryError) as error:          # review STR-06
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"{type(error).__name__}: {error}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText(f"Written {out}")

    def export_paraview(self):
        if not self._need_run():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the ParaView files", str(self.run.path))
        if folder:
            from .freeflight_io import export_flight_paraview, is_flight
            try:                                      # review STR-07
                out = (export_flight_paraview if is_flight(self.run) else export_paraview)(self.run, Path(folder)/"paraview")   # A2
            except (ValueError, OSError, MemoryError) as error:
                QtWidgets.QMessageBox.warning(self, APP_NAME, f"Export failed: {type(error).__name__}: {error}")
                return
            self.status.setText(f"Written {out}")

    def screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Screenshot", str(self.runs_root/"view.png"), "PNG (*.png)")
        if path:
            self.view3d.screenshot(path)
            self.status.setText(f"Written {path}")

    def about(self):
        QtWidgets.QMessageBox.about(self, APP_NAME, f"<b>{APP_NAME}</b> {__version__}<br>{COPYRIGHT}<br><br>"
                                    "Runs the OpenFAWE aerodynamic models (VSM, lifting line and vortex lattice with free "
                                    "wakes, rotor actuator discs with ring wakes) live. The Makani M600 is the bundled "
                                    "example.<br><br>This program uses Qt and Qt for Python (PySide6, shiboken6) "
                                    "under the GNU Lesser General Public License v3; they are separate, replaceable files "
                                    "in the program folder, and their source is at https://code.qt.io/. Licence texts: "
                                    "the <i>licenses</i> folder; third-party notices: THIRD_PARTY_NOTICES.md; this "
                                    "program's licence: LICENSE-BINARIES.md (next to the program).")

    def closeEvent(self, event):
        if self.process is not None:
            self.stop()
            self.process.join(timeout=10)                # a vortex period can take seconds (review STU-02)
            if self.process.is_alive():
                self.process.terminate()
        self.view3d.close()
        super().closeEvent(event)
