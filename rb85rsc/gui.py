"""PyQt6 desktop interface.  Physics lives in the other modules; this file only builds widgets.

Simulation work runs in a QThread; results and progress come back through signals.
Stop sets a threading.Event checked inside every ODE right-hand-side evaluation,
so it interrupts long integrations and scans, not only between runs.
"""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QTabWidget, QToolBox, QVBoxLayout, QWidget,
)

from . import plotting as pl
from . import polarization as pol
from .config import ConfigError, SimConfig, field_info, get_param
from .protocols import PROTOCOLS

GROUPS = [
    ("trap", "Trap"), ("initial", "Initial state"), ("magnetic", "Magnetic field"), ("raman", "Raman (calibrated)"),
    ("spin_pump", "Spin pump (F=3 -> F'=3)"), ("repump", "Repump (F=2 -> F'=3)"), ("optical", "Optical model"),
    ("timing", "Timing / schedule"), ("numerics", "Numerics"), ("analysis", "Analysis"),
]
LOG_MIN, LOG_MAX, LOG_STEPS = -6.0, -1.0, 500


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, list):
        if len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            return ", ".join(f"{x:.8g}" for x in v)
        return json.dumps(v)
    return str(v)


class FieldWidget:
    """One config leaf -> one Qt widget, with parse/set helpers."""

    def __init__(self, path, f, value):
        self.path, self.f = path, f
        md = f.metadata
        self.kind = "text"
        if isinstance(value, bool):
            self.kind = "bool"
            self.w = QCheckBox()
            self.w.setChecked(value)
        elif md.get("choices"):
            self.kind = "choice"
            self.w = QComboBox()
            self.w.addItems(md["choices"])
            self.w.setCurrentText(value)
        elif isinstance(value, int):
            self.kind = "int"
            self.w = QSpinBox()
            self.w.setRange(int(md.get("min") if md.get("min") is not None else -10**9), int(md.get("max") if md.get("max") is not None else 10**9))
            self.w.setValue(value)
        elif isinstance(value, list):
            self.kind = "vec" if (len(value) == 3 and path.endswith(("direction", "axis"))) else "json"
            self.w = QLineEdit(_fmt(value))
        else:
            self.kind = "optfloat" if value is None or f.type.__class__.__name__ != "type" else "float"
            if "Optional" in str(f.type):
                self.kind = "optfloat"
            elif isinstance(value, str):
                self.kind = "str"
            else:
                self.kind = "float"
            self.w = QLineEdit(_fmt(value))
            if self.kind == "float":
                v = QDoubleValidator()
                v.setNotation(QDoubleValidator.Notation.ScientificNotation)
                self.w.setValidator(v)
        tip = md.get("help", "")
        if md.get("unit"):
            tip += f"  [{md['unit']}]"
        if md.get("illustrative"):
            tip += "  (ILLUSTRATIVE default)"
        self.w.setToolTip(tip)

    def get(self):
        if self.kind == "bool":
            return self.w.isChecked()
        if self.kind == "choice":
            return self.w.currentText()
        if self.kind == "int":
            return self.w.value()
        t = self.w.text().strip()
        if self.kind == "vec":
            return [float(x) for x in t.replace("[", "").replace("]", "").split(",")]
        if self.kind == "json":
            return json.loads(t) if t else []
        if self.kind == "optfloat":
            return float(t) if t else None
        if self.kind == "str":
            return t
        return float(t)

    def set(self, v):
        if self.kind == "bool":
            self.w.setChecked(v)
        elif self.kind == "choice":
            self.w.setCurrentText(v)
        elif self.kind == "int":
            self.w.setValue(v)
        else:
            self.w.setText(_fmt(v))


class PolarizationWidget(QGroupBox):
    """Spherical fractions <-> linked (total impurity, pi share) <-> geometry, without feedback loops."""

    def __init__(self, title, get_bdir):
        super().__init__(title)
        self.get_bdir = get_bdir
        self._updating = False
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["spherical", "geometry"])
        self.preset = QComboBox()
        self.preset.addItems(["(preset)", "pure sigma+", "pi-contaminated 1%", "wrong-handed sigma- 1%", "pi 0.1%", "sigma- 0.1%"])
        top.addWidget(QLabel("mode"))
        top.addWidget(self.mode)
        top.addWidget(self.preset)
        lay.addLayout(top)
        # spherical
        self.sph = QWidget()
        f = QFormLayout(self.sph)
        self.pi = QLineEdit("0")
        self.sm = QLineEdit("0")
        self.tot = QLineEdit("0")
        self.share = QLineEdit("0.5")
        self.zero = QCheckBox("zero impurity")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, LOG_STEPS)
        self.share_slider = QSlider(Qt.Orientation.Horizontal)
        self.share_slider.setRange(0, 1000)
        self.derived = QLabel()
        for w, tip in ((self.pi, "pi_fraction: intensity fraction in q=0"), (self.sm, "sigma_minus_fraction: intensity fraction in q=-1"),
                       (self.tot, "total_impurity = pi + sigma- (intensity fraction, full range 0..1)"),
                       (self.share, "pi_share_of_impurity: pi = total*share, sigma- = total*(1-share)")):
            w.setToolTip(tip)
        self.slider.setToolTip("log-spaced total impurity 1e-6 ... 1e-1")
        f.addRow("pi_fraction", self.pi)
        f.addRow("sigma_minus_fraction", self.sm)
        f.addRow("total_impurity", self.tot)
        row = QHBoxLayout()
        row.addWidget(self.slider)
        row.addWidget(self.zero)
        f.addRow("log impurity", row)
        f.addRow("pi_share_of_impurity", self.share)
        f.addRow("", self.share_slider)
        lay.addWidget(self.sph)
        # geometry
        self.geo = QWidget()
        g = QFormLayout(self.geo)
        self.ang, self.azi, self.chi, self.psi = (QLineEdit(x) for x in ("0", "0", "45", "0"))
        g.addRow("beam angle to B (deg)", self.ang)
        g.addRow("beam azimuth (deg)", self.azi)
        g.addRow("ellipticity chi (deg; +45 = +helicity)", self.chi)
        g.addRow("ellipse orientation psi (deg)", self.psi)
        lay.addWidget(self.geo)
        lay.addWidget(self.derived)
        for w in (self.pi, self.sm):
            w.editingFinished.connect(self._from_fractions)
        for w in (self.tot, self.share):
            w.editingFinished.connect(self._from_linked)
        for w in (self.ang, self.azi, self.chi, self.psi):
            w.editingFinished.connect(self._refresh)
        self.slider.valueChanged.connect(self._from_slider)
        self.share_slider.valueChanged.connect(self._from_share_slider)
        self.zero.toggled.connect(self._from_zero)
        self.mode.currentTextChanged.connect(self._mode)
        self.preset.activated.connect(self._preset)
        self._mode("spherical")

    # -- sync logic -------------------------------------------------------
    def _set_all(self, pi, sm, share=None):
        self._updating = True
        try:
            tot = pi + sm
            if share is None:
                share = pi / tot if tot > 0 else float(self.share.text() or 0.5)
            self.pi.setText(f"{pi:.8g}")
            self.sm.setText(f"{sm:.8g}")
            self.tot.setText(f"{tot:.8g}")
            self.share.setText(f"{share:.6g}")
            self.zero.setChecked(tot == 0)
            if tot > 0:
                self.slider.setValue(int(round((np.clip(np.log10(tot), LOG_MIN, LOG_MAX) - LOG_MIN) / (LOG_MAX - LOG_MIN) * LOG_STEPS)))
            self.share_slider.setValue(int(round(share * 1000)))
        finally:
            self._updating = False
        self._refresh()

    def _from_fractions(self):
        if not self._updating:
            try:
                self._set_all(float(self.pi.text()), float(self.sm.text()))
            except ValueError:
                pass

    def _from_linked(self):
        if not self._updating:
            try:
                tot, sh = float(self.tot.text()), float(self.share.text())
                pi_, sm_ = pol.fractions_from_impurity(tot, sh)
                self._set_all(pi_, sm_, sh)
            except ValueError as e:
                self.derived.setText(f"<span style='color:#b00'>{e}</span>")

    def _from_slider(self, v):
        if not self._updating:
            tot = 10 ** (LOG_MIN + (LOG_MAX - LOG_MIN) * v / LOG_STEPS)
            sh = float(self.share.text() or 0.5)
            self._set_all(tot * sh, tot * (1 - sh), sh)

    def _from_share_slider(self, v):
        if not self._updating:
            sh = v / 1000
            tot = float(self.tot.text() or 0)
            self._set_all(tot * sh, tot * (1 - sh), sh)

    def _from_zero(self, on):
        if not self._updating and on:
            self._set_all(0.0, 0.0, float(self.share.text() or 0.5))

    def _preset(self, i):
        name = self.preset.itemText(i)
        vals = {"pure sigma+": (0, 0), "pi-contaminated 1%": (0.01, 0), "wrong-handed sigma- 1%": (0, 0.01),
                "pi 0.1%": (1e-3, 0), "sigma- 0.1%": (0, 1e-3)}
        if name in vals:
            self.mode.setCurrentText("spherical")
            self._set_all(*vals[name])

    def _mode(self, m):
        self.sph.setVisible(m == "spherical")
        self.geo.setVisible(m == "geometry")
        self._refresh()

    def _refresh(self):
        try:
            fr = self.fractions()
            msg = pol.describe(fr)
            if self.mode.currentText() == "geometry":
                msg = "derived (not overridable): " + msg
            self.derived.setText(msg)
        except Exception as e:  # noqa: BLE001
            self.derived.setText(f"<span style='color:#b00'>{e}</span>")

    def fractions(self):
        if self.mode.currentText() == "geometry":
            b = self.get_bdir()
            k = pol.beam_direction_from_angles(float(self.ang.text()), float(self.azi.text()), b)
            return pol.spherical_fractions(pol.jones_field_vector(k, float(self.chi.text()), float(self.psi.text()), b), b)
        pi_, sm_ = float(self.pi.text()), float(self.sm.text())
        fr = {0: pi_, -1: sm_, 1: 1 - pi_ - sm_}
        pol.check_fractions(fr[1], fr[0], fr[-1])
        return fr

    # -- config I/O ------------------------------------------------------
    def load(self, pc):
        self.mode.setCurrentText(pc.mode)
        self.ang.setText(f"{pc.beam_angle_deg:g}")
        self.azi.setText(f"{pc.beam_azimuth_deg:g}")
        self.chi.setText(f"{pc.ellipticity_deg:g}")
        self.psi.setText(f"{pc.orientation_deg:g}")
        self._set_all(pc.pi_fraction, pc.sigma_minus_fraction, pc.pi_share_of_impurity)

    def dump(self) -> dict:
        return {
            "mode": self.mode.currentText(), "pi_fraction": float(self.pi.text()), "sigma_minus_fraction": float(self.sm.text()),
            "beam_angle_deg": float(self.ang.text()), "beam_azimuth_deg": float(self.azi.text()),
            "ellipticity_deg": float(self.chi.text()), "orientation_deg": float(self.psi.text()),
            "linked_pi_share": float(self.share.text()),
        }


class Worker(QObject):
    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job, stop):
        super().__init__()
        self.job, self.stop = job, stop

    def run(self):
        try:
            self.done.emit(self.job(self))
        except Exception as e:  # noqa: BLE001
            from .dynamics import SimulationStopped

            self.failed.emit("Stopped by user." if isinstance(e, SimulationStopped) else f"{e}\n\n{traceback.format_exc()}")


class PlotTab(QWidget):
    def __init__(self, figsize=(9, 6)):
        super().__init__()
        self.fig = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)

    def draw(self, fn):
        self.fig.clear()
        fn(self.fig)
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, cfg: SimConfig | None = None, start_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Rb-85 optical pumping + Raman sideband cooling (1D model)")
        self.resize(1600, 1000)
        self.start_path = start_path  # config the window opened with; Reset returns to it
        self.fields: dict[str, FieldWidget] = {}
        self.last = None
        self.thread = None
        self.stop = threading.Event()

        # left: parameters
        self.toolbox = QToolBox()
        info = [(p, f) for p, f in field_info() if ".polarization." not in p and p != "timing.protocol"]
        base = cfg or SimConfig()
        self.pol_widgets = {}
        for key, title in GROUPS:
            page = QWidget()
            form = QFormLayout(page)
            for p, f in info:
                if p.split(".")[0] != key:
                    continue
                fw = FieldWidget(p, f, get_param(base, p))
                self.fields[p] = fw
                unit = f.metadata.get("unit", "")
                lab = QLabel(p.split(".", 1)[1] + (f" [{unit}]" if unit else ""))
                lab.setToolTip(fw.w.toolTip())
                form.addRow(lab, fw.w)
            if key in ("spin_pump", "repump"):
                pw = PolarizationWidget("Polarization (relative to bias field)", self._bdir)
                self.pol_widgets[key] = pw
                form.addRow(pw)
            self.toolbox.addItem(page, title)
        scroll = QScrollArea()
        scroll.setWidget(self.toolbox)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)

        # controls
        ctl = QWidget()
        cl = QHBoxLayout(ctl)
        self.protocol = QComboBox()
        for k, v in PROTOCOLS.items():
            self.protocol.addItem(v, k)
        self.run_btn = QPushButton("Run")
        self.cmp_btn = QPushButton("Compare protocols")
        self.stop_btn = QPushButton("Stop")
        self.reset_btn = QPushButton("Reset")
        self.load_btn = QPushButton("Load config...")
        self.save_btn = QPushButton("Save config...")
        self.stop_btn.setEnabled(False)
        for w in (QLabel("Protocol:"), self.protocol, self.run_btn, self.cmp_btn, self.stop_btn, self.reset_btn, self.load_btn, self.save_btn):
            cl.addWidget(w)
        cl.addStretch(1)
        self.run_btn.clicked.connect(self.on_run)
        self.cmp_btn.clicked.connect(self.on_compare)
        self.stop_btn.clicked.connect(self.on_stop)
        self.reset_btn.clicked.connect(self.on_reset)
        self.load_btn.clicked.connect(self.on_load)
        self.save_btn.clicked.connect(self.on_save)

        # right: plots
        self.tabs = QTabWidget()
        self.t_over = PlotTab((12, 8))
        self.t_spin = PlotTab()
        self.t_tr = PlotTab()
        self.t_pnt = PlotTab()
        self.t_joint = PlotTab()
        self.t_cmp = PlotTab((12, 8))
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        self.diag.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        for w, n in ((self.t_over, "Overview"), (self.t_spin, "Spin"), (self.t_tr, "Time traces"), (self.t_pnt, "p(n,t)"),
                     (self.t_joint, "Joint P(F,mF,n)"), (self.diag, "Diagnostics / validity"), (self.t_cmp, "Protocol comparison")):
            self.tabs.addTab(w, n)
        self.tabs.addTab(self._scan_tab(), "Scan")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(ctl)
        rl.addWidget(self.tabs)
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.status = QLabel("Ready.")
        self.statusBar().addWidget(self.status, 1)
        self.last_ok = QLabel("Last successful run: none")
        self.last_ok.setToolTip("Most recent run / comparison / scan that finished without error in this window")
        self.statusBar().addPermanentWidget(self.last_ok)
        self.statusBar().addPermanentWidget(self.prog)
        m = self.menuBar().addMenu("&File")
        for name, fn, key in (("Load config JSON...", self.on_load, "Ctrl+O"), ("Save config JSON...", self.on_save, "Ctrl+S"),
                              ("Export results...", self.on_export, "")):
            act = QAction(name, self)
            act.setShortcut(key)
            act.triggered.connect(fn)
            m.addAction(act)
        self.load_config(base)

    # ------------------------------------------------------------------
    def _scan_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        numeric = [p for p, f in field_info() if isinstance(get_param(SimConfig(), p), float) and not isinstance(get_param(SimConfig(), p), bool)]
        numeric += ["spin_pump.polarization.total_impurity", "repump.polarization.total_impurity",
                    "spin_pump.polarization.pi_share_of_impurity", "repump.polarization.pi_share_of_impurity"]
        self.scan_rows = []
        for i in range(2):
            p = QComboBox()
            p.setEditable(True)
            p.addItems((["(none)"] if i else []) + numeric)
            p.setToolTip("Dotted parameter path; separate several with ';' to set them together")
            if i == 0:
                p.setCurrentText("raman.red_sideband_offset_hz")
            s0, s1, n = QLineEdit("-150e3" if i == 0 else "0.01"), QLineEdit("50e3" if i == 0 else "0.3"), QSpinBox()
            n.setRange(1, 200)
            n.setValue(9 if i == 0 else 4)
            lg = QCheckBox("log")
            row = QHBoxLayout()
            for x in (p, QLabel("start"), s0, QLabel("stop"), s1, QLabel("num"), n, lg):
                row.addWidget(x)
            form.addRow(f"axis {i + 1}", row)
            self.scan_rows.append((p, s0, s1, n, lg))
        self.scan_btn = QPushButton("Run scan (uses current parameters as base)")
        self.scan_btn.clicked.connect(self.on_scan)
        lay.addLayout(form)
        lay.addWidget(self.scan_btn)
        self.t_scan = PlotTab((12, 7))
        lay.addWidget(self.t_scan)
        return w

    def _bdir(self):
        try:
            return self.fields["magnetic.direction"].get()
        except Exception:  # noqa: BLE001
            return [0, 0, 1]

    def load_config(self, cfg: SimConfig):
        for p, fw in self.fields.items():
            fw.set(get_param(cfg, p))
        for b, pw in self.pol_widgets.items():
            pw.load(getattr(cfg, b).polarization)
        i = self.protocol.findData(cfg.timing.protocol)
        self.protocol.setCurrentIndex(max(i, 0))

    def current_config(self) -> SimConfig:
        d = SimConfig().to_dict()
        for p, fw in self.fields.items():
            a, b = p.split(".", 1)
            d[a][b] = fw.get()
        for b, pw in self.pol_widgets.items():
            d[b]["polarization"] = pw.dump()
        d["timing"]["protocol"] = self.protocol.currentData()
        return SimConfig.from_dict(d)

    # ------------------------------------------------------------------
    def _start(self, job, on_done, label):
        if self.thread is not None:
            return
        try:
            self.current_config()
        except (ConfigError, ValueError) as e:
            QMessageBox.warning(self, "Invalid parameters", str(e))
            return
        self.stop.clear()
        self.thread = QThread()
        self.worker = Worker(job, self.stop)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda f, m: (self.prog.setValue(int(1000 * f)), self.status.setText(f"{label}: {m}")))
        self.worker.log.connect(self.status.setText)
        self.worker.done.connect(on_done)
        self.worker.done.connect(lambda *_: self._mark_success(label))
        self.worker.failed.connect(self._failed)
        self.worker.done.connect(self._finish)
        self.worker.failed.connect(self._finish)
        for b in (self.run_btn, self.cmp_btn, self.scan_btn):
            b.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText(f"{label}...")
        self.thread.start()

    def _mark_success(self, label):
        kind = {"Running": "run", "Comparing protocols": "comparison", "Scanning": "scan"}.get(label, label)
        self.last_ok.setText(f"Last successful {kind}: {datetime.now():%Y-%m-%d %H:%M:%S}")

    def _finish(self, *_):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        for b in (self.run_btn, self.cmp_btn, self.scan_btn):
            b.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _failed(self, msg):
        self.status.setText(msg.splitlines()[0])
        if not msg.startswith("Stopped"):
            QMessageBox.critical(self, "Simulation error", msg[:3000])

    def on_run(self):
        cfg = self.current_config()
        stop = self.stop

        def job(w):
            from .runner import run

            def est(e):
                w.log.emit(f"Estimated coherent runtime ~{e['est_coherent_runtime_s']:.1f} s (state dim {e['state_dimension_complex']})")
            return run(cfg, stop, lambda f, m="": w.progress.emit(f, m), est)

        self._start(job, self._show_result, "Running")

    def on_compare(self):
        cfg = self.current_config()
        stop = self.stop

        def job(w):
            from .runner import compare_protocols

            return compare_protocols(cfg, stop=stop, progress=lambda f, m="": w.progress.emit(f, m), log=w.log.emit)

        self._start(job, self._show_compare, "Comparing protocols")

    def on_scan(self):
        cfg = self.current_config()
        axes = []
        try:
            for p, s0, s1, n, lg in self.scan_rows:
                name = p.currentText().strip()
                if name in ("", "(none)"):
                    continue
                params = [x.strip() for x in name.split(";")]
                axes.append({"param": params if len(params) > 1 else params[0], "start": float(s0.text()), "stop": float(s1.text()),
                             "num": n.value(), "log": lg.isChecked(), "label": name})
        except ValueError as e:
            QMessageBox.warning(self, "Scan", str(e))
            return
        spec = {"title": "GUI scan", "axes": axes}
        stop = self.stop

        def job(w):
            from .scan import run_scan

            return run_scan(spec, cfg, stop, lambda f, m="": w.progress.emit(f, m), log=w.log.emit, workers=1)

        self._start(job, self._show_scan, "Scanning")

    def on_stop(self):
        self.stop.set()
        self.status.setText("Stopping...")

    def on_reset(self):
        cfg, msg = SimConfig(), "Parameters reset to built-in defaults."
        if self.start_path:  # back to the startup config, re-read so file edits apply
            try:
                cfg, msg = SimConfig.from_json(self.start_path), f"Parameters reset to {self.start_path}."
            except Exception as e:  # noqa: BLE001
                msg = f"Could not re-read {self.start_path} ({e}); reset to built-in defaults."
        self.load_config(cfg)
        self.last = None
        for t in (self.t_over, self.t_spin, self.t_tr, self.t_pnt, self.t_joint, self.t_cmp, self.t_scan):
            t.draw(lambda f: None)
        self.diag.clear()
        self.prog.setValue(0)
        self.status.setText(msg)

    def on_load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if fn:
            try:
                self.load_config(SimConfig.from_json(fn))
                self.status.setText(f"Config loaded from {fn}")
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Load", str(e))

    def on_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save config", "config.json", "JSON (*.json)")
        if fn:
            try:
                self.current_config().to_json(fn)
                self.status.setText(f"Config saved to {fn}")
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Save", str(e))

    def on_export(self):
        if self.last is None:
            QMessageBox.information(self, "Export", "Run a simulation first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Export directory")
        if d:
            from .runner import export, export_comparison

            if isinstance(self.last, dict) and "series" in self.last:
                export(self.last, d)
            else:
                export_comparison(self.last, d)
            self.status.setText(f"Exported to {d}")

    # ------------------------------------------------------------------
    def _show_result(self, a):
        self.last = a
        self.t_over.draw(lambda f: _overview(f, a))
        self.t_spin.draw(lambda f: pl.plot_spin_bars(pl._new(f, 111), a))
        self.t_tr.draw(lambda f: _traces(f, a))
        self.t_pnt.draw(lambda f: pl.plot_pnt(pl._new(f, 111), a, f))
        self.t_joint.draw(lambda f: pl.plot_joint(pl._new(f, 111), a, f, nshow=min(a["P_final"].shape[1], 30)))
        self.diag.setPlainText(pl.diagnostics_text(a))
        st = a["validity"].status.upper()
        self.status.setText(f"Done ({a['final']['wall_time_s']:.1f} s). P_target={a['final']['P_target']:.4f}. Validity: {st}")
        self.prog.setValue(1000)

    def _show_compare(self, res):
        self.last = res
        self.t_cmp.fig.clear()
        fig = pl.comparison_figure(res)
        self.t_cmp.fig = fig
        self.t_cmp.canvas.figure = fig
        fig.set_canvas(self.t_cmp.canvas)
        self.t_cmp.canvas.draw_idle()
        self.tabs.setCurrentWidget(self.t_cmp)
        self.diag.setPlainText("\n\n".join(f"===== {p} =====\n{pl.diagnostics_text(a)}" for p, a in res.items()))
        self.prog.setValue(1000)
        self.status.setText("Protocol comparison done.")

    def _show_scan(self, scan):
        fig = pl.scan_figure(scan)
        self.t_scan.fig = fig
        self.t_scan.canvas.figure = fig
        fig.set_canvas(self.t_scan.canvas)
        self.t_scan.canvas.draw_idle()
        self.prog.setValue(1000)
        self.status.setText(f"Scan done{' (stopped early)' if scan['stopped'] else ''}. Best valid: {scan['best_valid_P_target']}")


def _overview(fig, a):
    gs = fig.add_gridspec(3, 3, height_ratios=[0.22, 1, 1])
    pl.plot_schedule(pl._new(fig, gs[0, :]), a)
    pl.plot_spin_bars(pl._new(fig, gs[1, 0]), a)
    pl.plot_traces(pl._new(fig, gs[1, 1]), a)
    pl.plot_nbar(pl._new(fig, gs[1, 2]), a)
    pl.plot_pnt(pl._new(fig, gs[2, 0]), a, fig)
    pl.plot_joint(pl._new(fig, gs[2, 1]), a, fig)
    pl.plot_photons(pl._new(fig, gs[2, 2]), a)


def _traces(fig, a):
    gs = fig.add_gridspec(3, 1)
    pl.plot_traces(pl._new(fig, gs[0]), a)
    pl.plot_nbar(pl._new(fig, gs[1]), a)
    pl.plot_cooling_rate(pl._new(fig, gs[2]), a)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "experiment.json"


def launch(config_path: str | None = None) -> int:
    """Open the GUI with config_path, else examples/experiment.json, else the built-in defaults."""
    app = QApplication.instance() or QApplication([])
    path = config_path or (str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None)
    cfg, note = None, None
    if path:
        try:
            cfg = SimConfig.from_json(path)
            note = f"Loaded {path}"
        except Exception as e:  # noqa: BLE001
            if config_path:
                raise
            path, note = None, f"Could not load {DEFAULT_CONFIG} ({e}); using built-in defaults."
    w = MainWindow(cfg, start_path=path)
    if note:
        w.status.setText(note)
    w.show()
    return app.exec()
