"""Check 10: headless execution, config round trip, export, GUI smoke test."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rb85rsc.config import ConfigError, SimConfig

ROOT = Path(__file__).resolve().parents[1]


def test_config_round_trip(tmp_path):
    c = SimConfig()
    c.spin_pump.polarization.set_impurity(0.01, 0.3)
    c.timing.protocol = "pulsed"
    p = tmp_path / "c.json"
    c.to_json(p)
    assert SimConfig.from_json(p) == c


def test_unknown_key_rejected():
    d = SimConfig().to_dict()
    d["raman"]["rabi"] = 1
    with pytest.raises(ConfigError):
        SimConfig.from_dict(d)


def test_examples_load():
    for f in (ROOT / "examples").rglob("*.json"):
        d = json.loads(f.read_text())
        if "axes" in d:
            from rb85rsc.scan import load_scan, point_configs

            spec, base = load_scan(f)
            assert len(list(point_configs(spec, base))) > 0
        else:
            SimConfig.from_dict(d)


@pytest.mark.parametrize("entrypoint", [["-m", "rb85rsc"], [str(ROOT / "rb85_rsc_sim.py")]])
def test_cli_entrypoints_write_same_default_config(entrypoint, tmp_path):
    path = tmp_path / "default.json"
    result = subprocess.run(
        [sys.executable, *entrypoint, "--write-default-config", str(path)],
        capture_output=True, text=True, cwd=ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert SimConfig.from_json(path) == SimConfig()


def test_legacy_entrypoint_preserves_imported_configuration(monkeypatch):
    spec = importlib.util.spec_from_file_location("legacy_rate_calcs", ROOT / "rate-calcs.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.RUN_SWEEP = False
    module.INITIAL_POPULATIONS = {"g_F3_m+3": 1.0}
    assert module.main.__globals__ is vars(module)
    assert module.main.__globals__["RUN_SWEEP"] is False
    assert module.main.__globals__["INITIAL_POPULATIONS"] == {"g_F3_m+3": 1.0}
    assert module.State.__module__ == spec.name


def test_headless_cli_and_export(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "rb85_rsc_sim.py"), "--config", str(ROOT / "examples" / "arxiv" / "continuous.json"), "--headless",
           "--output", str(out), "--set", "timing.total_duration_ms=0.5", "--set", "timing.n_samples=11"]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.pop("QT_QPA_PLATFORM", None)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    assert p.returncode == 0, p.stderr
    for suffix in ("timeseries.csv", "joint.npz", "metadata.json", "config.json", "dashboard.png", "dashboard.pdf"):
        assert (out / f"run_{suffix}").exists(), suffix
    meta = json.loads((out / "run_metadata.json").read_text())
    for k in ("config", "assumptions", "versions", "constants_SI", "validity", "seed", "polarization", "atomic_data_metadata"):
        assert k in meta


def test_headless_does_not_import_qt():
    code = "import sys; sys.path.insert(0, r'%s'); import rb85rsc.runner, rb85rsc.scan, rb85rsc.plotting, rb85rsc.cli, rb85_rsc_sim; print('PyQt6' in sys.modules)" % ROOT
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.stdout.strip() == "False"


def test_gui_smoke():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtWidgets import QApplication

    from rb85rsc.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    c = SimConfig()
    c.timing.total_duration_ms = 0.3
    c.timing.n_samples = 11
    w = MainWindow(c)
    # linked impurity controls: repump edit must not change the spin pump
    pw = w.pol_widgets["repump"]
    pw.tot.setText("0.02")
    pw.share.setText("0.25")
    pw._from_linked()
    got = w.current_config()
    assert got.repump.polarization.pi_fraction == pytest.approx(0.005)
    assert got.repump.polarization.sigma_minus_fraction == pytest.approx(0.015)
    assert got.spin_pump.polarization.total_impurity == 0
    assert w.last_ok.text() == "Last successful run: none"
    w.on_run()
    t0 = time.time()
    while w.thread is not None and time.time() - t0 < 120:
        QCoreApplication.processEvents()
        time.sleep(0.02)
    assert w.last is not None and "series" in w.last
    stamp = w.last_ok.text()
    assert stamp.startswith("Last successful run: 20")
    # Stop during a long integration
    c.timing.total_duration_ms = 20.0
    w.load_config(c)
    w.on_run()
    t0 = time.time()
    while time.time() - t0 < 1.0:
        QCoreApplication.processEvents()
        time.sleep(0.02)
    w.on_stop()
    while w.thread is not None and time.time() - t0 < 30:
        QCoreApplication.processEvents()
        time.sleep(0.02)
    assert w.thread is None and "Stopped" in w.status.text()
    assert w.last_ok.text() == stamp  # a stopped run is not a successful one
    w.close()
