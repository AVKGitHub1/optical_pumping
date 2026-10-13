"""Coherent pumping through the shared runner and the real Qt worker path."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from rb85rsc.config import ConfigError, SimConfig
from rb85rsc.coherent_pumping import run_cloud
from rb85rsc.runner import export, metadata, run
from test_gui_parallel import qt_app, pump_until, children

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = SimConfig.from_json(ROOT/'examples/onlyOP_correlated.json')
    cfg.ensemble.recoil_dynamics = 'coherent_pumping'
    cfg.ensemble.samples = 2
    cfg.ensemble.workers = 2
    cfg.ensemble.trajectories_per_site = 128
    cfg.ensemble.trajectory_seed = 43
    cfg.numerics.recoil_quadrature_points = 4
    cfg.timing.n_samples = 5
    cfg.timing.custom_segments[0]['duration_ms'] = .01
    cfg.validate()
    return cfg


def test_coherent_config_round_trip_and_raman_validation(tmp_path):
    assert SimConfig().ensemble.recoil_dynamics == 'population'
    cfg = config()
    path = tmp_path/'coherent.json'
    cfg.to_json(path)
    assert SimConfig.from_json(path) == cfg
    cfg.timing.custom_segments[0]['raman'] = 1.
    with pytest.raises(ConfigError, match='Raman off'):
        cfg.validate()
    cfg.raman.enabled = False  # A disabled beam has no scheduled exposure.
    cfg.validate()
    cfg.ensemble.recoil_model = 'cartesian_marginals'
    with pytest.raises(ConfigError, match='correlated'):
        cfg.validate()
    cfg = config()
    cfg.ensemble.trajectories_per_site = 2.5
    with pytest.raises(ConfigError, match='integer'):
        cfg.validate()


def test_runner_matches_direct_trajectories_and_exports_uncertainty(tmp_path):
    from rb85rsc.plotting import diagnostics_text
    cfg = config()
    original = cfg.to_dict()
    a = run(cfg, ensemble_workers=1)
    expected = run_cloud(cfg, trajectories=128, seed=43, workers=1, modes=('coherent',))
    for key, value in expected['modes']['coherent']['mean'].items():
        np.testing.assert_array_equal(a['series'][key], value)
        np.testing.assert_array_equal(a['standard_error'][key], expected['modes']['coherent']['standard_error'][key])
    assert a['kind'] == 'coherent_pumping'
    assert 'one-sigma' in diagnostics_text(a)
    assert not any(category == 'motional secular' and status == 'invalid' for category, status, _ in a['validity'].items)
    assert any(category == 'weak excitation' for category, _, _ in a['validity'].items)
    assert cfg.to_dict() == original
    export(a, tmp_path)
    saved = json.loads((tmp_path/'run_metadata.json').read_text())
    assert saved['config'] == original
    assert saved['sampling']['trajectory_seed'] == 43
    assert saved['sampling']['workers'] == 1
    assert saved['final_estimates']['survival']['standard_error'] == a['standard_error']['survival'][-1]
    assert 'survival_standard_error' in (tmp_path/'run_timeseries.csv').read_text().splitlines()[0]
    assert (tmp_path/'run_dashboard.png').stat().st_size > 1000
    assert (tmp_path/'run_dashboard.pdf').stat().st_size > 1000
    assert (tmp_path/'nodes'/'site_001.npz').is_file()
    assert (tmp_path/'run_thermal_axes.png').stat().st_size > 1000
    assert (tmp_path/'run_motional_axes.pdf').stat().st_size > 1000
    arrays = np.load(tmp_path/'run_observables.npz')
    np.testing.assert_array_equal(arrays['pn_axes_z'], a['pn_axes']['z'])
    np.testing.assert_array_equal(arrays['spin_standard_error'], a['array_standard_error']['spin'])
    assert arrays['joint_absolute'].shape == a['joint_absolute'].shape
    assert 'cooling_power_x_W_standard_error' in (tmp_path/'run_timeseries.csv').read_text().splitlines()[0]
    assert metadata(a) == saved


def test_coherent_diagnostics_do_not_hide_weak_excitation_failure():
    cfg = config()
    cfg.optical.weak_excitation_invalid = 1e-8
    result = run(cfg, ensemble_workers=1)
    assert result['validity'].status == 'invalid'
    assert any(c == 'weak excitation' and s == 'invalid' for c, s, _ in result['validity'].items)


def test_coherent_cli_accepts_saved_gui_config(tmp_path):
    cfg = config()
    cfg.ensemble.workers = 1
    path = tmp_path/'config.json'
    cfg.to_json(path)
    result = subprocess.run([sys.executable, '-m', 'rb85rsc', '--headless', '--config', str(path),
                             '--output', str(tmp_path/'output'), '--no-figures'],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'Coherent pumping:' in result.stdout
    assert 'maximum split step' not in result.stdout
    assert (tmp_path/'output'/'cloud_trajectories.npz').exists()


def test_comparison_cli_honors_saved_trajectory_settings(tmp_path):
    cfg = config()
    cfg.ensemble.samples = 1
    cfg.ensemble.workers = 1
    path = tmp_path/'config.json'
    cfg.to_json(path)
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/'check_coherent_pumping.py'),
                             '--config', str(path), '--output', str(tmp_path/'output')],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    saved = SimConfig.from_json(tmp_path/'output'/'run_config.json')
    assert saved == cfg
    report = json.loads((tmp_path/'output'/'comparison.json').read_text())
    assert report['trajectories_per_site'] == 128 and report['trajectory_seed'] == 43
    control = SimConfig.from_json(tmp_path/'output'/'population_deterministic'/'run_config.json')
    assert control.ensemble.recoil_dynamics == 'population'


@pytest.fixture
def coherent_window(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from rb85rsc.gui import MainWindow
    failures = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: failures.append(args[2]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: failures.append(args[2]))
    window = MainWindow(config())
    try:
        yield window, failures
    finally:
        if window.thread is not None:
            window.on_stop()
            pump_until(lambda: window.thread is None)
        window.close()


def test_gui_solver_controls_validate_before_launch(coherent_window):
    w, failures = coherent_window
    assert w.current_config().ensemble.trajectory_seed == 43
    assert w.fields['ensemble.trajectories_per_site'].w.isEnabled()
    assert not w.fields['ensemble.time_step_us'].w.isEnabled()
    assert not w.fields['ensemble.recoil_model'].w.isEnabled()
    assert not w.cmp_btn.isEnabled() and not w.scan_btn.isEnabled()
    assert 'spin-motion' in w.coherent_note.text()
    w.fields['timing.custom_segments'].set([dict(duration_ms=.01, raman=1., pump=1., repump=1.)])
    assert not w.run_btn.isEnabled()
    assert 'requires Raman off' in w.schedule_note.text()
    w.on_run()
    assert w.thread is None and len(failures) == 1 and 'Raman off' in failures[0]
    w.fields['raman.enabled'].set(False)
    assert w.run_btn.isEnabled()
    w.current_config().validate()
    w.fields['ensemble.recoil_dynamics'].set('population')
    assert w.cmp_btn.isEnabled() and w.scan_btn.isEnabled()
    assert w.fields['ensemble.time_step_us'].w.isEnabled()
    assert not w.fields['ensemble.trajectories_per_site'].w.isEnabled()


def test_gui_coherent_run_export_stop_restart_and_switch(coherent_window, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog
    w, failures = coherent_window
    before = children()
    cfg = w.current_config()
    w.on_run()
    pump_until(lambda: w.thread is None, timeout=60)
    assert not failures and children() == before
    completed = w.last
    assert completed['kind'] == 'coherent_pumping'
    assert w.tabs.currentWidget() is w.t_over
    assert all(w.tabs.tabText(i) != 'Coherent pumping' for i in range(w.tabs.count()))
    for tab in (w.t_over, w.t_spin, w.t_tr, w.t_thermal, w.t_pnt, w.t_joint):
        assert w.tabs.isTabEnabled(w.tabs.indexOf(tab))
        assert tab.fig.axes
    assert 'Monte Carlo' in w.diag.toPlainText()
    assert len(w.t_thermal.fig.axes) == 6
    assert len(w.t_tr.fig.axes) == 4
    assert any(ax.collections for ax in w.t_over.fig.axes)
    expected = run(cfg, ensemble_workers=1)
    np.testing.assert_array_equal(completed['series']['P_target_absolute'], expected['series']['P_target_absolute'])
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    w.on_export()
    assert (tmp_path/'run_metadata.json').exists()
    stamp = w.last_ok.text()

    long = cfg.copy()
    long.ensemble.trajectories_per_site = 65536
    long.ensemble.samples = 8
    long.timing.custom_segments[0]['duration_ms'] = 20.
    w.load_config(long)
    w.on_run()
    pump_until(lambda: len(children()-before) == 2 or w.thread is None)
    assert w.thread is not None
    started = time.monotonic()
    w.on_stop()
    pump_until(lambda: w.thread is None)
    assert time.monotonic()-started < 10
    assert children() == before and not failures
    assert w.last is completed and w.last_ok.text() == stamp
    assert w.run_btn.isEnabled() and not w.cmp_btn.isEnabled()

    w.load_config(cfg)
    w.on_run()
    pump_until(lambda: w.thread is None, timeout=60)
    assert not failures and w.last is not completed and children() == before
    cfg.ensemble.recoil_dynamics = 'population'
    w.load_config(cfg)
    w.on_run()
    pump_until(lambda: w.thread is None, timeout=60)
    assert not failures and w.last.get('kind') != 'coherent_pumping'
    assert w.tabs.isTabEnabled(w.tabs.indexOf(w.t_thermal))
    assert w.tabs.currentWidget() is w.t_over
    assert 'standard_error' not in w.last
    assert w.t_over.fig._supxlabel is None


def test_closing_coherent_window_joins_workers(coherent_window):
    w, failures = coherent_window
    cfg = w.current_config()
    cfg.ensemble.samples = 8
    cfg.ensemble.trajectories_per_site = 65536
    w.load_config(cfg)
    before = children()
    w.show()
    w.on_run()
    pump_until(lambda: len(children()-before) == 2 or w.thread is None)
    assert w.thread is not None
    w.close()
    pump_until(lambda: w.thread is None and not w.isVisible())
    assert not failures and children() == before
