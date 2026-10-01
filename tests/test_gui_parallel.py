"""Real spawned GUI workers: numerical equivalence, cancellation, and cleanup."""
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from rb85rsc.config import SimConfig
from rb85rsc.dynamics import SimulationStopped

ROOT = Path(__file__).resolve().parents[1]


def experiment(duration_ms=.02, samples=8):
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.ensemble.samples = samples
    cfg.ensemble.workers = 8
    cfg.timing.custom_segments = [dict(duration_ms=duration_ms, raman=1., pump=1., repump=1.)]
    cfg.timing.n_samples = 2
    return cfg


def pump_until(predicate, timeout=30):
    from PyQt6.QtCore import QCoreApplication

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(.01)
    QCoreApplication.processEvents()
    assert predicate(), 'GUI did not reach the expected state before timeout'


def children():
    return {p.pid for p in multiprocessing.active_children()}


@pytest.fixture(scope='module')
def qt_app():
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from rb85rsc.gui import MainWindow

    failures = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: failures.append(args[2]))
    w = MainWindow(experiment())
    try:
        yield w, failures
    finally:
        if w.thread is not None:
            w.on_stop()
            pump_until(lambda: w.thread is None)
        w.close()


def assert_same_physics(actual, expected):
    for name in ('P', 'counts', 'overflow', 'min_eig', 'coherence'):
        np.testing.assert_array_equal(getattr(actual['result'], name), getattr(expected['result'], name))
    for (_, a), (_, b) in zip(actual['ensemble_nodes'], expected['ensemble_nodes']):
        for name in ('P', 'counts', 'loss', 'energy_abs_hz', 'min_eig', 'coherence',
                     'loss_flux', 'loss_energy_flux_hz', 'P_final_3d', 'rho_final'):
            np.testing.assert_array_equal(a[name], b[name])
        assert a['steps'] == b['steps']


def test_gui_eight_workers_complete_stop_and_restart(window, monkeypatch):
    import concurrent.futures
    from rb85rsc.runner import run

    w, failures = window
    original_pool = concurrent.futures.ProcessPoolExecutor
    pool_sizes = []

    def pool(*args, **kwargs):
        pool_sizes.append(kwargs['max_workers'])
        return original_pool(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', pool)
    w.fields['ensemble.workers'].set(8)
    short = w.current_config()
    before = children()
    w.on_run()
    pump_until(lambda: w.thread is None, timeout=60)
    assert not failures
    assert pool_sizes == [8]
    completed = w.last
    assert completed is not None
    assert children() == before
    expected = run(short, ensemble_workers=1)
    assert_same_physics(completed, expected)
    stamp = w.last_ok.text()

    w.load_config(experiment(duration_ms=5., samples=16))
    w.on_run()
    pump_until(lambda: len(children() - before) == 8 or w.thread is None)
    assert w.thread is not None and len(children() - before) == 8
    cancel_at = time.monotonic() + .5
    pump_until(lambda: time.monotonic() >= cancel_at)
    started = time.monotonic()
    w.on_stop()
    pump_until(lambda: w.thread is None)
    assert time.monotonic() - started < 10
    assert not failures and 'Stopped' in w.status.text()
    assert w.last is completed and w.last_ok.text() == stamp
    assert children() == before
    assert w.run_btn.isEnabled() and not w.stop_btn.isEnabled()

    # A fresh pool and cleared stop signal must permit another successful run.
    w.load_config(short)
    w.on_run()
    pump_until(lambda: w.thread is None, timeout=60)
    assert not failures and w.last is not completed
    assert pool_sizes == [8, 8, 8]
    assert_same_physics(w.last, expected)
    assert children() == before


def test_gui_close_stops_and_joins_workers(window):
    w, failures = window
    before = children()
    w.load_config(experiment(duration_ms=5., samples=16))
    w.show()
    w.on_run()
    pump_until(lambda: len(children() - before) == 8 or w.thread is None)
    assert w.thread is not None
    w.close()
    assert w._close_when_finished
    pump_until(lambda: w.thread is None and not w.isVisible())
    assert not failures and children() == before
    assert w.last is None


def test_stop_between_split_steps_with_sparse_output(monkeypatch):
    from rb85rsc import ensemble
    from rb85rsc.atomic import load_atomic_data
    from rb85rsc.far_detuned import DLineResponse
    from rb85rsc.dynamics import _single_thread

    cfg = experiment(duration_ms=.02, samples=1)
    response = DLineResponse(load_atomic_data(), cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
    calibration = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    local = ensemble.LocalExperiment(cfg, response, calibration, [110, 90, 70], .73)
    stop = threading.Event()
    original = ensemble._UniformizedScattering.step
    calls = []

    def step(self, populations, dt):
        calls.append(dt)
        result = original(self, populations, dt)
        stop.set()
        return result

    monkeypatch.setattr(ensemble._UniformizedScattering, 'step', step)
    with _single_thread(), pytest.raises(SimulationStopped):
        ensemble.solve_local(local, cfg, stop)
    assert len(calls) == 1


def test_already_stopped_ensemble_skips_setup(monkeypatch):
    from rb85rsc import ensemble

    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(ensemble.at, 'load_atomic_data', lambda: pytest.fail('setup should not start'))
    with pytest.raises(SimulationStopped):
        ensemble.run_ensemble(experiment(), stop=stop)


def failing_node(data, calibration, position, phase, beat):
    """Spawn-importable fixture: one error must stop another running task."""
    from rb85rsc.ensemble import _WORKER_STOP

    if position[0] == 0:
        raise RuntimeError('test node failure')
    if not _WORKER_STOP.wait(timeout=15):
        raise AssertionError('remaining worker was not cancelled')
    raise SimulationStopped()


def test_worker_error_cancels_other_tasks_and_joins_pool(monkeypatch):
    from rb85rsc import ensemble

    monkeypatch.setattr(ensemble, '_solve_node', failing_node)
    before = children()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='test node failure'):
        ensemble._parallel_components(experiment(), {}, [[1, 0, 0], [0, 0, 0]],
                                      [0., 0.], 0., 2, threading.Event(), None)
    assert time.monotonic() - started < 10
    assert children() == before
