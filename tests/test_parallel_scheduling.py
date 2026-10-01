"""Outer parameter pools own the worker budget without changing provenance."""
import importlib
import json
import concurrent.futures
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from rb85rsc import runner, scan
from rb85rsc.config import SimConfig

ROOT = Path(__file__).resolve().parents[1]


def test_parallel_scan_keeps_config_and_orders_results(monkeypatch):
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    original = cfg.to_dict()
    calls = []

    def run(point, *, ensemble_workers=None):
        calls.append((point.ensemble.workers, ensemble_workers))
        value = point.raman.carrier_rabi_hz
        final = dict.fromkeys(scan.METRIC_KEYS, value)
        final.update(solver='test', wall_time_s=0.)
        return dict(final=final, validity=SimpleNamespace(status='ok', items=[]))

    monkeypatch.setattr(scan, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(scan, '_reject_weak_excitation', lambda _: None)
    monkeypatch.setattr(runner, 'run', run)
    spec = dict(axes=[dict(param='raman.carrier_rabi_hz', values=[2000., 1000.])], workers=2)
    result = scan.run_scan(spec, cfg, log=lambda _: None)

    assert calls == [(cfg.ensemble.workers, 1)] * 2
    assert result['metrics']['P_target_absolute'] == [[2000., 1000.]]
    assert result['best_valid_objective']['index'] == [0, 0]
    assert result['base_config'] == original == cfg.to_dict()


def test_batch_worker_override_preserves_metadata_and_resume(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    workflow = importlib.import_module('optimize_experiment')
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    original = cfg.to_dict()
    calls = []

    def run(point, *, ensemble_workers=None):
        calls.append((point.to_dict(), ensemble_workers))
        return dict(final=dict(P_target_absolute=.25, P_trapped=.5),
                    validity=SimpleNamespace(status='ok', items=[]))

    monkeypatch.setattr(workflow, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(workflow, 'run', run)
    path = tmp_path / 'batch.json'
    items = [('point', original)]
    rows = workflow.batch(items, path, workers=2)
    resumed = workflow.batch(items, path, workers=1)

    assert calls == [(original, 1)]
    assert rows == resumed == json.loads(path.read_text())
    assert rows[0]['config'] == original == cfg.to_dict()


@pytest.mark.parametrize('override', [None, 1, 2])
def test_ensemble_worker_override_keeps_sample_order_and_config(monkeypatch, override):
    from rb85rsc import ensemble

    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.raman.frequency_mode = 'absolute_beat'
    original = cfg.to_dict()
    pool_workers = []

    class ImmediatePool:
        def __init__(self, max_workers):
            pool_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    def local(point, response, calibration, position, phase, beat=None):
        return SimpleNamespace(cfg=point, position=position)

    def analyze(point, central, components, calibration, beat):
        return dict(final=dict(wall_time_s=0.), result=SimpleNamespace(),
                    positions=[result for _, result in components], config=point.to_dict())

    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', ImmediatePool)
    # Deliberately complete samples in reverse order; averaging must retain the
    # spatial quadrature order, including when changing the process count.
    monkeypatch.setattr(concurrent.futures, 'as_completed', lambda futures: reversed(list(futures)))
    monkeypatch.setattr(ensemble, '_single_thread', nullcontext)
    monkeypatch.setattr(ensemble.at, 'load_atomic_data', lambda: None)
    monkeypatch.setattr(ensemble, 'DLineResponse', lambda *args: None)
    monkeypatch.setattr(ensemble, 'calibrate_carrier', lambda *args: {})
    monkeypatch.setattr(ensemble, 'LocalExperiment', local)
    monkeypatch.setattr(ensemble, 'spatial_samples', lambda _: ([[10., 0., 0.], [20., 0., 0.]], [0., 0.]))
    monkeypatch.setattr(ensemble, 'solve_local', lambda local, *args: local.position[0])
    monkeypatch.setattr(ensemble, 'analyze_ensemble', analyze)

    result = runner.run(cfg, **({} if override is None else dict(ensemble_workers=override)))
    expected = cfg.ensemble.workers if override is None else override
    assert pool_workers == ([] if expected == 1 else [expected])
    assert result['positions'] == [10., 20.]
    assert result['config'] == original == cfg.to_dict()


@pytest.mark.parametrize('workers', [0, -1, 1.5, '2', True, False])
def test_invalid_worker_override_fails_before_setup(monkeypatch, workers):
    from rb85rsc import ensemble

    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    original = cfg.to_dict()
    monkeypatch.setattr(ensemble.at, 'load_atomic_data', lambda: pytest.fail('setup must not run'))
    with pytest.raises(ValueError, match='workers must be a positive integer'):
        runner.run(cfg, ensemble_workers=workers)
    assert cfg.to_dict() == original
