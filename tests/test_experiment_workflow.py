"""Result provenance checks: stale scans must not become new predictions."""
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rb85rsc.config import SimConfig
from rb85rsc.scan import run_scan

ROOT = Path(__file__).resolve().parents[1]


def test_scan_resume_recomputes_changed_config(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    workflow = importlib.import_module('optimize_experiment')
    monkeypatch.setattr(workflow, 'ProcessPoolExecutor', ThreadPoolExecutor)
    calls = []

    def evaluate(item):
        tag, data = item
        calls.append(tag)
        return dict(tag=tag, config=data, status='ok',
                    final=dict(P_target_absolute=data['initial']['temperature_k'], P_trapped=1.))

    monkeypatch.setattr(workflow, 'evaluate', evaluate)
    cfg = SimConfig()
    path = tmp_path / 'scan.json'
    workflow.batch([('same_coordinates', cfg.to_dict())], path, 1)
    workflow.batch([('same_coordinates', cfg.to_dict())], path, 1)
    assert len(calls) == 1
    cfg.initial.temperature_k *= 2
    rows = workflow.batch([('same_coordinates', cfg.to_dict())], path, 1)
    assert len(calls) == 2 and len(rows) == 1
    assert rows[0]['final']['P_target_absolute'] == cfg.initial.temperature_k
    assert json.loads(path.read_text()) == rows


def test_control_comparison_keeps_physics_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    workflow = importlib.import_module('finalize_experiment')
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    original = workflow.comparison_config(cfg.to_dict())
    cfg.raman.red_sideband_offset_hz = 1234.
    cfg.ensemble.workers = 1
    assert workflow.comparison_config(cfg.to_dict()) == original
    cfg.raman.carrier_decay_mode = 'measured_envelope'
    assert workflow.comparison_config(cfg.to_dict()) != original


def test_bad_scan_objective_fails_before_running(monkeypatch):
    from rb85rsc import runner
    monkeypatch.setattr(runner, 'estimate', lambda _: pytest.fail('cost estimation must not run'))
    spec = dict(axes=[dict(param='raman.carrier_rabi_hz', values=[5000.])], objective='not_a_metric')
    with pytest.raises(ValueError, match='unknown scan objective'):
        run_scan(spec, SimConfig(), log=lambda _: None)
