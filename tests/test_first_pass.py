"""Regressions for visible validity and scheduled optical-pumping controls."""
import os
from pathlib import Path

import numpy as np
import pytest

from rb85rsc.config import SimConfig, ConfigError
from rb85rsc.protocols import raman_is_active
from rb85rsc.runner import run, export, metadata

ROOT = Path(__file__).resolve().parents[1]


def optical_only():
    c = SimConfig.from_json(ROOT/'examples/onlyOP.json')
    c.ensemble.samples = 1
    c.ensemble.workers = 1
    c.timing.n_samples = 3
    c.timing.custom_segments[0]['duration_ms'] = .01
    return c


@pytest.mark.parametrize('field', [0., 3.])
def test_off_schedule_skips_forbidden_carrier_and_exports_validity(monkeypatch, tmp_path, field):
    import rb85rsc.ensemble as ensemble
    def forbidden(*_):
        pytest.fail('Raman-off run must not request carrier calibration')
    monkeypatch.setattr(ensemble, 'calibrate_carrier', forbidden)
    c = optical_only()
    c.magnetic.magnitude_gauss = field
    c.magnetic.direction = [0., 0., 1.]  # Circular parallel beams cannot drive this delta-m=1 pair.
    c.raman.carrier_rabi_hz = c.raman.flop_contrast = 0.
    c.validate()
    a = run(c)
    assert not a['final']['raman_active']
    assert a['ensemble_details']['calibration']['status'] == 'skipped'
    assert a['final']['photons']['raman_scatter'] == 0
    assert np.max(a['series']['raman_coherence_l1']) == 0
    v = {item['category']: item['status'] for item in metadata(a)['validity']}
    assert v['magnetic field'] == ('warning' if field == 3 else 'ok')
    if field == 0:
        assert v['ground coherences'] == 'invalid'
    export(a, tmp_path, figures=field == 3)
    assert not (tmp_path/'run_carrier_calibration.png').exists()
    assert (tmp_path/'run_metadata.json').exists()


def test_calibration_inputs_required_only_for_active_schedule():
    c = optical_only()
    c.raman.carrier_rabi_hz = c.raman.flop_contrast = 0.
    c.validate()
    assert not raman_is_active(c)
    c.timing.custom_segments[0]['raman'] = 1.
    with pytest.raises(ConfigError, match='positive measured'):
        c.validate()
    c.raman.enabled = False
    c.validate()
    assert not raman_is_active(c)


def test_gui_shows_active_duration_and_preserves_inactive_inputs():
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from rb85rsc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    c = optical_only()
    w = MainWindow(c)
    try:
        assert not w.fields['timing.total_duration_ms'].w.isEnabled()
        assert not w.fields['raman.carrier_rabi_hz'].w.isEnabled()
        assert '0.01 ms' in w.schedule_note.text()
        assert 'Raman off' in w.schedule_note.text()
        preserved = w.current_config()
        assert preserved.timing.total_duration_ms == c.timing.total_duration_ms
        assert preserved.raman.carrier_rabi_hz == c.raman.carrier_rabi_hz
        w.fields['timing.custom_segments'].w.setText('[{"duration_ms":2,"raman":1}]')
        assert w.fields['raman.carrier_rabi_hz'].w.isEnabled()
        assert '2 ms' in w.schedule_note.text()
        w.protocol.setCurrentIndex(w.protocol.findData('continuous'))
        assert w.fields['timing.total_duration_ms'].w.isEnabled()
        assert not w.fields['timing.custom_segments'].w.isEnabled()
        w.fields['raman.enabled'].w.setChecked(False)
        assert not w.fields['ensemble.calibration_samples'].w.isEnabled()
    finally:
        w.close()
