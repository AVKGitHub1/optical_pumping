"""Rotating the whole apparatus must preserve calibration and dynamics."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rb85rsc import atomic as at, far_detuned as fd, polarization as pol
from rb85rsc.config import ConfigError, SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.ensemble import LocalExperiment, solve_local

ROOT = Path(__file__).resolve().parents[1]
ROTATIONS = [
    np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),  # z -> x
    np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),  # z -> y
    Rotation.from_rotvec([.4, -.7, .2]).as_matrix(),
]


def config():
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.ensemble.samples = 1
    cfg.ensemble.workers = 1
    cfg.ensemble.calibration_samples = 8
    cfg.timing.custom_segments = [dict(duration_ms=.006, raman=1., pump=1., repump=1.)]
    cfg.timing.n_samples = 3
    # Spherical inputs allow the pump lab vectors to be rotated explicitly,
    # without redefining azimuth/ellipse angles in the magnetic-field frame.
    cfg.repump.polarization.mode = 'spherical'
    cfg.repump.direction = [1., 0., 0.]
    cfg.raman.scattering_rate_s = .4  # Exercise the extra recoil term too.
    return cfg


def rotate(cfg, rotation):
    cfg = cfg.copy()
    for obj, name in ((cfg.trap, 'axis'), (cfg.magnetic, 'direction'),
                      (cfg.raman, 'beam_low_direction'), (cfg.raman, 'beam_high_direction'),
                      (cfg.spin_pump, 'direction'), (cfg.repump, 'direction'),
                      (cfg.ensemble, 'transverse_reference_direction')):
        setattr(obj, name, (rotation @ getattr(obj, name)).tolist())
    cfg.ensemble.lattice_linear_polarizations = (np.asarray(cfg.ensemble.lattice_linear_polarizations) @ rotation.T).tolist()
    return cfg


@pytest.mark.parametrize('rotation', ROTATIONS)
@pytest.mark.parametrize('reverse', [False, True])
def test_accepts_rotated_counterpropagating_beams(rotation, reverse):
    cfg = rotate(config(), rotation)
    sign = -1 if reverse else 1
    cfg.raman.beam_low_direction = (sign * 3 * np.asarray(cfg.trap.axis)).tolist()
    cfg.raman.beam_high_direction = (-sign * 2 * np.asarray(cfg.trap.axis)).tolist()
    cfg.validate()
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize('low,high,message', [
    ([0, 0, 1], [0, 0, 2], 'counterpropagating'),
    ([0, 0, 1], [1, 0, -1], 'counterpropagating'),
    ([1, 0, 1], [-1, 0, -1], 'one lattice axis'),
    ([0, 0, 0], [0, 0, -1], 'nonzero 3-vector'),
    ([0, 1], [0, 0, -1], 'nonzero 3-vector'),
    ([np.nan, 0, 1], [0, 0, -1], 'nonzero 3-vector'),
])
def test_rejects_unsupported_or_invalid_directions(low, high, message):
    cfg = config()
    cfg.raman.beam_low_direction = low
    cfg.raman.beam_high_direction = high
    with pytest.raises(ConfigError, match=message):
        cfg.validate()


def test_frame_default_parallel_reference_and_polarization_validation():
    np.testing.assert_array_equal(pol.lattice_frame([0, 0, 1], [1, 0, 0]), np.eye(3))
    axes = pol.lattice_frame([1, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(axes, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    cfg = rotate(config(), ROTATIONS[0])
    cfg.ensemble.lattice_linear_polarizations[0] = axes[0].tolist()
    with pytest.raises(ConfigError, match='perpendicular'):
        cfg.validate()


@pytest.mark.parametrize('rotation', ROTATIONS)
def test_rotated_calibration_and_scattering_dynamics(rotation, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    base = config()
    tilted = rotate(base, rotation)
    base.validate()
    tilted.validate()
    xyz, phases = fd.spatial_samples(base, 8)
    rotated_xyz, rotated_phases = fd.spatial_samples(tilted, 8)
    np.testing.assert_allclose(rotated_xyz, xyz @ rotation.T, atol=1e-12)
    np.testing.assert_array_equal(rotated_phases, phases)
    atom = at.load_atomic_data()
    position = np.array([110., 90., 70.])  # Noncentral, different local depths.
    locals_, calibrations, results = [], [], []
    with _single_thread():
        for cfg, pos in ((base, position), (tilted, rotation @ position)):
            response = fd.DLineResponse(atom, cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
            cal = fd.calibrate_carrier(cfg, response)
            local = LocalExperiment(cfg, response, cal, pos, .73)
            calibrations.append(cal)
            locals_.append(local)
            results.append(solve_local(local, cfg))
    for key in ('intensity_per_unit_tone_w_m2', 'fitted_trace', 'inhomogeneous_envelope',
                'modeled_rms_carrier_hz'):
        np.testing.assert_allclose(calibrations[0][key], calibrations[1][key], rtol=2e-11, atol=2e-12)
    assert calibrations[0]['cache_key'] != calibrations[1]['cache_key']
    left, right = locals_
    np.testing.assert_allclose(right.axes_lab, left.axes_lab @ rotation.T, atol=1e-14)
    for name in ('depths', 'intensity', 'raman_shifts_hz', 'lattice_shifts_hz', 'P0'):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=2e-10, atol=2e-10)
    for local in locals_:
        assert local.model.eta_R > 0
    # Gauge phases of the spin basis may change on rotation; populations,
    # photon counts, loss, motional energies, and coherence magnitudes may not.
    for key in ('P', 'P_final_3d', 'counts', 'loss', 'energy_abs_hz', 'coherence'):
        np.testing.assert_allclose(results[0][key], results[1][key], rtol=2e-9, atol=2e-11)
    assert results[0]['loss'][-1] > 0
    assert np.all(results[0]['counts'][-1] > 0)


def test_rotated_run_exports_axis_directions(tmp_path, monkeypatch):
    from rb85rsc.runner import export, metadata, run
    monkeypatch.setattr(fd, 'CACHE', tmp_path / 'cache')
    cfg = rotate(config(), ROTATIONS[0])
    result = run(cfg)
    axes = np.asarray([result['model_summary']['lattice_axes_lab'][a] for a in 'xyz'])
    np.testing.assert_allclose(axes, ROTATIONS[0].T, atol=1e-14)
    assert metadata(result)['config'] == cfg.to_dict()
    export(result, tmp_path / 'out', figures=False)
    with np.load(tmp_path / 'out/run_ensemble.npz') as data:
        np.testing.assert_array_equal(data['lattice_axes_lab'], axes)


@pytest.mark.parametrize('action', ['on_run', 'on_compare', 'on_scan'])
def test_gui_invalid_geometry_shows_dialog(action, monkeypatch):
    import os
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from rb85rsc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    window = MainWindow(config())
    try:
        valid = rotate(config(), ROTATIONS[2])
        window.load_config(valid)
        cfg = window.current_config()  # Also exercise GUI vector precision.
        np.testing.assert_allclose(cfg.trap.axis, valid.trap.axis, atol=1e-8)
        window.fields['raman.beam_high_direction'].set(valid.raman.beam_low_direction)
        getattr(window, action)()
        assert len(warnings) == 1 and 'counterpropagating' in warnings[0]
        assert window.thread is None
    finally:
        window.close()
