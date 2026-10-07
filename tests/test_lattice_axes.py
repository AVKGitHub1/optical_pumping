"""Independent lattice parameters and Raman cooling along a fixed x/y/z well."""
import os
from pathlib import Path

import numpy as np
import pytest

from rb85rsc import atomic as at, far_detuned as fd, motion as mo
from rb85rsc.config import LatticeAxisConfig, SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.ensemble import LocalExperiment, solve_local

ROOT = Path(__file__).resolve().parents[1]


def config(axis=0):
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.ensemble.samples = cfg.ensemble.workers = 1
    cfg.ensemble.calibration_samples = 8
    cfg.timing.custom_segments = [dict(duration_ms=.02, raman=1., pump=1., repump=1.)]
    cfg.timing.n_samples = 3
    cfg.raman.beam_low_direction = np.eye(3)[axis].tolist()
    cfg.raman.beam_high_direction = (-np.eye(3)[axis]).tolist()
    cfg.magnetic.direction = [.3, .4, .5]
    cfg.trap.lattice_x = LatticeAxisConfig(wavelength_nm=1064., depth_uk=10., waist_um=140.)
    cfg.trap.lattice_y = LatticeAxisConfig(wavelength_nm=1188., depth_uk=8., waist_um=210.)
    cfg.trap.lattice_z = LatticeAxisConfig(wavelength_nm=785., depth_uk=15., waist_um=270.)
    return cfg


def local_experiment(cfg, position=(60., 70., 80.)):
    atom = at.load_atomic_data()
    response = fd.DLineResponse(atom, cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
    calibration = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    return LocalExperiment(cfg, response, calibration, position, .73)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_each_raman_axis_uses_its_well_and_preserves_other_axes(axis):
    cfg = config(axis)
    cfg.spin_pump.enabled = cfg.repump.enabled = False
    cfg.initial.spin_preset = 'stretched'
    cfg.initial.motion_mode = 'fock'
    cfg.initial.fock_n = 1
    cfg.validate()
    original = cfg.to_dict()
    local = local_experiment(cfg)
    assert cfg.to_dict() == original  # Do not rotate or rewrite the trap config.
    assert cfg.trap.axis == [0., 0., 1.]
    assert local.raman_axis == axis
    assert local.model.motion is local.sites[axis]
    expected_spacing = np.diff(local.sites[axis].energies_hz)[0]
    assert local.model.nu_ref == pytest.approx(expected_spacing)
    assert local.model.eta_R == pytest.approx(4 * np.pi / 783e-9 * local.sites[axis].x0)
    p = np.array([60., 70., 80.])
    waists = [140., 210., 270.]
    expected_depths = [depth * np.exp(-2 * (p @ p - p[i]**2) / waists[i]**2)
                       for i, depth in enumerate([10., 8., 15.])]
    np.testing.assert_allclose(local.depths, expected_depths)
    assert local.intensity == pytest.approx(10000. * np.exp(-2 * (p @ p - p[axis]**2) / cfg.raman.waist_um**2))
    # The axial Fock preparation belongs to z, even when x/y is driven.
    assert local.P0.sum(axis=(0, 1, 2))[1] == pytest.approx(1.)
    local.terms = []  # Isolate coherent Raman motion from recoil heating/loss.
    with _single_thread():
        result = solve_local(local, cfg)
    final = result['P_final_3d']
    np.testing.assert_allclose(final.sum(), 1., atol=2e-12)
    marginal_axes = tuple(i for i in range(1, 4) if i != axis + 1)
    np.testing.assert_allclose(result['P'][-1], final.sum(axis=marginal_axes), atol=2e-13)
    assert final[at.DOWN].sum() > 1e-7
    for other in range(3):
        if other == axis:
            continue
        summation = tuple(i for i in range(4) if i != other + 1)
        np.testing.assert_allclose(final.sum(axis=summation), local.P0.sum(axis=summation), atol=2e-12)


def test_frequency_input_sets_depth_and_leaves_other_lattices_unchanged():
    cfg = config(1)
    cfg.trap.lattice_y.confinement = 'frequency'
    cfg.trap.lattice_y.frequency_hz = 35000.
    # Conflicting stored depth is inactive in frequency mode.
    cfg.trap.lattice_y.depth_uk = 900.
    local = local_experiment(cfg, (0., 0., 0.))
    assert local.sites[1].trap_hz == pytest.approx(35000.)
    atom = at.load_atomic_data()
    expected_depth = .5 * atom.mass_kg * (1188e-9 * 35000.)**2 / at.K_B * 1e6
    assert local.depths[1] == pytest.approx(expected_depth)
    np.testing.assert_allclose(local.depths[[0, 2]], [10., 15.])
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize('axis', [0, 1])
def test_transverse_run_calibration_outputs_and_export(axis, tmp_path, monkeypatch):
    from rb85rsc.runner import export, run
    monkeypatch.setattr(fd, 'CACHE', tmp_path / 'cache')
    cfg = config(axis)
    result = run(cfg)
    label = 'xyz'[axis]
    assert result['final']['raman_axis'] == label
    cal = result['model_summary']['calibration']
    assert cal['lattice_axis'] == label
    assert cal['lattice_parameters']['wavelength_nm'] == cfg.lattice_config(axis).wavelength_nm
    assert cal['temperature_k'] == cfg.ensemble.transverse_temperature_k
    assert result['final']['max_conservation_error'] < 1e-10
    local, node = result['ensemble_nodes'][0]
    rate = local.operators(local.model.schedule[-1])[1]
    target = [at.UP, slice(None), slice(None), slice(None)]
    target[axis + 1] = 0
    weights = node['P_final_3d'][tuple(target)]
    assert result['final']['P_target_absolute'] == pytest.approx(weights.sum())
    assert result['final']['target_state_scattering_rate_per_s'] == pytest.approx((rate[tuple(target)] * weights).sum() / weights.sum())
    np.testing.assert_allclose(result['series']['mean_excitation_energy_hz'], result['series'][f'mean_excitation_energy_{label}_hz'])
    export(result, tmp_path / 'out', figures=False)
    with np.load(tmp_path / 'out/run_ensemble.npz') as saved:
        assert saved['raman_axis'].item() == label
        np.testing.assert_array_equal(saved['node_0_P_final_spin_nx_ny_nz'], node['P_final_3d'])


def test_calibration_follows_selected_lattice_and_cache_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    cfg = config(0)
    response = fd.DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
    first = fd.calibrate_carrier(cfg, response)
    cfg.trap.lattice_x.waist_um = 190.
    second = fd.calibrate_carrier(cfg, response)
    assert second['cache_key'] != first['cache_key']
    assert second['intensity_per_unit_tone_w_m2'] != first['intensity_per_unit_tone_w_m2']
    cfg.trap.lattice_y.depth_uk = 11.
    third = fd.calibrate_carrier(cfg, response)
    assert third['intensity_per_unit_tone_w_m2'] == second['intensity_per_unit_tone_w_m2']


def test_parallel_to_magnetic_field_has_clear_physical_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    cfg = config(0)
    cfg.magnetic.direction = [1., 0., 0.]
    cfg.validate()  # Geometry is accepted; this is a selection-rule limitation.
    response = fd.DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
    with pytest.raises(ValueError, match='parallel to the magnetic field'):
        fd.calibrate_carrier(cfg, response)


def test_gui_lattice_inputs_roundtrip_and_calculate_frequency(tmp_path):
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from rb85rsc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    cfg = config(0)
    window = MainWindow(cfg)
    try:
        assert set(window.lattice_widgets) == set('xyz')
        x = window.lattice_widgets['x']
        x.inputs['wavelength_nm'].setText('900')
        x.inputs['depth_uk'].setText('12')
        x.inputs['waist_um'].setText('180')
        mass = at.load_atomic_data().mass_kg
        assert float(x.inputs['frequency_hz'].text()) == pytest.approx(mo.lattice_frequency_hz(mass, 900, 12))
        assert x.inputs['frequency_hz'].isReadOnly()
        x.mode.setCurrentIndex(x.mode.findData('frequency'))
        x.inputs['frequency_hz'].setText('55000')
        assert x.inputs['depth_uk'].isReadOnly()
        assert not x.inputs['frequency_hz'].isReadOnly()
        actual = window.current_config()
        assert actual.trap.axis == [0., 0., 1.]
        assert actual.trap.lattice_x.frequency_hz == 55000.
        assert actual.lattice_parameters(0, mass).frequency_hz == 55000.
        assert actual.trap.lattice_y == cfg.trap.lattice_y
        assert actual.trap.lattice_z == cfg.trap.lattice_z
        actual.to_json(tmp_path / 'axes.json')
        window.load_config(SimConfig.from_json(tmp_path / 'axes.json'))
        assert window.current_config().trap.lattice_x == actual.trap.lattice_x
        assert x.mode.currentData() == 'frequency'
    finally:
        window.close()
