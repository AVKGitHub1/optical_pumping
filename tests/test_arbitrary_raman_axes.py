"""Oblique Raman motion must evolve all coupled wells and preserve spectators."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from rb85rsc import atomic as at, far_detuned as fd
from rb85rsc.config import LatticeAxisConfig, SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.ensemble import LocalExperiment, solve_local

ROOT = Path(__file__).resolve().parents[1]


def config(low=(1., 0., 1.5), high=(-1., 0., -1.5), layout='one_tone_per_beam'):
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.ensemble.samples = cfg.ensemble.workers = 1
    cfg.ensemble.calibration_samples = 8
    cfg.ensemble.cloud_radius_um = 1.
    cfg.ensemble.transverse_temperature_k = 1e-9
    cfg.ensemble.time_step_us = 1.
    cfg.timing.custom_segments = [dict(duration_ms=.006, raman=1., pump=0., repump=0.)]
    cfg.timing.n_samples = 4
    cfg.raman.beam_low_direction = list(low)
    cfg.raman.beam_high_direction = list(high)
    cfg.raman.tone_layout = layout
    cfg.raman.extra_coherence_decay_rate_s = 0.
    cfg.magnetic.direction = [.3, .4, .5]
    cfg.spin_pump.enabled = cfg.repump.enabled = False
    cfg.initial.spin_preset = 'stretched'
    cfg.initial.motion_mode = 'fock'
    cfg.initial.fock_n = 1
    # Unequal basis dimensions expose incorrect transposes/reshapes; these
    # shallow wells keep even the complete 3D reference basis small.
    for label, wavelength in zip('xyz', (785., 1064., 1188.)):
        setattr(cfg.trap, f'lattice_{label}',
                LatticeAxisConfig(wavelength_nm=wavelength, depth_uk=3., waist_um=200.))
    return cfg


@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
@pytest.mark.parametrize('low,high', [
    ([1, 0, 1.5], [-2, 0, -3]),
    ([0, 0, 1], [1, 0, -1]),
    ([1, 2, 3], [2, 4, 6]),
])
def test_arbitrary_directions_and_tone_layout_roundtrip(low, high, layout):
    cfg = config(low, high, layout)
    cfg.validate()
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
@pytest.mark.parametrize('low,high,coupled', [
    ([1, 1.5, 0], [-1, -1.5, 0], (0, 1)),
    ([1, 0, 1.5], [-1, 0, -1.5], (0, 2)),
    ([0, 1, 1.5], [0, -1, -1.5], (1, 2)),
    ([1, .7, 1.5], [-1, -.7, -1.5], (0, 1, 2)),
    ([1, 0, 0], [0, 0, 1], (0, 2)),
])
def test_coherent_oblique_motion_changes_every_projected_axis(low, high, coupled, layout):
    cfg = config(low, high, layout)
    cfg.validate()
    original = cfg.to_dict()
    response = fd.DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss,
                               cfg.magnetic.direction)
    calibration = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    local = LocalExperiment(cfg, response, calibration, [0., 0., 0.], .73)
    assert cfg.to_dict() == original
    local.terms = []  # Isolate coherent motion from photon recoil.
    with _single_thread():
        result = solve_local(local, cfg)
    final = result['P_final_3d']
    np.testing.assert_allclose(result['P'].sum(axis=(1, 2)), 1., atol=2e-12)
    np.testing.assert_allclose(final.sum(), 1., atol=2e-12)
    assert np.min(result['min_eig']) > -2e-12
    assert final[local.model.down].sum() > 1e-7
    for axis in range(3):
        summation = tuple(i for i in range(4) if i != axis + 1)
        before, after = local.P0.sum(axis=summation), final.sum(axis=summation)
        if axis in coupled:
            assert np.linalg.norm(after - before) > 1e-9
        else:
            np.testing.assert_allclose(after, before, atol=2e-12)
    marginal_axes = tuple(i for i in range(1, 4) if i != local.raman_axis + 1)
    np.testing.assert_allclose(result['P'][-1], final.sum(axis=marginal_axes), atol=2e-12)


def test_oblique_run_calibration_and_export(tmp_path, monkeypatch):
    from rb85rsc.runner import export, run
    monkeypatch.setattr(fd, 'CACHE', tmp_path / 'cache')
    cfg = config()
    result = run(cfg)
    assert result['final']['raman_axis'] == 'z'
    assert result['model_summary']['raman_coupled_axes'] == ['x', 'z']
    assert result['model_summary']['calibration']['intensity_per_unit_tone_w_m2'] > 0
    np.testing.assert_allclose(result['series']['P_trapped'] + result['result'].overflow,
                               1., atol=2e-11)
    export(result, tmp_path / 'out', figures=False)
    saved_metadata = json.loads((tmp_path / 'out/run_metadata.json').read_text())
    assert saved_metadata['model_summary']['raman_coupled_axes'] == ['x', 'z']
    with np.load(tmp_path / 'out/run_ensemble.npz') as saved:
        assert saved['raman_axis'].item() == 'z'
        np.testing.assert_array_equal(saved['node_0_P_final_spin_nx_ny_nz'],
                                      result['ensemble_nodes'][0][1]['P_final_3d'])


@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
def test_copropagating_run_uses_fixed_phase_light_shift(tmp_path, monkeypatch, layout):
    from rb85rsc.runner import run
    monkeypatch.setattr(fd, 'CACHE', tmp_path / 'cache')
    cfg = config([1, 0, 1.5], [2, 0, 3], layout)
    cfg.ensemble.phase_offset_deg = 37.
    result = run(cfg)
    local, node = result['ensemble_nodes'][0]
    assert result['model_summary']['raman_coupled_axes'] == []
    assert local.phase == pytest.approx(np.deg2rad(37.))
    central = LocalExperiment(cfg, local.response, result['model_summary']['calibration'],
                              [0, 0, 0], local.phase)
    m = central.model
    excited = [0, 0, 0]
    excited[central.raman_axis] = min(1, m.N - 1)
    expected = (m.nu_ud_zeeman + m.nu_ref
                + central.raman_shifts_hz[m.up, 0, 0, 0] - central.raman_shifts_hz[m.down, 0, 0, 0]
                + central.lattice_shifts_hz[(m.up, *excited)] - central.lattice_shifts_hz[m.down, 0, 0, 0]
                + cfg.raman.differential_light_shift_hz + cfg.raman.red_sideband_offset_hz)
    assert result['model_summary']['nu_beat_hz'] == pytest.approx(expected, abs=1e-6, rel=0)
    np.testing.assert_allclose(node['P'].sum(axis=(1, 2)) + node['loss'], 1., atol=2e-11)


def test_oblique_evolution_matches_full_three_axis_matrix_exponential():
    cfg = config()
    cfg.raman.max_sideband_order = 20  # Retain every matrix element in these wells.
    response = fd.DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss,
                               cfg.magnetic.direction)
    intensity = 10000.
    calibration = dict(intensity_per_unit_tone_w_m2=intensity, residual_coherence_decay_rate_s=0.)
    local = LocalExperiment(cfg, response, calibration, [0., 0., 0.], .73)
    local.terms = []
    m = local.model
    # Independent reference: keep the spectator y basis explicitly in a single
    # full density matrix instead of using the solver's coherent-axis blocks.
    delta = np.asarray(cfg.raman.beam_low_direction, float)
    delta /= np.linalg.norm(delta)
    high = np.asarray(cfg.raman.beam_high_direction, float)
    delta -= high / np.linalg.norm(high)
    delta *= 2 * np.pi / (cfg.raman.wavelength_nm * 1e-9)
    displacement = np.ones((1, 1), complex)
    for site, direction in zip(local.sites, local.axes_lab):
        displacement = np.kron(displacement, site.displacement(float(delta @ direction) * site.x0))
    _, _, _, paths = fd.raman_fields(cfg, response)
    downward = paths[0, 1] * displacement  # The sole one-tone-per-beam path.
    size = displacement.shape[0]
    levels = np.indices(local.shape[1:]).reshape(3, -1)
    energy = sum(site.energies_hz[n] - site.energies_hz[0]
                 for site, n in zip(local.sites, levels))
    shifts = local.raman_shifts_hz + local.lattice_shifts_hz
    up = energy + m.nu_ud_zeeman - m.nu_beat + shifts[m.up].ravel()
    down = energy + shifts[m.down].ravel()
    hamiltonian = np.diag(2 * np.pi * np.r_[up, down]).astype(complex)
    hamiltonian[:size, size:] = np.pi * intensity * downward.conj().T
    hamiltonian[size:, :size] = np.pi * intensity * downward
    rho = np.diag(np.r_[local.P0[m.up].ravel(), local.P0[m.down].ravel()])
    with _single_thread():
        unitary = expm(-1j * hamiltonian * m.schedule[-1].t1)
        expected = np.diag(unitary @ rho @ unitary.conj().T).real
        result = solve_local(local, cfg)
    actual = np.r_[result['P_final_3d'][m.up].ravel(), result['P_final_3d'][m.down].ravel()]
    np.testing.assert_allclose(actual, expected, atol=3e-12)


def test_gui_accepts_oblique_one_tone_geometry():
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from rb85rsc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    cfg = config()
    window = MainWindow(cfg)
    try:
        actual = window.current_config()
        actual.validate()
        assert actual.raman.tone_layout == 'one_tone_per_beam'
        np.testing.assert_allclose(actual.raman.beam_low_direction, cfg.raman.beam_low_direction)
        np.testing.assert_allclose(actual.raman.beam_high_direction, cfg.raman.beam_high_direction)
    finally:
        window.close()


def test_gui_oblique_reference_with_raman_off_trajectories():
    pytest.importorskip('PyQt6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from rb85rsc.gui import MainWindow
    from rb85rsc.runner import run
    app = QApplication.instance() or QApplication([])
    cfg = config()
    cfg.ensemble.recoil_dynamics = 'coherent_pumping'
    cfg.ensemble.recoil_model = 'correlated'
    cfg.ensemble.trajectories_per_site = 16
    cfg.numerics.recoil_quadrature_points = 4
    cfg.raman.enabled = False
    cfg.timing.custom_segments[0]['raman'] = 0.
    window = MainWindow(cfg)
    try:
        actual = window.current_config()
        result = run(actual)
        assert result['kind'] == 'coherent_pumping'
        assert result['final']['raman_axis'] == 'z'
        assert not result['final']['raman_active']
        np.testing.assert_allclose(result['series']['nbar'], result['series']['nbar_z'])
        np.testing.assert_allclose(result['series']['P_trapped'] + result['series']['loss'], 1.)
    finally:
        window.close()
