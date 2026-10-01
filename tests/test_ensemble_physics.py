"""Independent limits for the clarified experimental model."""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from rb85rsc import atomic as at, motion as mo, polarization as pol
from rb85rsc.config import SimConfig, ConfigError
from rb85rsc.ensemble import RecoilTerm, uniform_step, cross_kernel, LocalExperiment, solve_local, mixture_temperature
from rb85rsc.far_detuned import DLineResponse, raman_fields, spatial_samples, envelope, calibrate_carrier
from rb85rsc.dynamics import _single_thread
from rb85rsc.model import Model

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = SimConfig.from_json(ROOT / 'examples/experiment.json')
    c.ensemble.samples = 1
    c.timing.custom_segments = [dict(duration_ms=.01, raman=1., pump=1., repump=1.)]
    c.timing.n_samples = 3
    return c


@pytest.fixture(scope='module')
def response():
    c = config()
    return DLineResponse(at.load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)


def test_depth_wavelength_sets_frequency_and_transverse_recoil():
    atom = at.load_atomic_data()
    assert mo.lattice_frequency_hz(atom.mass_kg, 785, 15) == pytest.approx(69043.7, rel=2e-6)
    assert mo.lattice_frequency_hz(atom.mass_kg, 1188, 15) == pytest.approx(45622.3, rel=2e-6)
    c = config()
    with pytest.raises(ConfigError, match='runner.run'):
        Model(c)  # never silently reinterpret the measured carrier as a path


def test_helicities_are_defined_along_each_beam(response):
    c = config()
    _, _, eps, paths = raman_fields(c, response)
    np.testing.assert_allclose(eps[0], eps[1], atol=1e-14)
    for e, k in zip(eps, (c.raman.beam_low_direction, c.raman.beam_high_direction)):
        assert abs(np.dot(e, k)) < 1e-14
    np.testing.assert_allclose(paths, paths[0, 0], atol=1e-12)
    assert abs(paths[0, 0]) > 0


def test_gaussian_radius_and_phase_sampling():
    c = config()
    xyz, phase = spatial_samples(c, 4096)
    np.testing.assert_allclose(xyz.std(axis=0), c.ensemble.cloud_radius_um / 2, rtol=.003)
    assert abs(np.exp(1j * phase).mean()) < 1e-3
    assert envelope([0, 200, 0], [1, 0, 0], 200) == pytest.approx(np.exp(-2))
    assert envelope([200, 0, 0], [1, 0, 0], 200) == 1


def test_high_power_scan_point_is_rejected_without_a_false_optimum():
    from rb85rsc.scan import _reject_weak_excitation
    c = config()
    c.spin_pump.power_mw = c.repump.power_mw = .003  # 3 uW each
    rejected = _reject_weak_excitation(c)
    assert rejected['status'] == 'invalid'
    assert np.isnan(rejected['P_target_absolute'])
    c.spin_pump.power_mw = c.repump.power_mw = 3e-6  # 3 nW each
    assert _reject_weak_excitation(c) is None


def test_dline_shift_and_scattering_normalization(response):
    # Near D2 cycling line, RWA shift = Omega²/(4 Delta), rate = Gamma Omega²/(4 Delta²).
    atom = response.atom
    eps = pol.jones_field_vector([1, 0, 0], 45, 0, [1, 0, 0])
    ie = at.EXCITED_STATES.index((4, 4))
    nu0 = atom.d2_centroid_hz + atom.excited_hfs_hz[4] + atom.g_excited[4] * 4 * at.MU_B * 1e-4 / at.H - response.eg[at.UP]
    freq = nu0 - 1e9
    omega = response.eunit * atom.dipole_si[ie, at.UP] / at.HBAR
    expected_shift = omega**2 / (4 * 2 * np.pi * (-1e9)) / (2 * np.pi)
    assert response.shift_hz(freq, eps)[at.UP] == pytest.approx(expected_shift, rel=3e-4)
    rates = abs(response.scattering_amplitudes(freq, eps))**2
    expected_rate = atom.gamma * omega**2 / (4 * (2 * np.pi * 1e9)**2)
    assert rates[at.UP].sum() == pytest.approx(expected_rate, rel=.003)
    assert rates[at.UP, at.UP].sum() == pytest.approx(rates[at.UP].sum(), rel=1e-6)


def test_uniformization_matches_independent_augmented_exponential():
    rng = np.random.default_rng(9)
    W = rng.uniform(0, 20, (12, 12))
    K = np.array([[.7, .1], [.2, .6]])
    kernels = (np.eye(1), np.eye(1), K)
    total = np.broadcast_to(W.sum(axis=1)[:, None, None, None], (12, 1, 1, 2)).copy()
    term = RecoilTerm(W, kernels, total, 0, 'pump')
    loss = total - term.retained_hazard()
    counts = np.array([total, total * 0, total * 0, total * 0])
    p = rng.uniform(size=total.shape)
    p /= p.sum()
    dim = p.size
    G = np.kron(W.T, K)
    aug = np.zeros((dim + 2, dim + 2))
    aug[:dim, :dim] = G - np.diag(total.ravel())
    aug[dim, :dim] = loss.ravel()
    aug[dim + 1, :dim] = total.ravel()
    dt = .007
    expected = expm(dt * aug) @ np.r_[p.ravel(), 0., 0.]
    got, lf, cn = uniform_step(p, [(term, 1.)], total, loss, counts, dt)
    np.testing.assert_allclose(got.ravel(), expected[:dim], atol=2e-14)
    assert lf.sum() == pytest.approx(expected[dim], abs=2e-14)
    assert cn[0] == pytest.approx(expected[dim + 1], abs=2e-14)
    assert got.sum() + lf.sum() == pytest.approx(1., abs=2e-14)


def test_zero_kick_does_not_cause_loss():
    atom = at.load_atomic_data()
    nu = mo.lattice_frequency_hz(atom.mass_kg, 785, 15)
    site = mo.lattice_site(atom.mass_kg, nu, 15e-6 * at.K_B / at.H, 20)
    K, total = cross_kernel(site, 0., 0., 0., 0., 'sigma', 16)
    np.testing.assert_allclose(K, np.eye(site.N), atol=2e-13)
    np.testing.assert_allclose(total, 1., atol=2e-13)


def test_local_unitary_matches_static_matrix_exponential(response):
    c = config()
    c.spin_pump.enabled = c.repump.enabled = False
    calibration = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    local = LocalExperiment(c, response, calibration, [110, 90, 70], .73)
    local.terms = []
    seg = local.model.schedule[0]
    H = local.operators(seg)[-1]
    p = local.P0.copy()
    nz = p.shape[-1]
    rho = np.zeros(p.shape[1:3] + (2 * nz, 2 * nz), complex)
    di = np.arange(2 * nz)
    rho[..., di, di] = np.concatenate([p[at.UP], p[at.DOWN]], axis=-1)
    U = expm(-1j * H * seg.t1)
    expected = U @ rho @ U.conj().swapaxes(-1, -2)
    with _single_thread():
        r = solve_local(local, c)
    np.testing.assert_allclose(r['rho_final'], expected.sum(axis=(0, 1)), atol=2e-13)
    assert r['P'][-1].sum() == pytest.approx(1., abs=2e-13)


def test_custom_spin_populations_and_three_axis_loss(response):
    c = config()
    c.initial.spin_preset = 'custom'
    c.initial.custom_spin_populations = [0.] * 12
    c.initial.custom_spin_populations[at.UP] = 1.
    calibration = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    local = LocalExperiment(c, response, calibration, [90, 100, 120], .2)
    np.testing.assert_allclose(local.P0.sum(axis=(1, 2, 3)), c.initial.custom_spin_populations, atol=1e-14)
    active, gout, hazard, counts, _ = local.operators(local.model.schedule[0])
    assert hazard.max() > 0
    assert np.min(hazard) >= 0
    assert np.min(gout) >= 0
    with _single_thread():
        r = solve_local(local, c)
    np.testing.assert_allclose(r['P'].sum(axis=(1, 2)) + r['loss'], 1., atol=1e-11)
    assert np.min(r['min_eig']) > -1e-12
    assert r['loss'][-1] > 0


def test_mixture_temperature_roundtrip_different_local_depths(response):
    c = config()
    cal = dict(intensity_per_unit_tone_w_m2=10000., residual_coherence_decay_rate_s=0.)
    locals = [LocalExperiment(c, response, cal, p, 0.) for p in ([0, 0, 0], [110, 90, 60])]
    parts = [(l, {'P': np.ones((1, 1, 1)) * w}) for l, w in zip(locals, (.3, .7))]
    energy = sum(w * (mo.boltzmann_pn(l.sites[0].energies_hz, 10e-6) @ (l.sites[0].energies_hz - l.sites[0].energies_hz[0])) for l, w in zip(locals, (.3, .7)))
    assert mixture_temperature(energy, parts, 0) == pytest.approx(10e-6, rel=1e-9)


def test_modeled_carrier_matches_two_frequency_analytic_limit(response, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from rb85rsc import far_detuned as fd
    c = config()
    c.raman.carrier_decay_mode = 'modeled'
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    low, high, eps, _ = raman_fields(c, response)
    monkeypatch.setattr(fd, 'raman_fields', lambda *_: (low, high, eps, np.ones((2, 2))))
    monkeypatch.setattr(fd, 'spatial_samples', lambda *_: (np.zeros((2, 3)), np.array([0., np.pi/2])))
    site = SimpleNamespace(N=1, n_bound=1, x0=0., energies_hz=np.array([0.]),
                           displacement=lambda _: np.ones((1, 1)))
    monkeypatch.setattr(fd.mo, 'lattice_site', lambda *_: site)
    # Equal subensembles have local Rabi frequencies 4I and 2I. Thus
    # P=.9/2*[1-cos(6pi I t) cos(2pi I t)] and f_RMS=sqrt(10)*I.
    result = calibrate_carrier(c, response)
    intensity = c.raman.carrier_rabi_hz / np.sqrt(10)
    t = np.array(result['times_s'])
    expected = .9 / 2 * (1 - np.cos(6*np.pi*intensity*t) * np.cos(2*np.pi*intensity*t))
    assert result['intensity_per_unit_tone_w_m2'] == pytest.approx(intensity)
    np.testing.assert_allclose(result['fitted_trace'], expected, atol=2e-14)
    assert result['modeled_rms_carrier_hz'] == pytest.approx(5000.)
    assert result['residual_coherence_decay_rate_s'] == 0.
    assert result['calibration_compatible'] is None  # measurement agreement not asserted
    assert result['reported_envelope_used_for_calibration'] is False
    tau = np.arccos(np.exp(-1)) / (2*np.pi*intensity)
    assert result['modeled_inhomogeneous_envelope_first_1e_time_s'] == pytest.approx(tau, rel=1e-5)
    # The discarded decay summary and readout contrast must not change physics.
    c.raman.flop_decay_periods = 40.
    c.raman.flop_contrast = .3
    changed = calibrate_carrier(c, response)
    assert changed['cache_key'] != result['cache_key']
    assert changed['intensity_per_unit_tone_w_m2'] == result['intensity_per_unit_tone_w_m2']
    assert changed['residual_coherence_decay_rate_s'] == 0.
    np.testing.assert_allclose(changed['fitted_trace'], expected / 3, atol=2e-14)
    assert calibrate_carrier(c, response) == changed  # portable cache round trip


def test_modeled_decay_diagnostic_does_not_claim_measured_agreement():
    from rb85rsc.runner import run
    c = config()
    with _single_thread():
        result = run(c)
    assert result['final']['carrier_decay_mode'] == 'modeled'
    assert result['final']['calibration_compatible'] is None
    messages = [v for v in result['validity'].items if v[0] == 'carrier calibration']
    assert len(messages) == 1 and messages[0][1] == 'ok'
    assert 'reported decay is not a calibration constraint' in messages[0][2]


def test_ensemble_requires_measured_carrier_before_calibration():
    c = config()
    c.raman.calibration = 'pathway'
    with pytest.raises(ConfigError, match='ensemble mode requires raman.calibration'):
        c.validate()


def test_custom_motion_reports_discarded_weight():
    from rb85rsc.runner import run
    c = config()
    c.initial.motion_mode = 'custom'
    c.initial.custom_pn = [1.] * 20
    with _single_thread():
        result = run(c)
    nz = result['ensemble_nodes'][0][0].sites[2].N
    assert result['final']['initial_tail_discarded'] == pytest.approx((20 - nz) / 20)
    assert any(category == 'initial distribution' and status == 'warning'
               for category, status, _ in result['validity'].items)


def test_cache_failed_write_preserves_previous_result(tmp_path):
    import json
    from rb85rsc.far_detuned import _write_cache
    target = tmp_path / 'cache.json'
    _write_cache(target, {'intensity': 1.})
    with pytest.raises(TypeError):
        _write_cache(target, {'not_json': object()})
    assert json.loads(target.read_text()) == {'intensity': 1.}
    assert not list(tmp_path.glob('*.tmp'))
