"""Independent spherical moments, photon jumps, and Markov propagation."""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from rb85rsc import atomic as at, motion as mo, polarization as pol
from rb85rsc.config import SimConfig, ConfigError
from rb85rsc.correlated_recoil import CorrelatedRecoil, emission_sphere
from rb85rsc.dynamics import _single_thread, SimulationStopped
from rb85rsc.ensemble import RecoilTerm, _RecoilGain, _UniformizedScattering, _term, LocalExperiment, solve_local
from rb85rsc.far_detuned import DLineResponse
from rb85rsc.runner import run, metadata

ROOT = Path(__file__).resolve().parents[1]


def small_sites():
    sites = [mo.HarmonicBasis(n, 50e3) for n in (2, 3, 2)]
    for site, x0 in zip(sites, (.2, .3, .4)):
        site.x0 = x0
    return sites


@pytest.mark.parametrize('pattern,transverse,parallel', [('isotropic', 1/3, 1/3), ('pi', .4, .2), ('sigma', .3, .4)])
def test_sphere_normalization_and_analytic_dipole_moments(pattern, transverse, parallel):
    b = pol.unit([.2, .7, -.3])
    u, w = emission_sphere(b, 12, pattern)
    assert np.all(w > 0)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1, atol=1e-15)
    assert w.sum() == pytest.approx(1., abs=1e-15)
    np.testing.assert_allclose(w @ u, 0, atol=1e-15)
    np.testing.assert_allclose((u.T*w) @ u, transverse*np.eye(3)+(parallel-transverse)*np.outer(b, b), atol=1e-15)
    if pattern == 'isotropic':
        # One direction obeys |u|=1: its squared Cartesian components cannot
        # be independent. Independent marginals would give 1/9, not 1/15.
        assert w @ (u[:, 0]**2*u[:, 1]**2) == pytest.approx(1/15, abs=1e-15)


def test_bulk_lattice_overlaps_match_individual_grid_integrals():
    atom = at.load_atomic_data()
    with _single_thread():
        site = mo.lattice_site(atom.mass_kg, 60e3, 300e3, 16)
        etas = np.array([0., .03, -.3, 1.1])
        expected = np.array([site.displacement(eta) for eta in etas])
        before = set(site._cache)
        np.testing.assert_allclose(site.displacements(etas), expected, rtol=3e-13, atol=2e-15)
        assert set(site._cache) == before


@pytest.mark.parametrize('pattern', ['isotropic', 'pi', 'sigma'])
def test_joint_kernel_matches_direct_kronecker_photon_jumps(pattern):
    sites = small_sites()
    ka, kb = np.array([.3, -.8, .6]), np.array([-.1, .4, -.2])
    builder = CorrelatedRecoil(sites, [1, 2, 3], 8)
    got = builder.kernel(ka, kb, 1., pattern)
    u, w = emission_sphere([1, 2, 3], 8, pattern)
    expected = np.zeros_like(got, dtype=complex)
    for direction, weight in zip(u, w):
        da = db = np.ones((1, 1), complex)
        for i, site in enumerate(sites):
            da = np.kron(da, site.displacement((ka[i]-direction[i])*site.x0))
            db = np.kron(db, site.displacement((kb[i]-direction[i])*site.x0))
        expected += weight*da*db.conj()
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=2e-15)
    np.testing.assert_array_equal(builder.kernel(kb, ka, 1., pattern), got.conj())


def test_zero_emission_reduces_to_product_of_axis_displacements():
    sites = small_sites()
    ka = np.array([.4, .3, -.2])
    got = CorrelatedRecoil(sites, [0, 0, 1], 4).kernel(ka, ka, 0., 'pi')
    expected = np.ones((1, 1))
    for k, site in zip(ka, sites):
        expected = np.kron(expected, abs(site.displacement(k*site.x0))**2)
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_coherent_absorption_positive_gain_and_loss_match_direct_amplitude_sum():
    sites = small_sites()
    ks = [np.array([0., 0., 1.]), np.array([0., 0., -1.])]
    amps = [1., .8*np.exp(.7j)]
    joint = CorrelatedRecoil(sites, [1, 0, 0], 8)
    terms = [_term(sites, np.eye(12)*a*b.conjugate(), ka, kb, 1., np.array([1., 0., 0.]),
                   'sigma', 8, 0, 'pump', np.eye(3), joint)
             for a, ka in zip(amps, ks) for b, kb in zip(amps, ks)]
    expected = np.zeros((12, 12))
    u, w = emission_sphere([1, 0, 0], 8, 'sigma')
    for direction, weight in zip(u, w):
        amplitude = 0
        for a, k in zip(amps, ks):
            d = np.ones((1, 1), complex)
            for i, site in enumerate(sites):
                d = np.kron(d, site.displacement((k[i]-direction[i])*site.x0))
            amplitude = amplitude+a*d
        expected += weight*abs(amplitude)**2
    p = np.arange(1., 145.).reshape(12, 2, 3, 2)
    p /= p.sum()
    got = _RecoilGain([(t, 1.) for t in terms])(p)
    np.testing.assert_allclose(got.reshape(12, 12), p.reshape(12, 12) @ expected.T, atol=1e-15)
    assert got.min() >= 0
    hazard = sum(t.total-t.retained_hazard() for t in terms)
    assert hazard.min() >= -1e-14


def test_correlated_uniformization_matches_independent_augmented_exponential():
    sites = small_sites()
    k = np.array([0., 0., 1.])
    kernel = CorrelatedRecoil(sites, [0, 0, 1], 8).kernel(k, k, 1., 'pi')
    rng = np.random.default_rng(47)
    spin = rng.uniform(0, 2, (12, 12))
    shape = (12, 2, 3, 2)
    total = np.broadcast_to(spin.sum(axis=1)[:, None, None, None], shape).copy()
    term = RecoilTerm(spin, (), total, 0, 'pump', kernel)
    loss = total-term.retained_hazard()
    counts = np.array([total, total*0, total*0, total*0])
    generator = np.zeros((149, 149))
    generator[:144, :144] = np.kron(spin.T, kernel)-np.diag(total.ravel())
    generator[144, :144] = loss.ravel()
    generator[145:, :144] = counts.reshape(4, -1)
    p = rng.random(shape)
    p /= p.sum()
    expected = expm(.03*generator) @ np.r_[p.ravel(), np.zeros(5)]
    actual, escaped, photons = _UniformizedScattering([(term, 1.)], total, loss, counts).step(p, .03)
    np.testing.assert_allclose(np.r_[actual.ravel(), escaped.sum(), photons], expected, atol=2e-14)
    np.testing.assert_allclose(term.gain(p), _RecoilGain([(term, 1.)])(p), atol=1e-15)
    assert actual.sum()+escaped.sum() == pytest.approx(1, abs=2e-14)


def test_correlated_kernel_rigid_apparatus_rotation():
    sites = small_sites()
    axes = pol.lattice_frame([0, 0, 1], [1, 0, 0])
    b, k = pol.unit([1, 2, 3]), pol.unit([.2, -.4, .7])
    rotation = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
    a = _term(sites, np.eye(12), k, k, 1., b, 'pi', 8, 0, 'pump', axes,
              CorrelatedRecoil(sites, axes @ b, 8))
    ax = axes @ rotation.T
    c = _term(sites, np.eye(12), rotation @ k, rotation @ k, 1., rotation @ b, 'pi', 8, 0, 'pump', ax,
              CorrelatedRecoil(sites, ax @ rotation @ b, 8))
    np.testing.assert_allclose(a.joint_kernel, c.joint_kernel, atol=2e-15)
    np.testing.assert_allclose(a.total, c.total, atol=2e-15)


def config():
    c = SimConfig.from_json(ROOT/'examples/onlyOP_diagonalized.json')
    c.ensemble.recoil_model = 'correlated'
    c.ensemble.samples = 2
    c.ensemble.workers = 1
    c.trap.depth_uk = c.trap.lattice_x.depth_uk = c.ensemble.transverse_depth_uk = 3.
    c.timing.custom_segments[0]['duration_ms'] = .006
    c.timing.n_samples = 3
    c.numerics.recoil_quadrature_points = 8
    return c


def test_correlated_ensemble_serial_parallel_conservation_and_metadata():
    c = config()
    serial = run(c)
    parallel = run(c, ensemble_workers=2)
    for name in ('P', 'counts', 'overflow'):
        np.testing.assert_array_equal(getattr(serial['result'], name), getattr(parallel['result'], name))
    for (_, a), (_, b) in zip(serial['ensemble_nodes'], parallel['ensemble_nodes']):
        np.testing.assert_array_equal(a['P_final_3d'], b['P_final_3d'])
    assert serial['final']['max_conservation_error'] < 1e-12
    assert serial['result'].P.min() >= 0
    meta = metadata(serial)
    assert meta['model_summary']['recoil_model'] == 'correlated'
    assert any('one shared 3D photon direction' in x for x in meta['assumptions'])
    assert {v['category']: v['status'] for v in meta['validity']}['recoil coherence'] == 'warning'


def test_correlated_raman_and_lattice_scattering_are_included():
    c = config()
    c.timing.custom_segments[0]['raman'] = 1.
    c.raman.scattering_rate_s = .4
    calibration = dict(intensity_per_unit_tone_w_m2=2e4, residual_coherence_decay_rate_s=0.)
    with _single_thread():
        response = DLineResponse(at.load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction, c.optical.zeeman_model)
        local = LocalExperiment(c, response, calibration, [20, 30, 40], .73)
        assert all(t.joint_kernel is not None for t in local.terms)
        assert {t.counter for t in local.terms} == {0, 1, 2, 3}
        r = solve_local(local, c)
    np.testing.assert_allclose(r['P'].sum(axis=(1, 2))+r['loss'], 1, atol=2e-13)
    assert np.all(r['counts'][-1] > 0)
    assert r['coherence'][-1] > 0


def test_config_limits_and_cancellation_during_construction():
    c = config()
    c.ensemble.enabled = False
    with pytest.raises(ConfigError, match='correlated recoil requires'):
        c.validate()
    c.ensemble.enabled = True
    c.numerics.recoil_quadrature_points = 2.5
    with pytest.raises(ConfigError, match='must be an integer'):
        c.validate()
    with pytest.raises(ValueError, match='at most 1024'):
        CorrelatedRecoil([mo.HarmonicBasis(11, 50e3)]*3, [1, 0, 0], 8)
    def stop():
        raise SimulationStopped()
    builder = CorrelatedRecoil(small_sites(), [1, 0, 0], 8, stop)
    with pytest.raises(SimulationStopped):
        builder.kernel(np.zeros(3), np.zeros(3), 1., 'sigma')


@pytest.mark.parametrize('delta,exit_code', [(1e-10, 0), (1e-3, 1), (np.nan, 1)])
def test_comparison_checks_each_node_even_when_cloud_average_cancels(tmp_path, monkeypatch, delta, exit_code):
    import importlib.util
    import json
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('correlated_workflow', ROOT/'scripts/check_correlated_recoil.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    def simulate(cfg, **kwargs):
        change = delta if cfg.numerics.recoil_quadrature_points == 8 else 0.
        nodes = [(None, {key: np.array([sign*change]) for key in ('P', 'counts', 'loss', 'P_final_3d', 'rho_final')})
                 for sign in (-1, 1)]
        return dict(result=SimpleNamespace(P=np.zeros(1), counts=np.zeros(1), overflow=np.zeros(1)),
                    final=dict(P_target_absolute=.2, P_trapped=.5, P_up=.4, photons_total=2., wall_time_s=0.),
                    validity=SimpleNamespace(status='invalid'), ensemble_nodes=nodes)
    monkeypatch.setattr(script, 'run', simulate)
    monkeypatch.setattr(script, 'export', lambda *args, **kwargs: None)
    assert script.main(['--output', str(tmp_path), '--quadratures', '4', '8', '--samples', '2']) == exit_code
    result = json.loads((tmp_path/'comparison.json').read_text())
    assert result['angular_convergence']['max_absolute_differences']['P'] == 0.
    assert result['angular_convergence']['status'] == ('passed' if exit_code == 0 else 'failed')
