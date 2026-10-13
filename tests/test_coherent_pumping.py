"""Independent master-equation checks for event-driven 3D pumping."""
from threading import Event

import numpy as np
import pytest
from scipy.linalg import expm

from rb85rsc.coherent_pumping import PhotonGroup, average_nodes, prepare_local, run_cloud, simulate
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import SimulationStopped, _single_thread
from rb85rsc.protocols import Segment


def toy():
    """Two spins, three two-level axes, shared recoil and absorbing loss."""
    rotation = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], complex)
    kicks = tuple(np.array([rotation(a), rotation(-a/2)]) for a in (.7, .4, .6))
    weights = np.array([[.3, .7], [.6, .4]])
    rates = np.array([[[.4, .2], [1.1, .3]], [[.7, .2], [.3, .6]]])
    travel = PhotonGroup(rates, kicks, weights, None, None, 0, 'pump')
    absorption = np.array([[1.1, .3j], [-.3j, .7]])
    standing_kicks = (kicks[0] @ absorption, .85*kicks[1], kicks[2])
    standing = PhotonGroup(rates*.6, standing_kicks, weights, 0,
                           absorption.conj().T @ absorption, 3, 'always')
    h = [np.array([[0., w], [0., w*1.2]]) for w in (2.3, 1.5, 3.2)]
    initial = np.zeros((2, 2, 2, 2))
    initial[0, 0, 0, 0] = 1
    return h, [travel, standing], initial


def dense_reference(h, groups, initial, schedule, mode):
    """Build the full joint density generator independently, including loss.

    Uses neither separable trajectory evolution nor production recoil kernels.
    Photon counters integrate the FULL absorption hazard before bound projection.
    """
    shape = initial.shape
    spins, sizes = shape[0], shape[1:]
    d = int(np.prod(sizes))
    eye = [np.eye(n) for n in sizes]
    tensor = lambda v: np.kron(np.kron(v[0], v[1]), v[2])
    ham = [sum(tensor([np.diag(h[a][s]) if b == a else eye[b] for b in range(3)])
               for a in range(3)) for s in range(spins)]
    size = spins*d*d
    state = np.zeros(size+4, complex)
    state[:size] = np.array([np.diag(p.ravel()) for p in initial]).ravel()
    for segment in schedule:
        scales = dict(pump=segment.pump, repump=segment.repump, always=1.)
        generator = np.zeros((size+4, size+4), complex)
        for col in range(size):
            rho = np.zeros((spins, d, d), complex)
            rho.ravel()[col] = 1
            drho = np.array([-1j*(H @ r-r @ H) for H, r in zip(ham, rho)])
            counts = np.zeros(4, complex)
            for g in groups:
                rates = g.rates*scales[g.scale_name]
                full = tensor([g.full_absorption if a == g.absorption_axis else eye[a] for a in range(3)])
                if mode == 'population':
                    full = np.diag(np.diag(full))
                for s in range(spins):
                    outgoing = rates[s].sum()
                    drho[s] -= outgoing*(full @ rho[s]+rho[s] @ full)/2
                    counts[g.counter] += outgoing*np.trace(full @ rho[s])
                    for dest in range(spins):
                        for channel in (0, 1):
                            for q, weight in enumerate(g.weights[channel]):
                                J = tensor([axis[q] for axis in g.kicks])
                                if mode == 'coherent':
                                    gain = J @ rho[s] @ J.conj().T
                                else:
                                    gain = np.diag(abs(J)**2 @ np.diag(rho[s]))
                                drho[dest] += rates[s, dest, channel]*weight*gain
            generator[:, col] = np.r_[drho.ravel(), counts]
        state = expm(generator*(segment.t1-segment.t0)) @ state
    rho = state[:size].reshape(spins, d, d)
    P = np.diagonal(rho, axis1=1, axis2=2).real.reshape(shape)
    return dict(survival=P.sum(), loss=1-P.sum(), P_up_absolute=P[1].sum(),
                P_n0_absolute=P[:, :, :, 0].sum(), P_target_absolute=P[1, :, :, 0].sum(),
                P_3d_ground_absolute=P[:, 0, 0, 0].sum(), P_target_3d_absolute=P[1, 0, 0, 0],
                photons_spin_pump=state[size].real,
                photons_lattice_scatter=state[size+3].real, photons_total=state[size:].real.sum()), rho


@pytest.mark.parametrize('mode', ['coherent', 'population'])
def test_full_joint_master_equation_with_loss_and_unsampled_boundary(mode):
    h, groups, initial = toy()
    schedule = [Segment(0., .31, 0., 1., 1., 'on'), Segment(.31, .8, 0., 0., 0., 'lattice only')]
    with _single_thread():
        exact, rho = dense_reference(h, groups, initial, schedule, mode)
        result = simulate(h, groups, initial, schedule, np.linspace(0, .8, 5), 40000, 276,
                          mode=mode, target_spin=1)
    for key, expected in exact.items():
        assert abs(result['mean'][key][-1]-expected) < 7*result['standard_error'][key][-1]+1/40000, key
    for axis, marginal in enumerate(result['rho_axes_final']):
        joint = rho.reshape((2, 2, 2, 2, 2, 2, 2))
        # Partial trace using unique spin/row/column labels.
        labels = ['s', 'a', 'b', 'c', 'a', 'b', 'c']
        labels[axis+4] = 'd'
        reduced = np.einsum(''.join(labels)+'->s'+labels[axis+1]+'d', joint)
        np.testing.assert_allclose(marginal, reduced, atol=.015, rtol=0)
        assert np.linalg.eigvalsh(marginal).min() > -1e-12
    assert result['max_conservation_error'] < 1e-15
    assert result['photon_events'] <= result['proposed_events']


def test_coherence_changes_multiple_recoil_events():
    h, groups, initial = toy()
    h = [np.zeros_like(v) for v in h]
    groups = groups[:1]
    groups[0].kicks = tuple(v[:1] for v in groups[0].kicks)
    groups[0].weights = np.ones((2, 1))
    schedule = [Segment(0., 1.5, 0., 1., 1., '')]
    with _single_thread():
        coherent, _ = dense_reference(h, groups, initial, schedule, 'coherent')
        population, _ = dense_reference(h, groups, initial, schedule, 'population')
        observed = {}
        for mode, exact in (('coherent', coherent), ('population', population)):
            r = simulate(h, groups, initial, schedule, [0., 1.5], 40000, 56,
                         mode=mode, target_spin=1)
            mean = r['mean']['P_target_absolute'][-1]
            error = r['standard_error']['P_target_absolute'][-1]
            assert abs(mean-exact['P_target_absolute']) < 6*error
            observed[mode] = (mean, error)
    assert abs(coherent['P_target_absolute']-population['P_target_absolute']) > .01
    a, b = observed['coherent'], observed['population']
    assert abs(a[0]-b[0]) > 5*np.hypot(a[1], b[1])


def test_terminal_photon_is_counted_once():
    h = [np.zeros((1, 1)) for _ in range(3)]
    rate = np.array([[[2., 0.]]])
    g = PhotonGroup(rate, tuple(np.zeros((1, 1, 1)) for _ in range(3)), np.ones((2, 1)),
                    None, None, 0, 'pump')
    result = simulate(h, [g], np.ones((1, 1, 1, 1)), [Segment(0., 1., 0., 1., 0., '')],
                      [0., 1.], 30000, 16, target_spin=0)
    assert abs(result['mean']['survival'][-1]-np.exp(-2)) < 7*result['standard_error']['survival'][-1]
    np.testing.assert_array_equal(result['mean']['photons_total'], result['mean']['loss'])


def test_zero_rates_and_seed_reproducibility():
    h, groups, initial = toy()
    for g in groups:
        g.rates[:] = 0
    schedule = [Segment(0., 1., 0., 1., 1., '')]
    a = simulate(h, groups, initial, schedule, [0., 1.], 8, 0, target_spin=0)
    b = simulate(h, groups, initial, schedule, [0., 1.], 8, 0, target_spin=0)
    np.testing.assert_array_equal(a['mean']['P_target_absolute'], [1., 1.])
    np.testing.assert_array_equal(a['rho_axes_final'], b['rho_axes_final'])
    assert a['photon_events'] == 0


def test_invalid_trajectory_inputs_and_cancellation():
    h, groups, initial = toy()
    schedule = [Segment(0., 1., 0., 1., 1., '')]
    for count in (1, 2.5, True):
        with pytest.raises(ValueError, match='integer trajectories'):
            simulate(h, groups, initial, schedule, [0., 1.], count, 0)
    stop = Event()
    stop.set()
    with pytest.raises(SimulationStopped):
        simulate(h, groups, initial, schedule, [0., 1.], 2, 0, stop=stop)
    groups[0].weights *= .5
    with pytest.raises(ValueError, match='angular weights'):
        simulate(h, groups, initial, schedule, [0., 1.], 2, 0)


def test_active_raman_rejected():
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    cfg.timing.custom_segments[0]['raman'] = 1.
    with pytest.raises(ValueError, match='Raman off'):
        prepare_local(cfg)


def test_real_photon_operators_reproduce_production_population_generator():
    from rb85rsc.ensemble import LocalExperiment
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    cfg.trap.depth_uk = 2.
    cfg.ensemble.transverse_depth_uk = 2.
    cfg.numerics.recoil_quadrature_points = 6
    original = cfg.to_dict()
    with _single_thread():
        local, h, groups = prepare_local(cfg)
        control = LocalExperiment(cfg, local.response, local.calibration, [0., 0., 0.], 0.)
        P = np.random.default_rng(71).random(local.shape)
        P /= P.sum()
        gain = np.zeros_like(P)
        total = np.zeros_like(P)
        for g in groups:
            full = np.ones(local.shape[1:])
            if g.absorption_axis is not None:
                shape = [1, 1, 1]
                shape[g.absorption_axis] = local.shape[g.absorption_axis+1]
                full *= np.diag(g.full_absorption).real.reshape(shape)
            total += g.rates.sum(axis=(1, 2))[:, None, None, None]*full
            for channel in (0, 1):
                K = sum(w*np.kron(np.kron(abs(g.kicks[0][q])**2, abs(g.kicks[1][q])**2),
                                    abs(g.kicks[2][q])**2) for q, w in enumerate(g.weights[channel]))
                gain += (g.rates[:, :, channel].T @ P.reshape(12, -1) @ K.T).reshape(P.shape)
        np.testing.assert_allclose(total, sum(t.total for t in control.terms), atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(gain, sum(t.gain(P) for t in control.terms), atol=1e-10, rtol=1e-11)
        expected_h = 2*np.pi*control.lattice_shifts_hz.copy()
        actual_h = np.zeros(local.shape)
        for a, site in enumerate(local.sites):
            assert h[a].shape == (12, site.N)
            dims = [1, 1, 1, 1]
            dims[a+1] = site.N
            expected_h += 2*np.pi*(site.energies_hz-site.energies_hz[0]).reshape(dims)
            dims[0] = 12
            actual_h += h[a].reshape(dims)
        np.testing.assert_allclose(actual_h, expected_h, atol=2e-9, rtol=1e-13)
    assert cfg.to_dict() == original
    assert local.terms == []


def test_cloud_average_uses_within_site_monte_carlo_variances():
    nodes = [dict(modes=dict(coherent=dict(t_s=np.array([0., 1.]),
                mean=dict(survival=np.array([1., mean])),
                standard_error=dict(survival=np.array([0., error])))))
             for mean, error in ((.8, .03), (.4, .04))]
    result = average_nodes(nodes, 'coherent')
    np.testing.assert_allclose(result['mean']['survival'], [1., .6])
    np.testing.assert_allclose(result['standard_error']['survival'], [0., .025])


def test_cloud_seed_is_independent_of_workers_and_requested_modes():
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    cfg.ensemble.samples = 2
    cfg.numerics.recoil_quadrature_points = 4
    cfg.timing.custom_segments[0]['duration_ms'] = .01
    cfg.timing.n_samples = 3
    original = cfg.to_dict()
    single = run_cloud(cfg, trajectories=32, seed=72, workers=1)
    parallel = run_cloud(cfg, trajectories=32, seed=72, workers=2, modes=('population', 'coherent'))
    for left, right in zip(single['nodes'], parallel['nodes']):
        assert left['index'] == right['index']
        for mode in single['modes']:
            assert left['modes'][mode]['seed_entropy'] == right['modes'][mode]['seed_entropy']
            for key, value in left['modes'][mode]['mean'].items():
                np.testing.assert_array_equal(value, right['modes'][mode]['mean'][key])
    assert cfg.to_dict() == original


def test_cloud_validation_before_launch():
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    for kwargs in (dict(seed=-1), dict(workers=0), dict(trajectories=1), dict(modes=('other',))):
        with pytest.raises(ValueError):
            run_cloud(cfg, **kwargs)
    stop = Event()
    stop.set()
    with pytest.raises(SimulationStopped):
        run_cloud(cfg, stop=stop)


def test_cli_population_check_detects_failure_and_exports(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    script = importlib.import_module('check_coherent_pumping')
    estimate = dict(mean=dict(survival=np.array([1., .5])), standard_error=dict(survival=np.array([0., .001])))
    assert script.compare_control(estimate, dict(survival=.502), 10000)['passed']
    assert not script.compare_control(estimate, dict(survival=.6), 10000)['passed']
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    cfg.ensemble.samples = 1
    cfg.numerics.recoil_quadrature_points = 4
    cfg.timing.n_samples = 3
    cfg.timing.custom_segments[0]['duration_ms'] = .01
    result = run_cloud(cfg, trajectories=32, workers=1)
    script.save_trajectories(result, tmp_path)
    saved = np.load(tmp_path/'cloud_trajectories.npz')
    np.testing.assert_array_equal(saved['coherent_mean_survival'], result['modes']['coherent']['mean']['survival'])
    node = np.load(tmp_path/'nodes'/'site_000.npz')
    np.testing.assert_array_equal(node['coherent_rho_x_final'], result['nodes'][0]['modes']['coherent']['rho_axes_final'][0])


def test_independent_repeat_comparison_guards_and_statistics(monkeypatch):
    import importlib
    from copy import deepcopy
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    script = importlib.import_module('check_coherent_pumping')
    a = dict(trajectory_seed=0, config=SimConfig.from_json('examples/onlyOP_correlated.json').to_dict(),
             trajectories_per_site=4096, sites=64,
             final={m: dict(survival=dict(mean=.5, standard_error=.001)) for m in ('coherent', 'population')})
    b = deepcopy(a)
    with pytest.raises(ValueError, match='independent'):
        script.compare_reports(a, b)
    b['trajectory_seed'] = 1
    b['config']['ensemble']['trajectories_per_site'] = 8192
    b['config']['ensemble']['trajectory_seed'] = 1
    b['config']['numerics']['recoil_quadrature_points'] = 48
    b['final']['coherent']['survival']['mean'] += .002
    result = script.compare_reports(a, b)
    assert result['statistically_consistent']
    assert result['modes']['coherent']['survival']['combined_standard_error'] == pytest.approx(np.sqrt(2)*.001)
    b['final']['coherent']['survival']['mean'] += .1
    assert not script.compare_reports(a, b)['statistically_consistent']
    b['config']['ensemble']['samples'] = 128
    with pytest.raises(ValueError, match='matching configs'):
        script.compare_reports(a, b)
