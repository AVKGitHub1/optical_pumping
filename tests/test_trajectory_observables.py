"""Independent checks of survivor conditioning and thermal uncertainty."""
import numpy as np
import pytest

from rb85rsc.analysis import _derivative
from rb85rsc.coherent_pumping import average_nodes
from rb85rsc.config import SimConfig
from rb85rsc.ensemble import canonical_mixture_temperature
from rb85rsc.trajectory_observables import (
    ObservableRecorder, analyze_observables, derivative_error, ratio_estimate,
)


def test_ratio_error_matches_direct_influence_with_spatial_strata():
    # Y vanishes for lost atoms. Different sites have different survival and
    # conditional means; averaging site-conditioned means is incorrect.
    samples = [(np.array([1., 2., 0., 0.]), np.array([1., 1., 0., 0.])),
               (np.array([2., 3., 5., 0., 0., 0.]), np.array([1., 1., 1., 0., 0., 0.]))]
    y, sy, s, ss = [np.array([x[which].mean() if stat == 'mean' else x[which].std(ddof=1)/np.sqrt(len(x[which]))
                            for x in samples])[:, None] for which, stat in ((0, 'mean'), (0, 'se'), (1, 'mean'), (1, 'se'))]
    ratio, error = ratio_estimate(y, sy, s, ss, [4, 6])
    expected = y.mean()/s.mean()
    variance = sum(np.var((a-expected*b)/s.mean(), ddof=1)/len(a) for a, b in samples)/4
    assert ratio[0] == pytest.approx(expected)
    assert error[0] == pytest.approx(np.sqrt(variance))
    # Perfectly correlated numerator/denominator has no conditional error.
    same, zero = ratio_estimate(s, ss, s, ss, [4, 6])
    np.testing.assert_allclose(same, 1.)
    np.testing.assert_allclose(zero, 0., atol=1e-8)
    missing, error = ratio_estimate(y*0, sy*0, s*0, ss*0, [4, 6])
    assert np.isnan(missing).all() and np.isnan(error).all()


def recorded_cloud(extinct=False):
    """Known per-trajectory observations with different local basis sizes."""
    rng = np.random.default_rng(136)
    times = np.linspace(0, 1e-3, 7)
    nodes, histories = [], []
    for j, (count, sizes) in enumerate(((7, (2, 3, 2)), (13, (3, 2, 4)))):
        levels = [np.arange(size)*16000*(axis+1+j) for axis, size in enumerate(sizes)]
        record = ObservableRecorder(count, times, levels, 2)
        means, errors, history = {}, {}, []
        spin = np.full(count, 11)
        initial = [rng.dirichlet(np.r_[12., np.ones(size-1)], count) for size in sizes]
        for it in range(len(times)):
            alive = np.arange(count) >= (it//3 + j)
            if extinct and it == len(times)-1:
                alive[:] = False
            # Identical trajectory membership over time is needed to verify
            # temporal covariance. Keep within the positive-temperature range.
            p = [(v*(1+.02*it*np.arange(v.shape[1]))) for v in initial]
            p = [v/v.sum(axis=1)[:, None]*alive[:, None] for v in p]
            record.record(it, spin, [np.sqrt(v).astype(complex) for v in p], alive, 12)
            p0 = np.array([v[:, 0] for v in p])
            values = dict(survival=alive.astype(float), loss=(~alive).astype(float),
                P_up_absolute=alive.astype(float), P_n0_absolute=p0[2], P_target_absolute=p0[2],
                P_3d_ground_absolute=p0.prod(axis=0), P_target_3d_absolute=p0.prod(axis=0),
                photons_spin_pump=np.zeros(count), photons_repump=np.zeros(count),
                photons_lattice_scatter=np.zeros(count), photons_total=np.zeros(count))
            for key, value in values.items():
                means.setdefault(key, []).append(value.mean())
                errors.setdefault(key, []).append(value.std(ddof=1)/np.sqrt(count))
            history.append(np.column_stack((alive, *(v @ e for v, e in zip(p, levels)))))
        result = dict(t_s=times, mean={k: np.array(v) for k, v in means.items()},
                      standard_error={k: np.array(v) for k, v in errors.items()},
                      trajectories=count, observables=record.result())
        nodes.append(dict(energies_hz=[e.tolist() for e in levels], modes=dict(coherent=result)))
        histories.append(np.array(history))
    return dict(nodes=nodes, modes=dict(coherent=average_nodes(nodes, 'coherent'))), histories


def test_shared_distributions_condition_cloud_survival_and_pad_local_spectra():
    raw, histories = recorded_cloud()
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    a = analyze_observables(cfg, raw)
    survival = np.mean([h[:, :, 0].mean(axis=1) for h in histories], axis=0)
    np.testing.assert_allclose(a['series']['P_trapped'], survival)
    for axis, label in enumerate('xyz'):
        np.testing.assert_allclose(a['pn_axes'][label].sum(axis=1), 1.)
        np.testing.assert_allclose(a['pn_axes_absolute'][label].sum(axis=1), survival)
        expected = np.mean([h[:, :, axis+1].mean(axis=1) for h in histories], axis=0)/survival
        np.testing.assert_allclose(a['series'][f'mean_excitation_energy_{label}_hz'], expected)
        assert a['pn_axes'][label].shape[1] == (3, 3, 4)[axis]
    np.testing.assert_allclose(a['spin'].sum(axis=1), 1.)
    np.testing.assert_allclose(a['P_final'].sum(), 1.)
    np.testing.assert_allclose(a['P_final_absolute'].sum(), survival[-1])
    np.testing.assert_allclose(a['series']['P_up'], 1.)
    np.testing.assert_allclose(a['standard_error']['P_up'], 0., atol=1e-8)
    assert not a['final']['raman_active']
    assert 'selection_cooling_power_x_W' not in a['series']


def test_derivative_error_preserves_temporal_covariance():
    raw, histories = recorded_cloud()
    results = [node['modes']['coherent'] for node in raw['nodes']]
    times = results[0]['t_s']
    survival = raw['modes']['coherent']['mean']['survival']
    energy = raw['modes']['coherent']['observables']['mean']['energy_hz'][:, 0]/survival
    expected_variance = 0.
    for h in histories:
        influence = (h[:, :, 1]-energy[:, None]*h[:, :, 0])/survival[:, None]
        direct = np.array([_derivative(times, v, 5) for v in influence.T])
        expected_variance += np.var(direct, axis=0, ddof=1)/len(direct)/len(histories)**2
    np.testing.assert_allclose(derivative_error(times, results, 0, energy, survival, 5),
                               np.sqrt(expected_variance), rtol=2e-12)


def test_temperature_error_matches_numerical_implicit_gradient():
    raw, histories = recorded_cloud()
    cfg = SimConfig.from_json('examples/onlyOP_correlated.json')
    a = analyze_observables(cfg, raw)
    for axis, label in enumerate('xyz'):
        levels = [np.array(n['energies_hz'][axis]) for n in raw['nodes']]
        # Four random means: each site's absolute energy and survival.
        samples = [h[-1][:, [axis+1, 0]] for h in histories]
        means = np.array([s.mean(axis=0) for s in samples])
        fit = lambda x: canonical_mixture_temperature(x[:, 0].sum()/x[:, 1].sum(), levels, x[:, 1])
        assert a['series'][f'T_equiv_{label}_K'][-1] == pytest.approx(fit(means))
        variance = 0.
        for site, sample in enumerate(samples):
            gradient = []
            for col, step in enumerate((.01, 1e-6)):
                left, right = means.copy(), means.copy()
                left[site, col] -= step
                right[site, col] += step
                gradient.append((fit(right)-fit(left))/(2*step))
            gradient = np.array(gradient)
            variance += gradient @ np.cov(sample.T, ddof=1) @ gradient/len(sample)
        assert a['standard_error'][f'T_equiv_{label}_K'][-1] == pytest.approx(np.sqrt(variance), rel=1e-6)


def test_no_survivors_leave_conditional_observables_undefined():
    raw, _ = recorded_cloud(extinct=True)
    a = analyze_observables(SimConfig.from_json('examples/onlyOP_correlated.json'), raw)
    assert a['final']['P_trapped'] == 0.
    for key in ('P_up', 'P_n0_given_up', 'nbar', 'T_equiv_K', 'mean_excitation_energy_J'):
        assert a['final'][key] is None
        assert np.isnan(a['series'][key][-1]) and np.isnan(a['standard_error'][key][-1])
    assert np.isnan(a['P_final']).all()
    np.testing.assert_array_equal(a['P_final_absolute'], 0.)
    assert np.isfinite(a['series']['cooling_power_x_W'][:-1]).all()
    assert np.isnan(a['series']['cooling_power_x_W'][-1])


def test_derivative_does_not_bridge_undefined_survivor_intervals():
    t = np.arange(8.)
    y = np.array([0., 2., 4., np.nan, 100., 103., np.nan, 2.])
    actual = _derivative(t, y, 5)
    np.testing.assert_allclose(actual, [2., 2., 2., np.nan, 3., 3., np.nan, np.nan])


def test_shared_plots_accept_an_extinct_ensemble():
    from matplotlib.figure import Figure
    from rb85rsc import plotting as pl
    raw, _ = recorded_cloud(extinct=True)
    a = analyze_observables(SimConfig.from_json('examples/onlyOP_correlated.json'), raw)
    for draw in (pl.draw_thermal_axes, pl.draw_motional_axes):
        fig = Figure(figsize=(13, 8), layout='constrained')
        draw(fig, a)
        fig.canvas.draw()
    fig = Figure(figsize=(12, 8), layout='constrained')
    for i, plot in enumerate((pl.plot_spin_bars, pl.plot_traces, pl.plot_joint), 1):
        plot(pl._new(fig, 3, 1, i), a)
    # Use an actual renderer so Matplotlib processes all-NaN bars and maps.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
