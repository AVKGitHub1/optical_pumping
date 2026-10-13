"""Three-axis thermometry, survivor conditioning, and thermal result exports."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rb85rsc import atomic as at, far_detuned as fd, motion as mo
from rb85rsc.config import SimConfig
from rb85rsc.ensemble import axis_thermal_data, mixture_temperature
from rb85rsc.runner import export, run

ROOT = Path(__file__).resolve().parents[1]


def component(spectra, marginals):
    sites = [SimpleNamespace(N=len(e), energies_hz=np.asarray(e, dtype=float)) for e in spectra]
    survival = np.asarray(marginals[0]).sum(axis=1)
    return SimpleNamespace(sites=sites), dict(P=survival[:, None, None], pn_axes=marginals)


def test_temperature_mixture_uses_current_survivors_and_local_spectra():
    ts = np.linspace(0., .001, 5)
    temperatures = np.array([2., 4., 6., 8., 10.]) * 1e-6
    spectra = (([0., 13000., 21000.], [0., 8000.], [0., 27000., 43000., 50000.]),
               ([0., 19000.], [0., 6000., 10000.], [0., 24000., 38000.]))
    weights = (np.array([1., .9, .7, .5, .3]), np.array([1., .6, .3, .15, .05]))
    parts = []
    energy_absolute = np.zeros((len(ts), 3))
    for node, (levels, survival) in enumerate(zip(spectra, weights)):
        marginals = []
        shifted = []
        for axis, e in enumerate(levels):
            # Arbitrary site-dependent ground energies must not enter excitation energy.
            shifted.append(np.asarray(e) + 7000. * (node + axis))
            p = np.array([mo.boltzmann_pn(np.asarray(e), t) for t in temperatures]) * survival[:, None]
            marginals.append(p)
            energy_absolute[:, axis] += (p @ e) / 2
        parts.append(component(shifted, marginals))
    survival = sum(weights) / 2
    energy = energy_absolute / survival[:, None]
    series, pn, absolute = axis_thermal_data(ts, parts, survival, energy, np.zeros_like(energy), 5)
    for axis, name in enumerate('xyz'):
        np.testing.assert_allclose(series[f'T_equiv_{name}_K'], temperatures, rtol=1e-9)
        np.testing.assert_allclose(absolute[name].sum(axis=1), survival)
        np.testing.assert_allclose(pn[name].sum(axis=1), 1.)
        nbar = sum(np.asarray(r['pn_axes'][axis]) @ np.arange(l.sites[axis].N)
                   for l, r in parts) / 2 / survival
        np.testing.assert_allclose(series[f'nbar_{name}'], nbar)
        np.testing.assert_allclose(series[f'P_n0_{name}'], pn[name][:, 0])
        np.testing.assert_allclose(series[f'P_n0_{name}_absolute'], absolute[name][:, 0])
        np.testing.assert_allclose(series[f'mean_excitation_energy_{name}_J'], at.H * energy[:, axis])
        assert pn[name].shape[1] == max(len(levels[axis]) for levels in spectra)


def test_undefined_temperatures_and_no_survivors_are_not_zero_temperature():
    ts = np.array([0., .001, .002])
    marginals = [np.array([[.2, .8], [.5, .5], [0., 0.]])] * 3
    parts = [component([[0., 10000.]] * 3, marginals)]
    energy = np.array([[8000.] * 3, [5000.] * 3, [0.] * 3])
    series, pn, absolute = axis_thermal_data(ts, parts, np.array([1., 1., 0.]), energy, energy * 0, 5)
    for axis in 'xyz':
        assert np.isnan(series[f'T_equiv_{axis}_K'][0])  # Above positive-T canonical range.
        assert np.isinf(series[f'T_equiv_{axis}_K'][1])  # Infinite-T endpoint.
        assert np.isnan(series[f'T_equiv_{axis}_K'][2])
        assert np.isnan(pn[axis][2]).all()
        assert np.isnan(series[f'nbar_{axis}'][2])
        assert np.isnan(series[f'mean_excitation_energy_{axis}_J'][2])
        assert absolute[axis][2].sum() == 0.
    ground_only = [component([[1200.]] * 3, [np.ones((3, 1))] * 3)]
    assert np.isnan(mixture_temperature(0., ground_only, 0))  # One level cannot define T.
    assert mixture_temperature(0., parts, 0, 0) == 0.


def test_pure_loss_cooling_has_no_retained_state_cooling():
    # Independent two-level wells with only hot-state escape: no Raman or recoil.
    ts = np.linspace(0., .001, 201)
    gaps = np.array([14000., 19000., 31000.])
    loss_rates = np.array([70., 150., 210.])
    unnormalized = np.array([np.column_stack((np.full(len(ts), .7), .3 * np.exp(-r * ts)))
                             for r in loss_rates])
    axis_survival = unnormalized.sum(axis=2)
    survival = axis_survival.prod(axis=0)
    marginals = [p * (survival / norm)[:, None] for p, norm in zip(unnormalized, axis_survival)]
    hot_fraction = (unnormalized[:, :, 1] / axis_survival).T
    energy = hot_fraction * gaps
    # Exact loss covariance: cooling = h * gap * gamma * p_hot * (1 - p_hot).
    selection = at.H * gaps * loss_rates * hot_fraction * (1 - hot_fraction)
    parts = [component([[0., gap] for gap in gaps], marginals)]
    series, _, _ = axis_thermal_data(ts, parts, survival, energy, selection, 7)
    for axis in 'xyz':
        net = series[f'cooling_power_{axis}_W']
        assert np.all(net > 0.)
        np.testing.assert_allclose(net, series[f'selection_cooling_power_{axis}_W'], rtol=1e-7, atol=0.)
        assert np.max(np.abs(series[f'bound_cooling_power_{axis}_W'])) < np.max(net) * 1e-7


def test_three_axis_run_and_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'CACHE', tmp_path / 'cache')
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    cfg.ensemble.samples = 2
    cfg.ensemble.workers = 1
    cfg.ensemble.calibration_samples = 8
    cfg.timing.custom_segments = [dict(duration_ms=.02, raman=0., pump=0., repump=0.)]
    cfg.timing.n_samples = 5
    cfg.initial.temperature_k = 6e-6
    cfg.ensemble.transverse_temperature_k = 10e-6
    result = run(cfg)
    for index, (axis, temperature) in enumerate(zip('xyz', [10e-6, 10e-6, 6e-6])):
        # The trapping lattices still scatter after t=0, even with the Raman/pumps off.
        assert result['series'][f'T_equiv_{axis}_K'][0] == pytest.approx(temperature, rel=1e-9)
        np.testing.assert_allclose(result['pn_axes'][axis].sum(axis=1), 1.)
        energy_abs = sum(np.asarray(r['pn_axes'][index]) @ (l.sites[index].energies_hz - l.sites[index].energies_hz[0])
                         for l, r in result['ensemble_nodes']) / cfg.ensemble.samples
        np.testing.assert_allclose(result['series'][f'mean_excitation_energy_{axis}_hz'],
                                   energy_abs / result['series']['P_trapped'])
    driven = result['final']['raman_axis']
    np.testing.assert_allclose(result['pn_axes'][driven], result['pn'])
    np.testing.assert_allclose(result['series'][f'nbar_{driven}'], result['series']['nbar'])
    export(result, tmp_path / 'export')
    saved = np.load(tmp_path / 'export/run_ensemble.npz')
    np.testing.assert_array_equal(saved['t_s'], result['series']['t_s'])
    for axis in 'xyz':
        np.testing.assert_array_equal(saved[f'pn_{axis}'], result['pn_axes'][axis])
        np.testing.assert_array_equal(saved[f'pn_{axis}_absolute'], result['pn_axes_absolute'][axis])
        np.testing.assert_allclose(saved[f'pn_{axis}_absolute'].sum(axis=1), saved['survival'])
    columns = (tmp_path / 'export/run_timeseries.csv').read_text().splitlines()[0].split(',')
    assert all(f'T_equiv_{axis}_K' in columns and f'cooling_power_{axis}_W' in columns for axis in 'xyz')
    for name in ('thermal_axes', 'motional_axes'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / f'export/run_{name}.{extension}').stat().st_size > 1000
