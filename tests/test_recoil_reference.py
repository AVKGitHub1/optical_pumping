"""Analytic jump-process and escape limits for the motion-coherent benchmark."""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from rb85rsc.config import SimConfig
from rb85rsc.recoil_reference import RecoilChannel, generators, _augment, compare_recoil


def test_coherent_poisson_kicks_match_analytic_rotation_not_population_jumps():
    # N kicks rotate a two-state wavefunction by N*theta; N is Poisson(R*t).
    # Averaging the amplitudes yields an extra cos(R*t*sin(2theta)) factor
    # absent in population-resolving jumps.
    theta, rate, t = .37, 4., .6
    U = expm(1j*theta*np.array([[0., 1.], [1., 0.]]))
    G, Q, _ = generators([0., 0.], [RecoilChannel(np.array([[rate]]), U[None], np.ones(1))])
    rho = (expm(G.toarray()*t) @ np.array([1., 0., 0., 0.])).reshape(2, 2, order='F')
    expected = (1-np.exp(rate*t*(np.cos(2*theta)-1))*np.cos(rate*t*np.sin(2*theta)))/2
    assert rho[1, 1].real == pytest.approx(expected, abs=1e-13)
    p = expm(Q.toarray()*t) @ [1., 0.]
    assert abs(p[1]-expected) > .1
    assert np.trace(rho).real == pytest.approx(1., abs=1e-13)
    assert np.linalg.eigvalsh(rho).min() >= -1e-13


def test_resolved_fast_motion_approaches_population_limit():
    theta, rate, t = .37, 4., .6
    U = expm(1j*theta*np.array([[0., 1.], [1., 0.]]))
    channels = [RecoilChannel(np.array([[rate]]), U[None], np.ones(1))]
    G, Q, _ = generators([0., 10000.], channels)
    rho = (expm(G.toarray()*t) @ [1., 0., 0., 0.]).reshape(2, 2, order='F')
    p = expm(Q.toarray()*t) @ [1., 0.]
    np.testing.assert_allclose(rho.diagonal().real, p, atol=1e-7)


def test_bound_kick_escape_and_photons_match_exact_exponential():
    retained, rate, t = .8, 5., .3
    D = np.eye(2)*np.sqrt(retained)
    G, _, outgoing = generators([0., 1.], [RecoilChannel(np.array([[rate]]), D[None], np.ones(1))])
    trace = np.eye(2).ravel(order='F')
    A = _augment(G, trace, [trace*rate])
    y = expm(A.toarray()*t) @ [1., 0., 0., 0., 0., 0.]
    survival = np.exp(-rate*(1-retained)*t)
    assert (trace @ y[:4]).real == pytest.approx(survival)
    assert y[-2].real == pytest.approx(1-survival)
    assert y[-1].real == pytest.approx((1-survival)/(1-retained))


def test_zero_kick_reduces_to_spin_markov_equation():
    W = np.array([[2., 3.], [7., 1.]])
    G, Q, _ = generators([0., 1000.], [RecoilChannel(W, np.eye(2)[None], np.ones(1))])
    initial = np.array([.8, 0., 0., .2, 0., 0., 0., 0.])
    rho = (expm(G.toarray()*.2) @ initial).reshape(2, 2, 2).swapaxes(-1, -2)
    expected_spin = expm((W.T-np.diag(W.sum(axis=1)))*.2) @ [1., 0.]
    np.testing.assert_allclose(rho, expected_spin[:, None, None]*np.diag([.8, .2]), atol=1e-12)


def test_actual_pumping_reference_conserves_probability_and_refuses_raman():
    root = Path(__file__).resolve().parents[1]
    c = SimConfig.from_json(root/'examples/onlyOP.json')
    c.timing.custom_segments[0]['duration_ms'] = .02
    c.timing.n_samples = 5
    a = compare_recoil(c)
    assert a['coherent']['max_conservation_error'] < 1e-11
    assert a['population']['max_conservation_error'] < 1e-11
    assert a['coherent']['min_eigenvalue'] > -1e-10
    assert a['coherent']['motional_coherence_l1'][-1] > 1e-5
    assert a['coherent']['well_escape_no_recapture'][-1] > 0
    diagnostics = {v['category']: v for v in a['pumping_validity']}
    assert diagnostics['weak excitation']['status'] == 'ok'
    assert diagnostics['motional secular']['applies_to'] == 'population'
    c.timing.custom_segments[0]['raman'] = 1.
    with pytest.raises(ValueError, match='Raman off'):
        compare_recoil(c)
