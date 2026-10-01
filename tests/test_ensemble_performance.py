"""Independent equivalence checks for prepared recoil propagation."""
import numpy as np
import pytest
from scipy.linalg import expm

from rb85rsc.ensemble import (
    RecoilTerm, _RecoilGain, _UniformizedScattering, axis_apply,
)


@pytest.mark.parametrize('axis', [0, 1, 2, 3])
@pytest.mark.parametrize('complex_values', [False, True])
def test_fixed_axis_contraction_matches_tensor_sum(axis, complex_values):
    rng = np.random.default_rng(291)
    # Noncontiguous inputs, unequal axes, and a rectangular output kernel.
    p = rng.random((12, 4, 3, 2))[:, ::2]
    kernel = rng.random((5, p.shape[axis]))
    if complex_values:
        p = p + 1j * p[::-1]
        kernel = kernel + 1j * rng.random(kernel.shape)
    expected = np.tensordot(kernel, p, axes=(1, axis))
    expected = np.moveaxis(expected, 0, axis)
    np.testing.assert_allclose(axis_apply(kernel, p, axis), expected, rtol=2e-15, atol=2e-15)


def recoil_case():
    rng = np.random.default_rng(548)
    shape = (12, 2, 3, 2)
    kernels = tuple(rng.uniform(.1, .3, (n, n)) for n in shape[1:])
    W = rng.uniform(1., 2., (12, 12))
    # An identical full path, a shared x prefix, and complex interference paths
    # exercise reuse without changing the order of the physical term sum.
    imaginary = kernels[1].astype(complex) + .05j
    changed_x = kernels[0].copy()
    changed_x[0, 0] = np.nextafter(changed_x[0, 0], np.inf)
    definitions = [
        (W, kernels, .37),
        (W[::-1].copy(), tuple(k.copy() for k in kernels), 2.3),
        (.01j * W, (kernels[0], imaginary, kernels[2]), .71),
        (-.01j * W, (kernels[0], imaginary.conj(), kernels[2]), .71),
        (W, (changed_x, kernels[1], kernels[2]), .23),
    ]
    active, dense = [], np.zeros((np.prod(shape), np.prod(shape)))
    for counter, (spin, ks, scale) in enumerate(definitions):
        term = RecoilTerm(spin, ks, np.zeros(shape), counter % 4, 'pump')
        active.append((term, scale))
        matrix = spin.T
        for kernel in ks:
            matrix = np.kron(matrix, kernel)
        dense += scale * matrix.real
    assert dense.min() >= 0
    p = rng.random(shape)
    p /= p.sum()
    return active, dense, p


@pytest.mark.parametrize('population_dtype', [np.float64, np.float32])
def test_shared_recoil_prefixes_preserve_interference_and_nonunit_scales(population_dtype):
    active, dense, p = recoil_case()
    p = p.astype(population_dtype)
    gain = _RecoilGain(active)
    expected = (dense @ p.ravel()).reshape(p.shape)
    np.testing.assert_allclose(gain(p), expected, rtol=3e-15, atol=3e-15)
    # Repeated evaluations must use the new population, not a cached tensor.
    p2 = p[::-1].copy()
    np.testing.assert_allclose(gain(p2), (dense @ p2.ravel()).reshape(p.shape),
                               rtol=3e-15, atol=3e-15)


def test_prepared_scattering_matches_augmented_exponential_with_rewards():
    active, dense, p = recoil_case()
    hazard = np.linspace(.2, .8, p.size).reshape(p.shape)
    gout = dense.sum(axis=0).reshape(p.shape) + hazard
    counts = np.array([gout * .1, gout * .2, gout * .3, gout * .4])
    generator = np.zeros((p.size + 5, p.size + 5))
    generator[:p.size, :p.size] = dense - np.diag(gout.ravel())
    generator[p.size, :p.size] = hazard.ravel()
    generator[p.size + 1:, :p.size] = counts.reshape(4, -1)
    prepared = _UniformizedScattering(active, gout, hazard, counts)
    initial = p.copy()
    for dt in (.03, .07, .03):
        expected = expm(dt * generator) @ np.r_[p.ravel(), np.zeros(5)]
        got, loss, photons = prepared.step(p, dt)
        np.testing.assert_allclose(got.ravel(), expected[:p.size], rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(np.r_[loss.sum(), photons], expected[p.size:],
                                   rtol=2e-13, atol=2e-14)
        assert got.sum() + loss.sum() == pytest.approx(p.sum(), abs=3e-14)
        np.testing.assert_array_equal(p, initial)


def test_prepared_scattering_zero_rate_and_step_limit():
    p = np.ones((12, 1, 1, 1)) / 12
    zero = np.zeros_like(p)
    step = _UniformizedScattering([], zero, zero, np.zeros((4,) + p.shape))
    got, loss, counts = step.step(p, .1)
    np.testing.assert_array_equal(got, p)
    assert not np.shares_memory(got, p)
    assert not loss.any() and not counts.any()
    step = _UniformizedScattering([], zero + 100., zero, np.zeros((4,) + p.shape))
    with pytest.raises(ValueError, match='time_step_us too large'):
        step.step(p, .11)
