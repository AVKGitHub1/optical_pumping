"""Setup caches must preserve exact calculations and all input dependencies."""
import numpy as np
import pytest

from rb85rsc import atomic as at, motion as mo
from rb85rsc.far_detuned import DLineResponse


@pytest.mark.parametrize("order", [8, 16, 24, 48])
def test_quadrature_cache_matches_numpy_and_is_read_only(order):
    expected = np.polynomial.legendre.leggauss(order)
    actual = mo.gauss_legendre(order)
    assert mo.gauss_legendre(order) is actual
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
        with pytest.raises(ValueError):
            got[0] = 0


@pytest.fixture
def response():
    return DLineResponse(at.load_atomic_data(), 1., [1., 0., 0.])


@pytest.mark.parametrize("method", ["absorption", "shift_hz", "scattering_amplitudes"])
def test_atomic_response_cache_matches_uncached_and_protects_results(response, method, monkeypatch):
    eps = np.array([0., 1., 1j]) / np.sqrt(2.)
    args = (eps,) if method == "absorption" else (at.C_LIGHT / 783e-9, eps)
    uncached = getattr(response, "_" + method + "_uncached")
    expected = uncached(*args)
    cached = getattr(response, method)
    first = cached(*args)
    np.testing.assert_array_equal(first, expected)
    first[:] = 0  # Public callers retain a writable array without poisoning reuse.
    monkeypatch.setattr(response, "_" + method + "_uncached",
                        lambda *args: pytest.fail("identical response should be reused"))
    np.testing.assert_array_equal(cached(*args), expected)


@pytest.mark.parametrize("method", ["absorption", "shift_hz", "scattering_amplitudes"])
def test_atomic_response_cache_tracks_exact_mutable_inputs(response, method, monkeypatch):
    eps = np.array([0., 1., 1j]) / np.sqrt(2.)
    frequency = at.C_LIGHT / 783e-9
    uncached_name = "_" + method + "_uncached"
    uncached = getattr(response, uncached_name)
    calls = []

    def compute(*args):
        calls.append(1)
        return uncached(*args)

    monkeypatch.setattr(response, uncached_name, compute)

    def check():
        args = (eps,) if method == "absorption" else (frequency, eps)
        before = len(calls)
        np.testing.assert_array_equal(getattr(response, method)(*args), uncached(*args))
        assert len(calls) == before + 1

    check()
    eps[1] = np.nextafter(eps[1].real, np.inf)  # No rounding of polarization keys.
    check()
    response.bdir[:] = [0., 0., 1.]
    check()
    response.d *= 1.01
    check()
    if method != "absorption":
        frequency = np.nextafter(frequency, np.inf)  # Adjacent floats remain distinct.
        check()
        response.ee += 1e6
        check()
        response.eg += np.arange(12) * 100.
        check()
        response.eunit *= 1.01
        check()


def test_response_cache_is_bounded_and_instance_local(response):
    eps = np.array([0., 1., 0.])
    for i in range(140):
        response.shift_hz(at.C_LIGHT / 783e-9 + i, eps)
    assert len(response._response_cache) == 128
    other = DLineResponse(at.load_atomic_data(), 2., [1., 0., 0.])
    assert not other._response_cache
    frequency = at.C_LIGHT / 783e-9
    np.testing.assert_array_equal(other.shift_hz(frequency, eps), other._shift_hz_uncached(frequency, eps))
    assert np.any(other.shift_hz(frequency, eps) != response.shift_hz(frequency, eps))
