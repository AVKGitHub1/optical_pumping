"""Harmonic-oscillator motion: displacement matrix elements, recoil kernels, thermal states."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

from .atomic import HBAR, H, K_B


def x0_m(mass_kg: float, trap_hz: float) -> float:
    """Ground-state extent x0 = sqrt(hbar / (2 m omega_t))."""
    return float(np.sqrt(HBAR / (2.0 * mass_kg * 2.0 * np.pi * trap_hz)))


def lamb_dicke(wavelength_m: float, mass_kg: float, trap_hz: float, projection: float = 1.0) -> float:
    return float(abs(2.0 * np.pi / wavelength_m * projection) * x0_m(mass_kg, trap_hz))


def displacement_matrix(eta: float, n_rows: int, n_cols: int | None = None) -> np.ndarray:
    """Exact <n| exp(i eta (a + a^dagger)) |m> for n < n_rows, m < n_cols.

    <n|D(i eta)|m> = e^{-eta^2/2} sqrt(n_<! / n_>!) (i eta)^{|n-m|} L_{n_<}^{|n-m|}(eta^2)
    """
    n_cols = n_rows if n_cols is None else n_cols
    n = np.arange(n_rows)[:, None]
    m = np.arange(n_cols)[None, :]
    if eta == 0.0:
        return np.eye(n_rows, n_cols, dtype=complex)
    lo = np.minimum(n, m)
    d = np.abs(n - m)
    x = eta * eta
    logmag = 0.5 * (gammaln(lo + 1) - gammaln(lo + d + 1)) - 0.5 * x + d * np.log(abs(eta))
    lag = eval_genlaguerre(lo, d, x)
    phase = (1j * np.sign(eta)) ** d
    return np.exp(logmag) * lag * phase


def displacement_prob(eta: float, n_rows: int, n_cols: int) -> np.ndarray:
    """|<n'|D(i eta)|n>|^2 (depends only on |eta|)."""
    a = displacement_matrix(abs(eta), n_rows, n_cols)
    return (a * a.conj()).real


def emission_weight(c: np.ndarray, cos_beta: float, pattern: str) -> np.ndarray:
    """Probability density of c = s.u (emission direction projected on the trap axis).

    Integrated analytically over the azimuth around the trap axis for dipole
    patterns referenced to the field axis, where beta is the trap-axis/field angle.
    sigma: (3/16pi)(1+cos^2 theta_B);  pi: (3/8pi) sin^2 theta_B.
    """
    cb2 = cos_beta**2
    sb2 = 1.0 - cb2
    mean_cos2_theta_b = c**2 * cb2 + 0.5 * (1.0 - c**2) * sb2
    if pattern == "sigma":
        return 3.0 / 8.0 * (1.0 + mean_cos2_theta_b)
    if pattern == "pi":
        return 3.0 / 4.0 * (1.0 - mean_cos2_theta_b)
    if pattern == "isotropic":
        return 0.5 * np.ones_like(c)
    raise ValueError(pattern)


@lru_cache(maxsize=256)
def recoil_kernel(
    eta_photon: float,
    abs_projection: float,
    cos_beta: float,
    pattern: str,
    n_states: int,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Motional redistribution K[n', n] for one absorption + one spontaneous emission.

    The absorbed photon (wavevector projection eta_photon*abs_projection on the
    trap axis) and the emitted photon act through a single displacement
    exp(i x0 (k_abs - k_emit).u (a + a^+)); the excited state lives 26 ns, far
    shorter than the trap period, so the amplitudes are combined before squaring.
    Emission directions are integrated by Gauss-Legendre quadrature in c = s.u.

    Returns (K, overflow) with K restricted to n' < n_states and
    overflow[n] = 1 - sum_n' K[n', n] (probability pushed above n_max).
    The arrays are marked read-only because they are cached.
    """
    c, w = np.polynomial.legendre.leggauss(n_quad)
    wt = w * emission_weight(c, cos_beta, pattern)
    K = np.zeros((n_states, n_states))
    for ci, wi in zip(c, wt):
        K += wi * displacement_prob(eta_photon * (abs_projection - ci), n_states, n_states)
    ov = 1.0 - K.sum(axis=0)
    ov[ov < 0] = 0.0  # rounding at the 1e-16 level only
    K.setflags(write=False)
    ov.setflags(write=False)
    return K, ov


def mean_c2(cos_beta: float, pattern: str) -> float:
    """<c^2> for the emission pattern: per-photon emission recoil is eta^2 <c^2>."""
    c, w = np.polynomial.legendre.leggauss(16)
    return float(np.sum(w * c**2 * emission_weight(c, cos_beta, pattern)))


def thermal_pn(nbar: float, n_states: int) -> tuple[np.ndarray, float]:
    """Thermal p_n = nbar^n / (1+nbar)^(n+1), normalized; returns (p, discarded tail)."""
    if nbar < 0:
        raise ValueError("nbar must be >= 0")
    n = np.arange(n_states)
    if nbar == 0:
        p = np.zeros(n_states)
        p[0] = 1.0
        return p, 0.0
    r = nbar / (1.0 + nbar)
    p = (1.0 - r) * r**n
    tail = r**n_states
    return p / p.sum(), float(tail)


def nbar_from_temperature(temp_k: float, trap_hz: float) -> float:
    if temp_k <= 0:
        return 0.0
    return float(1.0 / np.expm1(H * trap_hz / (K_B * temp_k)))


def temperature_from_nbar(nbar: float, trap_hz: float) -> float:
    """Thermal-equivalent temperature; a proxy if p(n) is not thermal."""
    if nbar <= 0:
        return 0.0
    return float(H * trap_hz / (K_B * np.log1p(1.0 / nbar)))


def annihilation(n_states: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n_states)), 1)
