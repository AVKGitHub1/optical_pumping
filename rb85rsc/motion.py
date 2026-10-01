"""Harmonic-oscillator motion: displacement matrix elements, recoil kernels, thermal states."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

from .atomic import HBAR, H, K_B


@lru_cache(maxsize=32, typed=True)
def gauss_legendre(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only quadrature nodes and weights, cached by the exact order."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def x0_m(mass_kg: float, trap_hz: float) -> float:
    """Ground-state extent x0 = sqrt(hbar / (2 m omega_t))."""
    return float(np.sqrt(HBAR / (2.0 * mass_kg * 2.0 * np.pi * trap_hz)))


def lattice_frequency_hz(mass_kg: float, wavelength_nm: float, depth_uk: float) -> float:
    """Bottom frequency of V0 sin²(kz), using the optical wavelength."""
    recoil_hz = H / (2 * mass_kg * (wavelength_nm * 1e-9) ** 2)
    return float(2 * np.sqrt(depth_uk * 1e-6 * K_B / H * recoil_hz))


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
    c, w = gauss_legendre(n_quad)
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
    c, w = gauss_legendre(16)
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


def boltzmann_pn(energies_hz: np.ndarray, temp_k: float) -> np.ndarray:
    """Canonical distribution conditional on the supplied discrete levels."""
    e = np.asarray(energies_hz, float) - np.min(energies_hz)
    if temp_k <= 0:
        p = (e == 0).astype(float)
    else:
        p = np.exp(-e * H / (K_B * temp_k))
    return p / p.sum()


def temperature_from_energy(mean_hz: float, energies_hz: np.ndarray) -> float:
    """Positive canonical temperature matching energy above the lowest level.

    Uses the finite, actual spectrum (essential for a bound lattice basis).
    Above the infinite-temperature mean there is no positive-temperature fit;
    return NaN instead of assigning a misleading harmonic temperature.
    """
    from scipy.optimize import brentq

    e = np.asarray(energies_hz, float) - np.min(energies_hz)
    if mean_hz <= 0:
        return 0.0
    if mean_hz > e.mean() + 1e-12 * e[-1]:
        return np.nan
    if mean_hz >= e.mean() - 1e-12 * e[-1]:
        return np.inf
    scaled = e / e[-1]
    target = mean_hz / e[-1]

    def residual(beta):
        w = np.exp(-beta * scaled)
        return float(w @ scaled / w.sum() - target)

    upper = 1.0
    while residual(upper) > 0:
        upper *= 2
    beta = brentq(residual, 0.0, upper, xtol=1e-13)
    return float(H * e[-1] / (K_B * beta))


def annihilation(n_states: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n_states)), 1)


# ---------------------------------------------------------------------------
# Motional bases: harmonic oscillator (default) or one site of a 1D lattice.
# Both expose N, energies_hz, displacement(eta) and recoil_kernel(...), with
# eta = kappa * x0 (x0 of the harmonic frequency nu_t) as the common argument.
# ---------------------------------------------------------------------------
class HarmonicBasis:
    """Harmonic levels n = 0..N-1 with E_n = n h nu_t (infinitely deep); closed-form elements."""

    kind = "harmonic"

    def __init__(self, n_states: int, trap_hz: float):
        self.N = n_states
        self.n_bound = None
        self.energies_hz = trap_hz * np.arange(n_states)

    def displacement(self, eta: float) -> np.ndarray:
        return displacement_matrix(eta, self.N)

    def recoil_kernel(self, eta_photon, abs_projection, cos_beta, pattern, n_quad):
        return recoil_kernel(eta_photon, abs_projection, cos_beta, pattern, self.N, n_quad)


class LatticeSite:
    """Bound states of one site of a 1D lattice V(z) = V0 sin^2(k_L z).

    The lattice is fixed by its depth V0 and the harmonic frequency at the bottom
    of a site, h nu_t = 2 sqrt(V0 E_r), so E_r = (h nu_t)^2 / (4 V0) and
    k_L = sqrt(2 m E_r) / hbar.  Isolated-site approximation: inside |z| <= a/2
    (a = pi / k_L) the potential has the exact lattice shape, outside it is held
    at V0, so neighbouring wells and tunnelling are neglected (accurate for bands
    well below the barrier).  The Schrodinger equation is solved by finite
    differences; states with E < V0 are bound.  Energies are measured from the
    bottom of the well.

    Displacements exp(i kappa z) are overlaps on the grid.  The grid eigenbasis is
    complete, so 1 - sum_{bound n'} |<n'|exp(i kappa z)|n>|^2 is the probability
    of being promoted above V0 (the atom leaves its site), counted as loss.
    """

    kind = "lattice"

    def __init__(self, mass_kg: float, trap_hz: float, depth_hz: float, n_cap: int):
        from scipy.linalg import eigh_tridiagonal

        self.trap_hz, self.depth_hz = trap_hz, depth_hz
        self.recoil_hz = trap_hz**2 / (4.0 * depth_hz)
        self.k_lattice = np.sqrt(2.0 * mass_kg * H * self.recoil_hz) / HBAR
        self.wavelength_m = 2.0 * np.pi / self.k_lattice
        self.x0 = x0_m(mass_kg, trap_hz)
        a = np.pi / self.k_lattice  # site spacing (lambda_L / 2)
        dz = 0.05 * self.x0
        half = 2.5 * a  # site plus two spacings of flat V0 on each side before the hard wall
        m = int(np.ceil(half / dz))
        self.z = z = dz * np.arange(-m, m + 1)
        V = np.where(np.abs(z) <= a / 2, depth_hz * np.sin(self.k_lattice * z) ** 2, depth_hz)
        t = HBAR**2 / (2.0 * mass_kg * dz**2) / H  # finite-difference kinetic term, Hz
        E, vec = eigh_tridiagonal(2.0 * t + V, np.full(z.size - 1, -t), select="v", select_range=(-np.inf, depth_hz))
        for j in range(vec.shape[1]):  # sign convention of the oscillator states: outer lobe at z > 0 positive
            pos = vec[z >= 0, j]
            if pos[np.argmax(np.abs(pos))] < 0:
                vec[:, j] = -vec[:, j]
        self.n_bound = int(E.size)
        self.N = min(self.n_bound, int(n_cap))
        self.energies_hz = E[: self.N]
        self.all_bound_energies_hz = E
        self._vec = vec[:, : self.N]  # unit-norm columns (sum |v|^2 = 1)
        self._cache: dict = {}

    def tunneling_hz(self) -> float:
        """Lowest-band tunnelling J/h, deep-lattice formula (4/sqrt(pi)) E_r s^(3/4) exp(-2 sqrt(s))."""
        s = self.depth_hz / self.recoil_hz
        return float(4.0 / np.sqrt(np.pi) * self.recoil_hz * s**0.75 * np.exp(-2.0 * np.sqrt(s)))

    def displacement(self, eta: float) -> np.ndarray:
        key = ("D", float(eta))
        if key not in self._cache:
            v = self._vec
            self._cache[key] = v.T @ (np.exp(1j * (eta / self.x0) * self.z)[:, None] * v)
        return self._cache[key]

    def recoil_kernel(self, eta_photon, abs_projection, cos_beta, pattern, n_quad):
        key = ("K", float(eta_photon), float(abs_projection), float(cos_beta), pattern, int(n_quad))
        if key not in self._cache:
            c, w = gauss_legendre(n_quad)
            wt = w * emission_weight(c, cos_beta, pattern)
            K = np.zeros((self.N, self.N))
            for ci, wi in zip(c, wt):
                d = self.displacement(eta_photon * (abs_projection - ci))
                K += wi * (d * d.conj()).real
            ov = 1.0 - K.sum(axis=0)
            ov[ov < 0] = 0.0
            K.setflags(write=False)
            ov.setflags(write=False)
            self._cache[key] = (K, ov)
        return self._cache[key]


@lru_cache(maxsize=16)
def lattice_site(mass_kg: float, trap_hz: float, depth_hz: float, n_cap: int) -> LatticeSite:
    return LatticeSite(mass_kg, trap_hz, depth_hz, n_cap)
