"""Optical polarization bookkeeping relative to the bias-field (quantization) axis.

Two input modes, never combined:

* ``spherical``: the user gives *intensity* fractions of the spherical components
  q = +1 (sigma+), 0 (pi), -1 (sigma-) of the field in the bias-field frame.
  Field amplitudes are sqrt(fraction) * E0.  Whether a single transverse plane
  wave along the configured beam direction can realise these fractions is
  checked and reported; if not, the input is an *effective illumination model*.
* ``geometry``: the user gives the beam direction relative to B (polar angle,
  azimuth) and a transverse Jones polarization (ellipticity angle chi,
  ellipse orientation psi).  The fractions are derived and cannot be overridden.
"""
from __future__ import annotations

import numpy as np

# Spherical unit vectors in the field frame (z along B).
E_PLUS = -np.array([1.0, 1.0j, 0.0]) / np.sqrt(2.0)
E_ZERO = np.array([0.0, 0.0, 1.0], dtype=complex)
E_MINUS = np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0)
SPHERICAL = {+1: E_PLUS, 0: E_ZERO, -1: E_MINUS}


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction vector")
    return v / n


def field_frame(b_dir) -> np.ndarray:
    """Rows are (x_B, y_B, z_B) lab-frame unit vectors of the field frame."""
    z = unit(b_dir)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(ref, z)) if abs(z[2]) < 0.9 else unit(ref - z * np.dot(ref, z))
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def check_fractions(sigma_plus: float, pi: float, sigma_minus: float, tol: float = 1e-12) -> None:
    for name, v in (("sigma_plus", sigma_plus), ("pi", pi), ("sigma_minus", sigma_minus)):
        if not np.isfinite(v) or v < -tol or v > 1 + tol:
            raise ValueError(f"{name} fraction {v} outside [0, 1]")
    s = sigma_plus + pi + sigma_minus
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"polarization fractions sum to {s}, not 1")


def fractions_from_impurity(total_impurity: float, pi_share: float) -> tuple[float, float]:
    """Linked control -> (pi_fraction, sigma_minus_fraction)."""
    if not (0.0 <= total_impurity <= 1.0):
        raise ValueError("total_impurity must be in [0, 1]")
    if not (0.0 <= pi_share <= 1.0):
        raise ValueError("pi_share_of_impurity must be in [0, 1]")
    return total_impurity * pi_share, total_impurity * (1.0 - pi_share)


def impurity_from_fractions(pi: float, sigma_minus: float) -> tuple[float, float]:
    """(pi, sigma_minus) -> (total_impurity, pi_share); pi_share is 0.5 when undefined."""
    tot = pi + sigma_minus
    return tot, (pi / tot if tot > 0 else 0.5)


def jones_field_vector(k_dir_lab, chi_deg: float, psi_deg: float, b_dir) -> np.ndarray:
    """Complex unit field vector (lab frame) of a transverse plane wave.

    The transverse basis (e1, e2) satisfies e1 x e2 = k.  e1 is the projection of
    the field axis direction onto the transverse plane when that is defined
    (so psi=0 means "polarization ellipse major axis in the plane containing k and B"),
    otherwise the field-frame x axis.  chi = +45 deg is positive helicity
    (angular momentum +hbar along k); chi = 0 is linear polarization at angle psi.
    """
    k = unit(k_dir_lab)
    zb = unit(b_dir)
    t = zb - k * np.dot(zb, k)
    if np.linalg.norm(t) < 1e-9:
        t = field_frame(b_dir)[0] - k * np.dot(field_frame(b_dir)[0], k)
    e1 = unit(t)
    e2 = np.cross(k, e1)
    chi, psi = np.radians(chi_deg), np.radians(psi_deg)
    a1, a2 = np.cos(chi), 1j * np.sin(chi)  # ellipse in its own axes
    u1 = np.cos(psi) * e1 + np.sin(psi) * e2
    u2 = -np.sin(psi) * e1 + np.cos(psi) * e2
    eps = a1 * u1 + a2 * u2
    return eps / np.linalg.norm(eps)


def spherical_fractions(eps_lab: np.ndarray, b_dir) -> dict:
    """Intensity fractions |e_q^* . eps|^2 of a lab-frame field vector."""
    R = field_frame(b_dir)
    eps_b = R @ eps_lab  # components in field frame
    out = {q: float(abs(np.vdot(SPHERICAL[q], eps_b)) ** 2) for q in (+1, 0, -1)}
    s = sum(out.values())
    return {q: v / s for q, v in out.items()}


def beam_direction_from_angles(theta_deg: float, phi_deg: float, b_dir) -> np.ndarray:
    """Lab-frame beam direction at polar angle theta from B, azimuth phi in field frame."""
    R = field_frame(b_dir)
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    kb = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return R.T @ kb


def realizable_along(fracs: dict, k_dir_lab, b_dir, tol: float = 1e-9) -> bool:
    """Can a transverse plane wave along k produce these spherical intensity fractions?

    With free relative phases, sum_q sqrt(f_q) e^{i phi_q} (k . e_q) = 0 has a
    solution iff the three magnitudes satisfy the triangle inequality.
    """
    R = field_frame(b_dir)
    kb = R @ unit(k_dir_lab)
    mags = sorted(np.sqrt(max(fracs[q], 0.0)) * abs(np.dot(kb, SPHERICAL[q])) for q in (+1, 0, -1))
    return mags[2] <= mags[0] + mags[1] + tol


def describe(fracs: dict) -> str:
    imp = fracs[0] + fracs[-1]
    return (
        f"sigma+={fracs[1]:.6g}, pi={fracs[0]:.6g}, sigma-={fracs[-1]:.6g} "
        f"(total impurity {100 * imp:.4g}%)"
    )
