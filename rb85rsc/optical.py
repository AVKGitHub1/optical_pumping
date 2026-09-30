"""Weak-excitation optical pumping on D2 with adiabatically eliminated 5P3/2.

Convention: H_ge / hbar = Omega_ge / 2, Delta_ge = omega_L - omega_ge (rad/s),
Gamma = 1/tau (s^-1).  The eliminated excited amplitude driven from |g> is

    c_ge = (Omega_ge / 2) / (Delta_ge + i Gamma / 2),

so the isolated excitation (= scattering) rate is Gamma |c_ge|^2 =
Gamma |Omega|^2 / (Gamma^2 + 4 Delta^2) and the ground light shift is
|Omega|^2 Delta / (Gamma^2 + 4 Delta^2).

The two pumping beams are treated as mutually incoherent (no optical coherence
between their frequencies, so two-frequency CPT/EIT is excluded).  Ground-state
coherences created by a single beam are neglected (secular in the Zeeman splitting;
checked in diagnostics).  Each ground state therefore scatters independently.

path_model="independent" sums Gamma |c_ge|^2 b(e->g') over excited sublevels.
path_model="kramers_heisenberg" sums, for each absorbed polarization component,
the amplitudes through different F' sharing the same m' before squaring.  The two
agree on total scattering rates (sum rule) and differ only in the final-state
distribution when several F' contribute comparably.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import polarization as pol
from .atomic import (
    C_LIGHT,
    EPS0,
    EXCITED_STATES,
    GROUND_STATES,
    HBAR,
    MU_B,
    H,
    N_EXCITED,
    N_GROUND,
    AtomicData,
)

REFERENCE_TRANSITION = {"spin_pump": (3, 3), "repump": (2, 3)}  # (F, F') named zero-field lines
EMISSION_CLASSES = ("sigma", "pi")


@dataclass
class BeamSpec:
    name: str
    intensity_w_m2: float
    fractions: dict  # q -> intensity fraction
    laser_hz: float  # absolute optical frequency
    direction: np.ndarray  # lab unit vector
    fraction_source: str  # "spherical (realizable)", "effective illumination model", "geometry"


@dataclass
class BeamRates:
    """Rates at the nominal beam intensity (they scale linearly with intensity)."""

    W: np.ndarray  # (12, 12, 2) rate g -> g' with emitted class (sigma, pi), s^-1
    gamma_out: np.ndarray  # (12,) total scattering rate out of each g
    p_exc: np.ndarray  # (12,) steady excited fraction driven from g
    light_shift: np.ndarray  # (12,) ground-state shift, rad/s
    omega_max: float  # largest single Rabi frequency, rad/s
    detail: dict


def peak_intensity(cfg_pump) -> float:
    if cfg_pump.intensity_mode == "power_waist":
        return 2.0 * cfg_pump.power_w / (np.pi * cfg_pump.waist_m**2)
    return cfg_pump.peak_intensity_w_m2


def beam_spec(name: str, cfg_pump, atom: AtomicData, b_dir) -> BeamSpec:
    pc = cfg_pump.polarization
    if pc.mode == "geometry":
        k = pol.beam_direction_from_angles(pc.beam_angle_deg, pc.beam_azimuth_deg, b_dir)
        eps = pol.jones_field_vector(k, pc.ellipticity_deg, pc.orientation_deg, b_dir)
        fr = pol.spherical_fractions(eps, b_dir)
        src = "geometry (derived from transverse Jones vector)"
    else:
        k = pol.unit(cfg_pump.direction)
        fr = {0: pc.pi_fraction, -1: pc.sigma_minus_fraction}
        fr[1] = 1.0 - fr[0] - fr[-1]
        if abs(fr[1]) < 1e-15:
            fr[1] = 0.0
        pol.check_fractions(fr[1], fr[0], fr[-1])
        src = (
            "spherical (realizable by a transverse wave along the configured direction)"
            if pol.realizable_along(fr, k, b_dir)
            else "EFFECTIVE ILLUMINATION MODEL (not realizable by one transverse wave along the configured direction)"
        )
    f, fp = REFERENCE_TRANSITION[name]
    laser = atom.transition_hz(f, fp) + cfg_pump.detuning_mhz * 1e6
    inten = peak_intensity(cfg_pump) if cfg_pump.enabled else 0.0
    return BeamSpec(name, inten, fr, laser, k, src)


def beam_rates(
    atom: AtomicData,
    beam: BeamSpec,
    b_tesla: float,
    excited_shift_hz: float = 0.0,
    excited_manifold: str = "all_D2",
    path_model: str = "independent",
) -> BeamRates:
    gam = atom.gamma
    e0 = np.sqrt(2.0 * beam.intensity_w_m2 / (C_LIGHT * EPS0))
    d = atom.dipole_si  # (24, 12)
    eg = np.array([atom.ground_hfs_hz[f] + atom.g_ground[f] * m * MU_B * b_tesla / H for f, m in GROUND_STATES])
    ee = np.array(
        [atom.d2_centroid_hz + atom.excited_hfs_hz[f] + atom.g_excited[f] * m * MU_B * b_tesla / H for f, m in EXCITED_STATES]
    ) + excited_shift_hz
    mg = np.array([m for _, m in GROUND_STATES])
    me = np.array([m for _, m in EXCITED_STATES])
    fe = np.array([f for f, _ in EXCITED_STATES])
    q = me[:, None] - mg[None, :]
    frac = np.zeros_like(q, dtype=float)
    for qq in (-1, 0, 1):
        frac[q == qq] = beam.fractions[qq]
    allowed = np.abs(q) <= 1
    if excited_manifold == "F3_only_IDEALIZED":
        allowed &= (fe == 3)[:, None]
    omega = np.where(allowed, e0 * np.sqrt(frac) * d / HBAR, 0.0)  # rad/s, sqrt of intensity fraction
    delta = 2.0 * np.pi * (beam.laser_hz - (ee[:, None] - eg[None, :]))  # rad/s
    c = (omega / 2.0) / (delta + 0.5j * gam)
    pe = np.abs(c) ** 2  # (24, 12)
    p_exc = pe.sum(axis=0)
    light_shift = np.sum(omega**2 * delta / (gam**2 + 4 * delta**2), axis=0)
    W = np.zeros((N_GROUND, N_GROUND, 2))
    br = atom.branching
    D2 = atom.emission_norm()
    if path_model == "independent":
        for ig in range(N_GROUND):
            for ie in np.nonzero(pe[:, ig] > 0)[0]:
                for jg in range(N_GROUND):
                    qe = me[ie] - mg[jg]
                    if abs(qe) <= 1 and br[ie, jg] > 0:
                        W[ig, jg, 0 if abs(qe) == 1 else 1] += gam * pe[ie, ig] * br[ie, jg]
    elif path_model == "kramers_heisenberg":
        M = atom.dipole_au
        for ig in range(N_GROUND):
            for qa in (-1, 0, 1):
                m_e = mg[ig] + qa
                idx = np.nonzero((me == m_e) & (pe[:, ig] > 0))[0]
                if idx.size == 0:
                    continue
                for jg in range(N_GROUND):
                    qe = m_e - mg[jg]
                    if abs(qe) > 1:
                        continue
                    amp = np.sum(c[idx, ig] * M[idx, jg])
                    W[ig, jg, 0 if abs(qe) == 1 else 1] += gam * abs(amp) ** 2 / D2
    else:
        raise ValueError(path_model)
    gamma_out = W.sum(axis=(1, 2))
    detail = {
        "E0_V_per_m": float(e0),
        "intensity_w_m2": float(beam.intensity_w_m2),
        "fractions": {str(k): float(v) for k, v in beam.fractions.items()},
        "laser_hz": float(beam.laser_hz),
        "fraction_source": beam.fraction_source,
    }
    return BeamRates(W, gamma_out, p_exc, light_shift, float(np.max(np.abs(omega))), detail)


def dominant_channels(br: BeamRates, g: int, top: int = 4) -> list[tuple[str, float]]:
    """Largest destinations of scattering from ground state index g."""
    from .atomic import ground_label

    tot = br.W[g].sum(axis=1)
    order = np.argsort(tot)[::-1][:top]
    return [(ground_label(j), float(tot[j])) for j in order if tot[j] > 0]
