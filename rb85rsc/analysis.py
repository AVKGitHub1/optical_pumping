"""Derived observables and diagnostics from the joint populations P(F, mF, n, t)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .atomic import DOWN, GROUND_STATES, K_B, UP, H, ground_label
from .dynamics import RunResult
from .model import BEAMS, COUNTERS, Model, Validity
from .motion import temperature_from_nbar

NOT_REACHED = "not reached"


def _threshold_time(t, y, thr):
    idx = np.nonzero(y >= thr)[0]
    if idx.size == 0:
        return NOT_REACHED
    i = idx[0]
    if i == 0:
        return float(t[0])
    # linear interpolation between samples
    return float(t[i - 1] + (thr - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


def _derivative(t, y, window):
    if len(t) < 5:
        return np.gradient(y, t)
    w = min(window, len(t) - (1 - len(t) % 2))
    w = max(w if w % 2 else w - 1, 5)
    dt = t[1] - t[0]
    return savgol_filter(y, w, 3, deriv=1, delta=dt)


def analyze(model: Model, res: RunResult) -> dict:
    cfg = model.cfg
    t, P_abs = res.t, res.P
    N = model.N
    n = np.arange(N)
    # All populations are normalized to the atoms still in the basis ("trapped"): the survivors of a
    # lattice, or 1 - numerical overflow for a harmonic trap. Fractions of all atoms are kept as *_absolute.
    trace = P_abs.reshape(len(t), -1).sum(axis=1)
    safe = np.where(trace > 0, trace, 1.0)
    P = P_abs / safe[:, None, None]
    spin = P.sum(axis=2)  # (T, 12)
    pn = P.sum(axis=1)  # (T, N)
    f2 = spin[:, [i for i, (f, _) in enumerate(GROUND_STATES) if f == 2]].sum(axis=1)
    f3 = spin[:, [i for i, (f, _) in enumerate(GROUND_STATES) if f == 3]].sum(axis=1)
    p_up = spin[:, UP]
    p_n0 = pn[:, 0]
    p_target = P[:, UP, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_n0_given_up = np.where(p_up > 0, p_target / np.where(p_up > 0, p_up, 1), np.nan)
    nbar = pn @ n
    nbar0 = nbar[0]
    frac_red = 1.0 - nbar / nbar0 if nbar0 > 0 else np.full_like(nbar, np.nan)
    dndt = _derivative(t, nbar, cfg.analysis.derivative_window)
    cooling_rate = -dndt  # quanta/s removed (positive = cooling)
    energy_rate = cooling_rate * H * model.nu_t  # J/s
    temps = np.array([temperature_from_nbar(x, model.nu_t) for x in nbar])
    # thermality of final p(n): total-variation distance from thermal with same nbar
    nb = max(nbar[-1], 0)
    r = nb / (1 + nb) if nb > 0 else 0.0
    th = (1 - r) * r**n if nb > 0 else (n == 0).astype(float)
    tvd = 0.5 * np.abs(pn[-1] / max(pn[-1].sum(), 1e-300) - th / th.sum()).sum()
    counts = res.counts
    total_photons = counts[-1].sum()
    removed = nbar0 - nbar[-1]
    per_photon = removed / total_photons if total_photons > 0 else np.nan
    # target-state (|3,3>) scattering rate for the final segment configuration
    last = model.schedule[-1]
    o = model.ops(last)
    tgt_rate = float(o.gout[UP * N])
    tgt_by = {b: float(model.out[b][UP * N]) * (getattr(last, "pump" if b == "spin_pump" else "repump")) for b in BEAMS}

    cons = np.abs(trace + res.overflow - 1.0)
    boundary = P_abs[:, :, N - 2 :].sum(axis=(1, 2))
    series = {
        "t_s": t,
        "P_up": p_up,
        "P_down": spin[:, DOWN],
        "P_F2": f2,
        "P_F3": f3,
        "P_n0": p_n0,
        "P_target": p_target,
        "P_n0_given_up": p_n0_given_up,
        "P_trapped": trace,
        "P_up_absolute": p_up * trace,
        "P_n0_absolute": p_n0 * trace,
        "P_target_absolute": p_target * trace,
        "nbar": nbar,
        "fractional_energy_reduction": frac_red,
        "cooling_rate_quanta_per_s": cooling_rate,
        "cooling_power_W": energy_rate,
        "T_equiv_K": temps,
        "photons_spin_pump": counts[:, 0],
        "photons_repump": counts[:, 1],
        "photons_raman_scatter": counts[:, 2],
        "overflow": res.overflow,
        "trace_plus_overflow_minus_1": trace + res.overflow - 1.0,
        "boundary_population": boundary,
        "min_eigenvalue": res.min_eig,
        "raman_coherence_l1": res.coherence,
    }
    for i in range(len(GROUND_STATES)):
        series[f"P{ground_label(i)}"] = spin[:, i]

    num = cfg.numerics
    v: Validity = model.static_validity()
    ce = float(cons.max())
    v.add("conservation", "ok" if ce < num.conservation_tolerance else "invalid", f"max |trace + overflow - 1| = {ce:.3g}")
    mn = float(res.min_eig.min())
    v.add("positivity", "ok" if mn > -num.positivity_tolerance else "invalid", f"most negative eigenvalue/population = {mn:.3g}")
    bmax = float(boundary.max())
    ov = float(res.overflow[-1])
    if model.lattice_full:  # basis = every bound level: the top levels and the overflow are physical
        v.add("motional boundary", "ok", f"max population in the top two bound levels: {bmax:.3g} (physical; above them the atom is unbound)")
        v.add("lattice loss", "ok" if ov < 1e-2 else "warning", f"probability promoted above the lattice depth (atom leaves its site) = {ov:.3g}")
    else:
        v.add("motional boundary", "ok" if bmax < num.boundary_tolerance else "invalid", f"max population in n >= n_max-1: {bmax:.3g}")
        v.add("numerical overflow", "ok" if ov < num.boundary_tolerance else "invalid", f"probability pushed above n_max (numerical, not atom loss) = {ov:.3g}")
    # high-energy validity: harmonic, infinitely deep trap (the lattice entry comes from the model)
    if model.motion.kind == "lattice":
        pass
    elif cfg.trap.depth_uk:
        depth_hz = cfg.trap.depth_uk * 1e-6 * K_B / H
        nthr = 0.3 * depth_hz / model.nu_t
        hi = float(pn[:, n > nthr].sum(axis=1).max())
        st = "ok" if hi < 1e-3 else ("warning" if hi < 1e-2 else "invalid")
        v.add("harmonic approximation", st, f"max population with E_n > 0.3 x trap depth during run: {hi:.3g}")
    else:
        nmx = float(nbar.max())
        st = "ok" if nmx < 5 else "warning"
        v.add("harmonic approximation", st, f"trap depth not given; anharmonicity/finite-depth loss not assessed (max nbar during run {nmx:.3g})")
    # weak-excitation indicator for occupied states
    occ = spin.max(axis=0) > 1e-3
    worst = 0.0
    for seg in model.schedule:
        pe = seg.pump * model.rates["spin_pump"].p_exc * cfg.spin_pump.enabled + seg.repump * model.rates["repump"].p_exc * cfg.repump.enabled
        worst = max(worst, float(pe[occ].max()) if occ.any() else 0.0)
    oc = cfg.optical
    st = "ok" if worst < oc.weak_excitation_warning else ("warning" if worst < oc.weak_excitation_invalid else "invalid")
    v.add("weak excitation", st, f"max excited-state fraction for occupied ground states = {worst:.3g}")
    if res.solver == "rate":
        worst_r = max((w for _, w, _ in res.rate_validity), default=0.0)
        v.add("rate backend", "ok", f"rate elimination valid: max Omega/sqrt(gamma^2+delta^2) = {worst_r:.3g}")
    for note in res.notes:
        v.add("solver", "ok", note)
    for note in model.initial_notes:
        v.add("initial state", "ok", note)

    therm_note = "thermal-equivalent (p(n) near thermal)" if tvd < 0.05 else f"PROXY ONLY: final p(n) non-thermal (TVD from thermal {tvd:.2f})"
    final = {
        "P_up": float(p_up[-1]),
        "P_n0": float(p_n0[-1]),
        "P_target": float(p_target[-1]),
        "P_up_times_P_n0": float(p_up[-1] * p_n0[-1]),
        "P_n0_given_up": float(p_n0_given_up[-1]) if np.isfinite(p_n0_given_up[-1]) else None,
        "P_trapped": float(trace[-1]),
        "P_up_absolute": float(p_up[-1] * trace[-1]),
        "P_n0_absolute": float(p_n0[-1] * trace[-1]),
        "P_target_absolute": float(p_target[-1] * trace[-1]),
        "nbar": float(nbar[-1]),
        "nbar_initial": float(nbar0),
        "fractional_energy_reduction": float(frac_red[-1]) if nbar0 > 0 else None,
        "T_equiv_K": float(temps[-1]),
        "T_equiv_note": therm_note,
        "final_pn_TVD_from_thermal": float(tvd),
        "photons": {c: float(counts[-1, i]) for i, c in enumerate(COUNTERS)},
        "photons_total": float(total_photons),
        "net_quanta_removed": float(removed),
        "net_quanta_removed_per_photon": float(per_photon) if np.isfinite(per_photon) else None,
        "mean_cooling_rate_quanta_per_s": float(removed / t[-1]) if t[-1] > 0 else None,
        "target_state_scattering_rate_per_s": tgt_rate,
        "target_state_scattering_by_beam_per_s": tgt_by,
        "time_to_spin_threshold_s": _threshold_time(t, p_up, cfg.analysis.spin_threshold),
        "time_to_joint_threshold_s": _threshold_time(t, p_target, cfg.analysis.joint_threshold),
        "spin_threshold": cfg.analysis.spin_threshold,
        "joint_threshold": cfg.analysis.joint_threshold,
        "spin_populations_initial": {ground_label(i): float(spin[0, i]) for i in range(12)},
        "spin_populations_final": {ground_label(i): float(spin[-1, i]) for i in range(12)},
        "max_conservation_error": ce,
        "min_eigenvalue": mn,
        "max_boundary_population": bmax,
        "final_overflow": ov,
        "lattice_loss": ov if model.lattice_full else None,
        "initial_tail_discarded": float(model.initial_tail),
        "solver": res.solver,
        "wall_time_s": res.wall_time_s,
        "rhs_evaluations": res.n_rhs,
    }
    return {
        "series": series,
        "final": final,
        "validity": v,
        "pn": pn,
        "P_final": P[-1],
        "segments": res.segments,
        "cooling_rate_method": f"Savitzky-Golay (cubic, window {cfg.analysis.derivative_window} samples) derivative of nbar(t)",
    }
