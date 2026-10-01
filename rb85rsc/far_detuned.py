"""Cached ARC D-line amplitudes for Raman calibration and trap estimates.

Ground shifts include the counter-rotating D-line term. Raman and spontaneous
scattering amplitudes use the rotating-wave Kramers-Heisenberg sum over BOTH
fine-structure manifolds before squaring. Higher P levels and hyperfine-mediated
polarizability corrections are not included. This is an atomic estimate, not a
measurement of intensity or technical noise.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import qmc

from . import atomic as at, motion as mo, polarization as pol

CACHE = Path(__file__).parent / "cache"
FORMAT = 1


def _write_cache(path, data):
    """Publish complete JSON atomically when several scan workers calibrate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + "_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def dline_data():
    """Read portable atomic data; ARC's working database stays in memory."""
    path = CACHE / "rb85_d1_d2_estimates.json"
    version = at._arc_version()
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("format") == FORMAT and version in (data["arc_version"], "unavailable"):
            return data
    from arc import Rubidium85

    class MemoryRubidium85(Rubidium85):
        def _databaseInit(self):
            self.conn = sqlite3.connect(":memory:")
            self.c = self.conn.cursor()

    atom = MemoryRubidium85()
    rows = []
    for j, fs in ((0.5, (2, 3)), (1.5, (1, 2, 3, 4))):
        a, b = atom.getHFSCoefficients(5, 1, j)
        centroid = atom.getTransitionFrequency(5, 0, 0.5, 5, 1, j)
        for f in fs:
            for m in range(-f, f + 1):
                ds = []
                for fg, mg in at.GROUND_STATES:
                    q = m - mg
                    ds.append(float(atom.getDipoleMatrixElementHFS(5, 0, 0.5, fg, mg, 5, 1, j, f, m, q)) if abs(q) <= 1 else 0.)
                rows.append(dict(j=j, f=f, m=m, energy_hz=float(centroid + atom.getHFSEnergyShift(j, f, a, b)),
                                 g=float(atom.getLandegfExact(1, j, f)), dipole_au=ds))
    atom.conn.close()
    data = dict(format=FORMAT, arc_version=version, source="ARC Rubidium85 HFS energies and dipoles; 5P1/2 + 5P3/2",
                approximation="D-line sum; weak-field dipoles; no higher P levels", rows=rows)
    _write_cache(path, data)
    return data


def spherical_amplitudes(eps, bdir):
    eb = pol.field_frame(bdir) @ eps
    return {q: np.vdot(e, eb) for q, e in pol.SPHERICAL.items()}


class DLineResponse:
    def __init__(self, atom, b_gauss, bdir):
        self.atom = atom
        self.bdir = bdir
        data = dline_data()
        self.source = {k: v for k, v in data.items() if k != "rows"}
        rows = data["rows"]
        self.d = np.asarray([r["dipole_au"] for r in rows]) * at.EA0
        self.me = np.asarray([r["m"] for r in rows])
        self.mg = np.asarray([m for _, m in at.GROUND_STATES])
        self.ee = np.asarray([r["energy_hz"] + r["g"] * r["m"] * at.MU_B * b_gauss * 1e-4 / at.H for r in rows])
        self.eg = atom.ground_energy_hz(b_gauss * 1e-4)
        self.eunit = np.sqrt(2 / (at.C_LIGHT * at.EPS0))  # electric field at 1 W/m²
        self._response_cache = OrderedDict()

    def _response_key(self, kind, frequency_hz, eps):
        """Exact value keys also detect changes to mutable polarization inputs."""
        dependencies = (eps, self.bdir, self.d, self.me, self.mg, self.eunit)
        if kind != "absorption":
            dependencies += (self.ee, self.eg, frequency_hz)
        arrays = tuple((a.dtype.str, a.shape, a.tobytes()) for a in map(np.asarray, dependencies))
        return kind, arrays

    def _cached_response(self, key, compute):
        """Bound per-instance setup reuse; return copies to protect cached arrays."""
        if key in self._response_cache:
            value = self._response_cache.pop(key)
        else:
            value = compute()
            value.setflags(write=False)
        self._response_cache[key] = value
        if len(self._response_cache) > 128:
            self._response_cache.popitem(last=False)
        return value.copy()

    def absorption(self, eps):
        key = self._response_key("absorption", None, eps)
        return self._cached_response(key, lambda: self._absorption_uncached(eps))

    def _absorption_uncached(self, eps):
        qs = spherical_amplitudes(eps, self.bdir)
        q = self.me[:, None] - self.mg
        return self.d * sum(np.where(q == k, v, 0) for k, v in qs.items())

    def shift_hz(self, frequency_hz, eps):
        """Ground-state shift in Hz per W/m², including counter-rotation."""
        key = self._response_key("shift", frequency_hz, eps)
        return self._cached_response(key, lambda: self._shift_hz_uncached(frequency_hz, eps))

    def _shift_hz_uncached(self, frequency_hz, eps):
        d = self.absorption(eps)
        nu = self.ee[:, None] - self.eg
        denom = 1 / (2 * np.pi * (frequency_hz - nu)) - 1 / (2 * np.pi * (frequency_hz + nu))
        return np.sum(np.abs(d * self.eunit / at.HBAR) ** 2 * denom, axis=0) / (8 * np.pi)

    def raman_hz(self, low_hz, high_hz, eps_low, eps_high):
        """Complex up->down Rabi / 2pi per sqrt(I_low I_high)."""
        dl = self.absorption(eps_low)[:, at.UP]
        dh = self.absorption(eps_high)[:, at.DOWN]
        den = 1 / (2 * np.pi * (low_hz - (self.ee - self.eg[at.UP])))
        den += 1 / (2 * np.pi * (high_hz - (self.ee - self.eg[at.DOWN])))
        return np.sum(dl * dh.conj() * den) * self.eunit**2 / (8 * np.pi * at.HBAR**2)

    def scattering_amplitudes(self, frequency_hz, eps):
        """[initial, final, emitted q=-1,0,+1], sqrt(rate) per sqrt(W/m²).

        D1/D2 interference is kept. Different emitted spherical components are
        treated as resolved dipole channels, as in the pump model.
        """
        key = self._response_key("scattering", frequency_hz, eps)
        return self._cached_response(key, lambda: self._scattering_amplitudes_uncached(frequency_hz, eps))

    def _scattering_amplitudes_uncached(self, frequency_hz, eps):
        da = self.absorption(eps)
        den = 2 * np.pi * (frequency_hz - (self.ee[:, None] - self.eg))
        out = np.zeros((12, 12, 3), complex)
        for i in range(12):
            for j in range(12):
                nu_out = frequency_hz + self.eg[i] - self.eg[j]
                pref = np.sqrt((2 * np.pi * nu_out) ** 3 / (3 * np.pi * at.EPS0 * at.HBAR * at.C_LIGHT**3))
                for iq, q in enumerate((-1, 0, 1)):
                    de = np.where(self.me - self.mg[j] == q, self.d[:, j], 0)
                    out[i, j, iq] = pref * self.eunit / (2 * at.HBAR) * np.sum(de * da[:, i] / den[:, i])
        return out


def spatial_samples(cfg, count=None):
    """Frozen site centers: Gaussian cloud, independent uniform registration.

    The 783/785 nm mismatch spans many relative periods over this cloud, so
    uniform microscopic registration is used independently of the slow envelope.
    """
    n = count or cfg.ensemble.samples
    sampler = qmc.Sobol(4, scramble=True, seed=cfg.numerics.seed)
    points = sampler.random_base2(int(np.ceil(np.log2(n))))[:n]
    xyz = ndtri(np.clip(points[:, :3], 1e-12, 1 - 1e-12)) * cfg.ensemble.cloud_radius_um / 2
    phase = 2 * np.pi * points[:, 3] + np.deg2rad(cfg.ensemble.phase_offset_deg)
    return xyz, phase


def envelope(position_um, direction, waist_um):
    p = np.asarray(position_um)
    k = pol.unit(direction)
    return float(np.exp(-2 * max(p @ p - (p @ k) ** 2, 0) / waist_um**2))


def raman_fields(cfg, response):
    rc = cfg.raman
    dirs = (rc.beam_low_direction, rc.beam_high_direction)
    eps = [pol.jones_field_vector(k, 45 * h, 0, cfg.magnetic.direction) for k, h in zip(dirs, rc.beam_helicities)]
    low = at.C_LIGHT / (rc.wavelength_nm * 1e-9)
    high = low + response.eg[at.UP] - response.eg[at.DOWN]
    paths = np.asarray([[response.raman_hz(low, high, a, b) for b in eps] for a in eps])
    return low, high, eps, paths


def calibrate_carrier(cfg, response):
    """Infer intensity using the selected carrier-decay convention.

    Only z confinement is used (the stated calibration condition). Thermal
    carrier overlaps and Gaussian intensity / random position are included
    before either predicting decay from those overlaps (modeled mode) or fitting
    NONNEGATIVE residual homogeneous dephasing (measured_envelope mode).
    Modeled mode fixes intensity by RMS carrier frequency / initial curvature;
    the reported decay never sets intensity or adds homogeneous damping.
    Contrast is visibility only and never scales initial atom populations.
    """
    rc, ec = cfg.raman, cfg.ensemble
    ensemble_inputs = {k: getattr(ec, k) for k in ("cloud_radius_um", "lattice_waist_um", "phase_offset_deg", "calibration_samples")}
    raman_inputs = {k: getattr(rc, k) for k in ("carrier_rabi_hz", "wavelength_nm", "beam_low_direction", "beam_high_direction",
                    "tone_powers_low", "tone_powers_high", "beam_beat_phase_deg", "beam_helicities", "waist_um", "flop_contrast", "flop_decay_periods", "carrier_decay_mode")}
    keydata = dict(format=6, atom=response.source, raman=raman_inputs, ensemble=ensemble_inputs,
                   trap=cfg.trap.__dict__, magnetic=cfg.magnetic.__dict__, temperature=cfg.initial.temperature_k,
                   seed=cfg.numerics.seed)
    key = hashlib.sha256(json.dumps(keydata, sort_keys=True).encode()).hexdigest()[:24]
    path = CACHE / f"carrier_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    low, high, eps, paths = raman_fields(cfg, response)
    xyz, phases = spatial_samples(cfg, ec.calibration_samples)
    a = paths * np.sqrt(np.outer(rc.tone_powers_low, rc.tone_powers_high))
    chi = np.deg2rad(rc.beam_beat_phase_deg)
    cs, weights = [], []
    for p, th in zip(xyz, phases):
        depth = cfg.trap.depth_uk * envelope(p, [0, 0, 1], ec.lattice_waist_um)
        nu = mo.lattice_frequency_hz(response.atom.mass_kg, cfg.trap.wavelength_nm, depth)
        site = mo.lattice_site(response.atom.mass_kg, nu, depth * 1e-6 * at.K_B / at.H, cfg.trap.n_max + 1)
        if site.N != site.n_bound:
            raise ValueError("carrier calibration requires all bound axial levels; increase trap.n_max")
        eta = 4 * np.pi / (rc.wavelength_nm * 1e-9) * site.x0
        diag = np.diag(site.displacement(eta))
        op = a[0, 1] * np.exp(1j * th) * diag + a[1, 0] * np.exp(-1j * (chi + th)) * diag.conj()
        op += a[0, 0] + a[1, 1] * np.exp(-1j * chi)
        cs.extend(np.abs(op) * envelope(p, [0, 0, 1], rc.waist_um))
        weights.extend(mo.boltzmann_pn(site.energies_hz, cfg.initial.temperature_k) / len(xyz))
    cs, weights = np.asarray(cs), np.asarray(weights)
    if weights @ cs**2 < 1e-24:
        raise ValueError("No carrier coupling for the specified Raman polarizations/tone phases; cannot infer intensity from a positive measured Rabi frequency")
    curvature_intensity = rc.carrier_rabi_hz / np.sqrt(weights @ cs**2)
    t = np.linspace(0, 4 / rc.carrier_rabi_hz, 161)
    tau = rc.flop_decay_periods / rc.carrier_rabi_hz
    target = rc.flop_contrast * .5 * (1 - np.exp(-t / tau) * np.cos(2 * np.pi * rc.carrier_rabi_hz * t))

    def predicted(intensity, gamma):
        om = 2 * np.pi * intensity * cs[:, None]
        damped = np.sqrt(om**2 - gamma**2 / 4 + 0j)
        arg = damped * t
        # sin(wt)/w is regular at critical damping.
        z = np.exp(-gamma * t / 2) * (np.cos(arg) + gamma * t / 2 * np.sinc(arg / np.pi))
        return rc.flop_contrast * .5 * (1 - weights @ z.real)

    def residual(x):
        return predicted(curvature_intensity * x[0], x[1] / tau) - target

    modeled = rc.carrier_decay_mode == "modeled"
    if modeled:
        intensity, gamma = curvature_intensity, 0.
    else:
        fits = [least_squares(residual, [f, 1.0], bounds=([.25, 0], [4, 8])) for f in (.6, 1., 1.6, 2.5)]
        fit = min(fits, key=lambda f: np.sum(f.fun**2))
        intensity = curvature_intensity * fit.x[0]
        gamma = fit.x[1] / tau
    trace = predicted(intensity, gamma)
    rmse = float(np.sqrt(np.mean((trace - target)**2)))
    # Characteristic-function envelope of the distribution of local Rabi
    # frequencies. It need not be exponential or monotone; report FIRST 1/e
    # crossing rather than pretending it is a Markovian dephasing rate.
    envelope_t = np.linspace(0, 4 / rc.carrier_rabi_hz, 1601)
    envelope_curve = np.abs(weights @ np.exp(2j * np.pi * intensity * cs[:, None] * envelope_t))
    crossings = np.flatnonzero(envelope_curve <= np.exp(-1))
    decay_time = None
    if len(crossings):
        j = crossings[0]
        fraction = (envelope_curve[j-1] - np.exp(-1)) / (envelope_curve[j-1] - envelope_curve[j])
        decay_time = float(envelope_t[j-1] + fraction * (envelope_t[j] - envelope_t[j-1]))
    scattering, shifts = np.zeros(12), np.zeros(12)
    for freq, powers in ((low, rc.tone_powers_low), (high, rc.tone_powers_high)):
        for ep, power in zip(eps, powers):
            scattering += intensity * power * (abs(response.scattering_amplitudes(freq, ep))**2).sum(axis=(1, 2))
            shifts += intensity * power * response.shift_hz(freq, ep)
    result = dict(cache_key=key, calibration_model_version=6, carrier_decay_mode=rc.carrier_decay_mode,
                  intensity_calibration="rms_carrier_initial_curvature" if modeled else "synthetic_measured_envelope_fit",
                  intensity_per_unit_tone_w_m2=float(intensity),
                  power_per_unit_tone_w=float(intensity * np.pi * (rc.waist_um * 1e-6)**2 / 2),
                  residual_coherence_decay_rate_s=float(gamma), synthetic_trace_rmse=rmse,
                  calibration_compatible=None if modeled else rmse < .06,
                  reported_envelope_used_for_calibration=not modeled,
                  measured_rabi_hz=rc.carrier_rabi_hz,
                  modeled_rms_carrier_hz=float(intensity * np.sqrt(weights @ cs**2)),
                  modeled_inhomogeneous_envelope_first_1e_time_s=decay_time,
                  modeled_inhomogeneous_envelope_first_1e_periods=None if decay_time is None else decay_time * rc.carrier_rabi_hz,
                  inhomogeneous_envelope_times_s=envelope_t.tolist(), inhomogeneous_envelope=envelope_curve.tolist(),
                  measured_envelope_time_s=tau, contrast_visibility=rc.flop_contrast,
                  curvature_intensity_w_m2=float(curvature_intensity),
                  phase_averaged_central_raman_scattering_by_spin_s={at.ground_label(i): float(scattering[i]) for i in range(12)},
                  phase_averaged_central_raman_light_shifts_by_spin_hz={at.ground_label(i): float(shifts[i]) for i in range(12)},
                  phase_averaged_central_differential_shift_hz=float(shifts[at.UP] - shifts[at.DOWN]),
                  pathway_rabi_per_intensity_hz=[[dict(real=float(v.real), imag=float(v.imag)) for v in row] for row in paths],
                  note=("Modeled spatial/thermal carrier decay; intensity fixes RMS local carrier frequency to the supplied measured frequency (initial-curvature convention). No fitted homogeneous damping. The reported envelope is a comparison only, not a calibration constraint. The inhomogeneous 1/e time is a first crossing, not an exponential fit. Visibility does not remove atoms." if modeled else "Fit to a synthetic damped carrier trace reconstructed from the measurement summary, not raw data. Inhomogeneity included first; residual noise constrained nonnegative. Visibility does not remove atoms."),
                  times_s=t.tolist(), summary_trace=target.tolist(), fitted_trace=trace.tolist(),
                  source=response.source)
    _write_cache(path, result)
    return result
