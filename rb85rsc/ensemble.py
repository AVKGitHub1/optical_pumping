"""Frozen spatial ensemble, coherent axial Raman, and joint 3D recoil kinetics.

Each local atom has joint populations P[spin,nx,ny,nz] and a full axial density
matrix for the Raman pair at each (nx,ny). Recoil uses the existing
population-resolving approximation. Emission projections on the three axes
are sampled independently from their correct dipole marginals: correlations
between Cartesian components of ONE emitted photon are neglected. Spin/motion
and transverse/axial population correlations produced by cooling and loss ARE
retained. Loss is absorbing on departure from ANY local bound manifold.

A positivity-preserving Strang split applies the exact static Raman unitary
and a uniformized classical scattering generator. The configurable timestep
must be checked by halving; it is not controlled by the legacy ODE tolerances.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from . import atomic as at, motion as mo, polarization as pol
from .analysis import _derivative, _threshold_time
from .dynamics import RunResult, SimulationStopped, _single_thread
from .far_detuned import DLineResponse, calibrate_carrier, envelope, raman_fields, spatial_samples
from .model import Model, Validity


# Populations always have layout [spin, nx, ny, nz]. These are exactly the
# permutations used by moveaxis, without its repeated generic axis validation.
_AXIS_PERMUTATIONS = (
    ((0, 1, 2, 3), (0, 1, 2, 3)),
    ((1, 0, 2, 3), (1, 0, 2, 3)),
    ((2, 0, 1, 3), (1, 2, 0, 3)),
    ((3, 0, 1, 2), (1, 2, 3, 0)),
)


def axis_apply(K, P, axis):
    forward, backward = _AXIS_PERMUTATIONS[axis]
    moved = P.transpose(forward)
    result = (K @ moved.reshape(moved.shape[0], -1)).reshape((K.shape[0],) + moved.shape[1:])
    return result.transpose(backward)


@dataclass
class RecoilTerm:
    W: np.ndarray                 # initial spin, final spin
    kernels: tuple                # K_axis[n',n], possibly a coherent cross term
    total: np.ndarray             # total event hazard [spin,nx,ny,nz]
    counter: int                  # pump, repump, Raman, lattice
    scale_name: str               # pump, repump, raman, always

    def gain(self, p):
        q = p
        for axis, k in enumerate(self.kernels, 1):
            q = axis_apply(k, q, axis)
        return (self.W.T @ q.reshape(12, -1)).reshape(p.shape).real

    def retained_hazard(self):
        k = [x.sum(axis=0) for x in self.kernels]
        return np.einsum('s,i,j,k->sijk', self.W.sum(axis=1), *k).real


class _RecoilGain:
    """Reuse identical contraction prefixes without regrouping the term sum.

    Every path still applies x, y, z, then spin, then the segment scale. Exact
    array keys (including layout) preserve distinct interference kernels and
    floating-point evaluation order; no extra rounding or pruning is used.
    """

    def __init__(self, active):
        self.transforms = []
        self.terms = []
        prefixes = {}
        for term, scale in active:
            parent = 0  # the input population tensor
            for axis, kernel in enumerate(term.kernels, 1):
                key = (parent, axis, kernel.dtype.str, kernel.shape,
                       kernel.strides, kernel.tobytes())
                if key not in prefixes:
                    self.transforms.append((parent, axis, kernel))
                    prefixes[key] = len(self.transforms)
                parent = prefixes[key]
            self.terms.append((parent, term.W.T, scale))

    def __call__(self, P):
        values = [P]
        for parent, axis, kernel in self.transforms:
            values.append(axis_apply(kernel, values[parent], axis))
        gain = np.zeros_like(P)
        for leaf, spin, scale in self.terms:
            contribution = scale * (spin @ values[leaf].reshape(12, -1)).reshape(P.shape).real
            if gain.dtype == contribution.dtype:
                gain += contribution
            else:
                # Preserve the original sum's per-term dtype promotion for
                # standalone callers; simulator populations are float64.
                gain = gain + contribution
        return gain


def cross_kernel(site, kabs1, kabs2, kemit, bprojection, pattern, nq):
    """One-axis coherent absorption pair, integrated over emitted projection."""
    c, w = mo.gauss_legendre(nq)
    weights = w * mo.emission_weight(c, bprojection, pattern)
    K = np.zeros((site.N, site.N), complex)
    for direction, weight in zip(c, weights):
        d1 = site.displacement((kabs1 - kemit * direction) * site.x0)
        d2 = site.displacement((kabs2 - kemit * direction) * site.x0)
        K += weight * d1 * d2.conj()
    total = np.diag(site.displacement((kabs1 - kabs2) * site.x0))
    # The isolated wells have definite parity: each displacement element is
    # real or imaginary, so its absorption cross product is real. Retaining
    # roundoff imaginary parts would double tensor-contraction cost.
    return np.real_if_close(K, tol=1000), np.real_if_close(total, tol=1000)


def _term(sites, W, ka, kb, kemit, bdir, pattern, nq, counter, scale_name):
    kernels, totals = zip(*(cross_kernel(site, ka[i], kb[i], kemit, bdir[i], pattern, nq)
                           for i, site in enumerate(sites)))
    hazard = np.einsum('s,i,j,k->sijk', W.sum(axis=1), *totals).real
    return RecoilTerm(W, kernels, hazard, counter, scale_name)


class LocalExperiment:
    def __init__(self, cfg, response, calibration, position, phase, beat_hz=None):
        self.position, self.phase = np.asarray(position), phase
        self.response = response
        self.calibration = calibration
        atom = response.atom
        c = cfg.copy()
        ec, rc = cfg.ensemble, cfg.raman
        axes = np.eye(3)
        depths = np.asarray([ec.transverse_depth_uk, ec.transverse_depth_uk, cfg.trap.depth_uk])
        depths *= [envelope(position, k, ec.lattice_waist_um) for k in axes]
        wavelengths = [ec.transverse_wavelength_nm] * 2 + [cfg.trap.wavelength_nm]
        sites = []
        for lam, depth in zip(wavelengths, depths):
            nu = mo.lattice_frequency_hz(atom.mass_kg, lam, depth)
            site = mo.lattice_site(atom.mass_kg, nu, depth * 1e-6 * at.K_B / at.H, cfg.trap.n_max + 1)
            if site.N != site.n_bound:
                raise ValueError("3D well-loss mode requires all bound levels; increase trap.n_max")
            sites.append(site)
        self.sites = sites
        self.depths = depths
        c.trap.depth_uk = float(depths[2])
        c.raman.lattice_phase_deg = np.rad2deg(phase)
        intensity = calibration["intensity_per_unit_tone_w_m2"] * envelope(position, axes[2], rc.waist_um)
        self.intensity = intensity
        low, high, eps, paths = raman_fields(cfg, response)
        # Model's carrier field is now a units scale: microscopic complex path
        # coefficients contain Hz per W/m²; the measurement stays in parent cfg.
        c.raman.carrier_rabi_hz = intensity
        c.raman.scattering_rate_s = 0.0
        c.raman.extra_coherence_decay_rate_s = rc.extra_coherence_decay_rate_s + calibration["residual_coherence_decay_rate_s"]
        if beat_hz is not None:
            c.raman.frequency_mode = "absolute_beat"
            c.raman.beat_frequency_hz = beat_hz
        for name in ("spin_pump", "repump"):
            pc = getattr(c, name)
            # Obtain the ACTIVE geometry, not an inactive direction field.
            from .optical import beam_spec
            direction = beam_spec(name, pc, atom, pol.unit(cfg.magnetic.direction)).direction
            attenuation = envelope(position, direction, pc.waist_um)
            pc.power_mw *= attenuation
            pc.peak_intensity_w_m2 *= attenuation
        self.model = m = Model(c, atom, raman_path_coefficients=paths)
        self.shape = (12,) + tuple(s.N for s in sites)
        self.P0 = np.einsum('sn,i,j->sijn', m.P0,
                           mo.boltzmann_pn(sites[0].energies_hz, ec.transverse_temperature_k),
                           mo.boltzmann_pn(sites[1].energies_hz, ec.transverse_temperature_k))
        self.terms = []
        nq = cfg.numerics.recoil_quadrature_points
        bdir = pol.unit(cfg.magnetic.direction)
        for bi, name in enumerate(("spin_pump", "repump")):
            kvec = m.k_d2 * m.beams[name].direction
            for ci, pattern in enumerate(("sigma", "pi")):
                pat = pattern if cfg.optical.emission_pattern == "dipole" else "isotropic"
                W = m.rates[name].W[:, :, ci]
                self.terms.append(_term(sites, W, kvec, kvec, m.k_d2, bdir, pat, nq, bi,
                                        "pump" if bi == 0 else "repump"))
        # Elastic scattering also resolves n in this retained approximation.
        # Raman same-frequency beams interfere in absorption; different tones
        # do not interfere in the spontaneous rate (GHz beat is averaged).
        self.raman_shifts_hz = np.zeros((12, sites[2].N))
        dirs = (pol.unit(rc.beam_low_direction), pol.unit(rc.beam_high_direction))
        if rc.scattering_rate_s > 0:
            W = (np.eye(12) if rc.scattering_model == "spin_preserving" else np.ones((12, 12)) / 12) * rc.scattering_rate_s / 2
            k = 2 * np.pi / (rc.wavelength_nm * 1e-9)
            for direction in dirs:
                self.terms.append(_term(sites, W, k * direction, k * direction, k, bdir, "isotropic", nq, 2, "raman"))
        for freq, powers, phases in (
            (low, rc.tone_powers_low, [phase / 2, -phase / 2]),
            (high, rc.tone_powers_high, [phase / 2, -phase / 2 + np.deg2rad(rc.beam_beat_phase_deg)]),
        ):
            amplitudes = [response.scattering_amplitudes(freq, e) * np.sqrt(intensity * p) * np.exp(1j * ph)
                          for e, p, ph in zip(eps, powers, phases)]
            self._add_coherent_light(amplitudes, dirs, freq, bdir, nq, 2, "raman")
            # AC shifts: diagonal expectation of the same-tone interference.
            # All input fields for the clarified geometry have equal lab eps.
            for i in range(2):
                for j in range(2):
                    # Polarization cross term by the polarization identity.
                    if i == j:
                        sh = response.shift_hz(freq, eps[i])
                    else:
                        sh = (response.shift_hz(freq, eps[i] + eps[j]) - response.shift_hz(freq, eps[i] - eps[j])
                              + 1j * (response.shift_hz(freq, eps[i] + 1j * eps[j]) - response.shift_hz(freq, eps[i] - 1j * eps[j]))) / 4
                    phasefactor = np.sqrt(powers[i] * powers[j]) * np.exp(1j * (phases[i] - phases[j]))
                    d = np.diag(sites[2].displacement(2 * np.pi * freq / at.C_LIGHT * (dirs[i][2] - dirs[j][2]) * sites[2].x0))
                    self.raman_shifts_hz += intensity * (phasefactor * sh[:, None] * d).real
        self.lattice_shifts_hz = np.zeros(self.shape)
        self.lattice_estimates = []
        for axis, (lam, depth, ep) in enumerate(zip(wavelengths, depths, ec.lattice_linear_polarizations)):
            freq = at.C_LIGHT / (lam * 1e-9)
            if axis == 1:
                freq += ec.transverse_frequency_offset_hz
            ep = pol.unit(ep)
            sh = response.shift_hz(freq, ep)
            scalar = np.mean(sh)
            if scalar >= 0:
                raise ValueError("the configured lattice is not red detuned in the D-line estimate")
            peak = -depth * 1e-6 * at.K_B / at.H / scalar
            ds = [axes[axis], -axes[axis]]
            amps = [response.scattering_amplitudes(freq, ep) * np.sqrt(peak / 4)] * 2
            self._add_coherent_light(amps, ds, freq, bdir, nq, 3, "always")
            cosine = (1 + np.diag(sites[axis].displacement(4 * np.pi * freq / at.C_LIGHT * sites[axis].x0)).real) / 2
            # Common scalar potential is already in the site Hamiltonian.
            differential = (sh - scalar) * peak
            shape = [12, 1, 1, 1]
            shape[axis + 1] = sites[axis].N
            self.lattice_shifts_hz += (differential[:, None] * cosine).reshape(shape)
            self.lattice_estimates.append(dict(axis="xyz"[axis], peak_intensity_w_m2=float(peak),
                                               mean_scalar_shift_per_intensity_hz=float(scalar),
                                               polarization_lab=ep.tolist()))
        self.terms = self._combine_terms(self.terms)

    def _add_coherent_light(self, amps, dirs, freq, bdir, nq, counter, scale):
        k = 2 * np.pi * freq / at.C_LIGHT
        for i in range(2):
            for j in range(2):
                for iq, q in enumerate((-1, 0, 1)):
                    W = amps[i][:, :, iq] * amps[j][:, :, iq].conj()
                    if np.max(np.abs(W)) < 1e-14:
                        continue
                    self.terms.append(_term(self.sites, W, k * dirs[i], k * dirs[j], k, bdir,
                                            "pi" if q == 0 else "sigma", nq, counter, scale))

    @staticmethod
    def _combine_terms(terms):
        # Merge exactly equal recoil kernels, retaining interference in W.
        merged = {}
        for t in terms:
            parts = []
            for k in t.kernels:
                rounded = np.round(k, 12)
                rounded.real[rounded.real == 0] = 0.
                if np.iscomplexobj(rounded):
                    rounded.imag[rounded.imag == 0] = 0.
                parts.append(rounded.tobytes())
            key = (t.counter, t.scale_name, *parts)
            if key in merged:
                prev = merged[key]
                prev.W = prev.W + t.W
                prev.total = prev.total + t.total
            else:
                merged[key] = t
        for t in merged.values():
            if all(not np.iscomplexobj(k) for k in t.kernels):
                t.W = t.W.real
        return [t for t in merged.values() if np.any(t.W != 0)]

    def operators(self, seg):
        scales = dict(always=1., pump=seg.pump, repump=seg.repump, raman=seg.raman if self.model.cfg.raman.enabled else 0.)
        active = [(t, scales[t.scale_name]) for t in self.terms if scales[t.scale_name] > 0]
        gout = sum((scale * t.total for t, scale in active), np.zeros(self.shape))
        retained = sum((scale * t.retained_hazard() for t, scale in active), np.zeros(self.shape))
        hazard = gout - retained
        if gout.min() < -1e-6 or hazard.min() < -1e-6:
            raise ValueError("negative photon or loss hazard in coherent recoil construction")
        gout = np.maximum(gout, 0)
        hazard = np.maximum(hazard, 0)
        counts = np.asarray([sum((scale * t.total for t, scale in active if t.counter == i), np.zeros(self.shape)) for i in range(4)])
        m, nz = self.model, self.sites[2].N
        hu, hd = m.light_shifts(seg)
        eps = m.eps
        # Fixed lab beat for every atom, with local anharmonic spacings and
        # local Stark shifts. Never retune the beat separately for each node.
        up = eps + 2 * np.pi * (m.nu_ud_zeeman - m.nu_beat) + hu
        down = eps + hd
        up = up + 2 * np.pi * scales["raman"] * (self.raman_shifts_hz[at.UP] + m.cfg.raman.differential_light_shift_hz)
        down = down + 2 * np.pi * scales["raman"] * self.raman_shifts_hz[at.DOWN]
        H = np.zeros(self.shape[1:3] + (2 * nz, 2 * nz), complex)
        di = np.arange(2 * nz)
        shifts = np.concatenate([self.lattice_shifts_hz[at.UP], self.lattice_shifts_hz[at.DOWN]], axis=-1)
        H[..., di, di] = np.concatenate([up, down]) + 2 * np.pi * shifts
        V = .5 * m.omega_c * m.D * scales["raman"]
        n = np.arange(nz)
        V = np.where(np.abs(n[:, None] - n) <= m.cfg.raman.max_sideband_order, V, 0)
        H[..., :nz, nz:] = V
        H[..., nz:, :nz] = V.conj().T
        return active, gout, hazard, counts, H


class _UniformizedScattering:
    """Prepare a segment's Markov generator and exact-timestep Poisson weights."""

    def __init__(self, active, gout, hazard, count_rates):
        self.rate = float(gout.max())
        self.stay = 1 - gout / self.rate if self.rate else None
        self.hazard = hazard
        self.count_rates = count_rates
        self.gain = _RecoilGain(active)
        self._weights = {}

    def _poisson_weights(self, dt):
        # Unlike the unitary's existing rounded timestep cache, scattering
        # previously used the exact dt. Retain that convention here.
        if dt not in self._weights:
            mu = self.rate * dt
            if mu > 10:
                raise ValueError("time_step_us too large for the scattering rate")
            weight = np.exp(-mu)
            weights = [weight]
            cumulative = weight
            for n in range(1, 100):
                weight *= mu / n
                weights.append(weight)
                cumulative += weight
                if n > mu and weight < 1e-14:
                    break
            self._weights[dt] = (weights, max(1 - cumulative, 0.))
        return self._weights[dt]

    def step(self, P, dt):
        rate = self.rate
        if rate == 0:
            return P.copy(), np.zeros_like(P), np.zeros(4)
        weights, tail = self._poisson_weights(dt)
        term = P
        result = weights[0] * term
        losses = np.zeros_like(P)
        counts = np.zeros(4)
        term_loss = np.zeros_like(P)
        term_counts = np.zeros(4)
        for weight in weights[1:]:
            # Keep multiply-then-divide and the original ordered sum, including
            # nonunit pulse scales and signed coherent scattering cross terms.
            term_loss += self.hazard * term / rate
            term_counts += np.einsum('csijk,sijk->c', self.count_rates, term) / rate
            gain = self.gain(term)
            term = self.stay * term + gain / rate
            result += weight * term
            losses += weight * term_loss
            counts += weight * term_counts
        # Preserve the original Poisson-tail correction and positivity bound.
        result += tail * term
        losses += tail * term_loss
        counts += tail * term_counts
        return result, losses, counts


def uniform_step(P, active, gout, hazard, count_rates, dt):
    """Exact positive Markov exponential, loss flux, and photon counts.

    Poisson tail is bounded below 1e-13 per step. The solver prepares this
    operation once per segment; this entry point also supports standalone use.
    """
    return _UniformizedScattering(active, gout, hazard, count_rates).step(P, dt)


def solve_local(local, cfg, stop=None, progress=None):
    start = time.perf_counter()
    m = local.model
    nz = local.sites[2].N
    ts = np.linspace(0, m.schedule[-1].t1, cfg.timing.n_samples)
    P = local.P0.copy()
    rho = np.zeros(P.shape[1:3] + (2 * nz, 2 * nz), complex)
    di = np.arange(2 * nz)
    rho[..., di, di] = np.concatenate([P[at.UP], P[at.DOWN]], axis=-1)
    cn, loss = np.zeros(4), 0.
    histories, axial, transverse, energies, allcounts, losses, mins, coherences = [], [], [], [], [], [], [], []
    level_energies = [site.energies_hz - site.energies_hz[0] for site in local.sites]
    loss_fluxes, loss_energy_fluxes = [], []
    current_hazard = local.operators(m.schedule[0])[2]

    def sample():
        histories.append(P.sum(axis=(1, 2)))
        axial.append(P.sum(axis=(0, 1, 2)))
        transverse.append([P.sum(axis=(0, 2, 3)), P.sum(axis=(0, 1, 3))])
        paxis = [transverse[-1][0], transverse[-1][1], axial[-1]]
        energies.append([p @ e for p, e in zip(paxis, level_energies)])
        allcounts.append(cn.copy())
        losses.append(loss)
        mins.append(min(float(P.min()), float(np.linalg.eigvalsh(rho).min())))
        coherences.append(np.abs(rho[..., :nz, nz:]).sum())
        flux = current_hazard * P
        loss_fluxes.append(flux.sum())
        loss_energy_fluxes.append([flux.sum(axis=tuple(i for i in range(4) if i != axis + 1)) @ e
                                  for axis, e in enumerate(level_energies)])

    sample()
    sample_index, steps = 1, 0
    for seg in m.schedule:
        active, gout, hazard, count_rates, H = local.operators(seg)
        scattering = _UniformizedScattering(active, gout, hazard, count_rates)
        current_hazard = hazard
        gamma = m.cfg.raman.extra_coherence_decay_rate_s
        # Phenomenological extra gamma acts only on the Raman spin coherence.
        pair_out = np.concatenate([gout[at.UP], gout[at.DOWN]], axis=-1)
        damp = (pair_out[..., :, None] + pair_out[..., None, :]) / 2
        damp[..., :nz, nz:] += gamma
        damp[..., nz:, :nz] += gamma
        now = seg.t0
        cache = {}
        while now < seg.t1 - 1e-14:
            if stop is not None and stop.is_set():
                raise SimulationStopped()
            target = min(seg.t1, ts[sample_index] if sample_index < len(ts) else seg.t1)
            # With Raman off, diagonal phase evolution commutes with the
            # population-resolving dissipator: no splitting substeps are needed.
            max_dt = cfg.ensemble.time_step_us * 1e-6 if seg.raman > 0 and m.cfg.raman.enabled else np.inf
            nsteps = max(1, int(np.ceil((target - now) / max_dt)),
                         int(np.ceil((target - now) * scattering.rate / 5)))
            dt = (target - now) / nsteps
            key = round(dt, 15)
            if key not in cache:
                cache[key] = (expm(-.5j * H * dt), np.exp(-damp * dt))
            U, decay = cache[key]
            Uh = U.conj().swapaxes(-1, -2)
            for _ in range(nsteps):
                if stop is not None and stop.is_set():
                    raise SimulationStopped()
                rho = U @ rho @ Uh
                pd = rho[..., di, di].real
                P[at.UP], P[at.DOWN] = pd[..., :nz], pd[..., nz:]
                P, lf, dc = scattering.step(P, dt)
                loss += lf.sum()
                cn += dc
                rho *= decay
                rho[..., di, di] = np.concatenate([P[at.UP], P[at.DOWN]], axis=-1)
                rho = U @ rho @ Uh
                pd = rho[..., di, di].real
                P[at.UP], P[at.DOWN] = pd[..., :nz], pd[..., nz:]
                steps += 1
            now = target
            if sample_index < len(ts) and abs(now - ts[sample_index]) < 1e-13:
                sample()
                sample_index += 1
            if progress:
                progress(now / ts[-1])
    result = dict(t=ts, P=np.asarray(histories), counts=np.asarray(allcounts), loss=np.asarray(losses),
                  energy_abs_hz=np.asarray(energies),
                  pn_axes=[[x[0] for x in transverse], [x[1] for x in transverse], axial],
                  min_eig=np.asarray(mins), coherence=np.asarray(coherences),
                  loss_flux=np.asarray(loss_fluxes), loss_energy_flux_hz=np.asarray(loss_energy_fluxes),
                  P_final_3d=P.copy(),
                  wall_time_s=time.perf_counter() - start, steps=steps, rho_final=rho.sum(axis=(0, 1)))
    return result


def mixture_temperature(mean_hz, components, axis):
    """Fit energy to a mixture of local finite canonical spectra.

    Spatial weights are the FINAL survivor weights; averaging temperatures is
    not used. Returns NaN if no positive-temperature mixture can match energy.
    """
    from scipy.optimize import brentq
    levels, weights = [], []
    for local, result in components:
        site = local.sites[axis]
        levels.append(site.energies_hz - site.energies_hz[0])
        weights.append(result["P"][-1].sum())
    weights = np.asarray(weights) / max(sum(weights), 1e-300)
    upper_energy = sum(w * e.mean() for w, e in zip(weights, levels))
    if mean_hz > upper_energy + 1e-8:
        return np.nan
    if mean_hz <= 0:
        return 0.
    if abs(mean_hz - upper_energy) < 1e-8:
        return np.inf
    def f(logt):
        return sum(w * (mo.boltzmann_pn(e, np.exp(logt)) @ e) for w, e in zip(weights, levels)) - mean_hz
    return float(np.exp(brentq(f, -40., 10.)))


def _check_stop(stop):
    if stop is not None and stop.is_set():
        raise SimulationStopped()


_WORKER_STOP = None


def _initialize_ensemble_worker(stop):
    """Share cancellation at process startup (spawn-safe on Windows and Qt)."""
    global _WORKER_STOP
    _WORKER_STOP = stop


def _parallel_components(cfg, calibration, xyz, phases, beat, workers, stop, progress):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

    options = {}
    process_stop = None
    if stop is not None:
        import multiprocessing

        # A GUI QThread must not fork a process containing Qt/BLAS threads.
        # Pass the shared Event through the initializer, not the task queue.
        context = multiprocessing.get_context("spawn")
        process_stop = context.Event()
        options = dict(mp_context=context, initializer=_initialize_ensemble_worker,
                       initargs=(process_stop,))
    _check_stop(stop)
    if progress:
        progress(0., f"starting {workers} workers for {len(xyz)} ensemble samples")
    _check_stop(stop)
    with ProcessPoolExecutor(max_workers=workers, **options) as pool:
        futures, ordered = {}, {}
        try:
            data = cfg.to_dict()
            for i, (position, phase) in enumerate(zip(xyz, phases)):
                _check_stop(stop)
                futures[pool.submit(_solve_node, data, calibration, position, phase, beat)] = i

            def collect(future):
                _check_stop(stop)
                ordered[futures[future]] = future.result()
                if progress:
                    progress(len(ordered) / len(xyz), f"ensemble atoms {len(ordered)}/{len(xyz)} complete")

            if stop is None:
                for future in as_completed(futures):
                    collect(future)
            else:
                pending = set(futures)
                while pending:
                    _check_stop(stop)
                    completed, pending = wait(pending, timeout=.05, return_when=FIRST_COMPLETED)
                    for future in completed:
                        collect(future)
            _check_stop(stop)
        except BaseException:
            # Signal running work BEFORE the context manager joins workers.
            # Cancel queued tasks and wait for cleanup before allowing a rerun.
            if process_stop is not None:
                process_stop.set()
            for future in futures:
                future.cancel()
            raise
    return [ordered[i] for i in range(len(xyz))]


def run_ensemble(cfg, stop=None, progress=None, estimate_cb=None, *, workers=None):
    started = time.perf_counter()
    cfg.validate()
    worker_count = cfg.ensemble.workers if workers is None else workers
    if isinstance(worker_count, bool) or not isinstance(worker_count, (int, np.integer)) or worker_count < 1:
        raise ValueError("ensemble workers must be a positive integer")
    _check_stop(stop)
    with _single_thread():
        atom = at.load_atomic_data()
        response = DLineResponse(atom, cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
        calibration = calibrate_carrier(cfg, response)
        _check_stop(stop)
        central = LocalExperiment(cfg, response, calibration, [0, 0, 0], 0.)
        # One fixed experimental beat, resonant with the central n=1->0 red
        # sideband including phase-AVERAGED Raman and lattice differential shift.
        if cfg.raman.frequency_mode == "absolute_beat":
            beat = cfg.raman.beat_frequency_hz
        else:
            low, high, eps, _ = raman_fields(cfg, response)
            shift = np.zeros(12)
            for freq, powers in ((low, cfg.raman.tone_powers_low), (high, cfg.raman.tone_powers_high)):
                for ep, power in zip(eps, powers):
                    shift += response.shift_hz(freq, ep) * power * calibration["intensity_per_unit_tone_w_m2"]
            lattice_delta = central.lattice_shifts_hz[at.UP, 0, 0, min(1, central.sites[2].N - 1)] - central.lattice_shifts_hz[at.DOWN, 0, 0, 0]
            beat = (central.model.nu_ud_zeeman + central.model.nu_ref + shift[at.UP] - shift[at.DOWN]
                    + lattice_delta + cfg.raman.differential_light_shift_hz + cfg.raman.red_sideband_offset_hz)
        xyz, phases = spatial_samples(cfg)
        components = []
        if worker_count > 1:
            components = _parallel_components(cfg, calibration, xyz, phases, beat,
                                              worker_count, stop, progress)
        else:
            for i, (p, ph) in enumerate(zip(xyz, phases)):
                _check_stop(stop)
                if progress:
                    progress(i / len(xyz), f"ensemble atom {i + 1}/{len(xyz)}")
                local = LocalExperiment(cfg, response, calibration, p, ph, beat)
                result = solve_local(local, cfg, stop)
                components.append((local, result))
        _check_stop(stop)
        analysis = analyze_ensemble(cfg, central, components, calibration, beat)
        _check_stop(stop)
        analysis["final"]["node_solver_time_sum_s"] = analysis["final"]["wall_time_s"]
        analysis["final"]["wall_time_s"] = time.perf_counter() - started
        analysis["result"].wall_time_s = analysis["final"]["wall_time_s"]
        return analysis


def _solve_node(data, calibration, position, phase, beat):
    from .config import SimConfig
    _check_stop(_WORKER_STOP)
    cfg = SimConfig.from_dict(data)
    with _single_thread():
        response = DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
        local = LocalExperiment(cfg, response, calibration, position, phase, beat)
        _check_stop(_WORKER_STOP)
        return local, solve_local(local, cfg, _WORKER_STOP)


def analyze_ensemble(cfg, central, components, calibration, beat):
    n = len(components)
    ts = components[0][1]["t"]
    nz = max(local.sites[2].N for local, _ in components)
    P = np.zeros((len(ts), 12, nz))
    for _, r in components:
        P[:, :, :r["P"].shape[-1]] += r["P"] / n
    count = sum(r["counts"] for _, r in components) / n
    loss = sum(r["loss"] for _, r in components) / n
    energy = sum(r["energy_abs_hz"] for _, r in components) / n
    loss_flux = sum(r["loss_flux"] for _, r in components) / n
    loss_energy_flux = sum(r["loss_energy_flux_hz"] for _, r in components) / n
    survival = P.sum(axis=(1, 2))
    safe = np.maximum(survival, 1e-300)
    normalized = P / safe[:, None, None]
    spin, pn = normalized.sum(axis=2), normalized.sum(axis=1)
    energy /= safe[:, None]
    selection_power = at.H * (loss_energy_flux - energy * loss_flux[:, None]) / safe[:, None]
    temperatures = [mixture_temperature(energy[-1, a], components, a) for a in range(3)]
    mins = np.min([r["min_eig"] for _, r in components], axis=0)
    coh = sum(r["coherence"] for _, r in components) / n
    nbar = pn @ np.arange(nz)
    series = dict(t_s=ts, P_up=spin[:, at.UP], P_down=spin[:, at.DOWN], P_F2=spin[:, :5].sum(axis=1), P_F3=spin[:, 5:].sum(axis=1),
                  P_n0=pn[:, 0], P_target=normalized[:, at.UP, 0], P_n0_given_up=np.divide(normalized[:, at.UP, 0], spin[:, at.UP], out=np.zeros(len(ts)), where=spin[:, at.UP] > 0),
                  P_trapped=survival, P_up_absolute=P[:, at.UP].sum(axis=1), P_n0_absolute=P[:, :, 0].sum(axis=1), P_target_absolute=P[:, at.UP, 0],
                  nbar=nbar, mean_excitation_energy_J=at.H * energy[:, 2], mean_excitation_energy_hz=energy[:, 2],
                  fractional_energy_reduction=1 - energy[:, 2] / max(energy[0, 2], 1e-300),
                  cooling_rate_quanta_per_s=-_derivative(ts, nbar, cfg.analysis.derivative_window),
                  cooling_power_W=-at.H * _derivative(ts, energy[:, 2], cfg.analysis.derivative_window),
                  selection_cooling_power_W=selection_power[:, 2],
                  # Full time-dependent energy-matched mixture temperatures are
                  # expensive; compute with the appropriate weights at each time.
                  T_equiv_K=np.full(len(ts), np.nan), photons_spin_pump=count[:, 0], photons_repump=count[:, 1],
                  photons_raman_scatter=count[:, 2], photons_lattice_scatter=count[:, 3], overflow=loss,
                  trace_plus_overflow_minus_1=survival + loss - 1, boundary_population=np.full(len(ts), np.nan),
                  min_eigenvalue=mins, raman_coherence_l1=coh)
    for it in range(len(ts)):
        temp_components = [(local, {"P": r["P"][it:it+1]}) for local, r in components]
        series["T_equiv_K"][it] = mixture_temperature(energy[it, 2], temp_components, 2)
    for axis, a in enumerate("xyz"):
        series[f"mean_excitation_energy_{a}_hz"] = energy[:, axis]
        series[f"selection_cooling_power_{a}_W"] = selection_power[:, axis]
    for i in range(12):
        series[f"P{at.ground_label(i)}"] = spin[:, i]
    final = {k: float(v[-1]) for k, v in series.items() if k in ("P_up", "P_n0", "P_target", "P_n0_given_up", "P_trapped", "P_up_absolute", "P_n0_absolute", "P_target_absolute", "nbar", "mean_excitation_energy_J", "mean_excitation_energy_hz", "fractional_energy_reduction")}
    finite = lambda x: float(x) if np.isfinite(x) else None
    final.update(nbar_initial=float(nbar[0]), mean_excitation_energy_initial_J=float(at.H * energy[0, 2]),
                 P_up_times_P_n0=final["P_up"] * final["P_n0"], T_equiv_K=finite(temperatures[2]),
                 T_equiv_axes_K={a: finite(t) for a, t in zip("xyz", temperatures)},
                 mean_excitation_energy_axes_hz={a: float(e) for a, e in zip("xyz", energy[-1])},
                 T_equiv_note="Energy-matched mixture of local bound canonical distributions, weighted by surviving spatial populations; proxy only. None means no finite positive-temperature match.",
                 final_pn_TVD_from_thermal=None, energy_observable_note="Actual local excitation energies per survivor, including loss selection; axial n=0 is not the 3D ground state.",
                 photons={a: float(v) for a, v in zip(("spin_pump", "repump", "raman_scatter", "lattice_scatter"), count[-1])},
                 photons_total=float(count[-1].sum()), net_quanta_removed=float(nbar[0] - nbar[-1]),
                 net_quanta_removed_per_photon=None, mean_cooling_rate_quanta_per_s=float((nbar[0] - nbar[-1]) / max(ts[-1], 1e-300)),
                 target_state_scattering_rate_per_s=None, target_state_scattering_by_beam_per_s={},
                 time_to_spin_threshold_s=_threshold_time(ts, series["P_up"], cfg.analysis.spin_threshold),
                 time_to_joint_threshold_s=_threshold_time(ts, series["P_target"], cfg.analysis.joint_threshold),
                 spin_threshold=cfg.analysis.spin_threshold, joint_threshold=cfg.analysis.joint_threshold,
                 spin_populations_initial={at.ground_label(i): float(spin[0, i]) for i in range(12)},
                 spin_populations_final={at.ground_label(i): float(spin[-1, i]) for i in range(12)},
                 spin_populations_absolute_final={at.ground_label(i): float(P[-1, i].sum()) for i in range(12)},
                 max_conservation_error=float(np.max(np.abs(survival + loss - 1))), min_eigenvalue=float(mins.min()),
                 max_boundary_population=None, final_overflow=float(loss[-1]), lattice_loss=float(loss[-1]),
                 well_loss_fraction_no_recapture=float(loss[-1]), loss_label="Well-loss fraction, assuming no recapture",
                 initial_tail_discarded=float(np.mean([local.model.initial_tail for local, _ in components])),
                 initial_unbound_fraction=None, solver="coherent_3d_split_ensemble",
                 wall_time_s=sum(r["wall_time_s"] for _, r in components), rhs_evaluations=0)
    validity = Validity()
    validity.add("conservation", "ok" if final["max_conservation_error"] < cfg.numerics.conservation_tolerance else "invalid", f"max |survival + well loss - 1| = {final['max_conservation_error']:.3g}")
    validity.add("positivity", "ok" if final["min_eigenvalue"] > -cfg.numerics.positivity_tolerance else "invalid", f"minimum density eigenvalue / population = {final['min_eigenvalue']:.3g}")
    if final["initial_tail_discarded"] > cfg.numerics.boundary_tolerance:
        validity.add("initial distribution", "warning", f"Mean initial axial weight outside the local retained levels = {final['initial_tail_discarded']:.3g}; each node is conditioned on retained states. This is not a loading-survival estimate.")
    peak_exc = max(float(max((s.pump * local.model.rates["spin_pump"].p_exc + s.repump * local.model.rates["repump"].p_exc).max() for s in local.model.schedule)) for local, _ in components)
    status = "invalid" if peak_exc >= cfg.optical.weak_excitation_invalid else ("warning" if peak_exc >= cfg.optical.weak_excitation_warning else "ok")
    validity.add("weak excitation", status, f"maximum eliminated excited fraction across nodes and spin states = {peak_exc:.4g}")
    final["max_excited_fraction"] = peak_exc
    secular_ratio = 0.
    for local, _ in components:
        min_gap = min((np.diff(site.energies_hz).min() for site in local.sites if site.N > 1),
                      default=min(site.trap_hz for site in local.sites)) * 2 * np.pi
        for seg in local.model.schedule:
            photon_rate = (seg.pump * local.model.rates["spin_pump"].gamma_out
                           + seg.repump * local.model.rates["repump"].gamma_out).max()
            secular_ratio = max(secular_ratio, float(photon_rate / min_gap))
    final["max_motional_secular_ratio"] = secular_ratio
    validity.add("motional secular", "invalid" if secular_ratio >= .5 else ("warning" if secular_ratio >= .1 else "ok"),
                 f"max optical rate / smallest local motional angular spacing across all axes = {secular_ratio:.3g}")
    target_rate, target_weight = 0., 0.
    for local, r in components:
        rate = local.operators(local.model.schedule[-1])[1][at.UP, :, :, 0]
        weight = r["P_final_3d"][at.UP, :, :, 0]
        target_rate += np.sum(rate * weight)
        target_weight += weight.sum()
    final["target_state_scattering_rate_per_s"] = float(target_rate / target_weight) if target_weight > 0 else None
    final["P_3d_ground_absolute"] = float(sum(r["P_final_3d"][:, 0, 0, 0].sum() for _, r in components) / n)
    final["P_target_3d_absolute"] = float(sum(r["P_final_3d"][at.UP, 0, 0, 0] for _, r in components) / n)
    if calibration.get("carrier_decay_mode") == "modeled":
        validity.add("carrier calibration", "ok", f"Modeled spatial/thermal decay; RMS carrier frequency = {calibration['modeled_rms_carrier_hz']:.6g} Hz fixes initial curvature. No fitted homogeneous damping; reported decay is not a calibration constraint or an experimental validation.")
    else:
        validity.add("carrier calibration", "warning" if not calibration["calibration_compatible"] else "ok", f"Synthetic measured-envelope fit RMS = {calibration['synthetic_trace_rmse']:.3g}; residual gamma = {calibration['residual_coherence_decay_rate_s']:.3g}/s. A poor fit is an unresolved calibration-model mismatch, not a validated intensity estimate.")
    validity.add("three-dimensional recoil", "warning", "Joint spin/nx/ny/nz populations retained, but emission uses resolved spherical dipole channels with factorized Cartesian marginals. Interference between emitted spherical channels and correlations of one photon's Cartesian recoil components are neglected.")
    validity.add("recoil coherence", "warning", "Population-resolving scattering erases motional coherence and treats elastic spin amplitudes as distinguishable; this is an additional approximation beyond secular averaging.")
    validity.add("lattice", "warning", "Separable isolated wells with Gaussian local depths; no tunnelling, coherent Raman continuum excitation, envelope transport or recapture. Scalar trap basis with diagonal differential shifts; trap shape changes from spin-dependent shifts neglected.")
    validity.add("Stark shift approximation", "warning", "Raman/trap Stark operators use diagonal matrix elements; static off-diagonal motional dressing and trap-basis changes are omitted. Excited-state lattice shifts of the pumping transitions remain the configured common optical shift.")
    validity.add("spatial ensemble", "warning", f"{n} frozen Gaussian position / uniform registration samples. Refine samples and time_step_us for convergence; samples are numerical quadrature, not experimental error bars.")
    validity.add("temperature", "warning", "Axis temperatures are energy-matched bound-state mixture proxies; transverse survivor cooling may result from preferential loss.")
    final["calibration_compatible"] = calibration["calibration_compatible"]
    final["carrier_decay_mode"] = calibration.get("carrier_decay_mode", "measured_envelope")
    center_model = central.model
    # Reporting model holds the USER config and common fixed experimental beat.
    center_model.cfg = cfg
    center_model.nu_beat = beat
    summary = dict(N_motional=nz, eta_R=center_model.eta_R, nu_10_hz=center_model.nu_ref, nu_beat_hz=beat,
                   trap_potential="lattice", central_bottom_frequencies_hz={a: float(s.trap_hz) for a, s in zip("xyz", central.sites)},
                   central_bound_levels={a: s.N for a, s in zip("xyz", central.sites)},
                   calibration=calibration, lattice_optical_estimates=central.lattice_estimates,
                   selected_lattice_linear_polarizations=cfg.ensemble.lattice_linear_polarizations,
                   node_results=[dict(position_um=l.position.tolist(), phase_deg=float(np.rad2deg(l.phase) % 360),
                                      depths_uk=l.depths.tolist(), survival=float(r["P"][-1].sum()), target_absolute=float(r["P"][-1, at.UP, 0])) for l, r in components])
    result = RunResult(ts, P, loss, count, mins, coh, final["solver"], center_model.schedule, final["wall_time_s"],
                       notes=["3D recoil with Cartesian marginal approximation; absorbing local bound manifolds; Strang splitting"], rho_final=None)
    return dict(series=series, final=final, validity=validity, pn=pn, P_final=normalized[-1], segments=center_model.schedule,
                cooling_rate_method="Savitzky-Golay derivative of survivor mean axial energy / level index",
                model=center_model, model_summary=summary, result=result, ensemble_nodes=components,
                ensemble_details=dict(assumptions=[f"Initially loaded Gaussian cloud, conditional on local bound states; axial preparation mode {cfg.initial.motion_mode}, configured axial temperature {cfg.initial.temperature_k:g} K, transverse temperature {cfg.ensemble.transverse_temperature_k:g} K. Loading survival unspecified.",
                    "Uniform static Raman/lattice registration, not rapid temporal phase noise.",
                    "Transverse quantum numbers are population resolved; axial Raman pair remains coherent.",
                    "Emission Cartesian components use independent dipole marginals; recoil coherence transfer omitted.",
                    "ARC D1+D2 scattering includes fine/hyperfine interference and coherent same-tone beam absorption.",
                    "D-line diagonal trap shifts are estimates; higher excited states and hyperfine-mediated trap polarizability omitted.",
                    "Fixed beat shared by all spatial samples; Gaussian local depths and pump/Raman intensities.",
                    "Loss on leaving any local bound manifold; no recapture. Temperature is a survivor-conditioned proxy."],
                    calibration=calibration))
