"""Event-driven 3D optical-pumping trajectories with motional coherence.

Raman light must be off. A trajectory then has a resolved spin and three
motional wavefunctions: free evolution, diagonal Stark shifts, traveling-wave
pump jumps, and single-axis standing-wave lattice jumps preserve this product
form. Averaging trajectories retains spin/axis correlations; the ensemble
density matrix is NOT assumed to factorize.

No-jump propagation uses full bound-space absorption hazards, including their
off-diagonal matrix elements. Bound photon jumps use the combined absorption
and emission displacement. Missing jump norm is absorbing escape. Proposal
times use Poisson thinning with a spectral bound on the instantaneous hazard.
There is no integration timestep or population projection of motional recoil.

Spin and spherical emission channels remain resolved. The optical elimination,
fixed separable wells, diagonal Stark model, and no-recapture assumptions remain.
Framework: https://arxiv.org/abs/1112.2806 and QuTiP's Monte Carlo solver guide.
"""
from dataclasses import dataclass
import time

import numpy as np

from . import atomic as at, motion as mo, polarization as pol
from .correlated_recoil import emission_sphere
from .dynamics import SimulationStopped, _single_thread
from .far_detuned import DLineResponse
from .protocols import raman_is_active


def _check_stop(stop):
    if stop is not None and stop.is_set():
        raise SimulationStopped()


@dataclass
class PhotonGroup:
    rates: np.ndarray             # initial spin, final spin, sigma/pi
    kicks: tuple                 # axis -> [direction, final n, initial n]
    weights: np.ndarray          # sigma/pi, direction
    absorption_axis: int | None  # None for traveling waves with unit full norm
    full_absorption: np.ndarray | None
    counter: int                 # pump, repump, Raman, lattice
    scale_name: str


def _overlaps(site, momenta, stop):
    etas, inverse = np.unique(momenta*site.x0, return_inverse=True)
    values = []
    for start in range(0, len(etas), 64):
        _check_stop(stop)
        values.append(site.displacements(etas[start:start+64]))
    return np.concatenate(values)[inverse]


def validate_config(cfg):
    """Reject physics outside the separable-trajectory model before setup."""
    cfg.validate()
    if not cfg.ensemble.enabled or cfg.trap.potential != 'lattice':
        raise ValueError('coherent 3D pumping requires the lattice ensemble configuration')
    if raman_is_active(cfg):
        raise ValueError('coherent 3D pumping requires Raman off throughout the schedule')
    if cfg.trap.heating_rate_quanta_per_s != 0:
        raise ValueError('coherent 3D pumping requires zero background heating')


def prepare_local(cfg, position_um=(0., 0., 0.), stop=None):
    """Share production atomic/site inputs; construct coherent photon jumps."""
    from .ensemble import LocalExperiment
    validate_config(cfg)
    _check_stop(stop)
    with _single_thread():
        atom = at.load_atomic_data()
        response = DLineResponse(atom, cfg.magnetic.magnitude_gauss, cfg.magnetic.direction, cfg.optical.zeeman_model)
        calibration = dict(intensity_per_unit_tone_w_m2=0., residual_coherence_decay_rate_s=0.)
        local = LocalExperiment(cfg, response, calibration, position_um, 0., stop=stop, population_recoil=False)
        sites, m = local.sites, local.model
        b = local.axes_lab @ pol.unit(cfg.magnetic.direction)
        directions, sigma = emission_sphere(b, cfg.numerics.recoil_quadrature_points, 'sigma')
        _, pi = emission_sphere(b, cfg.numerics.recoil_quadrature_points, 'pi')
        weights = np.array([sigma, pi])
        pump_weights = weights
        if cfg.optical.emission_pattern == 'isotropic':
            _, iso = emission_sphere(b, cfg.numerics.recoil_quadrature_points, 'isotropic')
            pump_weights = np.array([iso, iso])
        groups = []
        hamiltonians = [np.tile(2*np.pi*(site.energies_hz-site.energies_hz[0]), (12, 1)) for site in sites]
        for counter, name in enumerate(('spin_pump', 'repump')):
            kabs = local.axes_lab @ (m.k_d2*m.beams[name].direction)
            kicks = tuple(_overlaps(site, kabs[axis]-m.k_d2*directions[:, axis], stop)
                          for axis, site in enumerate(sites))
            groups.append(PhotonGroup(m.rates[name].W, kicks, pump_weights, None, None,
                                      counter, 'pump' if counter == 0 else 'repump'))
        for axis, spec in enumerate(local.lattices):
            freq = at.C_LIGHT/(spec.wavelength_nm*1e-9)
            if axis == 1:
                freq += cfg.ensemble.transverse_frequency_offset_hz
            k = 2*np.pi*freq/at.C_LIGHT
            ep = pol.unit(cfg.ensemble.lattice_linear_polarizations[axis])
            peak = local.lattice_estimates[axis]['peak_intensity_w_m2']
            amps = response.scattering_amplitudes(freq, ep)*np.sqrt(peak/4)
            rates = np.stack([abs(amps[:, :, 0])**2+abs(amps[:, :, 2])**2, abs(amps[:, :, 1])**2], axis=-1)
            kicks = []
            for j, site in enumerate(sites):
                emission = -k*directions[:, j]
                kick = (_overlaps(site, k+emission, stop)+_overlaps(site, -k+emission, stop)
                        if j == axis else _overlaps(site, emission, stop))
                kicks.append(kick)
            site = sites[axis]
            d = site.displacement(2*k*site.x0)
            full = 2*np.eye(site.N)+d+d.conj().T  # P_bound |2 cos(k x)|^2 P_bound
            groups.append(PhotonGroup(rates, tuple(kicks), weights, axis, full, 3, 'always'))
            shift = response.shift_hz(freq, ep)
            cosine = (1+np.diag(d).real)/2
            hamiltonians[axis] += 2*np.pi*(shift-shift.mean())[:, None]*peak*cosine
        return local, hamiltonians, groups


class SegmentEvolution:
    """Separable conditional evolution and a rigorous photon-rate bound."""
    def __init__(self, hamiltonians, groups, scales, mode):
        self.groups, self.mode = groups, mode
        self.rates = [g.rates*scales[g.scale_name] for g in groups]
        self.out = np.array([r.sum(axis=(1, 2)) for r in self.rates])
        self.cdf = []
        for r, total in zip(self.rates, self.out):
            cdf = np.cumsum(r.reshape(len(r), -1), axis=1)
            np.divide(cdf, total[:, None], out=cdf, where=total[:, None] > 0)
            cdf[total > 0, -1] = 1.
            self.cdf.append(cdf)
        spins = len(hamiltonians[0])
        self.bound = np.zeros(spins)
        damping = [np.zeros((spins, h.shape[1], h.shape[1]), complex) for h in hamiltonians]
        for group, outgoing in zip(groups, self.out):
            if group.absorption_axis is None:
                self.bound += outgoing
            else:
                axis, full = group.absorption_axis, group.full_absorption
                if not np.allclose(full, full.conj().T) or np.linalg.eigvalsh(full).min() < -1e-10:
                    raise ValueError('full absorption hazard must be positive Hermitian')
                damping[axis] += outgoing[:, None, None]*full
        self.eigen = []
        for h, gamma in zip(hamiltonians, damping):
            self.bound += np.linalg.eigvalsh(gamma).max(axis=1)
            eigensystems = []
            for s in range(spins):
                generator = -1j*np.diag(h[s])-.5*gamma[s]
                values, vectors = np.linalg.eig(generator)
                if np.linalg.cond(vectors) > 1e8:
                    raise ValueError('ill-conditioned no-jump eigensystem; unsupported parameters')
                eigensystems.append((values, vectors, np.linalg.inv(vectors)))
            self.eigen.append(eigensystems)
        self.bound *= 1+1e-12  # protect the spectral bound against roundoff

    def propagate(self, spin, vectors, times):
        if self.mode == 'population':
            return [v.copy() for v in vectors]
        result = []
        for values, systems in zip(vectors, self.eigen):
            rates, eigenvectors, inverse = systems[spin]
            # A scalar real shift cancels on normalization and prevents
            # underflow in long photon-free intervals.
            rates = rates-np.max(rates.real)
            moved = ((values @ inverse.T)*np.exp(np.asarray(times)[:, None]*rates)) @ eigenvectors.T
            norms = np.linalg.norm(moved, axis=1)
            if np.any(norms == 0) or not np.all(np.isfinite(norms)):
                raise ValueError('nonfinite conditional wavefunction')
            result.append(moved/norms[:, None])
        return result

    def group_hazards(self, spin, vectors):
        values = np.broadcast_to(self.out[:, spin], (len(vectors[0]), len(self.groups))).copy()
        factors = np.ones_like(values)
        for g, group in enumerate(self.groups):
            if group.absorption_axis is not None:
                v = vectors[group.absorption_axis]
                factors[:, g] = np.einsum('bi,ij,bj->b', v.conj(), group.full_absorption, v).real
                values[:, g] *= factors[:, g]
        if values.min() < -1e-9 or np.any(values.sum(axis=1) > self.bound[spin]*(1+1e-9)+1e-12):
            raise ValueError('invalid thinning hazard or spectral bound')
        return np.maximum(values, 0), factors


def simulate(hamiltonians, groups, initial, schedule, times, trajectories, seed,
             *, mode='coherent', target_spin=11, target_axis=2, levels_hz=None, stop=None, progress=None):
    """Average independent trajectories; standard errors are Monte Carlo only.

    Population mode collapses motion after each jump and has no conditional
    motional evolution. It reproduces the classical population generator and
    provides an independent control for rates, recoil, counters and escape.
    """
    if mode not in ('coherent', 'population'):
        raise ValueError('trajectory mode must be coherent or population')
    if isinstance(trajectories, bool) or not isinstance(trajectories, (int, np.integer)) or trajectories < 2:
        raise ValueError('at least two integer trajectories are required')
    times = np.asarray(times, float)
    if (len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0)
            or abs(times[-1]-schedule[-1].t1) > 1e-13):
        raise ValueError('sample times must increase from zero to the schedule end')
    if (len(hamiltonians) != 3 or initial.ndim != 4 or not np.isclose(initial.sum(), 1)
            or np.any(initial < 0) or not np.all(np.isfinite(initial))):
        raise ValueError('initial populations must be normalized spin/nx/ny/nz probabilities')
    if not groups:
        raise ValueError('at least one photon group is required (zero rates allowed)')
    for group in groups:
        if np.any(group.rates < 0) or not np.all(np.isfinite(group.rates)):
            raise ValueError('photon rates must be finite and nonnegative')
        if np.any(group.weights < 0) or not np.allclose(group.weights.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError('angular weights must be normalized and nonnegative')
    rng = np.random.default_rng(seed)
    n, shape = trajectories, initial.shape
    sampled = np.array(np.unravel_index(rng.choice(initial.size, n, p=initial.ravel()), shape))
    spin = sampled[0]
    psi = [np.eye(size, dtype=complex)[sampled[a+1]] for a, size in enumerate(shape[1:])]
    alive = np.ones(n, bool)
    counts = np.zeros((n, 4))
    last = np.zeros(n)
    angular_cdf = [np.cumsum(g.weights/g.weights.sum(axis=1)[:, None], axis=1) for g in groups]
    for cdf in angular_cdf:
        cdf[:, -1] = 1.
    means, errors = {}, {}
    recorder = None
    if levels_hz is not None:
        from .trajectory_observables import ObservableRecorder
        recorder = ObservableRecorder(n, times, levels_hz, target_axis)
    proposed = accepted = 0
    started = time.perf_counter()

    def record(index, evolution, t):
        ids = np.flatnonzero(alive)
        current = [np.zeros_like(v) for v in psi]
        for s in np.unique(spin[ids]):
            take = ids[spin[ids] == s]
            evolved = evolution.propagate(s, [v[take] for v in psi], t-last[take])
            for axis in range(3):
                current[axis][take] = evolved[axis]
        p0 = np.array([abs(v[:, 0])**2 for v in current])
        selected = (spin == target_spin)*alive
        values = dict(survival=alive.astype(float), loss=(~alive).astype(float),
                      P_up_absolute=selected.astype(float), P_n0_absolute=p0[target_axis],
                      P_target_absolute=selected*p0[target_axis], P_3d_ground_absolute=p0.prod(axis=0),
                      P_target_3d_absolute=selected*p0.prod(axis=0),
                      photons_spin_pump=counts[:, 0], photons_repump=counts[:, 1],
                      photons_lattice_scatter=counts[:, 3], photons_total=counts.sum(axis=1))
        for key, value in values.items():
            means.setdefault(key, np.zeros(len(times)))[index] = value.mean()
            errors.setdefault(key, np.zeros(len(times)))[index] = value.std(ddof=1)/np.sqrt(n)
        if recorder is not None:
            recorder.record(index, spin, current, alive, shape[0])
        return current

    sample_index = 0
    current = psi
    for segment in schedule:
        _check_stop(stop)
        scales = dict(pump=segment.pump, repump=segment.repump, always=1.)
        evolution = SegmentEvolution(hamiltonians, groups, scales, mode)
        ids = np.flatnonzero(alive)
        last[ids] = segment.t0
        next_event = np.full(n, np.inf)
        positive = ids[evolution.bound[spin[ids]] > 0]
        next_event[positive] = segment.t0+rng.exponential(1/evolution.bound[spin[positive]])
        targets = list(times[(times >= segment.t0) & (times <= segment.t1+1e-14)])
        if not targets or targets[-1] < segment.t1-1e-14:
            targets.append(segment.t1)
        for target in targets:
            while True:
                _check_stop(stop)
                due = np.flatnonzero(alive & (next_event <= target))
                if not len(due):
                    break
                proposed += len(due)
                # Freeze membership before any jumps change spin: otherwise a
                # trajectory could be processed twice in this batch.
                due_spin = spin[due].copy()
                for s in np.unique(due_spin):
                    take = due[due_spin == s]
                    event_time = next_event[take].copy()
                    vectors = evolution.propagate(s, [v[take] for v in psi], event_time-last[take])
                    for a in range(3):
                        psi[a][take] = vectors[a]
                    last[take] = event_time
                    hazards, factors = evolution.group_hazards(s, vectors)
                    chosen = np.sum(rng.random(len(take))[:, None]*evolution.bound[s] >= np.cumsum(hazards, axis=1), axis=1)
                    for g, group in enumerate(groups):
                        selected = np.flatnonzero(chosen == g)
                        if not len(selected):
                            continue
                        hit = take[selected]
                        accepted += len(hit)
                        counts[hit, group.counter] += 1  # includes photons causing escape
                        branch = np.searchsorted(evolution.cdf[g][s], rng.random(len(hit)), side='right')
                        final_spin, pattern = branch//2, branch % 2
                        photon = np.empty(len(hit), int)
                        for channel in (0, 1):
                            subset = np.flatnonzero(pattern == channel)
                            photon[subset] = np.searchsorted(angular_cdf[g][channel], rng.random(len(subset)), side='right')
                        moved = [np.einsum('bij,bj->bi', kick[photon], v[hit]) for kick, v in zip(group.kicks, psi)]
                        norm = [np.sum(abs(v)**2, axis=1) for v in moved]
                        retained = np.prod(norm, axis=0)/factors[selected, g]
                        if retained.min() < -1e-10 or retained.max() > 1+1e-8:
                            raise ValueError('bound photon jump is not a contraction of the full absorption rate')
                        keep = rng.random(len(hit)) < np.clip(retained, 0, 1)
                        alive[hit[~keep]] = False
                        spin[hit[keep]] = final_spin[keep]
                        for a in range(3):
                            normalized = moved[a][keep]/np.sqrt(norm[a][keep, None])
                            if mode == 'population':
                                cdf = np.cumsum(abs(normalized)**2, axis=1)
                                cdf[:, -1] = 1.
                                state = np.sum(rng.random(len(normalized))[:, None] >= cdf, axis=1)
                                normalized = np.eye(shape[a+1], dtype=complex)[state]
                            psi[a][hit[keep]] = normalized
                    next_event[take] = np.inf
                    live = take[alive[take] & (evolution.bound[spin[take]] > 0)]
                    next_event[live] = last[live]+rng.exponential(1/evolution.bound[spin[live]])
            if sample_index < len(times) and abs(target-times[sample_index]) < 1e-13:
                current = record(sample_index, evolution, target)
                sample_index += 1
            if abs(target-segment.t1) < 1e-13:
                # Preserve the conditional state at a schedule boundary,
                # including boundaries that are not output times.
                ids = np.flatnonzero(alive)
                for s in np.unique(spin[ids]):
                    take = ids[spin[ids] == s]
                    values = evolution.propagate(s, [v[take] for v in psi], segment.t1-last[take])
                    for a in range(3):
                        psi[a][take] = values[a]
                last[ids] = segment.t1
            if progress:
                progress(target/times[-1])
    densities = []
    for vectors in current:
        rho = np.zeros((shape[0], vectors.shape[1], vectors.shape[1]), complex)
        for s in range(shape[0]):
            v = vectors[alive & (spin == s)]
            rho[s] = v.T @ v.conj()/n
        densities.append(rho)
    return dict(t_s=times, mean=means, standard_error=errors, rho_axes_final=densities,
                **({'observables': recorder.result()} if recorder is not None else {}),
                trajectories=n, mode=mode, proposed_events=proposed, photon_events=accepted,
                max_conservation_error=float(np.max(abs(means['survival']+means['loss']-1))),
                wall_time_s=time.perf_counter()-started)


def run_local(cfg, position_um=(0., 0., 0.), *, trajectories=4096, seed=0,
              mode='coherent', stop=None, progress=None):
    with _single_thread():
        local, hamiltonians, groups = prepare_local(cfg, position_um, stop)
        ts = np.linspace(0, local.model.schedule[-1].t1, cfg.timing.n_samples)
        result = simulate(hamiltonians, groups, local.P0, local.model.schedule, ts, trajectories, seed,
                          mode=mode, target_spin=local.model.up, target_axis=local.raman_axis,
                          levels_hz=[s.energies_hz for s in local.sites],
                          stop=stop, progress=progress)
        result['position_um'] = list(position_um)
        result['bound_levels'] = [s.N for s in local.sites]
        result['depths_uk'] = local.depths.tolist()
        return result


def average_nodes(nodes, mode):
    """Equal spatial weights; independent trajectory errors add in variance."""
    results = [node['modes'][mode] for node in nodes]
    n = len(results)
    means = {k: sum(r['mean'][k] for r in results)/n for k in results[0]['mean']}
    errors = {k: np.sqrt(sum(r['standard_error'][k]**2 for r in results))/n for k in means}
    output = dict(t_s=results[0]['t_s'], mean=means, standard_error=errors)
    if 'observables' in results[0]:
        from .trajectory_observables import average_observables
        output['observables'] = average_observables(results)
    return output


_WORKER_STOP = None


def _initialize_worker(stop):
    global _WORKER_STOP
    _WORKER_STOP = stop


def _cloud_node(cfg, index, position, trajectories, seed, modes, stop=None):
    stop = _WORKER_STOP if stop is None else stop
    with _single_thread():
        local, h, groups = prepare_local(cfg, position, stop)
        ts = np.linspace(0, local.model.schedule[-1].t1, cfg.timing.n_samples)
        output = dict(index=index, position_um=list(position), bound_levels=[s.N for s in local.sites],
                      depths_uk=local.depths.tolist(), initial_tail_discarded=local.initial_tail,
                      energies_hz=[s.energies_hz.tolist() for s in local.sites], modes={})
        # Evaluate the remaining validity criteria from the actual local
        # inputs, without constructing or running the population backend.
        bounds = np.array([g.rates.sum(axis=(1, 2))*(1. if g.absorption_axis is None
                           else np.linalg.eigvalsh(g.full_absorption).max()) for g in groups])
        photon_bound, excited = 0., 0.
        for seg in local.model.schedule:
            scales = dict(pump=seg.pump, repump=seg.repump, always=1.)
            photon_bound = max(photon_bound, float(sum(b*scales[g.scale_name] for b, g in zip(bounds, groups)).max()))
            excited = max(excited, float((seg.pump*local.model.rates['spin_pump'].p_exc
                                          +seg.repump*local.model.rates['repump'].p_exc).max()))
        atom, m = local.model.atom, local.model
        spacing = (at.minimum_ground_zeeman_spacing_hz(atom, m.B) if cfg.optical.zeeman_model == 'diagonalized'
                   else min(abs(atom.g_ground[f]) for f in (2, 3))*at.MU_B*abs(m.B)/at.H)*2*np.pi
        output['diagnostics'] = dict(max_photon_rate_bound_s=photon_bound, max_excited_fraction=excited,
                                     ground_angular_spacing_s=float(spacing),
                                     magnetic=at.zeeman_validity(atom, m.B, m.nu_t, m.up, m.down, cfg.optical.zeeman_model))
        for mode in modes:
            # Stable across worker counts, completion order and requested modes.
            entropy = [seed, index, 0 if mode == 'coherent' else 1]
            r = simulate(h, groups, local.P0, local.model.schedule, ts, trajectories,
                         np.random.SeedSequence(entropy), mode=mode, target_spin=local.model.up,
                         target_axis=local.raman_axis, levels_hz=[s.energies_hz for s in local.sites], stop=stop)
            r['seed_entropy'] = entropy
            output['modes'][mode] = r
        return output


def run_cloud(cfg, *, trajectories=4096, seed=1729, workers=None,
              modes=('coherent', 'population'), stop=None, progress=None):
    """Run the configured Sobol cloud with independent trajectories per site.

    ``seed`` controls Monte Carlo only; ``cfg.numerics.seed`` controls positions.
    All modes share correlated 3D recoil, regardless of ensemble.recoil_model.
    The input config is never changed. Output errors exclude spatial quadrature
    and shared model uncertainty. Use the comparison CLI for validity reporting.
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    import multiprocessing
    from .far_detuned import spatial_samples
    validate_config(cfg)
    count = cfg.ensemble.workers if workers is None else workers
    for name, value, minimum in (('workers', count, 1), ('trajectories', trajectories, 2), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not modes or len(set(modes)) != len(modes) or any(m not in ('coherent', 'population') for m in modes):
        raise ValueError('modes must be a nonempty unique subset of coherent, population')
    _check_stop(stop)
    positions, _ = spatial_samples(cfg)
    started = time.perf_counter()
    nodes = {}

    def collect(node):
        nodes[node['index']] = node
        if progress:
            progress(len(nodes)/len(positions), f"trajectory sites {len(nodes)}/{len(positions)} complete")

    if count == 1:
        for i, position in enumerate(positions):
            collect(_cloud_node(cfg, i, position, trajectories, seed, modes, stop))
    else:
        context = multiprocessing.get_context('spawn')
        process_stop = context.Event()
        with ProcessPoolExecutor(max_workers=min(count, len(positions)), mp_context=context,
                                 initializer=_initialize_worker, initargs=(process_stop,)) as pool:
            futures = []
            try:
                futures = [pool.submit(_cloud_node, cfg, i, position, trajectories, seed, modes)
                           for i, position in enumerate(positions)]
                pending = set(futures)
                while pending:
                    _check_stop(stop)
                    completed, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                    for future in completed:
                        collect(future.result())
            except BaseException:
                process_stop.set()
                for future in futures:
                    future.cancel()
                raise
    _check_stop(stop)
    ordered = [nodes[i] for i in range(len(positions))]
    return dict(nodes=ordered, modes={mode: average_nodes(ordered, mode) for mode in modes},
                trajectories_per_site=trajectories, sites=len(positions), trajectory_seed=seed,
                spatial_seed=cfg.numerics.seed, workers=count, wall_time_s=time.perf_counter()-started,
                recoil_model='correlated', solver='coherent_3d_pumping_trajectories')
