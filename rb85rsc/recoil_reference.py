"""Small one-well optical-pumping benchmark retaining motional coherences.

For each resolved initial/final spin channel and photon direction the bound
jump is sqrt(W*w) |g'><g| tensor P_bound exp(i*q*x) P_bound. All motional
matrix elements belong to ONE jump, rather than separate |n'><n| jumps.
The outgoing term is -Gamma_g*rho_g, including escape to an absorbing sink;
using only D^dagger D in this term would incorrectly prevent well escape.

This isolates the motional-population approximation. Spin coherences remain
secularized, pump beams are incoherent, and emitted spherical channels remain
resolved as in optical.py. No Raman, lattice scattering, transverse recoil,
recapture, or excited-state mixing is included. It is not a 3D replacement.
Effective-jump framework: https://arxiv.org/abs/1112.2806.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from . import atomic as at, motion as mo, polarization as pol
from .dynamics import _single_thread
from .far_detuned import envelope, spatial_samples
from .model import Model
from .optical import beam_spec
from .protocols import raman_is_active


@dataclass
class RecoilChannel:
    rates: np.ndarray  # initial spin, final spin; s^-1
    kicks: np.ndarray  # direction, final n, initial n; contractions on bound space
    weights: np.ndarray  # normalized directional probability


def validation_positions(cfg, cloud_samples=0, position_um=(0., 0., 0.)):
    """An explicit reference position plus the production Sobol cloud prefix.

    Positions are independent local checks, not weights for a 3D prediction.
    The reference is kept separately even if a zero-radius cloud duplicates it.
    """
    position = np.asarray(position_um, float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_um must be a finite lab 3-vector")
    if not isinstance(cloud_samples, (int, np.integer)) or cloud_samples < 0:
        raise ValueError("cloud_samples must be a nonnegative integer")
    positions = [dict(site_id='reference', position_um=position.tolist())]
    if cloud_samples:
        xyz, _ = spatial_samples(cfg, cloud_samples)
        positions.extend(dict(site_id=f'cloud_{i}', position_um=p.tolist()) for i, p in enumerate(xyz))
    return positions


def generators(energies_hz, channels):
    """Return coherent and population generators with identical photon physics.

    Density blocks are vectorized column-major, then concatenated over spin.
    Return outgoing spin rates too, for separately integrated photon counts.
    Each kick must be a contraction; its missing norm goes to the sink.
    """
    n = len(energies_hz)
    if not channels:
        raise ValueError("at least one recoil channel is required")
    spins = channels[0].rates.shape[0]
    omega = 2*np.pi*np.asarray(energies_hz)
    phase = (-1j*(omega[:, None] - omega[None, :])).ravel(order='F')
    coherent = sparse.kron(sparse.eye(spins), sparse.diags(phase), format='csr')
    population = sparse.csr_matrix((spins*n, spins*n))
    outgoing = np.zeros(spins)
    for channel in channels:
        W, kicks, weights = channel.rates, channel.kicks, channel.weights
        if W.shape != (spins, spins) or np.any(W < 0) or not np.all(np.isfinite(W)):
            raise ValueError("rates must be a finite nonnegative square matrix")
        if (kicks.shape != (len(weights), n, n) or not np.all(np.isfinite(kicks))
                or np.any(weights < 0) or not np.all(np.isfinite(weights))
                or not np.isclose(np.sum(weights), 1., rtol=0, atol=1e-12)):
            raise ValueError("kicks need finite matrices and normalized nonnegative angular weights")
        if any(np.linalg.norm(d, 2) > 1 + 1e-10 for d in kicks):
            raise ValueError("bound-space kicks must be contractions")
        # Independent population construction: square matrix elements, whereas
        # the reference retains their complex cross products D rho D^dagger.
        kernel = np.einsum('u,unm->nm', weights, np.abs(kicks)**2)
        superoperator = sum(w*np.kron(d.conj(), d) for w, d in zip(weights, kicks))
        coherent += sparse.kron(sparse.csr_matrix(W.T), sparse.csr_matrix(superoperator), format='csr')
        population += sparse.kron(sparse.csr_matrix(W.T), sparse.csr_matrix(kernel), format='csr')
        outgoing += W.sum(axis=1)
    coherent -= sparse.diags(np.repeat(outgoing, n*n))
    population -= sparse.diags(np.repeat(outgoing, n))
    return coherent.tocsr(), population.tocsr(), outgoing


def _augment(generator, trace, counters):
    """Absorbing sink and integrated event counts, including photons causing loss."""
    rewards = np.vstack([-np.asarray(trace @ generator).ravel(), *counters])
    extra = rewards.shape[0]
    return sparse.bmat([[generator, sparse.csr_matrix((generator.shape[0], extra))],
                        [sparse.csr_matrix(rewards), sparse.csr_matrix((extra, extra))]], format='csr')


def _site_model(cfg, axis, position_um):
    cfg.validate()
    if raman_is_active(cfg):
        raise ValueError("recoil reference requires a schedule with Raman off")
    if cfg.trap.potential != 'lattice' or cfg.trap.heating_rate_quanta_per_s != 0:
        raise ValueError("recoil reference requires an isolated lattice well and zero background heating")
    if axis not in 'xyz' or len(axis) != 1:
        raise ValueError("axis must be x, y, or z")
    position = np.asarray(position_um, float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_um must be a finite lab 3-vector")
    atom = at.load_atomic_data()
    ia = 'xyz'.index(axis)
    axes = pol.lattice_frame(cfg.trap.axis, cfg.ensemble.transverse_reference_direction)
    spec = cfg.lattice_parameters(ia, atom.mass_kg)
    depth = spec.depth_uk * envelope(position, axes[ia], spec.waist_um)
    nu = mo.lattice_frequency_hz(atom.mass_kg, spec.wavelength_nm, depth)
    site = mo.lattice_site(atom.mass_kg, nu, depth*1e-6*at.K_B/at.H, cfg.trap.n_max+1)
    if site.N != site.n_bound:
        raise ValueError("reference requires all local bound levels; increase trap.n_max")
    if not 2 <= site.N <= 16:
        raise ValueError("reference requires 2 to 16 bound levels; adjust the local well depth")
    c = cfg.copy()
    c.ensemble.enabled = False
    c.raman.calibration = 'pathway'
    c.raman.enabled = False
    if ia != 2:
        c.initial = replace(cfg.initial, motion_mode='thermal_temperature',
                            temperature_k=cfg.ensemble.transverse_temperature_k)
    for name in ('spin_pump', 'repump'):
        pc = getattr(c, name)
        direction = beam_spec(name, pc, atom, c.magnetic.direction).direction
        factor = envelope(position, direction, pc.waist_um)
        pc.power_mw *= factor
        pc.peak_intensity_w_m2 *= factor
    return Model(c, atom, motion_axis=axes[ia], motion_basis=site)


def compare_recoil(cfg, *, axis='z', position_um=(0., 0., 0.)):
    """Compare motion-coherent and population kinetics for the same one-site input."""
    with _single_thread():
        return _compare_recoil(cfg, axis, position_um)


def _compare_recoil(cfg, axis, position_um):
    m = _site_model(cfg, axis, position_um)
    n, spins = m.N, at.N_GROUND
    directions, quadrature = mo.gauss_legendre(cfg.numerics.recoil_quadrature_points)
    parts = {}
    for name in ('spin_pump', 'repump'):
        projection = np.dot(m.beams[name].direction, m.u)
        kicks = np.array([m.motion.displacement(m.eta_d2*(projection-u)) for u in directions])
        channels = []
        for ci, pattern in enumerate(('sigma', 'pi')):
            pattern = pattern if cfg.optical.emission_pattern == 'dipole' else 'isotropic'
            weights = quadrature * mo.emission_weight(directions, m.cos_beta, pattern)
            channels.append(RecoilChannel(m.rates[name].W[:, :, ci], kicks, weights))
        parts[name] = generators(m.motion.energies_hz, channels)
    zero_channels = [RecoilChannel(np.zeros((spins, spins)), np.eye(n)[None], np.ones(1))]
    free, free_pop, _ = generators(m.motion.energies_hz, zero_channels)
    trace = np.tile(np.eye(n).ravel(order='F'), spins)
    trace_pop = np.ones(spins*n)
    rho = np.array([np.diag(p) for p in m.P0], complex)
    y = np.r_[rho.transpose(0, 2, 1).ravel(), np.zeros(3)]
    p = np.r_[m.P0.ravel(), np.zeros(3)]
    ts = np.linspace(0., m.schedule[-1].t1, cfg.timing.n_samples)
    histories, populations = [y.copy()], [p.copy()]
    index = 1
    for seg in m.schedule:
        G, Q = free.copy(), free_pop.copy()
        counters, counters_pop = [], []
        for name, scale in (('spin_pump', seg.pump), ('repump', seg.repump)):
            gc, gp, gout = parts[name]
            G += scale*(gc-free)
            Q += scale*gp
            counters.append(np.repeat(scale*gout, n*n)*trace)
            counters_pop.append(np.repeat(scale*gout, n))
        A, B = _augment(G, trace, counters), _augment(Q, trace_pop, counters_pop)
        now = seg.t0
        while now < seg.t1 - 1e-14:
            end = min(seg.t1, ts[index] if index < len(ts) else seg.t1)
            dt = end-now
            y = expm_multiply(A*dt, y)
            p = expm_multiply(B*dt, p)
            now = end
            if index < len(ts) and abs(now-ts[index]) < 1e-13:
                histories.append(y.copy())
                populations.append(p.copy())
                index += 1
    histories, populations = np.asarray(histories), np.asarray(populations)
    densities = histories[:, :-3].reshape(len(ts), spins, n, n).swapaxes(-1, -2)
    diagonal = np.diagonal(densities, axis1=-2, axis2=-1).real
    offdiagonal = densities.copy()
    indices = np.arange(n)
    offdiagonal[..., indices, indices] = 0.
    energies = m.motion.energies_hz - m.motion.energies_hz[0]

    def observables(P, extra):
        survival = P.sum(axis=(1, 2))
        denominator = np.where(survival > 0, survival, np.nan)
        return dict(P_absolute=P, survival=survival, well_escape_no_recapture=extra[:, 0].real,
                    photons_spin_pump=extra[:, 1].real, photons_repump=extra[:, 2].real,
                    P_up_absolute=P[:, m.up].sum(axis=1), P_target_absolute=P[:, m.up, 0],
                    P_n0_absolute=P[:, :, 0].sum(axis=1),
                    mean_excitation_hz=(P.sum(axis=1) @ energies)/denominator,
                    mean_excitation_uk=(P.sum(axis=1) @ energies)/denominator*at.H/at.K_B*1e6,
                    nbar=(P.sum(axis=1) @ np.arange(n))/denominator,
                    max_conservation_error=float(np.max(np.abs(survival+extra[:, 0].real-1))))

    coherent = observables(diagonal, histories[:, -3:])
    population = observables(populations[:, :-3].real.reshape(len(ts), spins, n), populations[:, -3:])
    coherent['motional_coherence_l1'] = np.abs(offdiagonal).sum(axis=(1, 2, 3))
    coherent['min_eigenvalue'] = float(np.linalg.eigvalsh(densities).min())
    population['min_population'] = float(population['P_absolute'].min())
    coherent['rho_final'] = densities[-1]
    validity = [dict(category=c, status=s, message=msg,
                     applies_to='population' if c == 'motional secular' else 'both')
                for c, s, msg in m.static_validity().items
                if c in ('magnetic field', 'ground coherences', 'motional secular')]
    peak_exc = max(float((seg.pump*m.rates['spin_pump'].p_exc + seg.repump*m.rates['repump'].p_exc).max())
                   for seg in m.schedule)
    status = ('invalid' if peak_exc >= cfg.optical.weak_excitation_invalid else
              'warning' if peak_exc >= cfg.optical.weak_excitation_warning else 'ok')
    validity.append(dict(category='weak excitation', status=status, applies_to='both',
                         message=f'maximum eliminated excited fraction = {peak_exc:.6g}'))
    differences = {}
    for key in ('survival', 'P_up_absolute', 'P_n0_absolute', 'P_target_absolute',
                'mean_excitation_uk', 'photons_spin_pump', 'photons_repump'):
        delta = coherent[key] - population[key]
        differences[key] = dict(final_signed=float(delta[-1]), max_absolute=float(np.max(np.abs(delta))))
    tv = .5*(np.abs(diagonal-population['P_absolute']).sum(axis=(1, 2))
             + np.abs(coherent['well_escape_no_recapture']-population['well_escape_no_recapture']))
    return dict(t_s=ts, axis=axis, position_um=list(position_um), bound_levels=n,
                energies_hz=m.motion.energies_hz, coherent=coherent, population=population,
                local_well=dict(direction_lab=m.u.tolist(), depth_uk=m.motion.depth_hz*at.H/at.K_B*1e6,
                                wavelength_nm=m.motion.wavelength_m*1e9, bottom_frequency_hz=m.nu_t,
                                minimum_spacing_hz=float(np.min(np.diff(m.motion.energies_hz))),
                                initial_discarded_weight=float(m.initial_tail),
                                pump_intensities_w_m2={key: beam.intensity_w_m2 for key, beam in m.beams.items()}),
                max_population_difference=float(np.max(np.abs(diagonal-population['P_absolute']))),
                max_total_variation_with_escape=float(tv.max()), differences=differences,
                pumping_validity=validity,
                scope='One local axis; optical pump and repump only. Same spin-resolved rates in both solvers. '
                      'Coherent version retains motional coherences in every spin block; population version discards them. '
                      'No Raman, lattice scattering, transverse escape, spin coherences, or recapture. '
                      'This comparison isolates one approximation; it does not validate the full experiment.')
