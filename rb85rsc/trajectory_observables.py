"""Trajectory moments for the shared population and thermal result views.

Observations use wavefunction probabilities, without measuring or collapsing
motion. All recorded moments are absolute (zero for escaped trajectories).
Small trajectory batches preserve temporal covariance for derivative errors.
"""
import numpy as np


class ObservableRecorder:
    def __init__(self, trajectories, times, levels_hz, target_axis):
        self.n, self.times = trajectories, times
        self.levels = [np.asarray(e)-e[0] for e in levels_hz]
        self.target_axis = target_axis
        self.mean, self.standard_error = {}, {}
        self.starts = np.linspace(0, trajectories, min(32, trajectories)+1, dtype=int)[:-1]
        self.sizes = np.diff(np.r_[self.starts, trajectories])
        self.batches = np.zeros((len(times), len(self.sizes), 4))

    def record(self, index, spin, current, alive, spins):
        probabilities = [abs(v)**2 for v in current]
        energy = np.stack([p @ e for p, e in zip(probabilities, self.levels)], axis=1)
        values = dict(energy_hz=energy,
                      nbar=np.stack([p @ np.arange(p.shape[1]) for p in probabilities], axis=1))
        values.update({f'pn_{a}': p for a, p in zip('xyz', probabilities)})
        for key, value in values.items():
            shape = (len(self.times), *value.shape[1:])
            if key not in self.mean:
                self.mean[key], self.standard_error[key] = np.zeros(shape), np.zeros(shape)
            self.mean[key][index] = value.mean(axis=0)
            self.standard_error[key][index] = value.std(axis=0, ddof=1)/np.sqrt(self.n)
        spin_mean = np.bincount(spin[alive], minlength=spins)/self.n
        joint_mean = np.zeros((spins, probabilities[self.target_axis].shape[1]))
        joint_second = np.zeros_like(joint_mean)
        for s in range(spins):
            p = probabilities[self.target_axis][alive & (spin == s)]
            joint_mean[s], joint_second[s] = p.sum(axis=0)/self.n, (p*p).sum(axis=0)/self.n
        for key, mean, second in (('spin', spin_mean, spin_mean), ('joint', joint_mean, joint_second)):
            shape = (len(self.times), *mean.shape)
            if key not in self.mean:
                self.mean[key], self.standard_error[key] = np.zeros(shape), np.zeros(shape)
            self.mean[key][index] = mean
            self.standard_error[key][index] = np.sqrt(np.maximum(second-mean**2, 0)/(self.n-1))
        self.batches[index] = np.add.reduceat(np.column_stack((alive, energy)), self.starts, axis=0)/self.sizes[:, None]

    def result(self):
        return dict(mean=self.mean, standard_error=self.standard_error,
                    thermal_batches=self.batches, batch_sizes=self.sizes)


def padded(values):
    """Pad different local bound spectra with zero absolute population."""
    shape = np.max([v.shape for v in values], axis=0)
    return np.array([np.pad(v, [(0, n-m) for n, m in zip(shape, v.shape)]) for v in values])


def average_observables(results):
    n = len(results)
    means, errors = {}, {}
    for key in results[0]['observables']['mean']:
        means[key] = padded([r['observables']['mean'][key] for r in results]).mean(axis=0)
        se = padded([r['observables']['standard_error'][key] for r in results])
        errors[key] = np.sqrt((se**2).sum(axis=0))/n
    return dict(mean=means, standard_error=errors)


def ratio_estimate(numerators, errors, denominators, denominator_errors, counts):
    """Delta-method ratio of cloud means, with within-site covariance.

    Each denominator is a Bernoulli indicator; each numerator vanishes whenever
    that indicator is zero. Hence Cov(mean Y, mean S) = mean Y*(1-mean S)/(N-1).
    Spatial strata have equal weights, and their independent variances add.
    """
    y, sy, d, sd = map(np.asarray, (numerators, errors, denominators, denominator_errors))
    extra = (1,)*(y.ndim-d.ndim)
    d, sd = d.reshape((*d.shape, *extra)), sd.reshape((*sd.shape, *extra))
    count = np.asarray(counts).reshape((-1,)+(1,)*(y.ndim-1))
    mean_d = d.mean(axis=0)
    safe = np.where(mean_d > 0, mean_d, np.nan)
    mean = y.mean(axis=0)/safe
    cov = y*(1-d)/(count-1)
    variance = np.sum(sy**2 + mean**2*sd**2 - 2*mean*cov, axis=0)/len(y)**2
    return mean, np.sqrt(np.maximum(variance, 0))/safe


def derivative_error(times, results, axis, energy, survival, window):
    """Batch estimate including correlations between output times and survival.

    Linearize the cloud ratio first, then differentiate the same batch histories
    as the plotted mean. Center within each spatial stratum; cloud heterogeneity
    is not Monte Carlo uncertainty. Unequal batch sizes are weighted by size.
    """
    from .analysis import _derivative
    variance = np.zeros(len(times))
    safe = np.where(survival > 0, survival, np.nan)
    for r in results:
        obs = r['observables']
        batches = obs['thermal_batches']
        influence = (batches[:, :, axis+1]-energy[:, None]*batches[:, :, 0])/safe[:, None]
        derivatives = np.array([_derivative(times, v, window) for v in influence.T])
        weights = obs['batch_sizes']/obs['batch_sizes'].sum()
        center = weights @ derivatives
        variance += np.sum(weights[:, None]*(derivatives-center)**2, axis=0)/(len(weights)-1)
    return np.sqrt(variance)/len(results)


def analyze_observables(cfg, raw, stop=None):
    """Adapt recorded moments to the result fields consumed by shared views."""
    from . import atomic as at
    from .analysis import _derivative
    from .coherent_pumping import _check_stop
    from .ensemble import canonical_mixture_temperature
    from .motion import boltzmann_pn
    from . import polarization as pol

    cloud = raw['modes']['coherent']
    results = [node['modes']['coherent'] for node in raw['nodes']]
    counts = np.array([r['trajectories'] for r in results])
    survivors = np.array([r['mean']['survival'] for r in results])
    survivor_se = np.array([r['standard_error']['survival'] for r in results])
    t, survival = cloud['t_s'], cloud['mean']['survival']
    means, errors = cloud['observables']['mean'], cloud['observables']['standard_error']
    series = dict(t_s=t, **cloud['mean'])
    se = dict(cloud['standard_error'])
    axes = pol.lattice_frame(cfg.trap.axis, cfg.ensemble.transverse_reference_direction)
    target_axis, _, _ = pol.raman_geometry(cfg.raman.beam_low_direction, cfg.raman.beam_high_direction, axes)
    up, down = cfg.raman.state_indices()

    def conditional(key):
        return ratio_estimate(padded([r['observables']['mean'][key] for r in results]),
                              padded([r['observables']['standard_error'][key] for r in results]),
                              survivors, survivor_se, counts)

    spin, spin_se = conditional('spin')
    joint, joint_se = conditional('joint')
    occupation, occupation_se = conditional('nbar')
    energy, energy_se = conditional('energy_hz')
    pn, pn_se = {}, {}
    for a in 'xyz':
        pn[a], pn_se[a] = conditional(f'pn_{a}')

    for key, raw_key in (('P_up', 'P_up_absolute'), ('P_n0', 'P_n0_absolute'),
                         ('P_target', 'P_target_absolute')):
        series[key], se[key] = ratio_estimate(
            [r['mean'][raw_key] for r in results], [r['standard_error'][raw_key] for r in results],
            survivors, survivor_se, counts)
    series['P_n0_given_up'], se['P_n0_given_up'] = ratio_estimate(
        [r['mean']['P_target_absolute'] for r in results], [r['standard_error']['P_target_absolute'] for r in results],
        [r['mean']['P_up_absolute'] for r in results], [r['standard_error']['P_up_absolute'] for r in results], counts)
    series['P_down'], se['P_down'] = spin[:, down], spin_se[:, down]
    for label, part in (('P_F2', slice(0, 5)), ('P_F3', slice(5, 12))):
        population = np.array([r['observables']['mean']['spin'][:, part].sum(axis=1) for r in results])
        error = np.sqrt(np.maximum(population*(1-population), 0)/(counts[:, None]-1))
        series[label], se[label] = ratio_estimate(population, error, survivors, survivor_se, counts)
    for i in range(12):
        series[f'P{at.ground_label(i)}'], se[f'P{at.ground_label(i)}'] = spin[:, i], spin_se[:, i]
    series['P_trapped'], se['P_trapped'] = survival, se['survival']
    series['overflow'], se['overflow'] = series['loss'], se['loss']
    # The validated solver has no Raman exposure, so this counter is exactly zero.
    series['photons_raman_scatter'] = se['photons_raman_scatter'] = np.zeros(len(t))

    for axis, a in enumerate('xyz'):
        _check_stop(stop)
        levels = [np.asarray(node['energies_hz'][axis])-node['energies_hz'][axis][0] for node in raw['nodes']]
        local_energy = np.array([r['observables']['mean']['energy_hz'][:, axis] for r in results])
        local_error = np.array([r['observables']['standard_error']['energy_hz'][:, axis] for r in results])
        temperatures, temperature_se = np.full(len(t), np.nan), np.full(len(t), np.nan)
        for it in range(len(t)):
            _check_stop(stop)
            temp = canonical_mixture_temperature(energy[it, axis], levels, survivors[:, it])
            temperatures[it] = temp
            if not np.isfinite(temp) or temp <= 0:
                continue
            canonical = [boltzmann_pn(e, temp) for e in levels]
            expected = np.array([p @ e for p, e in zip(canonical, levels)])
            slopes = np.array([p @ (e-mean)**2 for p, e, mean in zip(canonical, levels, expected)])*at.H/(at.K_B*temp**2)
            cov = local_energy[:, it]*(1-survivors[:, it])/(counts-1)
            variance = np.sum(local_error[:, it]**2 + expected**2*survivor_se[:, it]**2 - 2*expected*cov)
            slope = survivors[:, it] @ slopes
            if slope > 0:
                temperature_se[it] = np.sqrt(max(variance, 0))/slope
        for key, mean, error in (
                (f'nbar_{a}', occupation[:, axis], occupation_se[:, axis]),
                (f'P_n0_{a}', pn[a][:, 0], pn_se[a][:, 0]),
                (f'P_n0_{a}_absolute', means[f'pn_{a}'][:, 0], errors[f'pn_{a}'][:, 0]),
                (f'mean_excitation_energy_{a}_hz', energy[:, axis], energy_se[:, axis]),
                (f'mean_excitation_energy_{a}_J', at.H*energy[:, axis], at.H*energy_se[:, axis]),
                (f'T_equiv_{a}_K', temperatures, temperature_se),
                (f'cooling_power_{a}_W', -at.H*_derivative(t, energy[:, axis], cfg.analysis.derivative_window),
                 at.H*derivative_error(t, results, axis, energy[:, axis], survival, cfg.analysis.derivative_window))):
            series[key], se[key] = mean, error
    axis = 'xyz'[target_axis]
    for key, axis_key in (('nbar', f'nbar_{axis}'), ('T_equiv_K', f'T_equiv_{axis}_K'),
                           ('mean_excitation_energy_hz', f'mean_excitation_energy_{axis}_hz'),
                           ('mean_excitation_energy_J', f'mean_excitation_energy_{axis}_J'),
                           ('cooling_power_W', f'cooling_power_{axis}_W')):
        series[key], se[key] = series[axis_key], se[axis_key]
    finite = lambda value: float(value) if np.isfinite(value) else None
    final = {key: finite(value[-1]) for key, value in series.items() if key != 't_s'}
    final.update(raman_axis=axis, raman_active=False,
                 T_equiv_axes_K={a: finite(series[f'T_equiv_{a}_K'][-1]) for a in 'xyz'},
                 spin_populations_initial={at.ground_label(i): finite(spin[0, i]) for i in range(12)},
                 spin_populations_final={at.ground_label(i): finite(spin[-1, i]) for i in range(12)})
    return dict(series=series, standard_error=se, final=final,
                spin=spin, spin_absolute=means['spin'], pn=pn[axis], pn_axes=pn,
                pn_axes_absolute={a: means[f'pn_{a}'] for a in 'xyz'},
                P_final=joint[-1], P_final_absolute=means['joint'][-1],
                joint=joint, joint_absolute=means['joint'],
                array_standard_error=dict(spin=spin_se, spin_absolute=errors['spin'],
                    pn_axes=pn_se, pn_axes_absolute={a: errors[f'pn_{a}'] for a in 'xyz'},
                    joint=joint_se, joint_absolute=errors['joint']),
                model_summary=dict(raman_axis=axis, raman_state_up=at.GROUND_STATES[up],
                    raman_state_down=at.GROUND_STATES[down], potential='lattice'),
                cooling_rate_method='Smoothed derivative of survivor mean local energy; Monte Carlo errors use independent trajectory batches with temporal covariance.')
