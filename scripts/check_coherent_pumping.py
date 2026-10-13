"""3D coherent optical pumping, with a matched population-trajectory control."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rb85rsc.coherent_pumping import run_cloud, validate_config
from rb85rsc.coherent_results import final_estimates, save_trajectories
from rb85rsc.config import SimConfig
from rb85rsc.runner import run, export, versions


def compare_control(estimate, exact, total_trajectories):
    """Conservative regression check, not a physical-validity declaration."""
    values = {}
    for key, reference in exact.items():
        mean = float(estimate['mean'][key][-1])
        error = float(estimate['standard_error'][key][-1])
        difference = mean-reference
        # A finite rare-event sample can have zero empirical variance. Include
        # two trajectory weights without treating them as measured uncertainty.
        tolerance = 7*error+2/total_trajectories+1e-8
        values[key] = dict(difference=difference, standard_error=error,
                          sigma=difference/error if error > 0 else None,
                          tolerance=tolerance, passed=abs(difference) <= tolerance)
    return dict(passed=all(v['passed'] for v in values.values()), observables=values,
                criterion='Final cloud observables within 7 Monte Carlo standard errors + 2/N + 1e-8.')


def compare_reports(left, right):
    """Check independent repeats at identical inputs, allowing angular refinement."""
    if left['trajectory_seed'] == right['trajectory_seed']:
        raise ValueError('repeat comparisons require independent trajectory seeds')
    configs = []
    for report in (left, right):
        cfg = SimConfig.from_dict(report['config'])
        cfg.numerics.recoil_quadrature_points = 2
        # Monte Carlo controls and the GUI's selected backend do not change
        # the physical inputs of this paired-model comparison.
        cfg.ensemble.trajectories_per_site = 2
        cfg.ensemble.trajectory_seed = 0
        cfg.ensemble.recoil_dynamics = 'population'
        configs.append(cfg.to_dict())
    if configs[0] != configs[1]:
        raise ValueError('repeat comparisons require matching configs apart from angular quadrature')
    modes = {}
    n = min(r['trajectories_per_site']*r['sites'] for r in (left, right))
    for mode in ('coherent', 'population'):
        values = {}
        for key, a in left['final'][mode].items():
            b = right['final'][mode][key]
            error = float(np.hypot(a['standard_error'], b['standard_error']))
            difference = b['mean']-a['mean']
            values[key] = dict(difference=difference, combined_standard_error=error,
                               sigma=difference/error if error > 0 else None,
                               statistically_consistent=abs(difference) <= 7*error+2/n+1e-8)
        modes[mode] = values
    return dict(statistically_consistent=all(v['statistically_consistent'] for m in modes.values() for v in m.values()),
                modes=modes, interpretation='Independent final estimates compared within 7 combined Monte Carlo standard errors + 2/N + 1e-8. '
                'Agreement tests stability at this statistical precision; it is not a bound on physical or spatial quadrature errors.')


def plot_comparison(coherent, reference, output):
    """Shared external legend; pointwise MC intervals, not model uncertainty."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    t = coherent['t_s']*1e3
    for j, (key, title) in enumerate((('survival', 'Remaining in modeled wells'),
                                     ('P_target_absolute', 'Absolute target yield'))):
        mean = coherent['mean'][key]*100
        error = 1.96*coherent['standard_error'][key]*100
        exact = reference[key]*100
        top, bottom = axes[:, j]
        control, = top.plot(t, exact, '--', color='0.3', label='Population model')
        line, = top.plot(t, mean, color='#187b9b', label='Coherent recoil')
        top.fill_between(t, mean-error, mean+error, color='#187b9b', alpha=.2)
        top.set_title(title)
        top.set_ylabel('% of initially loaded atoms')
        bottom.axhline(0., color='0.3', linestyle='--', linewidth=1.)
        bottom.plot(t, mean-exact, color='#187b9b')
        bottom.fill_between(t, mean-exact-error, mean-exact+error, color='#187b9b', alpha=.2)
        bottom.set_ylabel('Coherent minus population\n(percentage points)')
        bottom.set_xlabel('Time (ms)')
        for ax in (top, bottom):
            ax.grid(alpha=.2)
    fig.legend(handles=[control, line, Patch(color='#187b9b', alpha=.2, label='95% Monte Carlo interval')],
               loc='outside upper center', ncol=3, frameon=False)
    fig.savefig(output/'comparison.png', dpi=160)
    fig.savefig(output/'comparison.pdf')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'examples/onlyOP_correlated.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/onlyOP_coherent_3d')
    p.add_argument('--trajectories', type=int, help='Independent trajectories per site and mode; defaults to the saved config (16384 if omitted there).')
    p.add_argument('--seed', type=int, help='Trajectory seed; defaults to the saved config (1729 if omitted there).')
    p.add_argument('--workers', type=int, choices=range(1, 9))
    p.add_argument('--samples', type=int, help='Optional cloud-size override, recorded in the config.')
    p.add_argument('--quadrature', type=int, help='Optional polar quadrature override; azimuth has twice as many nodes.')
    p.add_argument('--compare-to', type=Path, help='Earlier comparison.json for an independent-seed stability check.')
    args = p.parse_args(argv)
    if (args.trajectories is not None and args.trajectories < 2) or (args.seed is not None and args.seed < 0):
        p.error('trajectories must be >= 2 and seed >= 0')
    if args.samples is not None and args.samples < 1:
        p.error('samples must be positive')
    if args.quadrature is not None and not 2 <= args.quadrature <= 400:
        p.error('quadrature must be between 2 and 400')
    cfg = SimConfig.from_json(args.config)
    args.trajectories = cfg.ensemble.trajectories_per_site if args.trajectories is None else args.trajectories
    args.seed = cfg.ensemble.trajectory_seed if args.seed is None else args.seed
    cfg.ensemble.trajectories_per_site, cfg.ensemble.trajectory_seed = args.trajectories, args.seed
    cfg.ensemble.recoil_dynamics = 'coherent_pumping'
    if args.samples is not None:
        cfg.ensemble.samples = args.samples
    if args.quadrature is not None:
        cfg.numerics.recoil_quadrature_points = args.quadrature
    # All three calculations use precisely the same correlated recoil model.
    cfg.ensemble.recoil_model = 'correlated'
    validate_config(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg.to_json(args.output/'run_config.json')
    print(f'Running deterministic population control: {cfg.ensemble.samples} sites', flush=True)
    population_cfg = cfg.copy()
    population_cfg.ensemble.recoil_dynamics = 'population'
    baseline = run(population_cfg, ensemble_workers=args.workers)
    export(baseline, args.output/'population_deterministic', figures=False)
    reference = {k: baseline['final'][k] for k in ('P_up_absolute', 'P_n0_absolute', 'P_target_absolute',
                 'P_3d_ground_absolute', 'P_target_3d_absolute', 'photons_total')}
    reference.update(survival=baseline['final']['P_trapped'], loss=baseline['final']['final_overflow'],
                     photons_spin_pump=baseline['final']['photons']['spin_pump'],
                     photons_repump=baseline['final']['photons']['repump'],
                     photons_lattice_scatter=baseline['final']['photons']['lattice_scatter'])
    validity = [dict(category=c, status=s, message=m) for c, s, m in baseline['validity'].items]
    reference_series = dict(survival=baseline['series']['P_trapped'],
                            P_target_absolute=baseline['series']['P_target_absolute'])
    del baseline  # release dense population kernels before trajectory workers
    print(f'Running both trajectory modes: {args.trajectories} per site', flush=True)
    result = run_cloud(cfg, trajectories=args.trajectories, seed=args.seed, workers=args.workers,
                       progress=lambda f, message: print(message, flush=True))
    save_trajectories(result, args.output)
    coherent, population = (result['modes'][m] for m in ('coherent', 'population'))
    check = compare_control(population, reference, args.trajectories*cfg.ensemble.samples)
    differences = {k: dict(difference=float(coherent['mean'][k][-1]-value),
                           standard_error=float(coherent['standard_error'][k][-1])) for k, value in reference.items()}
    report = {k: v for k, v in result.items() if k not in ('nodes', 'modes')}
    report.update(config=cfg.to_dict(), versions=versions(),
                  final={mode: final_estimates(values) for mode, values in result['modes'].items()},
                  deterministic_population=reference, population_control=check,
                  coherent_minus_deterministic_population=differences, population_validity=validity,
                  interpretation=[
                      'Coherent trajectories retain motional phases on all axes and shared photon directions. Raman must be off.',
                      'The population motional-secular criterion does not apply to this trajectory solver. It still applies to the deterministic and population controls.',
                      'Errors are one-sigma Monte Carlo standard errors for the fixed spatial and angular quadratures, not experimental or model uncertainty.',
                      'Repeat with more trajectories and independent seeds; refine angular quadrature and cloud sampling separately.',
                      'Weak-excitation elimination, resolved spin and spherical emission channels, diagonal Stark shifts, frozen isolated wells, and no recapture remain assumptions.',
                      'Initial distributions are conditioned on local bound levels; survival is relative to that loaded ensemble.',
                      'Passing the population control validates stochastic propagation, not the shared physical approximations.'
                  ])
    if args.compare_to is not None:
        previous = json.loads(args.compare_to.read_text(encoding='utf-8'))
        report['independent_repeat'] = compare_reports(previous, report)
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_comparison(coherent, reference_series, args.output)
    print(json.dumps(dict(final=report['final'], population_control_passed=check['passed'],
                          coherent_minus_population=differences), indent=2), flush=True)
    return 0 if check['passed'] and report.get('independent_repeat', {}).get('statistically_consistent', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
