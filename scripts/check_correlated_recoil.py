"""Matched 3D recoil models and angular-convergence check of full sequences."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rb85rsc.config import SimConfig
from rb85rsc.runner import run, export


def differences(a, b):
    return {key: float(np.max(np.abs(getattr(a['result'], key)-getattr(b['result'], key))))
            for key in ('P', 'counts', 'overflow')}


def node_differences(a, b):
    if len(a['nodes']) != len(b['nodes']):
        raise ValueError('angular comparisons must use the same cloud samples')
    return {key: float(np.max([np.max(np.abs(left[key]-right[key]))
                              for left, right in zip(a['nodes'], b['nodes'])]))
            for key in ('P', 'counts', 'loss', 'P_final_3d', 'rho_final')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'examples/onlyOP_diagonalized.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/onlyOP_correlated_recoil')
    p.add_argument('--workers', type=int, choices=range(1, 9))
    p.add_argument('--samples', type=int, help='Optional cloud-size override, recorded in exported configs.')
    p.add_argument('--quadratures', type=int, nargs=2, default=[24, 48], metavar=('BASE', 'REFINED'))
    p.add_argument('--probability-tolerance', type=float, default=1e-7)
    p.add_argument('--photon-tolerance', type=float, default=1e-6)
    p.add_argument('--figures', action='store_true')
    args = p.parse_args(argv)
    if not 2 <= args.quadratures[0] < args.quadratures[1] <= 400:
        p.error('quadratures require 2 <= BASE < REFINED <= 400')
    if args.samples is not None and args.samples < 1:
        p.error('samples must be positive')
    if not all(np.isfinite(x) and x > 0 for x in (args.probability_tolerance, args.photon_tolerance)):
        p.error('convergence tolerances must be finite and positive')
    base = SimConfig.from_json(args.config)
    if args.samples is not None:
        base.ensemble.samples = args.samples
    if not base.ensemble.enabled:
        p.error('this comparison requires an enabled 3D ensemble')
    args.output.mkdir(parents=True, exist_ok=True)
    cases, outputs = {}, {}
    for name, mode, nq in [('cartesian_marginals', 'cartesian_marginals', args.quadratures[0]),
                           ('correlated', 'correlated', args.quadratures[0]),
                           ('correlated_refined', 'correlated', args.quadratures[1])]:
        cfg = base.copy()
        cfg.ensemble.recoil_model = mode
        cfg.numerics.recoil_quadrature_points = nq
        print(f'Running {name}: {cfg.ensemble.samples} sites, polar quadrature {nq}', flush=True)
        result = run(cfg, ensemble_workers=args.workers)
        export(result, args.output/name, figures=args.figures)
        cases[name] = dict(final=result['final'], validity_status=result['validity'].status,
                           recoil_model=mode, polar_quadrature=nq, azimuth_points=2*nq if mode == 'correlated' else None)
        # Comparisons only need exported observables, not all dense node kernels.
        outputs[name] = dict(result=result['result'], final=result['final'],
                            nodes=[{key: r[key] for key in ('P', 'counts', 'loss', 'P_final_3d', 'rho_final')}
                                   for _, r in result['ensemble_nodes']])
        print(f"  target={result['final']['P_target_absolute']:.9g}, survival={result['final']['P_trapped']:.9g}, "
              f"wall={result['final']['wall_time_s']:.2f}s", flush=True)
        del result
    angular = differences(outputs['correlated'], outputs['correlated_refined'])
    local_angular = node_differences(outputs['correlated'], outputs['correlated_refined'])
    converged = (angular['P'] <= args.probability_tolerance and angular['overflow'] <= args.probability_tolerance
                 and angular['counts'] <= args.photon_tolerance)
    converged = converged and all(value <= (args.photon_tolerance if key == 'counts' else args.probability_tolerance)
                                  for key, value in local_angular.items())
    changes = {key: outputs['correlated']['final'][key]-outputs['cartesian_marginals']['final'][key]
               for key in ('P_target_absolute', 'P_trapped', 'P_up', 'photons_total')}
    report = dict(config=base.to_dict(), cases=cases, correlated_minus_cartesian_marginals=changes,
                  angular_convergence=dict(status='passed' if converged else 'failed', max_absolute_differences=angular,
                      max_absolute_node_differences=local_angular,
                      probability_tolerance=args.probability_tolerance, photon_tolerance=args.photon_tolerance),
                  interpretation='Matched sequences differ only in Cartesian emission correlations. '
                  'Both retain population-resolving recoil, resolved spherical emission channels, and absorbing well loss. '
                  'Angular convergence does not establish physical validity or spatial/timestep convergence.')
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(changes=changes, angular_convergence=report['angular_convergence']), indent=2), flush=True)
    return 0 if converged else 1


if __name__ == '__main__':
    raise SystemExit(main())
