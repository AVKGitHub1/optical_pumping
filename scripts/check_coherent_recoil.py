"""Independent one-axis recoil checks across powers and cloud positions.

Local results are never combined into a prediction of full 3D survival.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rb85rsc.config import SimConfig
from rb85rsc.protocols import raman_is_active
from rb85rsc.recoil_reference import compare_recoil, validation_positions


PROBABILITIES = ('P_absolute', 'survival', 'P_up_absolute', 'P_n0_absolute', 'P_target_absolute')


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_case(result, cfg, output, stem):
    arrays = {'t_s': result['t_s'], 'energies_hz': result['energies_hz']}
    for name in ('coherent', 'population'):
        arrays.update({name+'_'+key: value for key, value in result[name].items() if isinstance(value, np.ndarray)})
    np.savez_compressed(output / f'{stem}.npz', **arrays)
    cfg.to_json(output / f'{stem}_config.json')


def numerical_checks(result, cfg):
    errors = []
    for name in ('coherent', 'population'):
        values = result[name]
        for key in (*PROBABILITIES, 'well_escape_no_recapture', 'photons_spin_pump', 'photons_repump'):
            if not np.all(np.isfinite(values[key])):
                errors.append(f'{name}: nonfinite {key}')
        error = values['max_conservation_error']
        if not np.isfinite(error) or error > cfg.numerics.conservation_tolerance:
            errors.append(f'{name}: probability conservation')
        minimum = values['min_eigenvalue' if name == 'coherent' else 'min_population']
        if not np.isfinite(minimum) or minimum < -cfg.numerics.positivity_tolerance:
            errors.append(f'{name}: positivity')
    return dict(status='failed' if errors else 'passed', failures=errors)


def quadrature_check(coarse, fine, tolerance):
    drift = {name: {key: float(np.max(np.abs(coarse[name][key] - fine[name][key])))
                    for key in PROBABILITIES} for name in ('coherent', 'population')}
    values = [value for row in drift.values() for value in row.values()]
    maximum = max(values) if np.all(np.isfinite(values)) else float('nan')
    return dict(status='passed' if np.isfinite(maximum) and maximum <= tolerance else 'failed',
                tolerance=tolerance, max_absolute_probability_change=maximum, changes=drift)


def summarize(rows, expected, check_quadrature):
    completed = [r for r in rows if r['status'] == 'complete']
    worst = {}
    for key in ('survival', 'P_up_absolute', 'P_n0_absolute', 'P_target_absolute', 'mean_excitation_uk'):
        finite = [r for r in completed if np.isfinite(r['differences'][key]['max_absolute'])]
        if finite:
            row = max(finite, key=lambda r: r['differences'][key]['max_absolute'])
            worst[key] = dict(max_absolute=row['differences'][key]['max_absolute'],
                              case_id=row['case_id'], axis=row['axis'], site_id=row['site_id'],
                              position_um=row['position_um'], power_scale=row['power_scale'])
    failed = [r['case_id'] for r in rows if r['status'] != 'complete' or r['numerics']['status'] != 'passed'
              or (check_quadrature and r['quadrature']['status'] != 'passed')]
    return dict(status='complete' if len(completed) == expected and not failed else 'incomplete_or_failed',
                expected_cases=expected, completed_cases=len(completed), failed_case_ids=failed,
                quadrature_checked=check_quadrature, worst_observable_differences=worst,
                scope=__doc__.strip(),
                interpretation='Differences are coherent minus population at the same sampled times. '
                'Maxima cover only requested sites, powers, axes, and output times; they are not error bounds. '
                'Passing numerical checks does not remove shared physical approximations or validity warnings.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'examples/onlyOP.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/coherent_recoil_reference')
    axes = parser.add_mutually_exclusive_group()
    axes.add_argument('--axis', choices=list('xyz'), help='One local axis (default: z).')
    axes.add_argument('--axes', choices=list('xyz'), nargs='+', help='Independent local-axis checks.')
    parser.add_argument('--position-um', type=float, nargs=3, default=[0., 0., 0.],
                        help='Reference lab position, checked in addition to cloud samples.')
    parser.add_argument('--cloud-samples', type=int, default=0,
                        help='Additional positions from the production Sobol cloud and configured seed.')
    parser.add_argument('--power-scales', type=float, nargs='+', default=[.03, .1, 1.])
    parser.add_argument('--check-quadrature', action='store_true', help='Repeat each case with twice the angular points.')
    parser.add_argument('--quadrature-tolerance', type=float, default=1e-6,
                        help='Maximum absolute probability drift allowed in the quadrature check.')
    args = parser.parse_args(argv)
    if any(not np.isfinite(x) or x <= 0 for x in args.power_scales):
        parser.error('power scales must be finite and positive')
    if not np.isfinite(args.quadrature_tolerance) or args.quadrature_tolerance <= 0:
        parser.error('quadrature tolerance must be finite and positive')
    cfg = SimConfig.from_json(args.config)
    if raman_is_active(cfg):
        parser.error('recoil reference requires a schedule with Raman off')
    if args.check_quadrature and cfg.numerics.recoil_quadrature_points > 200:
        parser.error('doubling quadrature would exceed the supported 400 points')
    try:
        positions = validation_positions(cfg, args.cloud_samples, args.position_um)
    except ValueError as error:
        parser.error(str(error))
    selected_axes = list(dict.fromkeys(args.axes or [args.axis or 'z']))
    expected = len(positions)*len(selected_axes)*len(args.power_scales)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = dict(config=cfg.to_dict(), axes=selected_axes, positions=positions,
                    power_scales=args.power_scales, seed=cfg.numerics.seed,
                    cloud_sampling='Production Sobol prefix; reference position has no ensemble weight.')

    def publish():
        write_json(args.output/'comparison.json', rows)
        summary = summarize(rows, expected, args.check_quadrature)
        write_json(args.output/'survey.json', dict(**manifest, **summary))
        return summary

    publish()  # Interrupted runs retain explicit incomplete coverage.
    for point in positions:
        for axis in selected_axes:
            for scale in args.power_scales:
                stem = f'case_{len(rows)}'
                c = cfg.copy()
                for beam in (c.spin_pump, c.repump):
                    beam.power_mw *= scale
                    beam.peak_intensity_w_m2 *= scale
                row = dict(case_id=stem, **point, axis=axis, power_scale=scale, status='failed')
                print(f'{stem}: {point["site_id"]}, {axis}, pump scale {scale:g}', flush=True)
                c.to_json(args.output/f'{stem}_config.json')
                try:
                    result = compare_recoil(c, axis=axis, position_um=point['position_um'])
                    save_case(result, c, args.output, stem)
                    row.update({key: result[key] for key in ('bound_levels', 'local_well', 'max_population_difference',
                               'max_total_variation_with_escape', 'differences', 'scope', 'pumping_validity')})
                    row['numerics'] = numerical_checks(result, c)
                    for name in ('coherent', 'population'):
                        row[name] = {key: float(result[name][key][-1]) for key in
                                     ('survival', 'well_escape_no_recapture', 'P_up_absolute', 'P_target_absolute', 'mean_excitation_hz')}
                        row[name]['max_conservation_error'] = result[name]['max_conservation_error']
                    row['coherent']['min_eigenvalue'] = result['coherent']['min_eigenvalue']
                    row['quadrature'] = dict(status='not_checked')
                    if args.check_quadrature:
                        fine_cfg = c.copy()
                        fine_cfg.numerics.recoil_quadrature_points *= 2
                        fine = compare_recoil(fine_cfg, axis=axis, position_um=point['position_um'])
                        save_case(fine, fine_cfg, args.output, stem+'_refined')
                        row['quadrature'] = quadrature_check(result, fine, args.quadrature_tolerance)
                        row['quadrature']['points'] = [c.numerics.recoil_quadrature_points,
                                                       fine_cfg.numerics.recoil_quadrature_points]
                        row['quadrature']['refined_numerics'] = numerical_checks(fine, fine_cfg)
                        if row['quadrature']['refined_numerics']['status'] != 'passed':
                            row['quadrature']['status'] = 'failed'
                    row['status'] = 'complete'
                    print(f"  max population difference {result['max_population_difference']:.6g}; "
                          f"quadrature {row['quadrature']['status']}", flush=True)
                except (ValueError, RuntimeError) as error:
                    row['error'] = str(error)
                    print(f'  FAILED: {error}', flush=True)
                rows.append(row)
                publish()
    summary = publish()
    print(f"{summary['completed_cases']}/{expected} cases completed; {summary['status']}", flush=True)
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
