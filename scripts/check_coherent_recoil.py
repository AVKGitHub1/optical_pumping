"""Compare one-well coherent recoil with population kinetics across pump powers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rb85rsc.config import SimConfig
from rb85rsc.recoil_reference import compare_recoil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'examples/onlyOP.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/coherent_recoil_reference')
    parser.add_argument('--axis', choices=list('xyz'), default='z')
    parser.add_argument('--position-um', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--power-scales', type=float, nargs='+', default=[.03, .1, 1.])
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.power_scales):
        parser.error('power scales must be finite and positive')
    cfg = SimConfig.from_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, scale in enumerate(args.power_scales):
        c = cfg.copy()
        for beam in (c.spin_pump, c.repump):
            beam.power_mw *= scale
            beam.peak_intensity_w_m2 *= scale
        print(f'One-well {args.axis} reference: pump power scale {scale:g}', flush=True)
        result = compare_recoil(c, axis=args.axis, position_um=args.position_um)
        arrays = {'t_s': result['t_s'], 'energies_hz': result['energies_hz']}
        for name in ('coherent', 'population'):
            arrays.update({name+'_'+key: value for key, value in result[name].items() if isinstance(value, np.ndarray)})
        np.savez_compressed(args.output/f'case_{i}.npz', **arrays)
        c.to_json(args.output/f'case_{i}_config.json')
        row = {key: result[key] for key in ('axis', 'position_um', 'bound_levels', 'max_population_difference', 'scope', 'pumping_validity')}
        row['power_scale'] = scale
        for name in ('coherent', 'population'):
            row[name] = {key: float(result[name][key][-1]) for key in
                         ('survival', 'well_escape_no_recapture', 'P_up_absolute', 'P_target_absolute', 'mean_excitation_hz')}
            row[name]['max_conservation_error'] = result[name]['max_conservation_error']
        row['coherent']['min_eigenvalue'] = result['coherent']['min_eigenvalue']
        summary.append(row)
        (args.output/'comparison.json').write_text(json.dumps(summary, indent=2))
        print(f"Maximum absolute population difference: {row['max_population_difference']:.6g}", flush=True)


if __name__ == '__main__':
    main()
