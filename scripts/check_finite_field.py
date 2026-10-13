"""Matched weak-field versus diagonalized-atom simulations, with all other inputs fixed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rb85rsc import atomic as at
from rb85rsc.config import SimConfig
from rb85rsc.runner import run, export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'examples/onlyOP.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/onlyOP_finite_field')
    parser.add_argument('--workers', type=int, choices=range(1, 9))
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args(argv)
    base = SimConfig.from_json(args.config)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for mode in ('weak_field', 'diagonalized'):
        cfg = base.copy()
        cfg.optical.zeeman_model = mode
        print(f'Running {mode}, B = {cfg.magnetic.magnitude_gauss:g} G', flush=True)
        result = run(cfg, ensemble_workers=args.workers)
        export(result, output/mode, figures=args.figures)
        rows[mode] = dict(final=result['final'], validity_status=result['validity'].status,
                          magnetic=at.zeeman_validity(result['model'].atom, result['model'].B,
                                                     result['model'].nu_t, result['model'].up,
                                                     result['model'].down, mode))
        print(f"  target = {result['final']['P_target_absolute']:.8g}, "
              f"survival = {result['final']['P_trapped']:.8g}, validity = {result['validity'].status}", flush=True)
    differences = {key: rows['diagonalized']['final'][key]-rows['weak_field']['final'][key]
                   for key in ('P_up', 'P_up_absolute', 'P_n0', 'P_target_absolute', 'P_trapped', 'photons_total')}
    summary = dict(config=base.to_dict(), cases=rows, diagonalized_minus_weak_field=differences,
                   interpretation='Matched full simulations. This comparison changes magnetic eigenstates and optical dipoles only. '
                   'It does not correct motional recoil coherence, photon Cartesian correlations, or recapture. '
                   'Physical validity diagnostics still apply to both runs.')
    (output/'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(differences, indent=2), flush=True)


if __name__ == '__main__':
    main()
