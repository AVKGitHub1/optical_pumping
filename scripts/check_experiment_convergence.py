"""Time/quadrature/order sensitivity on a common small spatial quadrature."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from rb85rsc.config import SimConfig
from optimize_experiment import batch, output_root


def main():
    base = SimConfig.from_json(ROOT / 'examples/experiment.json')
    out = output_root(base)
    out.mkdir(parents=True, exist_ok=True)
    base.ensemble.samples = 4
    base.ensemble.workers = 1
    base.timing.n_samples = 51
    cases = []
    for dt in (2., 1., .5):
        c = base.copy()
        c.ensemble.time_step_us = dt
        cases.append((f'timestep_{dt:g}_us', c.to_dict()))
    c = base.copy()
    c.numerics.recoil_quadrature_points = 48
    cases.append(('recoil_quadrature_48', c.to_dict()))
    c = base.copy()
    c.raman.max_sideband_order = 8
    cases.append(('all_bound_sidebands', c.to_dict()))
    batch(cases, out / 'numerical_convergence.json', 4)


if __name__ == '__main__':
    main()
