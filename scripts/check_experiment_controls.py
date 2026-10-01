"""Matched spatial-sample controls for the selected experimental configuration."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rb85rsc.config import SimConfig
from rb85rsc.runner import run, export, _json_default
from optimize_experiment import output_root


def execute(name, data):
    c = SimConfig.from_dict(data)
    a = run(c, progress=lambda f,m: print(name, f'{f:.0%}', m, flush=True))
    export(a, output_root(c) / name, figures=False)
    return name, a['final']


def main():
    base = SimConfig.from_json(ROOT / 'examples/experiment.json')
    tasks = []
    c = base.copy()
    c.raman.enabled = False
    tasks.append(('raman_off_control', c.to_dict()))
    c = base.copy()
    c.raman.red_sideband_offset_hz = 0.
    tasks.append(('zero_offset_check', c.to_dict()))
    results = {}
    # Each run already parallelizes its independent spatial samples.
    for name, data in tasks:
        name, result = execute(name, data)
        results[name] = result
        (output_root(base) / 'control_summary.json').write_text(json.dumps(results, indent=2, default=_json_default))


if __name__ == '__main__':
    main()
