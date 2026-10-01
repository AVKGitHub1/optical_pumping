"""Select the better of the matched 64-node detuning checks and update artifacts."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rb85rsc.config import SimConfig
from rb85rsc.model import Validity
from rb85rsc.plotting import dashboard, carrier_calibration_figure
from optimize_experiment import output_root

def render(folder, metadata):
    with (folder / 'run_timeseries.csv').open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    series = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
    with np.load(folder / 'run_joint.npz') as data:
        P = data['P'] / np.maximum(series['P_trapped'][:, None, None], 1e-300)
    cfg = SimConfig.from_dict(metadata['config'])
    a = dict(final=metadata['final'], series=series, pn=P.sum(axis=1), P_final=P[-1],
             validity=Validity([(v['category'],v['status'],v['message']) for v in metadata['validity']]),
             segments=[SimpleNamespace(**s) for s in metadata['schedule']],
             model=SimpleNamespace(cfg=cfg, motion=SimpleNamespace(kind='lattice')))
    fig = dashboard(a, title=cfg.name + ' - clarified 3D experiment')
    fig.savefig(folder / 'run_dashboard.png', dpi=130)
    fig.savefig(folder / 'run_dashboard.pdf')
    fig = carrier_calibration_figure(metadata['ensemble_details']['calibration'])
    fig.savefig(folder / 'run_carrier_calibration.png', dpi=140)
    fig.savefig(folder / 'run_carrier_calibration.pdf')


def comparison_config(data):
    """Controls may differ only in detuning, run label and worker count."""
    cfg = SimConfig.from_dict(data)
    if cfg.raman.frequency_mode != 'sideband_offset':
        raise ValueError('Detuning comparison requires frequency_mode = sideband_offset')
    cfg.name = ''
    cfg.raman.red_sideband_offset_hz = 0.
    cfg.ensemble.workers = 1
    return cfg.to_dict()


def main():
    cfg = SimConfig.from_json(ROOT / 'examples/experiment.json')
    out = output_root(cfg)
    current = json.loads((out / 'final/run_metadata.json').read_text())
    control = out / 'zero_offset_check/run_metadata.json'
    if not control.exists():
        raise ValueError('Run scripts/check_experiment_controls.py before selecting a detuning')
    alternative = json.loads(control.read_text())
    if not (comparison_config(cfg.to_dict()) == comparison_config(current['config'])
            == comparison_config(alternative['config'])):
        raise ValueError('Saved runs differ in physics or numerical settings; rerun the final configuration and matched controls')
    choices = [dict(offset_hz=m['config']['raman']['red_sideband_offset_hz'],
                    target_absolute=m['final']['P_target_absolute'], survival=m['final']['P_trapped'],
                    validity=m['validity_status']) for m in (current, alternative)]
    if alternative['validity_status'] != 'invalid' and alternative['final']['P_target_absolute'] > current['final']['P_target_absolute']:
        source = (out / 'final').resolve()
        archived = (out / 'previous_final').resolve()
        old_convergence = (out / 'spatial_convergence.json').resolve()
        archived_convergence = (out / 'previous_spatial_convergence.json').resolve()
        if not all(p.is_relative_to(out.resolve()) for p in (source, archived, old_convergence, archived_convergence)):
            raise ValueError('Archive paths must stay inside the experiment results directory')
        if archived.exists() or archived_convergence.exists():
            raise RuntimeError('Refusing to overwrite the archived detuning check')
        source.rename(archived)
        shutil.copytree(out / 'zero_offset_check', out / 'final')
        current = alternative
        SimConfig.from_dict(current['config']).to_json(ROOT / 'examples/experiment.json')
        if old_convergence.exists():
            old_convergence.rename(archived_convergence)
        nodes = current['model_summary']['node_results']
        convergence = []
        for n in (k for k in (4, 8, 16, 32, 64) if k <= len(nodes)):
            target = float(np.mean([v['target_absolute'] for v in nodes[:n]]))
            survival = float(np.mean([v['survival'] for v in nodes[:n]]))
            convergence.append(dict(samples=n, final=dict(P_target_absolute=target, P_trapped=survival, P_target=target/survival)))
        (out / 'spatial_convergence.json').write_text(json.dumps(convergence, indent=2))
    selection = dict(objective='P_target_absolute', checks=choices,
                     selected_offset_hz=current['config']['raman']['red_sideband_offset_hz'],
                     note='Best of the matched 64-node checks; finite search under the reported calibration/model assumptions.')
    (out / 'final_selection.json').write_text(json.dumps(selection, indent=2))
    render(out / 'final', current)
    print(selection, flush=True)


if __name__ == '__main__':
    main()
