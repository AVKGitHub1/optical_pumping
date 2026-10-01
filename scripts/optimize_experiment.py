"""Reproducible bounded search for the clarified example; run in conda scripts.

Coarse power scan covers 3 nW--3 uW. Points outside weak-excitation or
motional-secular validity are recorded and excluded, never used as optima.
The search is a finite coordinate/grid search, not proof of a global optimum.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rb85rsc.config import SimConfig
from rb85rsc.runner import run, export, _json_default
from rb85rsc.far_detuned import DLineResponse, calibrate_carrier, spatial_samples
from rb85rsc.ensemble import LocalExperiment
from rb85rsc.atomic import load_atomic_data
from rb85rsc.dynamics import _single_thread

def output_root(cfg):
    """Keep results under different calibration assumptions separate."""
    return ROOT / 'results' / ('astra_modeled_decay' if cfg.raman.carrier_decay_mode == 'modeled' else 'astra_clarified')


def evaluate(item):
    tag, data = item
    c = SimConfig.from_dict(data)
    # The batch pool owns the worker budget; retain the requested config for
    # result provenance and resume matching rather than rewriting its workers.
    a = run(c, ensemble_workers=1)
    return dict(tag=tag, config=data, final=a['final'], validity=a['validity'].items,
                status=a['validity'].status)


def batch(items, path, workers):
    items = list(items)
    requested = {tag: SimConfig.from_dict(data).to_dict() for tag, data in items}
    if len(requested) != len(items):
        raise ValueError('Batch tags must be unique')
    saved = json.loads(path.read_text()) if path.exists() else []
    # A tag names the scan coordinates, not all physics inputs. Never reuse
    # results from a different temperature, calibration, geometry or timestep.
    old = [r for r in saved if r['tag'] in requested
           and SimConfig.from_dict(r['config']).to_dict() == requested[r['tag']]]
    complete = {r['tag'] for r in old}
    remaining = [item for item in items if item[0] not in complete]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=2, default=_json_default))
    if not remaining:
        return old
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate, item): item[0] for item in remaining}
        for future in as_completed(futures):
            row = future.result()
            old.append(row)
            path.write_text(json.dumps(old, indent=2, default=_json_default))
            f = row['final']
            print(row['tag'], 'target_absolute', round(f['P_target_absolute'], 7),
                  'survival', round(f['P_trapped'], 6), row['status'], flush=True)
    return old


def best(rows):
    valid = [r for r in rows if r['status'] != 'invalid' and np.isfinite(r['final']['P_target_absolute'])]
    if not valid:
        raise RuntimeError('No valid candidate in the sampled domain')
    return max(valid, key=lambda r: r['final']['P_target_absolute'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'refine', 'detuning', 'spatial_refine', 'polarizations', 'final'], default='screen')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--use-selected', action='store_true', help='For --stage final, explicitly promote the saved optimization winner; otherwise rerun examples/experiment.json')
    args = ap.parse_args()
    c = SimConfig.from_json(ROOT / 'examples/experiment.json')
    out = output_root(c)
    out.mkdir(parents=True, exist_ok=True)
    c.ensemble.samples = 4
    c.timing.n_samples = 51
    if args.stage == 'screen':
        # Construct geometry once per repump polarization. Linearity of optical
        # rates allows exact static rejection throughout the coarse power grid.
        rejects, tasks = [], []
        response = DLineResponse(load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)
        calibration = calibrate_carrier(c, response)
        xyz, phases = spatial_samples(c)
        with _single_thread():
            for orient in (0., 90.):
                for impurity in (0., .0048861861):
                    c.repump.polarization.orientation_deg = orient
                    c.spin_pump.polarization.sigma_minus_fraction = impurity
                    c.spin_pump.power_mw = c.repump.power_mw = 1e-6  # 1 nW reference
                    nodes = [LocalExperiment(c, response, calibration, p, th) for p, th in zip(xyz, phases)]
                    for pump, repump in itertools.product((3., 10., 30., 100., 300., 1000., 3000.), repeat=2):
                        pe = max(float((pump * l.model.rates['spin_pump'].p_exc + repump * l.model.rates['repump'].p_exc).max()) for l in nodes)
                        secular = max(float((pump * l.model.rates['spin_pump'].gamma_out + repump * l.model.rates['repump'].gamma_out).max()) /
                                      (2 * np.pi * min(np.diff(s.energies_hz).min() for s in l.sites if s.N > 1)) for l in nodes)
                        tag = f'p{pump:g}_r{repump:g}_linear{orient:g}_imp{impurity:g}'
                        trial = c.copy()
                        trial.spin_pump.power_mw = pump * 1e-6
                        trial.repump.power_mw = repump * 1e-6
                        if pe >= c.optical.weak_excitation_invalid or secular >= .5:
                            rejects.append(dict(tag=tag, pump_nw=pump, repump_nw=repump,
                                                max_excited_fraction=pe, motional_secular_ratio=secular,
                                                reason='outside weak-excitation or motional-secular domain'))
                        else:
                            tasks.append((tag, trial.to_dict()))
        (out / 'power_scan_rejected.json').write_text(json.dumps(rejects, indent=2))
        print('Coarse grid:', len(tasks), 'admissible,', len(rejects), 'rejected', flush=True)
        rows = batch(tasks, out / 'power_scan.json', args.workers)
        winner = best(rows)
    elif args.stage == 'refine':
        previous = json.loads((out / 'power_scan.json').read_text())
        winner = best(previous)
        c = SimConfig.from_dict(winner['config'])
        p0, r0 = c.spin_pump.power_mw / 1e-6, c.repump.power_mw / 1e-6
        tasks = []
        for p, r in itertools.product(sorted(set([3., p0, p0 * 1.6, p0 * 2.5])), sorted(set([3., r0, r0 * 1.6, r0 * 2.5]))):
            cc = c.copy()
            cc.spin_pump.power_mw, cc.repump.power_mw = p * 1e-6, r * 1e-6
            tasks.append((f'p{p:g}_r{r:g}', cc.to_dict()))
        rows = batch(tasks, out / 'power_refinement.json', args.workers)
        winner = best(previous + rows)
    elif args.stage == 'detuning':
        winner = json.loads((out / 'selected.json').read_text())
        c = SimConfig.from_dict(winner['config'])
        tasks = []
        for det in (-30000., -20000., -10000., 0., 10000.):
            cc = c.copy()
            cc.raman.red_sideband_offset_hz = det
            tasks.append((f'red_offset_{det:g}_hz', cc.to_dict()))
        rows = batch(tasks, out / 'detuning_scan.json', args.workers)
        winner = best(rows)
    elif args.stage == 'spatial_refine':
        winner = json.loads((out / 'selected.json').read_text())
        c = SimConfig.from_dict(winner['config'])
        c.ensemble.samples = 64
        response = DLineResponse(load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)
        calibration = calibrate_carrier(c, response)
        xyz, phases = spatial_samples(c)
        with _single_thread():
            nodes = [LocalExperiment(c, response, calibration, p, ph) for p, ph in zip(xyz, phases)]
        checks, tasks = [], []
        for repump in (3., 6., 7.5, 10.):
            factor = repump * 1e-6 / c.repump.power_mw
            secular = max(float((l.model.rates['spin_pump'].gamma_out + factor * l.model.rates['repump'].gamma_out).max()) /
                          (2 * np.pi * min((np.diff(s.energies_hz).min() for s in l.sites if s.N > 1), default=min(s.trap_hz for s in l.sites))) for l in nodes)
            checks.append(dict(repump_nw=repump, max_motional_secular_ratio_64_nodes=secular, admissible=secular < .5))
            if secular >= .5:
                continue
            for delta in (-5000., 0., 5000.):
                cc = c.copy()
                cc.ensemble.samples = 8
                cc.repump.power_mw = repump * 1e-6
                cc.raman.red_sideband_offset_hz += delta
                tasks.append((f'repump_{repump:g}_offset_{cc.raman.red_sideband_offset_hz:g}', cc.to_dict()))
        (out / 'full_cloud_validity.json').write_text(json.dumps(checks, indent=2, default=_json_default))
        rows = batch(tasks, out / 'spatial_refinement.json', args.workers)
        winner = best(rows)
    elif args.stage == 'polarizations':
        winner = json.loads((out / 'selected.json').read_text())
        c = SimConfig.from_dict(winner['config'])
        c.ensemble.samples = 4
        # For k || B the two transverse linear directions are degenerate in
        # the population/weak-field model, so choose E_x = lab y once.
        choices = ([[0,1,0]], [[1,0,0], [0,0,1]], [[1,0,0], [0,1,0]])
        tasks = []
        for i, vectors in enumerate(itertools.product(*choices)):
            cc = c.copy()
            cc.ensemble.lattice_linear_polarizations = list(vectors)
            tasks.append((f'lattice_pol_{i}', cc.to_dict()))
        rows = batch(tasks, out / 'lattice_polarization_scan.json', args.workers)
        winner = best(rows)
        # Equivalent transverse linear orientations can differ at roundoff.
        # Prefer the first geometry when the objective is tied within 1e-5.
        ties = [r for r in rows if r['status'] != 'invalid' and r['final']['P_target_absolute'] >= winner['final']['P_target_absolute'] - 1e-5]
        winner = min(ties, key=lambda r: r['tag'])
    else:
        # Do not silently replace user edits with an old optimization winner.
        if args.use_selected:
            winner = json.loads((out / 'selected.json').read_text())
            c = SimConfig.from_dict(winner['config'])
        c.name = 'clarified_3d_experiment'
        from rb85rsc.motion import lattice_frequency_hz
        c.trap.frequency_hz = lattice_frequency_hz(load_atomic_data().mass_kg, c.trap.wavelength_nm, c.trap.depth_uk)
        c.raman.lattice_phase_deg = 0.  # inactive: uniform spatial registration is enabled
        c.ensemble.samples = 64
        c.ensemble.workers = args.workers
        c.timing.n_samples = 201
        c.to_json(ROOT / 'examples/experiment.json')
        a = run(c, progress=lambda f, m: print(f'{f:.1%} {m}', flush=True))
        export(a, out / 'final', figures=True)
        from rb85rsc.ensemble import analyze_ensemble
        response = DLineResponse(load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)
        calibration = calibrate_carrier(c, response)
        central = LocalExperiment(c, response, calibration, [0,0,0], 0.)
        convergence = []
        for count in (4, 8, 16, 32, 64):
            cc = c.copy()
            cc.ensemble.samples = count
            aa = analyze_ensemble(cc, central, a['ensemble_nodes'][:count], calibration, a['model_summary']['nu_beat_hz'])
            convergence.append(dict(samples=count, final=aa['final']))
        (out / 'spatial_convergence.json').write_text(json.dumps(convergence, indent=2, default=_json_default))
        print({k:a['final'][k] for k in ('P_target_absolute','P_trapped','T_equiv_axes_K','well_loss_fraction_no_recapture','max_conservation_error')}, flush=True)
        return
    (out / 'selected.json').write_text(json.dumps(winner, indent=2, default=_json_default))
    print('SELECTED', winner['tag'], winner['final']['P_target_absolute'], flush=True)


if __name__ == '__main__':
    main()
