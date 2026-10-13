"""GUI/headless results, diagnostics and export for coherent pumping."""
import csv
import json
from pathlib import Path

import numpy as np

from . import atomic as at
from .coherent_pumping import run_cloud
from .model import Validity
from .protocols import build_schedule
from .trajectory_observables import analyze_observables


ASSUMPTIONS = [
    'Weak-excitation optical elimination; ground spins and spherical emission channels remain resolved.',
    'Motional recoil coherence and a shared photon direction are retained on all three axes. Raman light is off.',
    'Fixed separable scalar wells and diagonal differential Stark shifts; no tunnelling, transport or recapture.',
    'Initial states are conditioned on local bound levels; loading survival is unspecified.',
    'Monte Carlo errors describe estimated means at fixed spatial and angular quadratures, not experimental or model uncertainty.',
]


def final_estimates(result):
    return {key: dict(mean=float(value[-1]), standard_error=float(result['standard_error'][key][-1]))
            for key, value in result['mean'].items()}


def run(cfg, *, stop=None, progress=None, workers=None):
    cfg = cfg.copy()
    raw = run_cloud(cfg, trajectories=cfg.ensemble.trajectories_per_site, seed=cfg.ensemble.trajectory_seed,
                    workers=workers, modes=('coherent',), stop=stop, progress=progress)
    validity = Validity()
    diagnostics = [node['diagnostics'] for node in raw['nodes']]
    peak = max(d['max_excited_fraction'] for d in diagnostics)
    validity.add('weak excitation', 'invalid' if peak >= cfg.optical.weak_excitation_invalid else
                 ('warning' if peak >= cfg.optical.weak_excitation_warning else 'ok'),
                 f'Maximum eliminated excited fraction across cloud sites = {peak:.4g}.')
    ratio = min((d['ground_angular_spacing_s']/d['max_photon_rate_bound_s']
                 if d['max_photon_rate_bound_s'] > 0 else np.inf) for d in diagnostics)
    validity.add('ground coherences', 'ok' if ratio > 10 else ('warning' if ratio > 1 else 'invalid'),
                 f'Adjacent-spin angular splitting / conservative full photon-rate bound = {ratio:.3g}; requires >> 1.')
    magnetic = max((d['magnetic'] for d in diagnostics), key=lambda m: {'ok': 0, 'warning': 1, 'invalid': 2}[m['status']])
    validity.add('magnetic field', magnetic['status'], '; '.join(magnetic['notes']) or at.zeeman_model_description(cfg.optical.zeeman_model))
    validity.add('motional recoil', 'ok', 'Coherent wavefunctions retained on x/y/z; the population motional-secular criterion does not apply to this solver.')
    conservation = max(node['modes']['coherent']['max_conservation_error'] for node in raw['nodes'])
    min_eigenvalue = min(float(np.linalg.eigvalsh(rho).min()) for node in raw['nodes']
                        for rho in node['modes']['coherent']['rho_axes_final'])
    validity.add('conservation', 'ok' if conservation < cfg.numerics.conservation_tolerance else 'invalid',
                 f'Maximum |survival + escape - 1| = {conservation:.3g}.')
    validity.add('positivity', 'ok' if min_eigenvalue >= -cfg.numerics.positivity_tolerance else 'invalid',
                 f'Minimum final reduced-density eigenvalue = {min_eigenvalue:.3g}.')
    tail = float(np.mean([node['initial_tail_discarded'] for node in raw['nodes']]))
    if tail > cfg.numerics.boundary_tolerance:
        validity.add('initial distribution', 'warning', f'Mean axial weight outside local bound states = {tail:.3g}; each site is conditioned on retained levels.')
    validity.add('Monte Carlo precision', 'warning', f"{raw['trajectories_per_site']} trajectories per site; repeat with more trajectories and independent seeds. Rare unobserved events may have zero empirical error.")
    validity.add('spatial and angular quadrature', 'warning', f"{raw['sites']} frozen cloud sites, {cfg.numerics.recoil_quadrature_points} polar nodes; refine separately. Error bands exclude quadrature error.")
    validity.add('remaining model assumptions', 'warning', 'Resolved spin/emission channels, diagonal Stark shifts, frozen isolated wells, no recapture; Raman coupling is unsupported.')
    validity.add('thermal observables', 'warning', 'Energy and temperature describe survivors and include loss selection. Temperature is an energy-matched proxy; coherent loss-selection decomposition is not recorded.')
    analysis = analyze_observables(cfg, raw, stop)
    analysis['final'].update(wall_time_s=raw['wall_time_s'], solver=raw['solver'],
                 max_excited_fraction=peak, max_conservation_error=conservation, min_eigenvalue=min_eigenvalue)
    return dict(kind='coherent_pumping', config=cfg, trajectory_result=raw,
                **analysis, validity=validity, segments=build_schedule(cfg.timing))


def diagnostics_text(a):
    cfg, raw = a['config'], a['trajectory_result']
    target = at.ground_label(cfg.raman.state_indices()[0])
    lines = [f"MODEL VALIDITY: {a['validity'].status.upper()}",
             'Solver: 3D coherent optical-pumping trajectories (Raman off).',
             f"Target: spin {target}, local {a['final']['raman_axis']} ground state.",
             f"{raw['sites']} cloud sites x {raw['trajectories_per_site']} trajectories; "
             f"trajectory seed {raw['trajectory_seed']}, spatial seed {raw['spatial_seed']}.",
             'Means +/- one-sigma Monte Carlo standard errors.',
             'Absolute probabilities use initially loaded bound atoms; conditional distributions use survivors.',
             'Conditional errors include survival covariance. Thermal derivatives use trajectory batches with temporal covariance.',
             'Temperature is an energy-matched proxy; net cooling includes loss selection (decomposition unavailable).', '']
    for key, value in final_estimates(raw['modes']['coherent']).items():
        lines.append(f"{key}: {value['mean']:.7g} +/- {value['standard_error']:.3g}")
    for axis in 'xyz':
        for key in (f'nbar_{axis}', f'T_equiv_{axis}_K', f'mean_excitation_energy_{axis}_hz'):
            lines.append(f"{key}: {a['series'][key][-1]:.7g} +/- {a['standard_error'][key][-1]:.3g}")
    lines += ['', f"Wall time: {a['final']['wall_time_s']:.2f} s.", '']
    lines.extend(f'[{status.upper()}] {category}: {message}' for category, status, message in a['validity'].items)
    return '\n'.join(lines)


def metadata(a):
    from .runner import versions, constants
    raw = a['trajectory_result']
    return dict(config=a['config'].to_dict(), final=a['final'], final_estimates=final_estimates(raw['modes']['coherent']),
                validity_status=a['validity'].status,
                validity=[dict(category=c, status=s, message=m) for c, s, m in a['validity'].items],
                schedule=[s.__dict__ for s in a['segments']], assumptions=ASSUMPTIONS,
                sampling={k: raw[k] for k in ('sites', 'trajectories_per_site', 'trajectory_seed', 'spatial_seed', 'workers')},
                versions=versions(), constants_SI=constants(), solver=raw['solver'],
                uncertainty=dict(interval='Pointwise approximate 95% Monte Carlo (mean +/- 1.96 SE)',
                    conditional='Delta method with within-site survival covariance; undefined when the denominator is zero.',
                    temperature='Implicit energy-match derivative including uncertain spatial survivor weights.',
                    cooling=a['cooling_rate_method'],
                    unavailable='Coherent loss-selection decomposition; uncertainty of heatmaps is stored in run_observables.npz.'),
                observable_final_estimates={k: dict(mean=a['final'][k],
                    standard_error=float(v[-1]) if np.isfinite(v[-1]) else None) for k, v in a['standard_error'].items()})


def save_trajectories(result, output):
    """Persist means, their errors, and spin-resolved per-axis density matrices."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays, summaries = {}, []
    for mode, values in result['modes'].items():
        arrays['t_s'] = values['t_s']
        arrays.update({f'{mode}_mean_{k}': v for k, v in values['mean'].items()})
        arrays.update({f'{mode}_se_{k}': v for k, v in values['standard_error'].items()})
        for stat, items in values.get('observables', {}).items():
            arrays.update({f'{mode}_observables_{stat}_{k}': v for k, v in items.items()})
    np.savez_compressed(output/'cloud_trajectories.npz', **arrays)
    (output/'nodes').mkdir(exist_ok=True)
    for node in result['nodes']:
        summary = {k: v for k, v in node.items() if k != 'modes'}
        summary['modes'], arrays = {}, {}
        for mode, values in node['modes'].items():
            summary['modes'][mode] = {k: v for k, v in values.items()
                                      if k not in ('t_s', 'mean', 'standard_error', 'rho_axes_final', 'observables')}
            summary['modes'][mode]['final'] = final_estimates(values)
            arrays['t_s'] = values['t_s']
            arrays.update({f'{mode}_mean_{k}': v for k, v in values['mean'].items()})
            arrays.update({f'{mode}_se_{k}': v for k, v in values['standard_error'].items()})
            arrays.update({f'{mode}_rho_{a}_final': v for a, v in zip('xyz', values['rho_axes_final'])})
            for stat, items in values.get('observables', {}).items():
                if isinstance(items, dict):
                    arrays.update({f'{mode}_observables_{stat}_{k}': v for k, v in items.items()})
                else:
                    arrays[f'{mode}_{stat}'] = items
        np.savez_compressed(output/'nodes'/f"site_{node['index']:03d}.npz", **arrays)
        summaries.append(summary)
    (output/'nodes.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


def export(a, outdir, *, figures=True, stem='run'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    a['config'].to_json(outdir/f'{stem}_config.json')
    (outdir/f'{stem}_metadata.json').write_text(json.dumps(metadata(a), indent=2), encoding='utf-8')
    (outdir/f'{stem}_diagnostics.txt').write_text(diagnostics_text(a), encoding='utf-8')
    save_trajectories(a['trajectory_result'], outdir)
    arrays = dict(t_s=a['series']['t_s'], spin=a['spin'], spin_absolute=a['spin_absolute'],
                  joint=a['joint'], joint_absolute=a['joint_absolute'], P_final=a['P_final'],
                  spin_labels=np.array([at.ground_label(i) for i in range(12)]),
                  target_axis=np.array(a['final']['raman_axis']),
                  description=np.array('spin[t,s], joint[t,s,n_target], pn_axes_axis[t,n] are conditioned on cloud survival; '
                      '*_absolute arrays are per initially loaded bound atom. Errors are one-sigma Monte Carlo errors.'))
    for key, value in a['array_standard_error'].items():
        if isinstance(value, dict):
            arrays.update({f'{key}_{axis}_standard_error': v for axis, v in value.items()})
        else:
            arrays[f'{key}_standard_error'] = value
    for key in ('pn_axes', 'pn_axes_absolute'):
        arrays.update({f'{key}_{axis}': v for axis, v in a[key].items()})
    np.savez_compressed(outdir/f'{stem}_observables.npz', **arrays)
    columns = {'t_s': a['series']['t_s']}
    for key, se in a['standard_error'].items():
        columns[key], columns[key+'_standard_error'] = a['series'][key], se
    with (outdir/f'{stem}_timeseries.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    if figures:
        from .plotting import dashboard, thermal_axes_figure, motional_axes_figure
        for name, fig in (('dashboard', dashboard(a, title=a['config'].name)),
                          ('thermal_axes', thermal_axes_figure(a)), ('motional_axes', motional_axes_figure(a))):
            fig.savefig(outdir/f'{stem}_{name}.png', dpi=150)
            fig.savefig(outdir/f'{stem}_{name}.pdf')
    return outdir
