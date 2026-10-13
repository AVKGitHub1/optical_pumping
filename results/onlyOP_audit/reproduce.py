"""Read-only physics review of onlyOP; saves separate audit results.

Run from the repository root in the scripts conda environment.
The production model, example config, and original outputs are not edited.
"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import expm
from rb85rsc import atomic as at, optical as op
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.runner import run

OUT = Path(__file__).resolve().parent


def input_checks(cfg, atom):
    """Independent, inexpensive interpretation of the input and saved curves."""
    checks = {'central_beams': {}, 'pure_sigma_plus_target': {}}
    for name in ('spin_pump', 'repump'):
        spec = op.beam_spec(name, getattr(cfg, name), atom, cfg.magnetic.direction)
        rates = op.beam_rates(atom, spec, cfg.magnetic.magnitude_gauss * 1e-4,
                             path_model=cfg.optical.path_model)
        checks['central_beams'][name] = dict(
            direction=spec.direction.tolist(), fractions=spec.fractions,
            intensity_w_m2=spec.intensity_w_m2, max_scattering_s=float(rates.gamma_out.max()),
            target_scattering_s=float(rates.gamma_out[at.UP]),
            target_spin_changing_s=float(rates.gamma_out[at.UP] - rates.W[at.UP, at.UP].sum()))
    pure = cfg.copy()
    pure.spin_pump.polarization.sigma_minus_fraction = 0.
    spec = op.beam_spec('spin_pump', pure.spin_pump, atom, pure.magnetic.direction)
    rates = op.beam_rates(atom, spec, pure.magnetic.magnitude_gauss * 1e-4,
                         path_model=pure.optical.path_model)
    checks['pure_sigma_plus_target'] = dict(scattering_s=float(rates.gamma_out[at.UP]),
        spin_changing_s=float(rates.gamma_out[at.UP] - rates.W[at.UP, at.UP].sum()))
    with (ROOT / 'examples/onlyOP/run_timeseries.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checks['peaks_in_saved_run'] = {}
    for key in ('P_up_absolute', 'P_target_absolute'):
        row = max(rows, key=lambda r: float(r[key]))
        checks['peaks_in_saved_run'][key] = {k: float(row[k]) for k in ('t_s', key, 'P_up', 'P_trapped')}
    checks['net_minus_selection_cooling_power_W'] = {}
    for axis in 'xyz':
        values = [float(r[f'bound_cooling_power_{axis}_W']) for r in rows]
        checks['net_minus_selection_cooling_power_W'][axis] = dict(min=min(values), max=max(values))
    return checks


def main():
    cfg = SimConfig.from_json(ROOT / 'examples/onlyOP.json')
    saved = json.loads((ROOT / 'examples/onlyOP/run_metadata.json').read_text())
    atom = at.load_atomic_data()
    report = dict(
        config_matches_saved=cfg.to_dict() == saved['config'],
        magnetic_validity=at.zeeman_validity(atom, cfg.magnetic.magnitude_gauss * 1e-4, cfg.trap.frequency_hz),
        atomic_checks=at.steck_comparison(atom),
        recoil_energy_uk=at.HBAR**2 * (2*np.pi/atom.d2_wavelength_m)**2 / (2*atom.mass_kg*at.K_B) * 1e6,
        cases={},
    )
    report['input_checks'] = input_checks(cfg, atom)
    keys = ('P_up', 'P_n0', 'P_target_absolute', 'P_trapped', 'T_equiv_axes_K',
            'photons_total', 'max_motional_secular_ratio', 'max_excited_fraction',
            'max_conservation_error', 'P_target_3d_absolute', 'wall_time_s')
    for name, samples, impurity in [('reproduce64', 64, .01), ('spatial128', 128, .01),
                                    ('spatial256', 256, .01), ('pure_sigma64', 64, 0.)]:
        c = cfg.copy()
        c.ensemble.samples = samples
        c.spin_pump.polarization.sigma_minus_fraction = impurity
        print('Running', name, flush=True)
        a = run(c)
        row = {k: a['final'][k] for k in keys}
        row['validity_status'] = a['validity'].status
        if name == 'reproduce64':
            row['saved_final_max_abs_difference'] = max(abs(a['final'][k] - saved['final'][k])
                for k in keys if k not in ('T_equiv_axes_K', 'wall_time_s'))
            old = list(csv.DictReader((ROOT / 'examples/onlyOP/run_timeseries.csv').open()))
            row['saved_csv_max_abs_difference'] = max(
                np.max(np.abs(a['series'][k] - np.asarray([float(r[k]) for r in old])))
                for k in ('P_up', 'P_n0', 'P_trapped', 'P_target_absolute'))
            # Hypothetical spin-only control: same local illumination, no recoil,
            # trap loss or lattice scattering. Exactly exponentiate 12-state rates.
            spins = []
            details = []
            for local, result in a['ensemble_nodes']:
                m = local.model
                W = sum(m.rates[b].W.sum(axis=2) for b in ('spin_pump', 'repump'))
                generator = W.T - np.diag(W.sum(axis=1))
                p0 = local.P0.sum(axis=(1, 2, 3))
                spins.append(expm(generator * .001) @ p0)
                gaps = [np.min(np.diff(s.energies_hz)) if s.N > 1 else s.trap_hz for s in local.sites]
                rate = max(sum(m.rates[b].gamma_out for b in ('spin_pump', 'repump')))
                details.append(dict(position_um=local.position.tolist(), depths_uk=local.depths.tolist(),
                    min_gap_hz=float(min(gaps)), max_rate_s=float(rate),
                    secular_ratio=float(rate/(2*np.pi*min(gaps)))))
            row['spin_only_no_loss_P_up'] = float(np.mean(spins, axis=0)[at.UP])
            row['node_diagnostics'] = details
            row['nodes_above_secular_invalid_threshold'] = sum(d['secular_ratio'] >= .5 for d in details)
        report['cases'][name] = row
        (OUT / 'audit_results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k != 'node_diagnostics'}), flush=True)


if __name__ == '__main__':
    with _single_thread():
        main()
