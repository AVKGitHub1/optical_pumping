"""Single-photon check of the Cartesian-marginal approximation in local wells.

This diagnoses one recoil event on the configured initial state. It does not bound
the cumulative error of a cooling sequence or the resolved-dipole approximation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rb85rsc import atomic as at, motion as mo, polarization as pol
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.far_detuned import envelope
from rb85rsc.model import Model
from rb85rsc.recoil_reference import validation_positions
from optimize_experiment import output_root


def compare_single_photon(c, position_um=(0., 0., 0.), quadratures=(16, 32)):
    c.validate()
    position = np.asarray(validation_positions(c, position_um=position_um)[0]['position_um'])
    if c.trap.potential != 'lattice':
        raise ValueError('single-photon well-loss check requires a lattice')
    if not quadratures or any(not isinstance(n, (int, np.integer)) or n < 2 for n in quadratures):
        raise ValueError('quadratures must contain integers >= 2')
    atom = at.load_atomic_data()
    axes = pol.lattice_frame(c.trap.axis, c.ensemble.transverse_reference_direction)
    specs = [c.lattice_parameters(i, atom.mass_kg) for i in range(3)]
    wavelengths = [s.wavelength_nm for s in specs]
    depths = [s.depth_uk*envelope(position, direction, s.waist_um) for s, direction in zip(specs, axes)]
    sites = [mo.lattice_site(atom.mass_kg, mo.lattice_frequency_hz(atom.mass_kg, w, d), d * 1e-6 * at.K_B / at.H, c.trap.n_max + 1)
             for w, d in zip(wavelengths, depths)]
    if any(s.N != s.n_bound for s in sites):
        raise ValueError('single-photon check requires all local bound levels; increase trap.n_max')
    # Match production preparation, including a nonthermal axial input.
    preparation = c.copy()
    preparation.ensemble.enabled = False
    preparation.ensemble.recoil_model = 'cartesian_marginals'  # used only to prepare one-axis populations
    preparation.raman.enabled = False
    preparation.raman.calibration = 'pathway'
    model = Model(preparation, atom, motion_axis=axes[2], motion_basis=sites[2])
    pn = [mo.boltzmann_pn(s.energies_hz, c.ensemble.transverse_temperature_k) for s in sites[:2]]
    pn.append(model.P0.sum(axis=0))
    bdir = pol.unit(c.magnetic.direction)
    b_local = axes @ bdir
    k = 2 * np.pi / atom.d2_wavelength_m
    rows = []
    with _single_thread():
        for nq in quadratures:
            cosb, weights = np.polynomial.legendre.leggauss(nq)
            phi = 2 * np.pi * np.arange(2 * nq) / (2 * nq)
            cb, ph = np.meshgrid(cosb, phi, indexing='ij')
            ub = np.stack([np.sqrt(1 - cb**2) * np.cos(ph), np.sqrt(1 - cb**2) * np.sin(ph), cb], axis=-1).reshape(-1, 3)
            # Integrate in a B-aligned frame defined relative to the lattice.
            # Rigid rotation then preserves even the finite angular quadrature.
            u = ub @ pol.field_frame(b_local)
            for absorb_axis in range(3):
                direction = np.eye(3)[absorb_axis]
                retained = np.ones(len(u))
                for axis, (site, p) in enumerate(zip(sites, pn)):
                    retained *= np.asarray([(abs(site.displacement(k * site.x0 * (direction[axis] - v)))**2).sum(axis=0) @ p for v in u[:, axis]])
                patterns = ('sigma', 'pi') if c.optical.emission_pattern == 'dipole' else ('isotropic',)
                for pattern in patterns:
                    w = np.repeat(weights * mo.emission_weight(cosb, 1., pattern) / len(phi), len(phi))
                    exact = float(w @ retained)
                    factor = 1.
                    for axis, (site, p) in enumerate(zip(sites, pn)):
                        K, _ = site.recoil_kernel(k * site.x0, direction[axis], b_local[axis], pattern, nq)
                        factor *= K.sum(axis=0) @ p
                    rows.append(dict(polar_quadrature=nq, azimuth_points=2*nq, absorbed_along='xyz'[absorb_axis],
                                     absorption_direction_lab=axes[absorb_axis].tolist(),
                                     emitted_dipole_class=pattern, correlated_3d_survival=exact,
                                     marginal_product_survival=float(factor), difference=float(factor-exact)))
    return dict(position_um=position.tolist(), depths_uk=depths, wavelengths_nm=wavelengths,
                lattice_axes_lab=axes.tolist(), bound_levels=[s.N for s in sites],
                initial_motional_populations=[p.tolist() for p in pn], cases=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'examples/experiment.json')
    parser.add_argument('--output', type=Path, help='JSON output file.')
    parser.add_argument('--position-um', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--cloud-samples', type=int, default=0)
    parser.add_argument('--quadratures', type=int, nargs='+', default=[16, 32])
    args = parser.parse_args(argv)
    c = SimConfig.from_json(args.config)
    try:
        positions = validation_positions(c, args.cloud_samples, args.position_um)
    except ValueError as error:
        parser.error(str(error))
    if any(n < 2 for n in args.quadratures):
        parser.error('quadratures must contain integers >= 2')
    out = args.output or output_root(c) / 'one_photon_recoil_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    for point in positions:
        try:
            check = dict(site_id=point['site_id'], status='complete',
                         **compare_single_photon(c, point['position_um'], args.quadratures))
        except (ValueError, RuntimeError) as error:
            check = dict(**point, status='failed', error=str(error), cases=[])
        checks.append(check)
        # Preserve the original central-case field for existing consumers.
        complete = sum(s['status'] == 'complete' for s in checks)
        out.write_text(json.dumps(dict(note=__doc__, config=c.to_dict(), sites=checks,
                                       expected_sites=len(positions), completed_sites=complete,
                                       cases=checks[0]['cases']), indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {len(checks)} local single-photon checks to {out}', flush=True)
    return 0 if complete == len(positions) else 1


if __name__ == '__main__':
    raise SystemExit(main())
