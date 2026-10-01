"""Single-photon check of the Cartesian-marginal approximation at cloud center.

This diagnoses one recoil event on the initial thermal state. It does not bound
the cumulative error of a cooling sequence or the resolved-dipole approximation.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rb85rsc import atomic as at, motion as mo, polarization as pol
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import _single_thread
from optimize_experiment import output_root


def main():
    c = SimConfig.from_json(ROOT / 'examples/experiment.json')
    atom = at.load_atomic_data()
    wavelengths = [c.ensemble.transverse_wavelength_nm] * 2 + [c.trap.wavelength_nm]
    depths = [c.ensemble.transverse_depth_uk] * 2 + [c.trap.depth_uk]
    temps = [c.ensemble.transverse_temperature_k] * 2 + [c.initial.temperature_k]
    sites = [mo.lattice_site(atom.mass_kg, mo.lattice_frequency_hz(atom.mass_kg, w, d), d * 1e-6 * at.K_B / at.H, c.trap.n_max + 1)
             for w, d in zip(wavelengths, depths)]
    pn = [mo.boltzmann_pn(s.energies_hz, t) for s, t in zip(sites, temps)]
    bdir = pol.unit(c.magnetic.direction)
    k = 2 * np.pi / atom.d2_wavelength_m
    rows = []
    with _single_thread():
        for nq in (16, 32):
            cosb, weights = np.polynomial.legendre.leggauss(nq)
            phi = 2 * np.pi * np.arange(2 * nq) / (2 * nq)
            cb, ph = np.meshgrid(cosb, phi, indexing='ij')
            ub = np.stack([np.sqrt(1 - cb**2) * np.cos(ph), np.sqrt(1 - cb**2) * np.sin(ph), cb], axis=-1).reshape(-1, 3)
            u = ub @ pol.field_frame(bdir)
            for absorb_axis in (0, 2):
                direction = np.eye(3)[absorb_axis]
                retained = np.ones(len(u))
                for axis, (site, p) in enumerate(zip(sites, pn)):
                    retained *= np.asarray([(abs(site.displacement(k * site.x0 * (direction[axis] - v)))**2).sum(axis=0) @ p for v in u[:, axis]])
                for pattern in ('sigma', 'pi'):
                    w = np.repeat(weights * mo.emission_weight(cosb, 1., pattern) / len(phi), len(phi))
                    exact = float(w @ retained)
                    factor = 1.
                    for axis, (site, p) in enumerate(zip(sites, pn)):
                        K, _ = site.recoil_kernel(k * site.x0, direction[axis], bdir[axis], pattern, nq)
                        factor *= K.sum(axis=0) @ p
                    rows.append(dict(polar_quadrature=nq, azimuth_points=2*nq, absorbed_along='xyz'[absorb_axis],
                                     emitted_dipole_class=pattern, correlated_3d_survival=exact,
                                     marginal_product_survival=float(factor), difference=float(factor-exact)))
    out = output_root(c) / 'one_photon_recoil_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(note=__doc__, cases=rows), indent=2))
    print(json.dumps(rows[-4:], indent=2), flush=True)


if __name__ == '__main__':
    main()
