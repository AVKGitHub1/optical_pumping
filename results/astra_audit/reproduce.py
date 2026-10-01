"""Run from the repository root using `conda run -n scripts python ...`.

Baseline artifacts in before/ were generated before the corrections. This
script regenerates after/, convergence and explicitly hypothetical sensitivities.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import eigh_tridiagonal

from rb85rsc.atomic import H, HBAR
from rb85rsc.config import SimConfig, set_param
from rb85rsc.runner import export, metadata, run

OUT = ROOT / "results/astra_audit"
base = SimConfig.from_json(ROOT / "tests/fixtures/audit_experiment_1d.json")
cases = {
    "corrected": {},
    "rtol_1e-9": {"numerics.rtol": 1e-9, "numerics.atol": 1e-12},
    "quadrature_48": {"numerics.recoil_quadrature_points": 48},
    "all_bound_sidebands": {"raman.max_sideband_order": 5},
    "nmax_20": {"trap.n_max": 20},
    "repump_along_z": {"repump.polarization.beam_azimuth_deg": 90},
    "kramers_heisenberg": {"optical.path_model": "kramers_heisenberg"},
    "phase_0": {"raman.lattice_phase_deg": 0},
    "phase_45": {"raman.lattice_phase_deg": 45},
    "phase_135": {"raman.lattice_phase_deg": 135},
    "raman_scatter_100_per_s": {"raman.scattering_rate_s": 100},
    "raman_scatter_1000_per_s": {"raman.scattering_rate_s": 1000},
    "optical_pumping_control": {"raman.enabled": False},
}
results = {}
for name, changes in cases.items():
    cfg = base.copy()
    for path, value in changes.items():
        set_param(cfg, path, value)
    a = run(cfg)
    if name == "corrected":
        export(a, OUT / "after")
        reference = a
    results[name] = {"changes": changes, "final": a["final"],
                     "validity": metadata(a)["validity"], "model_summary": a["model_summary"]}
    f = a["final"]
    print(name, {k: f[k] for k in ["P_target", "P_target_absolute", "P_trapped", "nbar", "T_equiv_K"]}, flush=True)

# Independent finer finite-difference grid and a doubled exterior domain.
m = reference["model"]
site = m.motion
grid_checks = []
for step, half_sites in [(0.025, 2.5), (0.05, 5.0), (0.025, 5.0)]:
    dz = step * site.x0
    spacing = np.pi / site.k_lattice
    count = int(np.ceil(half_sites * spacing / dz))
    z = dz * np.arange(-count, count + 1)
    v = np.where(np.abs(z) <= spacing / 2,
                 site.depth_hz * np.sin(site.k_lattice * z)**2, site.depth_hz)
    kin = HBAR**2 / (2 * m.atom.mass_kg * dz**2 * H)
    e, vec = eigh_tridiagonal(v + 2 * kin, np.full(len(z) - 1, -kin),
                             select="v", select_range=(-np.inf, site.depth_hz))
    overlap = vec.T @ (np.exp(1j * m.eta_R_signed / site.x0 * z)[:, None] * vec)
    grid_checks.append({"step_x0": step, "half_width_sites": half_sites, "bound_levels": len(e),
                        "max_level_difference_hz": float(np.max(np.abs(e - site.energies_hz))),
                        "max_gap_difference_hz": float(np.max(np.abs(np.diff(e) - np.diff(site.energies_hz)))),
                        "max_overlap_magnitude_difference": float(np.max(np.abs(np.abs(overlap) - np.abs(site.displacement(m.eta_R_signed))))),
                        "levels_hz": e.tolist()})

(OUT / "audit_results.json").write_text(json.dumps({"cases": results, "grid_checks": grid_checks}, indent=2), encoding="utf-8")
print("GRID", json.dumps(grid_checks, indent=2), flush=True)
