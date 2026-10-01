"""High-level run / compare / export helpers shared by the CLI and GUI (no Qt imports)."""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import atomic
from .analysis import analyze
from .config import SimConfig
from .dynamics import estimate_cost, solve
from .model import Model
from .protocols import COMPARISON_SET

ASSUMPTIONS = [
    "Single atom, one-dimensional motion along trap.axis; P_n0 refers to the modeled axis only (not a 3D ground-state fraction).",
    "Same configured potential for all ground spin states; no collisions or photon reabsorption. Harmonic motion is infinitely deep; lattice motion uses anharmonic bound levels and recoil escape.",
    "D2 excited manifold (F'=1..4, all sublevels) adiabatically eliminated in the weak-excitation limit; excited fraction is checked.",
    "Spin-pump and repump beams (both on D2 near 780 nm; an explicit assumption) are mutually incoherent: two-frequency CPT/optical coherences excluded.",
    "Rates through different excited sublevels are summed independently unless optical.path_model = kramers_heisenberg.",
    "Pump-induced ground Zeeman coherences neglected (secular in the Zeeman splitting, checked).",
    "Ground energies use analytic Breit-Rabi with cached ARC constants. Excited energies and all dipoles use the weak-field basis; excited-state mixing is diagnosed.",
    "Calibrated Raman mode: only the |3,3> <-> |2,2> pair is coupled; Raman polarization imperfections enter only through the calibrated decay, shift, and scattering inputs.",
    "The up->down Raman operator imparts hbar*(k_low-k_high); its adjoint occupies the |up><down| Hamiltonian block. Exact motional overlaps are used; |n-m| > raman.max_sideband_order is dropped.",
    "raman.tone_layout = both_tones_both_beams: the four tone pathways add coherently (co-propagating ones carrier-only); the result depends on raman.lattice_phase_deg.",
    "Every optical scattering event applies absorption + emission recoil via a single displacement (excited lifetime << trap period); emission directions integrated over the dipole pattern.",
    "Recoil uses individual |s,n> population jumps. This omits coherence transfer even between degenerate motional transitions and elastic-spin interference, an additional approximation beyond secular averaging.",
    "Overflow is numerical in a harmonic basis, physical escape when all isolated-site bound levels are kept, and a mixture when the lattice basis is capped; it is never reflected into the basis.",
    "trap.potential = lattice: motion is one isolated site of V0 sin^2(k_L z) (V0 = trap.depth_uk, k_L from frequency_hz) with exact bound levels; "
    "promotion above V0 is counted as loss (overflow). Tunnelling, coherent Raman coupling into unbound states, and anharmonic heating matrix elements are neglected.",
    "Lattice thermal_temperature initializes a canonical distribution conditional on initially bound atoms; the initially unbound fraction is unspecified. Initial tail means discarded bound-state weight in this mode.",
    "Energy is mean excitation above the motional ground state per survivor, using actual level energies. Its change includes loss selection; selection_cooling_power_W reports that contribution. Lattice temperature is an energy-matched canonical proxy on the bound spectrum.",
]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("numpy", "scipy", "matplotlib", "qutip", "PyQt6", "ARC-Alkali-Rydberg-Calculator", "threadpoolctl"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def constants() -> dict:
    return {k: getattr(atomic, k) for k in ("H", "HBAR", "C_LIGHT", "EPS0", "E_CHARGE", "A0", "MU_B", "K_B")}


def estimate(cfg: SimConfig) -> dict:
    if cfg.ensemble.enabled:
        return {"backend": "coherent_3d_split_ensemble", "ensemble_samples": cfg.ensemble.samples,
                "time_step_us": cfg.ensemble.time_step_us,
                "note": "Runtime depends on local bound-state counts; legacy 1D ODE estimates do not apply.",
                "est_coherent_runtime_s": float("nan")}
    return estimate_cost(Model(cfg))


def run(cfg: SimConfig, stop=None, progress=None, estimate_cb=None, *, ensemble_workers=None):
    """Run physics unchanged, optionally sharing an outer scan's worker budget."""
    if cfg.ensemble.enabled:
        from .ensemble import run_ensemble
        return run_ensemble(cfg, stop, progress, estimate_cb, workers=ensemble_workers)
    model = Model(cfg)
    if estimate_cb is not None:
        estimate_cb(estimate_cost(model))
    res = solve(model, stop, progress)
    a = analyze(model, res)
    a["segments"] = res.segments
    a["model_summary"] = model.summary()
    a["result"] = res
    a["model"] = model
    return a


def compare_protocols(cfg: SimConfig, protocols=COMPARISON_SET, stop=None, progress=None, log=print) -> dict:
    out = {}
    for i, p in enumerate(protocols):
        c = cfg.copy()
        c.timing.protocol = p
        c.name = f"{cfg.name}:{p}"
        log(f"[{i + 1}/{len(protocols)}] protocol {p}")
        sub = (lambda f, m="", i=i: progress((i + f) / len(protocols), f"{p} {m}")) if progress else None
        out[p] = run(c, stop, sub)
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def metadata(a: dict) -> dict:
    m: Model = a["model"]
    return {
        "config": m.cfg.to_dict(),
        "final": a["final"],
        "validity_status": a["validity"].status,
        "validity": [{"category": c, "status": s, "message": msg} for c, s, msg in a["validity"].items],
        "model_summary": a["model_summary"],
        "polarization": {
            b: {
                "config": getattr(m.cfg, b).polarization.__dict__,
                "effective_fractions": {str(q): v for q, v in m.beams[b].fractions.items()},
                "direction_used": m.beams[b].direction.tolist(),
                "source": m.beams[b].fraction_source,
            }
            for b in ("spin_pump", "repump")
        },
        "schedule": [s.__dict__ for s in a["segments"]],
        "cooling_rate_method": a["cooling_rate_method"],
        "assumptions": a.get("ensemble_details", {}).get("assumptions", ASSUMPTIONS),
        "ensemble_details": a.get("ensemble_details"),
        "atomic_data_metadata": m.atom.metadata,
        "constants_SI": constants(),
        "versions": versions(),
        "seed": m.cfg.numerics.seed,
        "solver_notes": a["result"].notes,
        "rate_validity": a["result"].rate_validity,
    }


def export(a: dict, outdir: str | Path, figures=True, stem="run") -> Path:
    from .plotting import dashboard

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s = a["series"]
    keys = list(s)
    with open(outdir / f"{stem}_timeseries.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(s["t_s"])):
            w.writerow([f"{s[k][i]:.10g}" for k in keys])
    res = a["result"]
    np.savez_compressed(
        outdir / f"{stem}_joint.npz",
        t_s=res.t,
        P=res.P,
        overflow=res.overflow,
        counts=res.counts,
        photon_counter_labels=np.array(["spin_pump", "repump", "raman_scatter"] + (["lattice_scatter"] if res.counts.shape[1] == 4 else [])),
        spin_labels=np.array([atomic.ground_label(i) for i in range(12)]),
        rho_raman_block_final=res.rho_final if res.rho_final is not None else np.zeros(0),
        description=np.array("P[t, s, n]: joint population of ground sublevel s (order F=2 m=-2..2, F=3 m=-3..3) and axial vibrational level n (transverse states marginalized in ensemble mode), "
                             "as fractions of all atoms (sum = 1 - overflow); divide by P[t].sum() for per-trapped-atom values"),
    )
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(metadata(a), indent=2, default=_json_default))
    (outdir / f"{stem}_config.json").write_text(a["model"].cfg.to_json())
    if "ensemble_nodes" in a:
        arrays = {}
        for i, (local, result) in enumerate(a["ensemble_nodes"]):
            arrays[f"node_{i}_position_um"] = local.position
            arrays[f"node_{i}_phase_rad"] = np.asarray(local.phase)
            arrays[f"node_{i}_P_final_spin_nx_ny_nz"] = result["P_final_3d"]
            for axis, site in enumerate(local.sites):
                arrays[f"node_{i}_{'xyz'[axis]}_energies_hz"] = site.energies_hz
                arrays[f"node_{i}_{'xyz'[axis]}_pn_absolute_t"] = np.asarray(result["pn_axes"][axis])
        np.savez_compressed(outdir / f"{stem}_ensemble.npz", **arrays)
    if figures:
        fig = dashboard(a, title=f"{a['model'].cfg.name} - protocol {a['model'].cfg.timing.protocol}")
        fig.savefig(outdir / f"{stem}_dashboard.png", dpi=130)
        fig.savefig(outdir / f"{stem}_dashboard.pdf")
        if "ensemble_details" in a:
            from .plotting import carrier_calibration_figure
            fig = carrier_calibration_figure(a["ensemble_details"]["calibration"])
            fig.savefig(outdir / f"{stem}_carrier_calibration.png", dpi=140)
            fig.savefig(outdir / f"{stem}_carrier_calibration.pdf")
    return outdir


def export_comparison(results: dict, outdir: str | Path) -> Path:
    from .plotting import comparison_figure

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p, a in results.items():
        export(a, outdir / p, figures=True, stem=p)
    keys = ["P_up", "P_n0", "P_target", "P_n0_given_up", "P_trapped", "P_target_absolute", "nbar", "fractional_energy_reduction", "photons_total",
            "net_quanta_removed_per_photon", "target_state_scattering_rate_per_s", "time_to_spin_threshold_s", "time_to_joint_threshold_s"]
    with open(outdir / "comparison_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", *keys, "validity", "solver"])
        for p, a in results.items():
            w.writerow([p, *[a["final"][k] for k in keys], a["validity"].status, a["final"]["solver"]])
    fig = comparison_figure(results)
    fig.savefig(outdir / "comparison.png", dpi=130)
    fig.savefig(outdir / "comparison.pdf")
    return outdir
