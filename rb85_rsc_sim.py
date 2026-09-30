#!/usr/bin/env python
"""Rb-85 optical pumping + Raman sideband cooling simulator (1D, calibrated Raman mode).

Examples
--------
  python rb85_rsc_sim.py --gui
  python rb85_rsc_sim.py --config examples/continuous.json --headless --output results/continuous
  python rb85_rsc_sim.py --compare-protocols --config examples/default.json --output results/comparison
  python rb85_rsc_sim.py --scan-config examples/intensity_scan.json --headless --output results/scan
  python rb85_rsc_sim.py --config examples/default.json --estimate-only
  python rb85_rsc_sim.py --write-default-config examples/default.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _progress(frac, msg=""):
    sys.stderr.write(f"\r  {100 * frac:5.1f}% {msg[:50]:50s}")
    sys.stderr.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true", help="launch the PyQt6 interface")
    ap.add_argument("--headless", action="store_true", help="run without GUI (default unless --gui)")
    ap.add_argument("--config", help="simulation JSON")
    ap.add_argument("--output", default="results/run", help="output directory")
    ap.add_argument("--protocol", help="override timing.protocol")
    ap.add_argument("--solver", choices=["coherent", "rate", "auto"], help="override numerics.solver")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="override any parameter, e.g. --set raman.carrier_rabi_hz=3e3")
    ap.add_argument("--compare-protocols", action="store_true", help="run every protocol with identical parameters")
    ap.add_argument("--scan-config", help="scan JSON")
    ap.add_argument("--workers", type=int, help="parallel processes for scans")
    ap.add_argument("--estimate-only", action="store_true", help="print resource estimate and exit")
    ap.add_argument("--write-default-config", metavar="PATH", help="write the default configuration JSON and exit")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    from rb85rsc.config import SimConfig, set_param

    if args.write_default_config:
        SimConfig().to_json(args.write_default_config)
        print(f"wrote {args.write_default_config}")
        return 0
    if args.gui:
        from rb85rsc.gui import launch

        return launch(args.config)

    from rb85rsc.dynamics import estimate_cost
    from rb85rsc.model import Model
    from rb85rsc import runner

    if args.scan_config:
        from rb85rsc.scan import export_scan, load_scan, run_scan

        spec, base = load_scan(args.scan_config)
        for kv in args.set:
            k, v = kv.split("=", 1)
            set_param(base, k, json.loads(v))
        if args.solver:
            base.numerics.solver = args.solver
        scan = run_scan(spec, base, progress=None, workers=args.workers)
        out = export_scan(scan, args.output)
        print(f"scan written to {out}; best valid P_target: {scan['best_valid_P_target']}")
        return 0

    cfg = SimConfig.from_json(args.config) if args.config else SimConfig()
    if args.protocol:
        cfg.timing.protocol = args.protocol
    if args.solver:
        cfg.numerics.solver = args.solver
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        set_param(cfg, k, val)
    cfg.validate()

    est = estimate_cost(Model(cfg))
    n_runs = 7 if args.compare_protocols else 1
    print(f"Resource estimate: state dim {est['state_dimension_complex']} complex, "
          f"{est['memory_transfer_matrices_MB']:.1f} MB operators, ~{est['est_rhs_evals']:.3g} RHS evals x "
          f"{est['rhs_eval_s'] * 1e6:.0f} us = ~{est['est_coherent_runtime_s']:.1f} s per coherent run"
          + (f" (x{n_runs} protocols)" if n_runs > 1 else ""))
    if args.estimate_only:
        print(json.dumps(est, indent=2, default=float))
        return 0

    if args.compare_protocols:
        results = runner.compare_protocols(cfg, progress=_progress)
        sys.stderr.write("\n")
        out = runner.export_comparison(results, args.output)
        print(f"\n{'protocol':28s} {'P_up':>8s} {'P_n0':>8s} {'P_target':>9s} {'nbar':>8s} {'photons':>8s}  validity")
        for p, a in results.items():
            f = a["final"]
            print(f"{p:28s} {f['P_up']:8.4f} {f['P_n0']:8.4f} {f['P_target']:9.4f} {f['nbar']:8.4f} {f['photons_total']:8.2f}  {a['validity'].status}")
        print(f"written to {out}")
        return 0

    a = runner.run(cfg, progress=_progress)
    sys.stderr.write("\n")
    out = runner.export(a, args.output, figures=not args.no_figures)
    from rb85rsc.plotting import diagnostics_text

    print(diagnostics_text(a))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
