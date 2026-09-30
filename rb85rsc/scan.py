"""1D / 2D parameter scans with invalid-point marking.

Scan JSON::

    {
      "title": "...",
      "base_config": "default.json",          # path relative to the scan file, or
      "config": {...},                         # inline overrides applied on top
      "overrides": {"timing.protocol": "continuous"},
      "axes": [
        {"param": "raman.carrier_rabi_hz", "values": [1e3, 2e3]},
        {"param": ["spin_pump.peak_intensity_w_m2", "repump.peak_intensity_w_m2"],
         "start": 0.01, "stop": 1.0, "num": 5, "log": true, "label": "pump intensity"}
      ],
      "series": [{"label": "pi", "overrides": {"spin_pump.polarization.pi_share_of_impurity": 1.0}}],
      "workers": 1
    }

A list-valued ``param`` sets every listed parameter to the same value.  Virtual
parameters ``<beam>.polarization.total_impurity`` / ``pi_share_of_impurity`` are
supported.  Every point uses the same starting distribution (the base config's).
Points whose validity status is ``invalid`` are marked, never ranked as optima.
"""
from __future__ import annotations

import csv
import itertools
import json
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import SimConfig, set_param
from .model import Model
from .dynamics import estimate_cost

METRIC_KEYS = ("P_up", "P_n0", "P_target", "nbar", "photons_total", "target_state_scattering_rate_per_s")


def axis_values(ax: dict) -> np.ndarray:
    if "values" in ax:
        return np.asarray(ax["values"], float)
    if ax.get("log"):
        return np.geomspace(ax["start"], ax["stop"], ax["num"])
    return np.linspace(ax["start"], ax["stop"], ax["num"])


def _apply(cfg: SimConfig, param, value):
    for p in param if isinstance(param, list) else [param]:
        set_param(cfg, p, value)


def load_scan(path: str | Path) -> tuple[dict, SimConfig]:
    path = Path(path)
    spec = json.loads(path.read_text())
    return spec, base_config(spec, path.parent)


def base_config(spec: dict, root: Path = Path(".")) -> SimConfig:
    if "base_config" in spec:
        d = json.loads((root / spec["base_config"]).read_text())
    else:
        d = SimConfig().to_dict()
    for k, v in spec.get("config", {}).items():
        d.setdefault(k, {}).update(v) if isinstance(v, dict) else d.__setitem__(k, v)
    cfg = SimConfig.from_dict(d)
    for k, v in spec.get("overrides", {}).items():
        set_param(cfg, k, v)
    cfg.validate()
    return cfg


def point_configs(spec: dict, base: SimConfig):
    axes = spec["axes"]
    vals = [axis_values(a) for a in axes]
    series = spec.get("series") or [{"label": "", "overrides": {}}]
    for si, s in enumerate(series):
        for idx in itertools.product(*[range(len(v)) for v in vals]):
            c = base.copy()
            for k, v in s.get("overrides", {}).items():
                set_param(c, k, v)
            for a, v, i in zip(axes, vals, idx):
                _apply(c, a["param"], float(v[i]))
            c.validate()
            yield (si, *idx), c


def _run_point(cfg_dict: dict):
    from .runner import run

    cfg = SimConfig.from_dict(cfg_dict)
    a = run(cfg)
    f = a["final"]
    return {
        **{k: f[k] for k in METRIC_KEYS},
        "status": a["validity"].status,
        "invalid_reasons": [f"{c}: {m}" for c, s, m in a["validity"].items if s == "invalid"],
        "solver": f["solver"],
        "wall_time_s": f["wall_time_s"],
    }


def run_scan(spec: dict, base: SimConfig, stop: threading.Event | None = None, progress=None, log=print, workers=None) -> dict:
    stop = stop or threading.Event()
    axes = spec["axes"]
    vals = [axis_values(a) for a in axes]
    series = spec.get("series") or [{"label": "", "overrides": {}}]
    shape = (len(series), *[len(v) for v in vals])
    pts = list(point_configs(spec, base))
    est = estimate_cost(Model(pts[0][1]))
    log(f"scan: {len(pts)} points; coherent estimate ~{est['est_coherent_runtime_s']:.1f} s/point "
        f"(rate backend is much faster where valid); total <= ~{len(pts) * est['est_coherent_runtime_s'] / 60:.1f} min serial")
    metrics = {k: np.full(shape, np.nan) for k in METRIC_KEYS}
    valid = np.zeros(shape, bool)
    status = np.full(shape, "not run", dtype=object)
    reasons = {}
    workers = workers if workers is not None else spec.get("workers", 1)
    done = 0

    def store(idx, r):
        nonlocal done
        for k in METRIC_KEYS:
            metrics[k][idx] = r[k]
        status[idx] = r["status"]
        valid[idx] = r["status"] != "invalid"
        if r["invalid_reasons"]:
            reasons[str(idx)] = r["invalid_reasons"]
        done += 1
        if progress:
            progress(done / len(pts), f"point {done}/{len(pts)}")

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_point, c.to_dict()): idx for idx, c in pts}
            for fu in as_completed(futs):
                if stop.is_set():
                    for f2 in futs:
                        f2.cancel()
                    break
                store(futs[fu], fu.result())
    else:
        from .runner import run

        for idx, c in pts:
            if stop.is_set():
                break
            a = run(c, stop)
            f = a["final"]
            store(idx, {**{k: f[k] for k in METRIC_KEYS}, "status": a["validity"].status,
                        "invalid_reasons": [f"{cc}: {m}" for cc, s, m in a["validity"].items if s == "invalid"],
                        "solver": f["solver"], "wall_time_s": f["wall_time_s"]})
            log(f"  {idx}: P_target={f['P_target']:.4f} nbar={f['nbar']:.4g} [{a['validity'].status}, {f['solver']}, {f['wall_time_s']:.1f}s]")
    best = None
    if valid.any():
        m = np.where(valid, metrics["P_target"], -np.inf)
        bi = np.unravel_index(np.argmax(m), shape)
        best = {"index": [int(i) for i in bi], "P_target": float(metrics["P_target"][bi])}
    return {
        "title": spec.get("title", "scan"),
        "axes": [
            {"param": a["param"], "label": a.get("label", a["param"] if isinstance(a["param"], str) else " = ".join(a["param"])),
             "values": v.tolist(), "log": bool(a.get("log", False))}
            for a, v in zip(axes, vals)
        ],
        "series_labels": [s.get("label", "") for s in series],
        "series_overrides": [s.get("overrides", {}) for s in series],
        "metrics": {k: v.tolist() for k, v in metrics.items()},
        "valid": valid.tolist(),
        "status": status.tolist(),
        "invalid_reasons": reasons,
        "best_valid_P_target": best,
        "base_config": base.to_dict(),
        "stopped": stop.is_set(),
    }


def export_scan(scan: dict, outdir: str | Path) -> Path:
    from .plotting import scan_figure
    from .runner import versions

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "scan_results.json").write_text(json.dumps({**scan, "versions": versions()}, indent=2, default=str))
    shape = np.asarray(scan["valid"]).shape
    with open(outdir / "scan_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", *[a["label"] for a in scan["axes"]], *METRIC_KEYS, "status"])
        for idx in itertools.product(*[range(s) for s in shape]):
            row = [scan["series_labels"][idx[0]]]
            row += [scan["axes"][j]["values"][i] for j, i in enumerate(idx[1:])]
            row += [np.asarray(scan["metrics"][k])[idx] for k in METRIC_KEYS]
            row.append(np.asarray(scan["status"], dtype=object)[idx])
            w.writerow(row)
    fig = scan_figure(scan)
    fig.savefig(outdir / "scan.png", dpi=130)
    fig.savefig(outdir / "scan.pdf")
    return outdir
