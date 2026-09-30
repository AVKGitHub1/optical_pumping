"""Matplotlib figures (Figure objects only; no pyplot, so the GUI and headless share code).

Style: fixed-order categorical palette, single-hue sequential ramp for heatmaps,
one y-scale per axes (no twin axes), thin 2 px lines, recessive grid.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

from .atomic import GROUND_STATES, UP, ground_label

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
STATUS_TEXT = {"ok": "OK", "warning": "WARNING", "invalid": "INVALID"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)
    ax.title.set_color(TEXT)


def _new(fig, *args, **kw):
    ax = fig.add_subplot(*args, **kw)
    _style(ax)
    return ax


def _ms(t):
    return np.asarray(t) * 1e3


def plot_spin_bars(ax, a):
    fin = a["final"]
    labels = [ground_label(i) for i in range(12)]
    x = np.arange(12)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, list(fin["spin_populations_initial"].values()), w, color=SERIES[0], label="initial")
    ax.bar(x + w / 2 + 0.01, list(fin["spin_populations_final"].values()), w, color=SERIES[1], label="final")
    ax.set_xticks(x, labels, rotation=60, fontsize=7)
    ax.set_ylabel("population")
    ax.set_title("Spin sublevels (F=2 left, F=3 right)", fontsize=9)
    ax.axvline(4.5, color=MUTED, lw=0.8)
    ax.legend(fontsize=7, frameon=False)


def plot_traces(ax, a, keys=(("P_up", "P_up"), ("P_n0", "P_n0"), ("P_target", "P_target = P(3,3,0)"), ("P_F3", "P(F=3)"))):
    s = a["series"]
    t = _ms(s["t_s"])
    for i, (k, lab) in enumerate(keys):
        ax.plot(t, s[k], color=SERIES[i], lw=2, label=lab)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Spin polarization and motional ground state", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    _shade_segments(ax, a)


def _shade_segments(ax, a):
    segs = a.get("segments") or []
    if len(segs) > 60:
        return
    for sg in segs:
        if sg.raman > 0 and (sg.pump > 0 or sg.repump > 0):
            continue
        if sg.raman > 0:
            ax.axvspan(sg.t0 * 1e3, sg.t1 * 1e3, color="#eef4fc", lw=0, zorder=0)


def plot_nbar(ax, a):
    s = a["series"]
    t = _ms(s["t_s"])
    ax.plot(t, s["nbar"], color=SERIES[0], lw=2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("nbar (quanta)")
    ax.set_title("Mean vibrational occupation (modeled 1D axis)", fontsize=9)


def plot_photons(ax, a):
    s = a["series"]
    t = _ms(s["t_s"])
    for i, (k, lab) in enumerate((("photons_spin_pump", "spin pump"), ("photons_repump", "repump"), ("photons_raman_scatter", "Raman scatter"))):
        ax.plot(t, s[k], color=SERIES[i], lw=2, label=lab)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("photons scattered / atom")
    ax.set_title("Integrated scattered photons", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_cooling_rate(ax, a):
    s = a["series"]
    ax.plot(_ms(s["t_s"]), s["cooling_rate_quanta_per_s"], color=SERIES[2], lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("-dnbar/dt (quanta/s)")
    ax.set_title("Net cooling rate (Savitzky-Golay derivative)", fontsize=9)


def plot_pnt(ax, a, fig=None):
    pn = np.clip(a["pn"], 1e-8, None)
    t = _ms(a["series"]["t_s"])
    im = ax.imshow(
        pn.T, origin="lower", aspect="auto", cmap=SEQ, norm=LogNorm(1e-6, 1),
        extent=[t[0], t[-1], -0.5, pn.shape[1] - 0.5], interpolation="nearest",
    )
    ax.grid(False)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("n")
    ax.set_title("Motional distribution p(n, t)", fontsize=9)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("p(n)", fontsize=8)


def plot_joint(ax, a, fig=None, nshow=None):
    Pf = np.clip(a["P_final"], 1e-8, None)
    nshow = nshow or min(Pf.shape[1], 20)
    im = ax.imshow(Pf[:, :nshow], origin="lower", aspect="auto", cmap=SEQ, norm=LogNorm(1e-6, 1), interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(12), [ground_label(i) for i in range(12)], fontsize=7)
    ax.set_xlabel("n")
    ax.set_title("Final joint P(F, mF, n)", fontsize=9)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(labelsize=7)


def diagnostics_text(a) -> str:
    f = a["final"]
    v = a["validity"]
    lines = [
        f"MODEL VALIDITY: {STATUS_TEXT[v.status]}",
        f"P_up={f['P_up']:.4f}  P_n0={f['P_n0']:.4f}  P_target={f['P_target']:.4f}  (P_up*P_n0={f['P_up_times_P_n0']:.4f})",
        f"nbar {f['nbar_initial']:.4g} -> {f['nbar']:.4g}   T_equiv={f['T_equiv_K'] * 1e6:.3g} uK [{f['T_equiv_note']}]",
        f"photons/atom: pump {f['photons']['spin_pump']:.3g}, repump {f['photons']['repump']:.3g}, Raman {f['photons']['raman_scatter']:.3g}",
        f"target |3,3> scattering rate {f['target_state_scattering_rate_per_s']:.3g} /s; quanta removed per photon {f['net_quanta_removed_per_photon']}",
        f"t(P_up>={f['spin_threshold']}) = {_fmt_t(f['time_to_spin_threshold_s'])};  t(P_target>={f['joint_threshold']}) = {_fmt_t(f['time_to_joint_threshold_s'])}",
        f"conservation err {f['max_conservation_error']:.2g}; min eig {f['min_eigenvalue']:.2g}; boundary pop {f['max_boundary_population']:.2g}; overflow {f['final_overflow']:.2g}",
        f"solver {f['solver']}, wall {f['wall_time_s']:.1f} s",
        "",
    ]
    for cat, st, msg in v.items:
        lines.append(f"[{STATUS_TEXT[st]:7s}] {cat}: {msg}")
    return "\n".join(lines)


def _fmt_t(x):
    return x if isinstance(x, str) else f"{x * 1e3:.3g} ms"


def dashboard(a, title="") -> Figure:
    fig = Figure(figsize=(15, 10), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(3, 3)
    plot_spin_bars(_new(fig, gs[0, 0]), a)
    plot_traces(_new(fig, gs[0, 1]), a)
    plot_nbar(_new(fig, gs[0, 2]), a)
    plot_pnt(_new(fig, gs[1, 0]), a, fig)
    plot_joint(_new(fig, gs[1, 1]), a, fig)
    plot_photons(_new(fig, gs[1, 2]), a)
    plot_cooling_rate(_new(fig, gs[2, 0]), a)
    axt = fig.add_subplot(gs[2, 1:])
    axt.axis("off")
    axt.text(0, 1, diagnostics_text(a), family="monospace", fontsize=6.5, va="top", color=TEXT, wrap=True)
    fig.suptitle(title, fontsize=11, color=TEXT)
    return fig


def comparison_figure(results: dict) -> Figure:
    """results: protocol -> analysis dict."""
    fig = Figure(figsize=(15, 9), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(2, 3)
    names = list(results)
    cols = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}
    for j, (key, lab) in enumerate((("P_up", "P_up"), ("P_target", "P_target"), ("nbar", "nbar"))):
        ax = _new(fig, gs[0, j])
        for n in names:
            s = results[n]["series"]
            ls = "--" if results[n]["validity"].status == "invalid" else "-"
            ax.plot(_ms(s["t_s"]), s[key], color=cols[n], lw=2, ls=ls, label=n)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(lab)
        ax.set_title(lab + " vs time", fontsize=9)
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False)
    for j, (key, lab) in enumerate((("P_target", "final P_target"), ("P_up", "final P_up"), ("photons_total", "photons / atom"))):
        ax = _new(fig, gs[1, j])
        vals = [results[n]["final"][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=[cols[n] for n in names])
        for i, n in enumerate(names):
            st = results[n]["validity"].status
            ax.annotate(f"{vals[i]:.3g}" + ("" if st != "invalid" else "\nINVALID"), (i, vals[i]), ha="center", va="bottom", fontsize=7, color=TEXT2)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=7)
        ax.set_title(lab + " (equal total duration)", fontsize=9)
    return fig


METRICS = (("P_up", "final P_up"), ("P_n0", "final P_n0"), ("P_target", "final P_target"), ("nbar", "final nbar"),
           ("photons_total", "photons / atom"), ("target_state_scattering_rate_per_s", "|3,3> scattering rate (1/s)"))


def scan_figure(scan) -> Figure:
    """scan: dict with axes, metrics arrays, valid mask, optional series labels."""
    fig = Figure(figsize=(15, 8.5), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(2, 3)
    axes = scan["axes"]
    valid = scan["valid"]
    if len(axes) == 1:
        x = np.asarray(axes[0]["values"])
        logx = axes[0].get("log", False)
        for j, (k, lab) in enumerate(METRICS):
            ax = _new(fig, gs[j // 3, j % 3])
            for si, slab in enumerate(scan["series_labels"]):
                y = np.asarray(scan["metrics"][k])[si]
                v = np.asarray(valid)[si]
                ax.plot(x, y, color=SERIES[si], lw=2, marker="o", ms=4, label=slab)
                if (~v).any():
                    ax.plot(x[~v], y[~v], ls="none", marker="x", ms=9, mew=2, color=TEXT, label="invalid/unconverged" if si == 0 else None)
            if logx:
                ax.set_xscale("symlog", linthresh=_linthresh(x))
            ax.set_xlabel(axes[0]["label"])
            ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.legend(fontsize=7, frameon=False)
    else:
        x = np.asarray(axes[0]["values"])
        y = np.asarray(axes[1]["values"])
        for j, (k, lab) in enumerate(METRICS):
            ax = _new(fig, gs[j // 3, j % 3])
            Z = np.asarray(scan["metrics"][k])[0].T
            V = np.asarray(valid)[0].T
            im = ax.pcolormesh(np.arange(len(x) + 1) - 0.5, np.arange(len(y) + 1) - 0.5, Z, cmap=SEQ, shading="flat")
            ax.grid(False)
            iy, ix = np.nonzero(~V)
            ax.plot(ix, iy, ls="none", marker="x", ms=10, mew=2, color="#e34948", label="invalid/unconverged")
            ax.set_xticks(range(len(x)), [f"{v:.3g}" for v in x], rotation=45, fontsize=7)
            ax.set_yticks(range(len(y)), [f"{v:.3g}" for v in y], fontsize=7)
            ax.set_xlabel(axes[0]["label"])
            ax.set_ylabel(axes[1]["label"])
            ax.set_title(lab, fontsize=9)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.ax.tick_params(labelsize=7)
            if j == 0 and len(ix):
                ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(scan.get("title", "scan"), fontsize=11, color=TEXT)
    return fig


def _linthresh(x):
    nz = np.abs(x[x != 0])
    return float(nz.min()) if nz.size else 1.0
