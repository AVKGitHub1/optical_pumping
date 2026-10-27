"""Matplotlib figures (Figure objects only; no pyplot, so the GUI and headless share code).

Style: fixed-order categorical palette, single-hue sequential ramp for heatmaps,
one y-scale per axes (no twin axes), thin 2 px lines, recessive grid.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation

from .atomic import GROUND_STATES, K_B, ground_label

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


def _trace(ax, a, key, *, scale=1., **kwargs):
    """Shared line renderer; deterministic results simply have no error band."""
    values = np.asarray(a['series'][key])*scale
    values = np.where(np.isfinite(values), values, np.nan)
    line, = ax.plot(_ms(a['series']['t_s']), values, **kwargs)
    error = a.get('standard_error', {}).get(key)
    if error is not None:
        width = 1.96*np.asarray(error)*abs(scale)
        ax.fill_between(_ms(a['series']['t_s']), values-width, values+width,
                        color=line.get_color(), alpha=.18, linewidth=0)


def uncertainty_note(fig, a):
    if 'standard_error' in a:
        fig.supxlabel('Bands / error bars: pointwise approximate 95% Monte Carlo intervals. Heatmaps show means.', fontsize=8)


def plot_spin_bars(ax, a):
    fin = a["final"]
    labels = [ground_label(i) for i in range(12)]
    x = np.arange(12)
    w = 0.38
    errors = a.get('array_standard_error', {}).get('spin')
    for offset, index, key, color, label in ((-w/2-.01, 0, 'spin_populations_initial', SERIES[0], 'initial'),
                                            (w/2+.01, -1, 'spin_populations_final', SERIES[1], 'final')):
        ax.bar(x+offset, np.asarray(list(fin[key].values()), dtype=float), w, color=color, label=label,
               yerr=None if errors is None else 1.96*errors[index], capsize=2,
               error_kw={'elinewidth': .8})
    ax.set_xticks(x, labels, rotation=60, fontsize=7)
    ax.set_ylabel("fraction among survivors")
    ax.set_title("Spin sublevels (F=2 left, F=3 right)", fontsize=9)
    ax.axvline(4.5, color=MUTED, lw=0.8)
    ax.legend(fontsize=7, frameon=False)


def _raman_labels(a):
    model = a.get("model")
    if model is not None:
        return ground_label(model.up), ground_label(model.down)
    summary = a.get("model_summary", {})
    return tuple(ground_label(GROUND_STATES.index(tuple(summary.get(key, default))))
                 for key, default in (("raman_state_up", [3, 3]), ("raman_state_down", [2, 2])))


def plot_traces(ax, a, keys=None):
    s = a["series"]
    ensemble = "T_equiv_axes_K" in a["final"]
    up, down = _raman_labels(a)
    if keys is None:
        keys = (("P_up", f"P_up = P{up}"), ("P_n0", "P_n0"),
                ("P_target", f"P_target: {up}, n=0"), ("P_F3", "P(F=3)"))
    if ensemble:
        axis = a["final"].get("raman_axis", "z")
        keys = (("P_up", f"spin {up} / survivors"), ("P_n0", f"{axis}: n=0 / survivors"),
                ("P_target_absolute", "joint target / initially loaded"),
                ("P_up_absolute", f"spin {up} / initially loaded"))
    for i, (k, lab) in enumerate(keys):
        _trace(ax, a, k, color=SERIES[i], lw=2, label=lab)
    m = a.get("model")
    lattice = (m is not None and m.motion.kind == "lattice") or a.get('model_summary', {}).get('potential') == 'lattice'
    if lattice and "P_trapped" in s and np.min(s["P_trapped"]) < 1 - 1e-4:
        _trace(ax, a, 'P_trapped', color=TEXT2, lw=1.5, ls="--", label="survival in modeled wells")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Survival and absolute target yield" if ensemble else "Spin polarization and motional ground state", fontsize=9)
    if ensemble:
        # Reserve space below the x label, even in short dashboard panels.
        legend_transform = ax.transAxes + ScaledTranslation(0, -36 / 72, ax.figure.dpi_scale_trans)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0),
                  bbox_transform=legend_transform, borderaxespad=0, labelspacing=0.3)
    else:
        ax.legend(fontsize=7, frameon=False, loc="center right")
    _shade_segments(ax, a)


def _shade_segments(ax, a):
    cfg = a['model'].cfg if a.get('model') is not None else a.get('config')
    if cfg is not None and not cfg.raman.enabled:
        return
    segs = a.get("segments") or []
    if len(segs) > 60:
        return
    for sg in segs:
        if sg.raman > 0 and (sg.pump > 0 or sg.repump > 0):
            continue
        if sg.raman > 0:
            ax.axvspan(sg.t0 * 1e3, sg.t1 * 1e3, color="#eef4fc", lw=0, zorder=0)


SCHEDULE_LANES = (("raman", "Raman (RSC)", SERIES[0]), ("pump", "spin pump", SERIES[1]), ("repump", "repump", SERIES[2]))


def plot_schedule(ax, a):
    """Narrow timeline of which beams are on; bar height = amplitude multiplier (relative to that beam's maximum)."""
    segs = a.get("segments") or []
    m = a.get("model")
    enabled = {"raman": True, "pump": True, "repump": True}
    cfg = m.cfg if m is not None else a.get('config')
    if cfg is not None:
        enabled = {"raman": cfg.raman.enabled, "pump": cfg.spin_pump.enabled, "repump": cfg.repump.enabled}
    labels = []
    for i, (key, lab, col) in enumerate(SCHEDULE_LANES):
        y = len(SCHEDULE_LANES) - 1 - i  # Raman on top
        amps = [getattr(sg, key) if enabled[key] else 0.0 for sg in segs]
        top = max(amps, default=0.0) or 1.0
        if segs:  # segments are contiguous: step trace at the boundaries, shaded down to the lane baseline
            ts = [sg.t0 * 1e3 for sg in segs] + [segs[-1].t1 * 1e3]
            ys = [y - 0.4 + 0.8 * x / top for x in amps] + [y - 0.4 + 0.8 * amps[-1] / top]
            ax.step(ts, ys, where="post", color=col, lw=1.5)
            ax.fill_between(ts, y - 0.4, ys, step="post", color=col, alpha=0.5, lw=0)
        ax.axhline(y - 0.4, color=GRID, lw=0.8, zorder=0)
        labels.append(lab if enabled[key] else f"{lab} (disabled)")
    ax.set_yticks(range(len(SCHEDULE_LANES)), labels[::-1], fontsize=7)
    ax.set_ylim(-0.5, len(SCHEDULE_LANES) - 0.5)
    ax.set_xlim(0, segs[-1].t1 * 1e3 if segs else 1.0)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("time (ms)", fontsize=7)
    ax.set_title("Beams on (trace height = amplitude multiplier)", fontsize=8)


def plot_nbar(ax, a):
    s = a["series"]
    all_axes = all(f"nbar_{axis}" in s for axis in "xyz")
    if all_axes:
        for i, axis in enumerate("xyz"):
            _trace(ax, a, f'nbar_{axis}', color=SERIES[i], lw=2, label=_axis_label(a, axis))
        ax.legend(fontsize=7, frameon=False)
    else:
        _trace(ax, a, 'nbar', color=SERIES[0], lw=2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean vibrational level index")
    axis = a["final"].get("raman_axis")
    ax.set_title("Mean occupation: x, y, z (survivors)" if all_axes else
                 f"Mean {axis + ' lattice' if axis else 'axial'} occupation, among survivors", fontsize=9)


def plot_photons(ax, a):
    s = a["series"]
    for i, (k, lab) in enumerate((("photons_spin_pump", "spin pump"), ("photons_repump", "repump"), ("photons_raman_scatter", "Raman scatter"))):
        _trace(ax, a, k, color=SERIES[i], lw=2, label=lab)
    if "photons_lattice_scatter" in s:
        _trace(ax, a, 'photons_lattice_scatter', color=SERIES[3], lw=2, label="lattice scatter")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("photons / initially loaded atom")
    ax.set_title("Integrated scattered photons", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_cooling_rate(ax, a):
    s = a["series"]
    scale = 1e3 / K_B  # J/s -> uK/ms
    for key, label, color, style in (("cooling_power_W", "Net", SERIES[0], "-"),
                                    ("selection_cooling_power_W", "Loss selection", SERIES[1], "--")):
        if key in s:
            _trace(ax, a, key, scale=scale, color=color, lw=2, ls=style, label=label)
    if "selection_cooling_power_W" in s:
        ax.plot(_ms(s["t_s"]), (s["cooling_power_W"] - s["selection_cooling_power_W"]) * scale,
                color=SERIES[2], ls=":", label="Net minus selection")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("-d<E>/dt / kB (uK/ms)")
    ax.set_title("Survivor energy change (positive = decrease)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_pnt(ax, a, fig=None, axis=None):
    if axis is None:
        axis = a["final"].get("raman_axis")
    distribution = a["pn_axes"][axis] if axis is not None and "pn_axes" in a else a["pn"]
    pn = np.clip(distribution, 1e-8, None)
    t = _ms(a["series"]["t_s"])
    im = ax.imshow(
        pn.T, origin="lower", aspect="auto", cmap=SEQ, norm=LogNorm(1e-6, 1),
        extent=[t[0], t[-1], -0.5, pn.shape[1] - 0.5], interpolation="nearest",
    )
    ax.grid(False)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("n")
    ax.set_title(f"{axis} lattice p(n, t), survivors" if axis else "Motional distribution p(n, t)", fontsize=9)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("p(n)", fontsize=8)


def _axis_label(a, axis):
    coupled = a["final"].get("raman_coupled_axes", [a["final"].get("raman_axis")])
    return axis + (" (Raman)" if axis in coupled and a["final"].get("raman_active", True) else "")


def draw_motional_axes(fig, a):
    """GUI and export share the same per-axis population maps."""
    if "pn_axes" not in a:
        plot_pnt(_new(fig, 111), a, fig)
        return
    gs = fig.add_gridspec(1, 3)
    for i, axis in enumerate("xyz"):
        plot_pnt(_new(fig, gs[i]), a, fig, axis)
    fig.suptitle("Motional distributions in local x, y, z — conditioned on survival", fontsize=10)
    if 'standard_error' in a:
        fig.supxlabel('Heatmaps show Monte Carlo means; population uncertainties are included in the exported observables.', fontsize=8)


def draw_thermal_axes(fig, a):
    """Temperatures, energies, ground fractions, and loss-selection diagnostics."""
    if "pn_axes" not in a:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(.5, .5, "Three-axis thermal plots require a 3D spatial ensemble run.\n"
                "This run models motion along one trap axis.", ha="center", va="center", wrap=True)
        return
    s = a["series"]
    gs = fig.add_gridspec(2, 3)
    panels = (("T_equiv_{}_K", 1e6, "Energy-matched temperature", "Temperature estimate (uK)"),
              ("mean_excitation_energy_{}_J", 1e6 / K_B, "Excitation above the ground state", "Mean energy / kB (uK)"),
              ("P_n0_{}", 1., "Motional ground-state fractions", "P(n=0 | survived)"))
    for column, (key, scale, title, ylabel) in enumerate(panels):
        ax = _new(fig, gs[0, column])
        for i, axis in enumerate("xyz"):
            _trace(ax, a, key.format(axis), scale=scale, color=SERIES[i], lw=2, label=_axis_label(a, axis))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False)
        if column == 2:
            ax.set_ylim(-.02, 1.02)
        if column == 0 and any(not np.all(np.isfinite(s[f"T_equiv_{axis}_K"])) for axis in "xyz"):
            ax.text(.02, .02, "Gaps: no finite temperature estimate", transform=ax.transAxes, fontsize=7)
    reference = None
    for i, axis in enumerate("xyz"):
        ax = _new(fig, gs[1, i], **({"sharey": reference} if reference is not None else {}))
        if reference is None:
            reference = ax
        for j, (key, label, style) in enumerate((("cooling_power", "Net", "-"),
                                                ("selection_cooling_power", "Loss selection", "--"),
                                                ("bound_cooling_power", "Net minus loss", ":"))):
            if f'{key}_{axis}_W' in s:
                _trace(ax, a, f'{key}_{axis}_W', scale=1e3/K_B, color=SERIES[j], lw=1.8, ls=style, label=label)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_title(f"{_axis_label(a, axis)}: survivor energy change", fontsize=9)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Energy removal / kB (uK/ms)")
        ax.legend(fontsize=7, frameon=False)
    if len(a.get("segments", [])) <= 8:
        for ax in fig.axes:
            for seg in a.get("segments", [])[:-1]:
                ax.axvline(seg.t1 * 1e3, color=MUTED, ls=":", lw=.7)
    fig.suptitle("Local x, y, z motion among surviving atoms\n"
                 "Temperature is an energy-matched estimate; distributions need not be thermal.", fontsize=10)
    if 'standard_error' in a:
        fig.supxlabel('Bands: pointwise approximate 95% Monte Carlo intervals; derivative bands include temporal covariance.\n'
                      'Positive = decreasing survivor energy. Net includes loss selection; its separate contribution is unavailable.', fontsize=8)
    else:
        fig.supxlabel("Positive = decreasing survivor energy; negative = increasing energy. Net uses a smoothed derivative.\n"
                      "Loss selection is preferential escape; net minus loss includes Raman and recoil dynamics.", fontsize=8)


def thermal_axes_figure(a):
    fig = Figure(figsize=(13, 8), facecolor="white", layout="constrained")
    draw_thermal_axes(fig, a)
    return fig


def motional_axes_figure(a):
    fig = Figure(figsize=(12, 4.5), facecolor="white", layout="constrained")
    draw_motional_axes(fig, a)
    return fig


def plot_joint(ax, a, fig=None, nshow=None):
    Pf = np.clip(a["P_final"], 1e-8, None)
    nshow = nshow or min(Pf.shape[1], 20)
    im = ax.imshow(Pf[:, :nshow], origin="lower", aspect="auto", cmap=SEQ, norm=LogNorm(1e-6, 1), interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(12), [ground_label(i) for i in range(12)], fontsize=7)
    ax.set_xlabel("n")
    ax.set_title("Final joint P(spin, n), survivors" if 'pn_axes' in a else "Final joint P(F, mF, n)", fontsize=9)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(labelsize=7)


def plot_ground_3d(ax, a):
    for i, (key, label) in enumerate((('P_3d_ground_absolute', 'All spins, nx=ny=nz=0'),
                                     ('P_target_3d_absolute', 'Target spin, nx=ny=nz=0'))):
        _trace(ax, a, key, color=SERIES[i], lw=2, label=label)
    ax.set(xlabel='time (ms)', ylabel='probability / initially loaded')
    ax.set_title('Three-dimensional ground state', fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def diagnostics_text(a) -> str:
    if a.get('kind') == 'coherent_pumping':
        from .coherent_results import diagnostics_text as coherent_diagnostics
        return coherent_diagnostics(a)
    f = a["final"]
    v = a["validity"]
    up, down = _raman_labels(a)
    lines = [
        f"MODEL VALIDITY: {STATUS_TEXT[v.status]}",
        f"Raman pair: A={up} <-> B={down}; P_up and target refer to A, P_down to B.",
        f"P_up={f['P_up']:.4f}  P_n0={f['P_n0']:.4f}  P_target={f['P_target']:.4f}  (P_up*P_n0={f['P_up_times_P_n0']:.4f})",
        *([f"populations are per survivor: remaining in wells {f['P_trapped']:.4f} (well escape, no recapture {f['lattice_loss']:.4f}); initially loaded fractions: "
           f"P_up={f['P_up_absolute']:.4f}  P_n0={f['P_n0_absolute']:.4f}  P_target={f['P_target_absolute']:.4f}"] if f.get("lattice_loss") is not None else []),
        f"nbar (trapped atoms) {f['nbar_initial']:.4g} -> {f['nbar']:.4g}   "
        + (f"T_equiv={f['T_equiv_K'] * 1e6:.3g} uK" if f['T_equiv_K'] is not None else "T_equiv undefined")
        + f" [{f['T_equiv_note']}]",
        f"photons/atom: pump {f['photons']['spin_pump']:.3g}, repump {f['photons']['repump']:.3g}, Raman {f['photons']['raman_scatter']:.3g}",
        f"target {up} scattering rate {_fmt_number(f['target_state_scattering_rate_per_s'])} /s; quanta removed per photon {f['net_quanta_removed_per_photon']}",
        f"t(P_up>={f['spin_threshold']}) = {_fmt_t(f['time_to_spin_threshold_s'])};  t(P_target>={f['joint_threshold']}) = {_fmt_t(f['time_to_joint_threshold_s'])}",
        f"conservation err {f['max_conservation_error']:.2g}; min eig {f['min_eigenvalue']:.2g}; boundary pop {_fmt_number(f['max_boundary_population'])}; overflow {f['final_overflow']:.2g}",
        f"solver {f['solver']}, wall {f['wall_time_s']:.1f} s",
        "",
    ]
    if "T_equiv_axes_K" in f:
        active = "active" if f.get("raman_active", True) else "off throughout schedule"
        lines.insert(1, f"Target motional axis: local {f.get('raman_axis', 'z')}; Raman light {active}.")
        lines.insert(4, "Axis temperature proxies (uK): " + ", ".join(f"{axis}={_fmt_number(temp * 1e6 if temp is not None else None)}" for axis, temp in f["T_equiv_axes_K"].items()))
        lines.insert(4, f["loss_label"] + f": {f['well_loss_fraction_no_recapture']:.4f}")
        if "T_equiv_axes_initial_K" in f:
            lines.insert(5, "Axis temperature estimates, initial -> final (uK): " + ", ".join(
                f"{axis}: {_fmt_number(f['T_equiv_axes_initial_K'][axis] * 1e6 if f['T_equiv_axes_initial_K'][axis] is not None else None)}"
                f" -> {_fmt_number(f['T_equiv_axes_K'][axis] * 1e6 if f['T_equiv_axes_K'][axis] is not None else None)}" for axis in "xyz"))
            lines.insert(6, "Thermal x/y/z plots separate net energy removal from preferential-loss selection; "
                         "a falling survivor temperature alone does not establish Raman cooling.")
    for cat, st, msg in v.items:
        lines.append(f"[{STATUS_TEXT[st]:7s}] {cat}: {msg}")
    return "\n".join(lines)


def _fmt_number(value):
    return "unspecified" if value is None else f"{value:.3g}"


def carrier_calibration_figure(calibration) -> Figure:
    fig = Figure(figsize=(8, 4.5), facecolor="white", layout="constrained")
    ax = _new(fig, 111)
    if calibration.get("status") == "skipped":
        ax.axis("off")
        ax.text(.5, .5, calibration["reason"], ha="center", va="center", wrap=True)
        return fig
    t = np.asarray(calibration["times_s"]) * 1e3
    modeled = calibration.get("carrier_decay_mode") == "modeled"
    ax.plot(t, calibration["summary_trace"], lw=1.5, ls="--" if modeled else "-", color=SERIES[0],
            label="Reported summary (comparison only)" if modeled else "Measured summary reconstructed (not raw data)")
    ax.plot(t, calibration["fitted_trace"], lw=2, color=SERIES[1],
            label="Modeled spatial/thermal decay" if modeled else "Spatial/thermal model fit")
    ax.set_xlabel("Carrier pulse duration (ms)")
    ax.set_ylabel("Apparent transferred population (visibility included)")
    ax.set_ylim(-.02, 1.02)
    ax.legend(fontsize=8, frameon=False)
    if modeled:
        ax.set_title(f"Modeled carrier decay; RMS Rabi {calibration['modeled_rms_carrier_hz'] / 1e3:.3g} kHz\n"
                     "Initial-curvature intensity calibration; no fitted damping", fontsize=10)
    else:
        ax.set_title(f"Carrier calibration: RMS mismatch {calibration['synthetic_trace_rmse']:.3f}; "
                     f"residual dephasing {calibration['residual_coherence_decay_rate_s']:.2g}/s", fontsize=10)
    return fig


def _fmt_t(x):
    return x if isinstance(x, str) else f"{x * 1e3:.3g} ms"


def dashboard(a, title="") -> Figure:
    import textwrap

    # Explicit wrapping stays within the diagnostics axes when saving a figure;
    # matplotlib's automatic wrap uses the figure edge and can clip long audits.
    diagnostic_lines = [textwrap.fill(line, width=132) for line in diagnostics_text(a).splitlines()]
    diagnostic_text = "\n".join(diagnostic_lines)
    text_height = max(2.5, len(diagnostic_text.splitlines()) * 8.0 / 72 + 0.3)
    fig = Figure(figsize=(15, 8 + text_height), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[0.22, 1, 1, text_height / 2.5])
    plot_schedule(_new(fig, gs[0, :]), a)
    plot_spin_bars(_new(fig, gs[1, 0]), a)
    plot_traces(_new(fig, gs[1, 1]), a)
    plot_nbar(_new(fig, gs[1, 2]), a)
    plot_pnt(_new(fig, gs[2, 0]), a, fig)
    plot_joint(_new(fig, gs[2, 1]), a, fig)
    plot_photons(_new(fig, gs[2, 2]), a)
    plot_cooling_rate(_new(fig, gs[3, 0]), a)
    axt = fig.add_subplot(gs[3, 1:])
    axt.axis("off")
    axt.text(0, 1, diagnostic_text, family="monospace", fontsize=6.5, va="top", color=TEXT)
    status = a["validity"].status
    notice = " — outside model validity limits" if status == "invalid" else ""
    fig.suptitle(f"{title}\nModel validity: {STATUS_TEXT[status]}{notice}", fontsize=11,
                 color="#a32121" if status == "invalid" else TEXT)
    uncertainty_note(fig, a)
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
           ("photons_total", "photons / atom"), ("target_state_scattering_rate_per_s", "target state scattering rate (1/s)"))


def scan_figure(scan) -> Figure:
    """scan: dict with axes, metrics arrays, valid mask, optional series labels."""
    metrics = list(METRICS)
    tr = scan["metrics"].get("P_trapped")
    if tr is not None and np.isfinite(tr).any() and np.nanmin(tr) < 1 - 1e-4:  # real loss (lattice)
        metrics.insert(3, ("P_trapped", "final P_trapped (survival)"))
    ncol = (len(metrics) + 1) // 2
    fig = Figure(figsize=(5 * ncol, 8.5), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(2, ncol)
    axes = scan["axes"]
    valid = scan["valid"]
    if len(axes) == 1:
        x = np.asarray(axes[0]["values"])
        logx = axes[0].get("log", False)
        for j, (k, lab) in enumerate(metrics):
            ax = _new(fig, gs[j // ncol, j % ncol])
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
        for j, (k, lab) in enumerate(metrics):
            ax = _new(fig, gs[j // ncol, j % ncol])
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
