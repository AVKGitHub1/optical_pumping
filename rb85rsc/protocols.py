"""Piecewise-constant operating schedules.

Each segment gives amplitude multipliers applied to the configured beams:
``raman`` multiplies Omega_c (and, linearly, the calibrated Raman light shift and
Raman scattering rate, i.e. both Raman beam intensities scaled together);
``pump`` and ``repump`` multiply the pump-beam intensities.  All protocols span
exactly ``timing.total_duration_s`` so comparisons use equal wall-clock time.
"""
from __future__ import annotations

from dataclasses import dataclass

PROTOCOLS = {
    "optical_pumping_only": "1. Both optical beams, Raman off",
    "continuous": "2. Continuous combined cooling: both beams + Raman",
    "prep_then_continuous": "3. Spin preparation, then simultaneous weak pumping + Raman",
    "pulsed": "4. Alternating optical reset and Raman pulses",
    "raman_only": "5a. Control: Raman only",
    "repump_raman_no_spin_pump": "5b. Control: repump + Raman, F=3 spin pump disabled",
    "all_off": "5c. Control: all fields off",
    "custom": "Custom piecewise schedule",
}
COMPARISON_SET = [
    "optical_pumping_only",
    "continuous",
    "prep_then_continuous",
    "pulsed",
    "raman_only",
    "repump_raman_no_spin_pump",
    "all_off",
]


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    raman: float
    pump: float
    repump: float
    label: str

    @property
    def key(self) -> tuple:
        return (self.raman, self.pump, self.repump)


def build_schedule(timing) -> list[Segment]:
    T = timing.total_duration_s
    p = timing.protocol
    segs: list[tuple[float, float, float, float, str]] = []  # (duration, raman, pump, repump, label)
    if p == "optical_pumping_only":
        segs = [(T, 0.0, 1.0, 1.0, "optical pumping")]
    elif p == "continuous":
        segs = [(T, timing.cooling_raman_amplitude, timing.cooling_pump_scale, timing.cooling_repump_scale, "continuous")]
    elif p == "prep_then_continuous":
        tp = timing.prep_duration_s
        segs = [
            (tp, 0.0, timing.prep_pump_scale, timing.prep_repump_scale, "preparation"),
            (T - tp, timing.cooling_raman_amplitude, timing.cooling_pump_scale, timing.cooling_repump_scale, "continuous"),
        ]
    elif p == "pulsed":
        per = timing.raman_pulse_s + timing.reset_pulse_s
        reps = timing.repetitions or int(T // per + 1e-9)
        reset = (timing.reset_pulse_s, 0.0, timing.reset_pump_scale, timing.reset_repump_scale, "reset")
        raman = (timing.raman_pulse_s, timing.cooling_raman_amplitude, 0.0, 0.0, "raman pulse")
        cycle = [reset, raman] if timing.pulsed_first == "reset" else [raman, reset]
        segs = cycle * reps
        rem = T - reps * per
        if rem > 1e-12 * max(T, 1e-30):
            segs.append((rem, 0.0, 0.0, 0.0, "idle"))
    elif p == "raman_only":
        segs = [(T, timing.cooling_raman_amplitude, 0.0, 0.0, "raman only")]
    elif p == "repump_raman_no_spin_pump":
        segs = [(T, timing.cooling_raman_amplitude, 0.0, timing.cooling_repump_scale, "repump + raman")]
    elif p == "all_off":
        segs = [(T, 0.0, 0.0, 0.0, "all off")]
    elif p == "custom":
        segs = [
            (float(s["duration_s"]), float(s.get("raman", 0.0)), float(s.get("pump", 0.0)), float(s.get("repump", 0.0)), s.get("label", f"seg{i}"))
            for i, s in enumerate(timing.custom_segments)
        ]
    else:
        raise ValueError(f"unknown protocol {p}")
    out = []
    t = 0.0
    for dur, r, pu, re, lab in segs:
        if dur <= 0:
            continue
        out.append(Segment(t, t + dur, r, pu, re, lab))
        t += dur
    if p != "custom" and out:
        # Snap the final boundary to exactly T (no accumulated rounding).
        last = out[-1]
        out[-1] = Segment(last.t0, T, last.raman, last.pump, last.repump, last.label)
    return out


def schedule_duration(segs: list[Segment]) -> float:
    return segs[-1].t1 if segs else 0.0
