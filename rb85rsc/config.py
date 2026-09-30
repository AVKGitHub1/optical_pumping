"""Typed configuration shared by the GUI, the command line, and saved JSON.

Every field carries metadata: ``unit``, ``help``, optional ``min``/``max`` and
``choices``.  The GUI builds its forms from this metadata.  Values marked
ILLUSTRATIVE are demonstration defaults, not measured experimental values.
"""
from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import numpy as np

from . import polarization as pol


class ConfigError(ValueError):
    pass


def P(default, unit="", help="", min=None, max=None, choices=None, illustrative=False, **kw):
    md = dict(unit=unit, help=help, min=min, max=max, choices=choices, illustrative=illustrative, **kw)
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: copy.deepcopy(d), metadata=md)
    return field(default=default, metadata=md)


@dataclass
class TrapConfig:
    frequency_hz: float = P(70e3, "Hz", "Harmonic vibration frequency nu_t (cycles/s, not angular). Experimental value.", min=1.0)
    n_max: int = P(40, "", "Highest vibrational level kept (basis n = 0..n_max).", min=2, max=250)
    axis: list = P([1.0, 0.0, 0.0], "lab unit vector", "Direction of the modeled 1D trap axis.")
    heating_rate_quanta_per_s: float = P(0.0, "quanta/s", "Background motional heating dnbar/dt (infinite-temperature reservoir, a and a^+ jumps).", min=0.0)
    depth_hz: Optional[float] = P(None, "Hz", "Optional trap depth in frequency units; enables the harmonic-validity diagnostic.", min=0.0)


@dataclass
class InitialStateConfig:
    spin_preset: str = P(
        "uniform_all",
        "",
        "uniform_all: 1/12 per ground sublevel; uniform_F3: 1/7 per F=3 sublevel; "
        "stretched: all in |3,3>; manifold_split: F3 fraction set below, uniform within each manifold; "
        "custom: 12 populations below.",
        choices=["uniform_all", "uniform_F3", "stretched", "manifold_split", "custom"],
    )
    f3_fraction: float = P(7.0 / 12.0, "", "Total F=3 population for manifold_split (uniform per sublevel within each F).", min=0.0, max=1.0)
    custom_spin_populations: list = P(
        [1.0 / 12] * 12,
        "",
        "Populations for custom preset, order F=2 mF=-2..+2 then F=3 mF=-3..+3 (normalized; must be >=0, nonzero sum).",
    )
    motion_mode: str = P(
        "thermal_nbar", "", "Initial vibrational distribution.", choices=["thermal_nbar", "thermal_temperature", "custom", "fock"]
    )
    nbar: float = P(3.0, "quanta", "Thermal mean occupation (ILLUSTRATIVE: MOT+PGC value is unknown).", min=0.0, illustrative=True)
    temperature_k: float = P(10e-6, "K", "Temperature for thermal_temperature mode.", min=0.0)
    fock_n: int = P(1, "", "Vibrational level for fock mode.", min=0)
    custom_pn: list = P([], "", "Custom p(n) list for custom mode (normalized; >=0, nonzero sum).")


@dataclass
class FieldConfig:
    magnitude_gauss: float = P(0.5, "G", "Bias field magnitude (ILLUSTRATIVE). Weak-field (linear) Zeeman model; validity is checked.", min=0.0, illustrative=True)
    direction: list = P([0.0, 0.0, 1.0], "lab unit vector", "Bias-field / quantization axis.")


@dataclass
class RamanConfig:
    enabled: bool = P(True, "", "Master enable for the Raman coupling.")
    carrier_rabi_hz: float = P(5e3, "Hz", "Calibrated carrier two-photon Rabi frequency Omega_c/2pi, before motional overlap (ILLUSTRATIVE).", min=0.0, illustrative=True)
    frequency_mode: str = P(
        "sideband_offset",
        "",
        "sideband_offset: beat = nu_ud_ref + nu_t + offset; absolute_beat: beat frequency given directly.",
        choices=["sideband_offset", "absolute_beat"],
    )
    red_sideband_offset_hz: float = P(0.0, "Hz", "delta_beat - nu_t; 0 is the first red-sideband (cooling) resonance of the reference nu_ud.")
    beat_frequency_hz: float = P(3.0357324e9 + 70e3, "Hz", "|nu_high - nu_low| in absolute_beat mode.")
    wavelength_nm: float = P(783.0, "nm", "Raman beam wavelength (sets |k|). Experimental value.", min=100.0)
    beam_low_direction: list = P([0.70710678, 0.70710678, 0.0], "lab unit vector", "Lower-frequency Raman beam (absorbed on |up,n> -> |down,n-1>).", illustrative=True)
    beam_high_direction: list = P([-0.70710678, 0.70710678, 0.0], "lab unit vector", "Higher-frequency Raman beam (stimulated emission on up -> down).", illustrative=True)
    extra_coherence_decay_rate_s: float = P(0.0, "1/s", "Additional up-down coherence decay beyond modeled scattering (laser phase noise, B noise...).", min=0.0)
    differential_light_shift_hz: float = P(0.0, "Hz", "Calibrated Raman-beam shift of (E_up - E_down)/h at full amplitude.")
    scattering_rate_s: float = P(
        0.0, "1/s", "Residual Raman-beam photon scattering rate per atom at full amplitude. 0 = IDEALIZED (unknown, neglected).", min=0.0
    )
    scattering_model: str = P(
        "spin_preserving",
        "",
        "spin_preserving: elastic, recoil + full which-state decoherence (conservative); "
        "depolarizing: spin reshuffled uniformly over 12 sublevels (crude worst case).",
        choices=["spin_preserving", "depolarizing"],
    )
    max_sideband_order: int = P(3, "", "Keep Raman couplings with |n-m| <= this order (exact matrix elements; convergence parameter).", min=1, max=20)


@dataclass
class PolarizationConfig:
    mode: str = P("spherical", "", "spherical: effective intensity fractions; geometry: derived from beam angle + Jones vector.", choices=["spherical", "geometry"])
    pi_fraction: float = P(0.0, "", "Intensity fraction in q=0 (pi) [spherical mode].", min=0.0, max=1.0)
    sigma_minus_fraction: float = P(0.0, "", "Intensity fraction in q=-1 (wrong-handed) [spherical mode].", min=0.0, max=1.0)
    beam_angle_deg: float = P(0.0, "deg", "Beam polar angle from the bias field [geometry mode].", min=0.0, max=180.0)
    beam_azimuth_deg: float = P(0.0, "deg", "Beam azimuth about the bias field [geometry mode].")
    ellipticity_deg: float = P(45.0, "deg", "Ellipticity angle chi: +45 = positive helicity, 0 = linear, -45 = negative [geometry mode].", min=-45.0, max=45.0)
    orientation_deg: float = P(0.0, "deg", "Polarization-ellipse orientation psi in the beam frame [geometry mode].")
    linked_pi_share: float = P(0.5, "", "pi share remembered by the linked impurity control (used only while total impurity is 0).", min=0.0, max=1.0)

    # linked representation ------------------------------------------------
    @property
    def total_impurity(self) -> float:
        return self.pi_fraction + self.sigma_minus_fraction

    @property
    def pi_share_of_impurity(self) -> float:
        tot = self.total_impurity
        return self.pi_fraction / tot if tot > 0 else self.linked_pi_share

    def set_impurity(self, total: float, pi_share: float) -> None:
        self.pi_fraction, self.sigma_minus_fraction = pol.fractions_from_impurity(total, pi_share)
        self.linked_pi_share = float(pi_share)


@dataclass
class PumpConfig:
    enabled: bool = P(True, "", "Enable this beam.")
    intensity_mode: str = P("intensity", "", "Specify peak intensity directly or power + 1/e^2 waist (I = 2P/(pi w^2)).", choices=["intensity", "power_waist"])
    peak_intensity_w_m2: float = P(0.1, "W/m^2", "Peak intensity (ILLUSTRATIVE weak value).", min=0.0, illustrative=True)
    power_w: float = P(1e-6, "W", "Beam power (power_waist mode).", min=0.0)
    waist_m: float = P(2e-3, "m", "1/e^2 intensity radius (power_waist mode).", min=1e-7)
    detuning_mhz: float = P(0.0, "MHz", "Laser detuning from the named zero-field hyperfine transition (positive = blue).")
    direction: list = P([0.0, 0.0, 1.0], "lab unit vector", "Propagation direction (spherical mode; derived in geometry mode).")
    polarization: PolarizationConfig = field(default_factory=PolarizationConfig)
    light_shift_override_up_hz: Optional[float] = P(None, "Hz", "Measured light shift of |3,3> from this beam at full intensity; replaces (not adds to) the computed value.")
    light_shift_override_down_hz: Optional[float] = P(None, "Hz", "Measured light shift of |2,2> from this beam at full intensity; replaces the computed value.")


@dataclass
class OpticalConfig:
    excited_manifold: str = P(
        "all_D2",
        "",
        "all_D2: F'=1,2,3,4 included; F3_only_IDEALIZED: only F'=3 absorbs (comparison switch, not physical).",
        choices=["all_D2", "F3_only_IDEALIZED"],
    )
    path_model: str = P(
        "independent",
        "",
        "independent: rates summed over excited levels; kramers_heisenberg: amplitudes via different F' (same m') summed before squaring.",
        choices=["independent", "kramers_heisenberg"],
    )
    emission_pattern: str = P("dipole", "", "Spontaneous-emission angular distribution for recoil.", choices=["dipole", "isotropic"])
    excited_state_shift_hz: float = P(0.0, "Hz", "Configurable shift of all 5P3/2 levels relative to 5S1/2 (e.g. differential trap light shift).")
    weak_excitation_warning: float = P(0.02, "", "Excited-fraction threshold for a warning.", min=0.0)
    weak_excitation_invalid: float = P(0.1, "", "Excited-fraction threshold marking weak-excitation model invalid.", min=0.0)


@dataclass
class TimingConfig:
    protocol: str = P(
        "continuous",
        "",
        "Operating protocol.",
        choices=[
            "optical_pumping_only",
            "continuous",
            "prep_then_continuous",
            "pulsed",
            "raman_only",
            "repump_raman_no_spin_pump",
            "all_off",
            "custom",
        ],
    )
    total_duration_s: float = P(10e-3, "s", "Total wall-clock duration including any preparation.", min=0.0)
    prep_duration_s: float = P(1e-3, "s", "Spin-preparation stage (prep_then_continuous).", min=0.0)
    prep_pump_scale: float = P(1.0, "", "Spin-pump intensity multiplier during preparation.", min=0.0)
    prep_repump_scale: float = P(1.0, "", "Repump intensity multiplier during preparation.", min=0.0)
    cooling_pump_scale: float = P(1.0, "", "Spin-pump intensity multiplier during continuous cooling.", min=0.0)
    cooling_repump_scale: float = P(1.0, "", "Repump intensity multiplier during continuous cooling.", min=0.0)
    cooling_raman_amplitude: float = P(1.0, "", "Raman amplitude multiplier (scales Omega_c, shift, scattering) during cooling.", min=0.0)
    raman_pulse_s: float = P(300e-6, "s", "Raman pulse duration (pulsed).", min=0.0)
    reset_pulse_s: float = P(100e-6, "s", "Optical reset pulse duration (pulsed).", min=0.0)
    reset_pump_scale: float = P(1.0, "", "Spin-pump multiplier during reset pulses.", min=0.0)
    reset_repump_scale: float = P(1.0, "", "Repump multiplier during reset pulses.", min=0.0)
    repetitions: int = P(0, "", "Pulse repetitions; 0 = as many as fit in total_duration (remainder idle).", min=0)
    pulsed_first: str = P("reset", "", "First pulse of each pulsed cycle.", choices=["reset", "raman"])
    custom_segments: list = P(
        [], "", "custom protocol: list of {duration_s, raman, pump, repump, label} amplitude multipliers."
    )
    n_samples: int = P(201, "", "Number of uniformly spaced output samples.", min=2, max=20001)


@dataclass
class NumericsConfig:
    solver: str = P(
        "coherent",
        "",
        "coherent: reference Lindblad solver; rate: coherences eliminated (checked, rejected when invalid); auto: rate if valid else coherent.",
        choices=["coherent", "rate", "auto"],
    )
    method: str = P("DOP853", "", "scipy solve_ivp method for the coherent solver.", choices=["DOP853", "RK45"])
    rtol: float = P(1e-7, "", "Relative tolerance.", min=1e-13, max=1e-2)
    atol: float = P(1e-10, "", "Absolute tolerance.", min=1e-15, max=1e-3)
    recoil_quadrature_points: int = P(24, "", "Gauss-Legendre points over emission-direction projection.", min=2, max=400)
    boundary_tolerance: float = P(1e-4, "", "Max allowed population in the top two vibrational levels / overflow.", min=0.0)
    conservation_tolerance: float = P(1e-6, "", "Max allowed |trace + overflow - 1|.", min=0.0)
    positivity_tolerance: float = P(1e-8, "", "Most negative eigenvalue/population tolerated before flagging.", min=0.0)
    rate_validity_epsilon: float = P(0.2, "", "Rate backend requires Omega_nm <= eps*sqrt(gamma^2 + delta^2) for occupied pairs.", min=0.0)
    seed: int = P(0, "", "Recorded for reproducibility (the solvers are deterministic).")


@dataclass
class AnalysisConfig:
    spin_threshold: float = P(0.9, "", "P_up threshold for the time-to-polarization metric.", min=0.0, max=1.0)
    joint_threshold: float = P(0.5, "", "P_target threshold for the time-to-joint-success metric.", min=0.0, max=1.0)
    derivative_window: int = P(11, "samples", "Savitzky-Golay window (odd) for cooling-rate derivative.", min=3)


@dataclass
class SimConfig:
    name: str = P("default", "", "Run label.")
    trap: TrapConfig = field(default_factory=TrapConfig)
    initial: InitialStateConfig = field(default_factory=InitialStateConfig)
    magnetic: FieldConfig = field(default_factory=FieldConfig)
    raman: RamanConfig = field(default_factory=RamanConfig)
    spin_pump: PumpConfig = field(default_factory=PumpConfig)
    repump: PumpConfig = field(default_factory=PumpConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        cfg = _build(cls, d, "")
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "SimConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> "SimConfig":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        errs = []
        for path, f, value in iter_fields(self):
            md = f.metadata
            if md.get("choices") and value not in md["choices"]:
                errs.append(f"{path}={value!r} not in {md['choices']}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not np.isfinite(value):
                    errs.append(f"{path} is not finite")
                if md.get("min") is not None and value < md["min"]:
                    errs.append(f"{path}={value} below minimum {md['min']}")
                if md.get("max") is not None and value > md["max"]:
                    errs.append(f"{path}={value} above maximum {md['max']}")
            if path.endswith(("direction", "axis")) and isinstance(value, list):
                if len(value) != 3 or not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
                    errs.append(f"{path} must be a nonzero 3-vector")
        for name in ("spin_pump", "repump"):
            p = getattr(self, name).polarization
            if p.mode == "spherical":
                sp = 1.0 - p.pi_fraction - p.sigma_minus_fraction
                if sp < -1e-12:
                    errs.append(f"{name}.polarization: pi + sigma_minus = {p.total_impurity} exceeds 1")
        ini = self.initial
        if ini.spin_preset == "custom":
            v = np.asarray(ini.custom_spin_populations, float)
            if v.shape != (12,) or np.any(v < 0) or v.sum() <= 0:
                errs.append("initial.custom_spin_populations must be 12 nonnegative numbers with nonzero sum")
        if ini.motion_mode == "custom":
            v = np.asarray(ini.custom_pn, float)
            if v.ndim != 1 or v.size == 0 or np.any(v < 0) or v.sum() <= 0:
                errs.append("initial.custom_pn must be nonnegative with nonzero sum")
        if self.analysis.derivative_window % 2 == 0:
            errs.append("analysis.derivative_window must be odd")
        t = self.timing
        if t.protocol == "prep_then_continuous" and t.prep_duration_s > t.total_duration_s:
            errs.append("timing.prep_duration_s exceeds total_duration_s")
        if t.protocol == "pulsed":
            per = t.raman_pulse_s + t.reset_pulse_s
            if per <= 0:
                errs.append("pulsed protocol needs positive pulse durations")
            elif t.repetitions and t.repetitions * per > t.total_duration_s * (1 + 1e-12):
                errs.append("repetitions * (raman_pulse + reset_pulse) exceeds total_duration_s")
        if t.protocol == "custom":
            if not t.custom_segments:
                errs.append("custom protocol requires timing.custom_segments")
            for i, s in enumerate(t.custom_segments):
                if float(s.get("duration_s", -1)) <= 0:
                    errs.append(f"custom_segments[{i}].duration_s must be > 0")
                for k in ("raman", "pump", "repump"):
                    if float(s.get(k, 0.0)) < 0:
                        errs.append(f"custom_segments[{i}].{k} must be >= 0")
        if self.numerics.atol <= 0 or self.numerics.rtol <= 0:
            errs.append("tolerances must be positive")
        if errs:
            raise ConfigError("; ".join(errs))


# ---------------------------------------------------------------------------
def _build(cls, d: dict, prefix: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{prefix or 'config'} must be an object")
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys in {prefix or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in d:
            continue
        t = hints[name]
        v = d[name]
        if is_dataclass(t):
            kwargs[name] = _build(t, v, f"{prefix}{name}.")
        else:
            kwargs[name] = _coerce(t, v, prefix + name)
    return cls(**kwargs)


def _coerce(t, v, path):
    origin = getattr(t, "__origin__", None)
    if v is None:
        return None
    if t is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{path} must be true/false")
        return v
    if t is int or (origin is not None and int in getattr(t, "__args__", ()) and float not in t.__args__):
        if isinstance(v, bool) or float(v) != int(v):
            raise ConfigError(f"{path} must be an integer")
        return int(v)
    if t is float or (origin is not None and float in getattr(t, "__args__", ())):
        return float(v)
    return v


def iter_fields(obj, prefix=""):
    """Yield (dotted_path, dataclass Field, value) for every leaf field."""
    for f in fields(obj):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            yield from iter_fields(v, f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}", f, v


VIRTUAL = ("total_impurity", "pi_share_of_impurity")


def get_param(cfg: SimConfig, path: str) -> Any:
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return getattr(obj, parts[-1])


def set_param(cfg: SimConfig, path: str, value: Any) -> None:
    """Set a dotted parameter; supports the virtual linked impurity controls."""
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, PolarizationConfig) and last in VIRTUAL:
        tot, share = obj.total_impurity, obj.pi_share_of_impurity
        if last == "total_impurity":
            obj.set_impurity(float(value), share)
        else:
            obj.set_impurity(tot, float(value))
        return
    if not hasattr(obj, last):
        raise ConfigError(f"unknown parameter {path}")
    cur = getattr(obj, last)
    if isinstance(cur, bool):
        value = bool(value)
    elif isinstance(cur, int) and not isinstance(cur, bool):
        value = int(value)
    elif isinstance(cur, float) or cur is None:
        value = float(value) if value is not None else None
    setattr(obj, last, value)


def field_info(cfg_cls=SimConfig, prefix="") -> list[tuple[str, Any]]:
    """[(dotted_path, Field)] for GUI construction."""
    out = []
    hints = get_type_hints(cfg_cls)
    for f in fields(cfg_cls):
        t = hints[f.name]
        if is_dataclass(t):
            out.extend(field_info(t, f"{prefix}{f.name}."))
        else:
            out.append((prefix + f.name, f))
    return out


def illustrative_parameters(cfg: SimConfig) -> list[str]:
    return [p for p, f, _ in iter_fields(cfg) if f.metadata.get("illustrative")]
