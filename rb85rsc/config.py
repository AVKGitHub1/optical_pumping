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
    wavelength_nm: Optional[float] = P(None, "nm", "Lattice optical wavelength. When supplied, depth and wavelength determine the bottom frequency; frequency_hz is inactive.", min=100.0)
    frequency_hz: float = P(
        70e3, "Hz",
        "Harmonic vibration frequency nu_t (cycles/s, not angular). Experimental value. Lattice: harmonic frequency at the bottom "
        "of a site, 2 sqrt(V0 E_r)/h; the n=1->0 spacing is lower by about E_r.", min=1.0,
    )
    potential: str = P(
        "harmonic", "",
        "harmonic: infinitely deep harmonic trap. lattice: one site of a 1D lattice V0 sin^2(k_L z) along trap.axis with V0 = depth_uk "
        "and k_L from frequency_hz; exact bound levels, anharmonic sidebands, promotion above V0 counted as loss.",
        choices=["harmonic", "lattice"],
    )
    n_max: int = P(40, "", "Highest vibrational level kept (basis n = 0..n_max). Lattice: cap on the bound levels kept.", min=2, max=250)
    axis: list = P([1.0, 0.0, 0.0], "lab unit vector", "Direction of the modeled 1D trap axis.")
    heating_rate_quanta_per_s: float = P(0.0, "quanta/s", "Background motional heating dnbar/dt (infinite-temperature reservoir, a and a^+ jumps).", min=0.0)
    depth_uk: Optional[float] = P(None, "uK", "Trap depth (k_B x temperature units). Harmonic: optional, enables the harmonic-validity diagnostic. Lattice: required, V0.", min=0.0)


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
    magnitude_gauss: float = P(0.5, "G", "Bias field magnitude (ILLUSTRATIVE). Breit-Rabi ground energies; excited shifts and dipoles use the weak-field basis, with validity checked.", min=0.0, illustrative=True)
    direction: list = P([0.0, 0.0, 1.0], "lab unit vector", "Bias-field / quantization axis.")


@dataclass
class RamanConfig:
    calibration: str = P("pathway", "", "pathway: legacy one-path Rabi. measured_carrier: calibrate the ensemble carrier using ARC D1+D2 amplitudes and carrier_decay_mode (ensemble mode).", choices=["pathway", "measured_carrier"])
    carrier_decay_mode: str = P("measured_envelope", "", "modeled: predict spatial/thermal dephasing, with carrier_rabi_hz interpreted as the RMS local carrier frequency (initial-curvature calibration), and no fitted damping. measured_envelope: fit intensity and nonnegative residual damping to the reported envelope.", choices=["modeled", "measured_envelope"])
    beam_helicities: list = P([1, -1], "", "Helicity along each beam's OWN propagation vector: +1 and -1 are opposite circular polarizations.")
    waist_um: float = P(200.0, "um", "Raman 1/e^2 intensity radius.", min=1.0)
    flop_contrast: float = P(0.9, "", "Observed carrier contrast; treated as preparation/readout visibility, not atom loss.", min=0.0, max=1.0)
    flop_decay_periods: float = P(3.5, "periods", "Reported 1/e carrier decay in measured periods. Fit input only in measured_envelope mode; comparison only in modeled mode.", min=0.1)
    enabled: bool = P(True, "", "Master enable for the Raman coupling.")
    carrier_rabi_hz: float = P(
        5e3, "Hz",
        "Two-photon Rabi frequency Omega/2pi. calibration=pathway: one-path value before motional overlap. "
        "calibration=measured_carrier: measured ensemble carrier-flop frequency, interpreted by the ARC/spatial calibration.", min=0.0, illustrative=True,
    )
    frequency_mode: str = P(
        "sideband_offset",
        "",
        "sideband_offset: beat = nu_ud_ref + nu_t + offset; absolute_beat: beat frequency given directly.",
        choices=["sideband_offset", "absolute_beat"],
    )
    red_sideband_offset_hz: float = P(0.0, "Hz", "delta_beat - nu_10; 0 is the first red-sideband (n=1->0) resonance of the reference nu_ud (nu_10 = nu_t harmonic, exact level spacing lattice).")
    beat_frequency_hz: float = P(3.0357324e9 + 70e3, "Hz", "|nu_high - nu_low| in absolute_beat mode.")
    wavelength_nm: float = P(783.0, "nm", "Raman beam wavelength (sets |k|). Experimental value.", min=100.0)
    beam_low_direction: list = P(
        [0.70710678, 0.70710678, 0.0], "lab unit vector",
        "Lower-frequency Raman beam (absorbed on |up,n> -> |down,n-1>); beam 1 in both_tones_both_beams.", illustrative=True,
    )
    beam_high_direction: list = P(
        [-0.70710678, 0.70710678, 0.0], "lab unit vector",
        "Higher-frequency Raman beam (stimulated emission on up -> down); beam 2 in both_tones_both_beams.", illustrative=True,
    )
    tone_layout: str = P(
        "one_tone_per_beam",
        "",
        "one_tone_per_beam: beam 1 carries only the low tone, beam 2 only the high tone. "
        "both_tones_both_beams: each beam carries both tones; the two co-propagating pathways (dk = 0, carrier only) and the two "
        "crossed pathways (dk = +/-k(k1 - k2)) add coherently with the phases below.",
        choices=["one_tone_per_beam", "both_tones_both_beams"],
    )
    tone_powers_low: list = P(
        [1.0, 1.0], "relative",
        "[beam 1, beam 2] power of the low tone [both_tones_both_beams]. Pathway (low from i, high from j) has Rabi "
        "carrier_rabi_hz * sqrt(p_low_i p_high_j).", illustrative=True,
    )
    tone_powers_high: list = P([1.0, 1.0], "relative", "[beam 1, beam 2] power of the high tone [both_tones_both_beams].", illustrative=True)
    lattice_phase_deg: float = P(
        90.0, "deg",
        "Phase theta of the crossed pathway (low from beam 1, high from beam 2) relative to the co-propagating beam-1 pathway "
        "= dk.X_atom + tone phases: where the atom sits in the static beat interference pattern. For balanced beams the red "
        "sideband scales as sin(theta) and the crossed carrier as cos(theta) [both_tones_both_beams]. "
        "Inactive in ensemble mode, which averages registration using ensemble.phase_offset_deg.", illustrative=True,
    )
    beam_beat_phase_deg: float = P(
        0.0, "deg", "Beat-note phase chi of beam 2 minus beam 1, (phi_high - phi_low)_2 - (phi_high - phi_low)_1 [both_tones_both_beams].",
        illustrative=True,
    )
    extra_coherence_decay_rate_s: float = P(0.0, "1/s", "Additional up-down coherence decay beyond modeled scattering (laser phase noise, B noise...).", min=0.0)
    differential_light_shift_hz: float = P(0.0, "Hz", "Calibrated Raman-beam shift of (E_up - E_down)/h at full amplitude.")
    scattering_rate_s: float = P(
        0.0, "1/s", "Additional Raman-beam photon scattering rate at full amplitude. In measured_carrier mode this adds to the ARC estimate; in legacy mode 0 neglects scattering.", min=0.0
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
    power_mw: float = P(1e-3, "mW", "Beam power (power_waist mode).", min=0.0)
    waist_um: float = P(2000.0, "um", "1/e^2 intensity radius (power_waist mode).", min=0.1)
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
    total_duration_ms: float = P(10.0, "ms", "Total wall-clock duration including any preparation.", min=0.0)
    prep_duration_ms: float = P(1.0, "ms", "Spin-preparation stage (prep_then_continuous).", min=0.0)
    prep_pump_scale: float = P(1.0, "", "Spin-pump intensity multiplier during preparation.", min=0.0)
    prep_repump_scale: float = P(1.0, "", "Repump intensity multiplier during preparation.", min=0.0)
    cooling_pump_scale: float = P(1.0, "", "Spin-pump intensity multiplier during continuous cooling.", min=0.0)
    cooling_repump_scale: float = P(1.0, "", "Repump intensity multiplier during continuous cooling.", min=0.0)
    cooling_raman_amplitude: float = P(1.0, "", "Raman amplitude multiplier (scales Omega_c, shift, scattering) during cooling.", min=0.0)
    raman_pulse_ms: float = P(0.3, "ms", "Raman pulse duration (pulsed).", min=0.0)
    reset_pulse_ms: float = P(0.1, "ms", "Optical reset pulse duration (pulsed).", min=0.0)
    reset_pump_scale: float = P(1.0, "", "Spin-pump multiplier during reset pulses.", min=0.0)
    reset_repump_scale: float = P(1.0, "", "Repump multiplier during reset pulses.", min=0.0)
    repetitions: int = P(0, "", "Pulse repetitions; 0 = as many as fit in total_duration (remainder idle).", min=0)
    pulsed_first: str = P("reset", "", "First pulse of each pulsed cycle.", choices=["reset", "raman"])
    custom_segments: list = P(
        [], "", "custom protocol: list of {duration_ms, raman, pump, repump, label} amplitude multipliers."
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
class EnsembleConfig:
    enabled: bool = P(False, "", "Spatial/phase ensemble with coherent axial Raman dynamics and three-axis bound-state recoil/loss.")
    samples: int = P(32, "", "Deterministic scrambled Sobol samples of Gaussian position and uniform Raman spatial phase; refine to check convergence.", min=1)
    workers: int = P(1, "", "Processes for independent spatial samples in headless runs; interactive cancellable runs use one process.", min=1, max=8)
    cloud_radius_um: float = P(150.0, "um", "Spherical Gaussian 1/e^2 DENSITY radius (coordinate standard deviation = radius/2).", min=0.0)
    lattice_waist_um: float = P(200.0, "um", "All lattice beams' 1/e^2 intensity radius.", min=1.0)
    transverse_wavelength_nm: float = P(1188.0, "nm", "Optical wavelength of both transverse standing waves.", min=100.0)
    transverse_depth_uk: float = P(15.0, "uK", "Central depth of EACH transverse standing wave.", min=0.01)
    transverse_temperature_k: float = P(10e-6, "K", "Loaded transverse temperature, conditional on local bound states.", min=0.0)
    transverse_frequency_offset_hz: float = P(160e6, "Hz", "Optical carrier offset BETWEEN the x and y lattices, suppressing inter-axis interference.", min=0.0)
    lattice_linear_polarizations: list = P([[0, 1, 0], [1, 0, 0], [1, 0, 0]], "lab vectors", "Linear electric field directions for x, y, z lattices, perpendicular to the respective axes.")
    phase_offset_deg: float = P(0.0, "deg", "Offset added to the uniformly sampled Raman spatial phase; use for quadrature sensitivity.")
    time_step_us: float = P(2.0, "us", "Maximum Strang splitting step; halve to verify integration convergence.", min=0.01)
    calibration_samples: int = P(128, "", "Position/phase samples for independent pump-off, z-lattice-only carrier calibration.", min=8)


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
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

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
        if self.trap.potential == "lattice" and not (self.trap.depth_uk and self.trap.depth_uk > 0):
            errs.append("trap.potential = lattice requires trap.depth_uk > 0")
        if self.raman.calibration == "measured_carrier" and not self.ensemble.enabled:
            errs.append("measured_carrier calibration requires ensemble.enabled")
        if self.raman.calibration == "measured_carrier" and (self.raman.carrier_rabi_hz <= 0 or self.raman.flop_contrast <= 0):
            errs.append("measured_carrier requires positive measured frequency and contrast; use raman.enabled=false to turn the drive off")
        hel = np.asarray(self.raman.beam_helicities)
        if hel.shape != (2,) or not np.all(np.isin(hel, [-1, 1])):
            errs.append("raman.beam_helicities must contain two values, each +1 or -1")
        if self.ensemble.enabled:
            if self.raman.calibration != "measured_carrier":
                errs.append("ensemble mode requires raman.calibration = measured_carrier")
            if self.trap.potential != "lattice" or self.trap.wavelength_nm is None:
                errs.append("ensemble mode requires lattice depth and optical wavelength")
            if not np.allclose(pol.unit(self.trap.axis), [0, 0, 1]):
                errs.append("ensemble mode currently requires the axial lattice along lab z")
            if not np.allclose(pol.unit(self.raman.beam_low_direction), [0, 0, 1]) or not np.allclose(pol.unit(self.raman.beam_high_direction), [0, 0, -1]):
                errs.append("ensemble mode requires Raman beams along +z and -z")
            if self.raman.tone_layout != "both_tones_both_beams":
                errs.append("ensemble mode requires both tones in both Raman beams")
            if self.numerics.solver == "rate":
                errs.append("ensemble mode uses coherent evolution; rate solver is unsupported")
            if self.trap.heating_rate_quanta_per_s != 0:
                errs.append("ensemble mode does not implement an anharmonic background-heating noise model")
            ep = np.asarray(self.ensemble.lattice_linear_polarizations, float)
            if ep.shape != (3, 3) or not np.all(np.isfinite(ep)) or np.any(np.linalg.norm(ep, axis=1) == 0) or not np.allclose(np.diag(ep), 0):
                errs.append("ensemble.lattice_linear_polarizations must be three nonzero transverse lab vectors for x,y,z")
        for name in ("tone_powers_low", "tone_powers_high"):
            v = np.asarray(getattr(self.raman, name), float)
            if v.shape != (2,) or not np.all(np.isfinite(v)) or np.any(v < 0):
                errs.append(f"raman.{name} must be 2 nonnegative numbers [beam 1, beam 2]")
        for name in ("spin_pump", "repump"):
            p = getattr(self, name).polarization
            if p.mode == "spherical":
                sp = 1.0 - p.pi_fraction - p.sigma_minus_fraction
                if sp < -1e-12:
                    errs.append(f"{name}.polarization: pi + sigma_minus = {p.total_impurity} exceeds 1")
        ini = self.initial
        if ini.spin_preset == "custom":
            v = np.asarray(ini.custom_spin_populations, float)
            if v.shape != (12,) or not np.all(np.isfinite(v)) or np.any(v < 0) or v.sum() <= 0:
                errs.append("initial.custom_spin_populations must be 12 nonnegative numbers with nonzero sum")
        if ini.motion_mode == "custom":
            v = np.asarray(ini.custom_pn, float)
            if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or np.any(v < 0) or v.sum() <= 0:
                errs.append("initial.custom_pn must be nonnegative with nonzero sum")
        if self.analysis.derivative_window % 2 == 0:
            errs.append("analysis.derivative_window must be odd")
        t = self.timing
        if t.protocol == "prep_then_continuous" and t.prep_duration_ms > t.total_duration_ms:
            errs.append("timing.prep_duration_ms exceeds total_duration_ms")
        if t.protocol == "pulsed":
            per = t.raman_pulse_ms + t.reset_pulse_ms
            if per <= 0:
                errs.append("pulsed protocol needs positive pulse durations")
            elif t.repetitions and t.repetitions * per > t.total_duration_ms * (1 + 1e-12):
                errs.append("repetitions * (raman_pulse + reset_pulse) exceeds total_duration_ms")
        if t.protocol == "custom":
            if not t.custom_segments:
                errs.append("custom protocol requires timing.custom_segments")
            for i, s in enumerate(t.custom_segments):
                if float(s.get("duration_ms", -1)) <= 0:
                    errs.append(f"custom_segments[{i}].duration_ms must be > 0")
                for k in ("raman", "pump", "repump"):
                    if float(s.get(k, 0.0)) < 0:
                        errs.append(f"custom_segments[{i}].{k} must be >= 0")
        if self.numerics.atol <= 0 or self.numerics.rtol <= 0:
            errs.append("tolerances must be positive")
        if errs:
            raise ConfigError("; ".join(errs))


# ---------------------------------------------------------------------------
# renamed fields: old JSON key -> (new key, factor old -> new units); lets saved configs keep loading
_LEGACY_KEYS = {
    "PumpConfig": (("power_w", "power_mw", 1e3), ("waist_m", "waist_um", 1e6)),
    "TrapConfig": (("depth_hz", "depth_uk", 6.62607015e-34 / 1.380649e-23 * 1e6),),
    "TimingConfig": tuple((f"{k}_s", f"{k}_ms", 1e3) for k in ("total_duration", "prep_duration", "raman_pulse", "reset_pulse")),
}


def _legacy(cls_name: str, key: str):
    return next(((new, scale) for old, new, scale in _LEGACY_KEYS.get(cls_name, ()) if old == key), None)


def _migrate_segments(segs):
    """custom_segments entries: duration_s -> duration_ms."""
    if not isinstance(segs, list):
        return segs
    return [{("duration_ms" if k == "duration_s" else k): (float(v) * 1e3 if k == "duration_s" else v) for k, v in s.items()}
            if isinstance(s, dict) and "duration_s" in s and "duration_ms" not in s else s for s in segs]


def _build(cls, d: dict, prefix: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{prefix or 'config'} must be an object")
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    for old, new, scale in _LEGACY_KEYS.get(cls.__name__, ()):
        if old in d:
            if new in d:
                raise ConfigError(f"{prefix}{old} and {prefix}{new} both given")
            d = {**{k: v for k, v in d.items() if k != old}, new: None if d[old] is None else float(d[old]) * scale}
    if cls.__name__ == "TimingConfig" and "custom_segments" in d:
        d = {**d, "custom_segments": _migrate_segments(d["custom_segments"])}
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
    leg = None if hasattr(obj, last) else _legacy(type(obj).__name__, last)
    if leg:  # old unit-suffixed name, e.g. timing.total_duration_s
        last, value = leg[0], (None if value is None else float(value) * leg[1])
    if last == "custom_segments":
        value = _migrate_segments(value)
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
