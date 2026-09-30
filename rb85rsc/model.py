"""Assemble all physical operators for one configuration.

State space
-----------
Ground sublevels s = 0..11 (see ``atomic.GROUND_STATES``) times vibrational
levels n = 0..n_max (N = n_max + 1).  Population index = s*N + n.

* The Raman pair {up=|3,3>, down=|2,2>} (x motion) keeps a full density matrix,
  because the calibrated Raman coupling acts only there.
* The other 10 sublevels keep motional populations only: nothing couples them
  coherently, and the secular approximation (all optical rates << omega_t, all
  pumping rates << ground Zeeman splittings) removes coherences created by
  spontaneous emission.  Both conditions are checked in the diagnostics.

Frame
-----
Rotating frame of the Raman beat omega_b, then the interaction picture with
respect to H0 = omega_t (a^+a - |up><up|).  The residual Hamiltonian is

  H_I(t)/hbar = -delta |up><up| + sum_{n,m} (Omega_c/2) D_nm e^{i(n-m-1) omega_t t} |up,n><down,m| + h.c.

with D = exp(i eta_R (a + a^+)) (exact matrix elements) and
delta = 2 pi (nu_beat - nu_ud - nu_t).  The first red sideband n -> n-1 is static;
the carrier rotates at -omega_t and the blue sideband at -2 omega_t, so they are
retained as off-resonant terms rather than dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import motion as mo
from . import polarization as pol
from .atomic import (
    DOWN,
    GROUND_STATES,
    MU_B,
    N_GROUND,
    SPECTATORS,
    UP,
    H,
    AtomicData,
    ground_label,
    load_atomic_data,
    zeeman_validity,
)
from .config import SimConfig, illustrative_parameters
from .optical import BeamRates, beam_rates, beam_spec
from .protocols import Segment, build_schedule

BEAMS = ("spin_pump", "repump")
COUNTERS = ("spin_pump", "repump", "raman_scatter")


@dataclass
class SegmentOps:
    seg: Segment
    G: np.ndarray  # (12N, 12N) population gain matrix from optical/Raman scattering
    gout: np.ndarray  # (12N,) total out-rate (includes self-returning events)
    ovr: np.ndarray  # (12N,) rate into the numerical overflow bin
    counts: np.ndarray  # (3, 12N) photon-emission rate per counter
    delta: float  # rad/s, detuning of the drive from the first red sideband
    vk: list  # [(k*omega_t, V_k)] with V_k (N,N) complex
    gamma_extra: float
    heating: float
    ls_up: float  # rad/s total pump light shift of up
    ls_down: float
    raman_amp: float


@dataclass
class Validity:
    items: list = field(default_factory=list)  # (category, status, message)

    def add(self, cat, status, msg):
        self.items.append((cat, status, msg))

    @property
    def status(self):
        s = [i[1] for i in self.items]
        return "invalid" if "invalid" in s else ("warning" if "warning" in s else "ok")


class Model:
    def __init__(self, cfg: SimConfig, atom: AtomicData | None = None):
        cfg.validate()
        self.cfg = cfg
        self.atom = atom or load_atomic_data()
        at = self.atom
        self.N = N = cfg.trap.n_max + 1
        self.b_dir = pol.unit(cfg.magnetic.direction)
        self.B = cfg.magnetic.magnitude_gauss * 1e-4
        self.u = pol.unit(cfg.trap.axis)
        self.cos_beta = float(np.dot(self.u, self.b_dir))
        self.nu_t = cfg.trap.frequency_hz
        self.omega_t = 2 * np.pi * self.nu_t
        self.x0 = mo.x0_m(at.mass_kg, self.nu_t)
        self.k_d2 = 2 * np.pi / at.d2_wavelength_m
        self.eta_d2 = self.k_d2 * self.x0  # single 780 nm photon along the trap axis
        nq = cfg.numerics.recoil_quadrature_points
        pat = lambda cls: cls if cfg.optical.emission_pattern == "dipole" else "isotropic"

        # ---- optical pumping beams ----------------------------------------
        self.beams = {b: beam_spec(b, getattr(cfg, b), at, self.b_dir) for b in BEAMS}
        self.rates: dict[str, BeamRates] = {
            b: beam_rates(at, self.beams[b], self.B, cfg.optical.excited_state_shift_hz, cfg.optical.excited_manifold, cfg.optical.path_model)
            for b in BEAMS
        }
        self.T, self.out, self.ovf = {}, {}, {}
        for b in BEAMS:
            proj = float(np.dot(self.beams[b].direction, self.u))
            kern = {cls: mo.recoil_kernel(self.eta_d2, proj, self.cos_beta, pat(cls), N, nq) for cls in ("sigma", "pi")}
            self.T[b], self.out[b], self.ovf[b] = self._transfer(self.rates[b].W, kern)

        # ---- Raman ------------------------------------------------------------
        rc = cfg.raman
        self.k_raman = 2 * np.pi / (rc.wavelength_nm * 1e-9)
        klo, khi = pol.unit(rc.beam_low_direction), pol.unit(rc.beam_high_direction)
        self.dk = self.k_raman * (klo - khi)  # k_abs - k_emit for up -> down
        self.eta_R_signed = float(np.dot(self.dk, self.u) * self.x0)
        self.eta_R = abs(self.eta_R_signed)
        self.D = mo.displacement_matrix(self.eta_R_signed, N)
        self.omega_c = 2 * np.pi * rc.carrier_rabi_hz if rc.enabled else 0.0
        order = rc.max_sideband_order
        n = np.arange(N)
        dn = n[:, None] - n[None, :]
        self.vk_unit = []
        for k in range(-order - 1, order):
            mask = (dn - 1 == k) & (np.abs(dn) <= order)
            if mask.any():
                Vk = np.where(mask, 0.5 * self.omega_c * self.D, 0.0)
                if np.any(Vk != 0):
                    self.vk_unit.append((k * self.omega_t, Vk))
        eg = at.ground_energy_hz(self.B)
        self.nu_ud_zeeman = float(eg[UP] - eg[DOWN])  # hyperfine + linear Zeeman
        self.nu_ud_ref = self.nu_ud_zeeman + rc.differential_light_shift_hz
        if rc.frequency_mode == "sideband_offset":
            self.nu_beat = self.nu_ud_ref + self.nu_t + rc.red_sideband_offset_hz
        else:
            self.nu_beat = rc.beat_frequency_hz
        # Raman-beam scattering (calibrated; recoil: absorb from either beam, isotropic emission)
        Wr = np.zeros((N_GROUND, N_GROUND, 2))
        if rc.scattering_model == "spin_preserving":
            for g in range(N_GROUND):
                Wr[g, g, 0] = rc.scattering_rate_s
        else:
            Wr[:, :, 0] = rc.scattering_rate_s / N_GROUND
        klo_p, khi_p = float(np.dot(klo, self.u)), float(np.dot(khi, self.u))
        eta_rp = self.k_raman * self.x0
        k1, o1 = mo.recoil_kernel(eta_rp, klo_p, self.cos_beta, "isotropic", N, nq)
        k2, o2 = mo.recoil_kernel(eta_rp, khi_p, self.cos_beta, "isotropic", N, nq)
        kr = (0.5 * (k1 + k2), 0.5 * (o1 + o2))
        self.T["raman_scatter"], self.out["raman_scatter"], self.ovf["raman_scatter"] = self._transfer(Wr, {"sigma": kr, "pi": kr})

        self.schedule = build_schedule(cfg.timing)
        self.P0, self.initial_tail, self.initial_notes = self._initial_state()

    # ------------------------------------------------------------------
    def _transfer(self, W, kern):
        N = self.N
        T = np.zeros((N_GROUND * N, N_GROUND * N))
        ovr = np.zeros(N_GROUND * N)
        for ci, cls in enumerate(("sigma", "pi")):
            K, ov = kern[cls]
            for g in range(N_GROUND):
                for g2 in range(N_GROUND):
                    w = W[g, g2, ci]
                    if w > 0:
                        T[g2 * N : (g2 + 1) * N, g * N : (g + 1) * N] += w * K
                        ovr[g * N : (g + 1) * N] += w * ov
        out = np.repeat(W.sum(axis=(1, 2)), N)
        return T, out, ovr

    def _initial_state(self):
        ic = self.cfg.initial
        N = self.N
        notes = []
        if ic.spin_preset == "uniform_all":
            spin = np.full(N_GROUND, 1.0 / N_GROUND)
        elif ic.spin_preset == "uniform_F3":
            spin = np.array([1.0 / 7 if f == 3 else 0.0 for f, _ in GROUND_STATES])
        elif ic.spin_preset == "stretched":
            spin = np.zeros(N_GROUND)
            spin[UP] = 1.0
        elif ic.spin_preset == "manifold_split":
            f3 = ic.f3_fraction
            spin = np.array([f3 / 7 if f == 3 else (1 - f3) / 5 for f, _ in GROUND_STATES])
        else:
            v = np.asarray(ic.custom_spin_populations, float)
            if abs(v.sum() - 1) > 1e-9:
                notes.append(f"custom spin populations normalized from sum {v.sum():.6g}")
            spin = v / v.sum()
        tail = 0.0
        if ic.motion_mode in ("thermal_nbar", "thermal_temperature"):
            nb = ic.nbar if ic.motion_mode == "thermal_nbar" else mo.nbar_from_temperature(ic.temperature_k, self.nu_t)
            pn, tail = mo.thermal_pn(nb, N)
            notes.append(f"thermal nbar={nb:.6g}; discarded tail above n_max before normalization = {tail:.3g}")
        elif ic.motion_mode == "fock":
            if ic.fock_n >= N:
                raise ValueError("fock_n exceeds n_max")
            pn = np.zeros(N)
            pn[ic.fock_n] = 1.0
        else:
            v = np.asarray(ic.custom_pn, float)
            tail = float(v[N:].sum() / v.sum()) if v.size > N else 0.0
            v = np.pad(v, (0, max(0, N - v.size)))[:N]
            if v.sum() <= 0:
                raise ValueError("custom p(n) has no weight below n_max")
            if abs(v.sum() - 1) > 1e-9:
                notes.append(f"custom p(n) normalized from in-basis sum {v.sum():.6g}")
            pn = v / v.sum()
            if tail:
                notes.append(f"custom p(n) tail above n_max discarded: {tail:.3g}")
        return np.outer(spin, pn), tail, notes

    # ------------------------------------------------------------------
    def light_shifts(self, seg: Segment) -> tuple[float, float]:
        ls_u = ls_d = 0.0
        for b, scale in (("spin_pump", seg.pump), ("repump", seg.repump)):
            pc = getattr(self.cfg, b)
            if not pc.enabled or scale == 0:
                continue
            r = self.rates[b]
            u = 2 * np.pi * pc.light_shift_override_up_hz if pc.light_shift_override_up_hz is not None else r.light_shift[UP]
            d = 2 * np.pi * pc.light_shift_override_down_hz if pc.light_shift_override_down_hz is not None else r.light_shift[DOWN]
            ls_u += scale * u
            ls_d += scale * d
        return ls_u, ls_d

    def detuning(self, seg: Segment) -> float:
        """delta = 2 pi (nu_beat - nu_ud(t) - nu_t), nu_ud including all configured shifts."""
        ls_u, ls_d = self.light_shifts(seg)
        amp = seg.raman if self.cfg.raman.enabled else 0.0
        nu_ud = self.nu_ud_zeeman + amp * self.cfg.raman.differential_light_shift_hz + (ls_u - ls_d) / (2 * np.pi)
        return 2 * np.pi * (self.nu_beat - nu_ud - self.nu_t)

    def ops(self, seg: Segment) -> SegmentOps:
        cfg = self.cfg
        sp = seg.pump if cfg.spin_pump.enabled else 0.0
        sr = seg.repump if cfg.repump.enabled else 0.0
        ar = seg.raman if cfg.raman.enabled else 0.0
        G = sp * self.T["spin_pump"] + sr * self.T["repump"] + ar * self.T["raman_scatter"]
        gout = sp * self.out["spin_pump"] + sr * self.out["repump"] + ar * self.out["raman_scatter"]
        ovr = sp * self.ovf["spin_pump"] + sr * self.ovf["repump"] + ar * self.ovf["raman_scatter"]
        counts = np.vstack([sp * self.out["spin_pump"], sr * self.out["repump"], ar * self.out["raman_scatter"]])
        ls_u, ls_d = self.light_shifts(seg)
        vk = [(w, ar * V) for w, V in self.vk_unit] if ar > 0 else []
        return SegmentOps(
            seg, G, gout, ovr, counts, self.detuning(seg), vk, cfg.raman.extra_coherence_decay_rate_s,
            cfg.trap.heating_rate_quanta_per_s, ls_u, ls_d, ar,
        )

    # ------------------------------------------------------------------
    def raman_isolation(self) -> dict:
        """Nearest unmodeled inter-hyperfine Raman channel relative to the drive."""
        at = self.atom
        eg = at.ground_energy_hz(self.B)
        best = None
        for i, (f, m) in enumerate(GROUND_STATES):
            if f != 3:
                continue
            for j, (f2, m2) in enumerate(GROUND_STATES):
                if f2 != 2 or abs(m - m2) > 2 or (i, j) == (UP, DOWN):
                    continue
                nu = eg[i] - eg[j]
                for k in range(-2, 3):
                    det = self.nu_beat - (nu + k * self.nu_t)
                    if best is None or abs(det) < abs(best[0]):
                        best = (det, ground_label(i), ground_label(j), k)
        det, a, b, k = best
        scale = max(self.cfg.raman.carrier_rabi_hz, 1e-30)
        return {"nearest_detuning_hz": det, "pair": f"{a}<->{b}", "sideband": k, "ratio_to_carrier_rabi": abs(det) / scale}

    def static_validity(self) -> Validity:
        cfg = self.cfg
        v = Validity()
        z = zeeman_validity(self.atom, self.B, self.nu_t)
        self.zeeman = z
        v.add("magnetic field", z["status"], "; ".join(z["notes"]) or
              f"weak-field Zeeman OK (quadratic error on nu_ud {z['raman_quadratic_zeeman_error_hz']:.3g} Hz, "
              f"excited quadratic shift {z['excited_quadratic_over_gamma']:.2g} Gamma)")
        # ground-state secular approximation (no Zeeman coherences from pumping)
        zs = min(abs(self.atom.g_ground[f]) for f in (2, 3)) * MU_B * self.B / H * 2 * np.pi
        max_rate = 0.0
        for seg in self.schedule:
            o = self.ops(seg)
            max_rate = max(max_rate, float(o.gout.max()))
        ratio = zs / max_rate if max_rate > 0 else np.inf
        st = "ok" if ratio > 10 else ("warning" if ratio > 1 else "invalid")
        v.add("ground coherences", st, f"adjacent-mF Zeeman splitting / max pumping rate = {ratio:.3g} (neglect of pump-induced Zeeman coherences needs >>1)")
        rmax = max_rate / self.omega_t
        st = "ok" if rmax < 0.1 else ("warning" if rmax < 0.5 else "invalid")
        v.add("motional secular", st, f"max optical scattering rate / omega_t = {rmax:.3g} (secular recoil jumps need <<1)")
        iso = self.raman_isolation()
        if self.omega_c > 0:
            st = "ok" if iso["ratio_to_carrier_rabi"] > 20 else ("warning" if iso["ratio_to_carrier_rabi"] > 3 else "invalid")
            v.add("Raman isolation", st, f"only |3,3>-|2,2> Raman pair modeled; nearest unmodeled channel {iso['pair']} (sideband {iso['sideband']:+d}) "
                  f"is {iso['nearest_detuning_hz'] / 1e3:.4g} kHz from the drive = {iso['ratio_to_carrier_rabi']:.3g} x carrier Rabi (assumes comparable coupling)")
            if cfg.raman.scattering_rate_s == 0:
                v.add("Raman scattering", "warning", "IDEALIZED ASSUMPTION: residual Raman photon scattering rate = 0 (unknown)")
            v.add("Raman polarization", "ok", "calibrated mode: Raman polarization imperfections enter only via the calibrated extra decay, shift, and scattering inputs")
        for b in BEAMS:
            src = self.beams[b].fraction_source
            if "EFFECTIVE" in src:
                v.add(f"{b} polarization", "warning", src)
        if cfg.optical.excited_manifold != "all_D2":
            v.add("excited manifold", "warning", "IDEALIZED F'=3-only absorption (F'=1,2,4 excluded)")
        if cfg.optical.emission_pattern == "isotropic":
            v.add("recoil", "warning", "isotropic emission approximation (optional; dipole pattern is the reference)")
        ld = self.eta_R**2 * (2 * self._nbar0() + 1)
        v.add("Lamb-Dicke", "ok", f"eta_R = {self.eta_R:.4f}, eta_R^2(2 nbar0+1) = {ld:.3g}; exact displacement elements used (no LD expansion)")
        if self.initial_tail > cfg.numerics.boundary_tolerance:
            v.add("initial tail", "invalid", f"initial distribution tail above n_max = {self.initial_tail:.3g}")
        return v

    def _nbar0(self) -> float:
        return float(np.sum(self.P0.sum(axis=0) * np.arange(self.N)))

    def summary(self) -> dict:
        return {
            "N_motional": self.N,
            "x0_m": self.x0,
            "eta_single_photon_D2_along_axis": self.eta_d2,
            "eta_R": self.eta_R,
            "raman_dk_lab_per_m": self.dk.tolist(),
            "nu_ud_zeeman_hz": self.nu_ud_zeeman,
            "nu_ud_ref_hz": self.nu_ud_ref,
            "nu_beat_hz": self.nu_beat,
            "delta_beat_minus_nu_t_ref_hz": self.nu_beat - self.nu_ud_ref - self.nu_t,
            "cos_trap_axis_field": self.cos_beta,
            "pump_rates_per_s": {
                b: {
                    "gamma_out_up": float(self.rates[b].gamma_out[UP]),
                    "gamma_out_down": float(self.rates[b].gamma_out[DOWN]),
                    "gamma_out_max": float(self.rates[b].gamma_out.max()),
                    "p_exc_max": float(self.rates[b].p_exc.max()),
                    "light_shift_up_hz": float(self.rates[b].light_shift[UP] / (2 * np.pi)),
                    "light_shift_down_hz": float(self.rates[b].light_shift[DOWN] / (2 * np.pi)),
                    **self.rates[b].detail,
                }
                for b in BEAMS
            },
            "illustrative_parameters": illustrative_parameters(self.cfg),
        }


SPECTATOR_IDX = np.array(SPECTATORS)
