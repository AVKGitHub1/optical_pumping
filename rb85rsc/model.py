"""Assemble all physical operators for one configuration.

State space
-----------
Ground sublevels s = 0..11 (see ``atomic.GROUND_STATES``) times vibrational
levels n = 0..n_max (N = n_max + 1).  Population index = s*N + n.

* The Raman pair {up=|3,3>, down=|2,2>} (x motion) keeps a full density matrix,
  because the calibrated Raman coupling acts only there.
* The other 10 sublevels keep motional populations only. Coherence transfer
  through scattering is omitted, including degenerate motional transitions.
  Rate/secular conditions are checked, but do not justify this additional
  population-recoil approximation; it carries a separate diagnostic warning.

Frame
-----
Rotating frame of the Raman beat omega_b, then the interaction picture with
respect to H0 = omega_t (a^+a - |up><up|).  The residual Hamiltonian is

  H_I(t)/hbar = -delta |up><up| + sum_{n,m} (Omega_c/2) D_nm e^{i(n-m-1) omega_t t} |up,n><down,m| + h.c.

with D = exp(-i eta_R (a + a^+)) for the raising block (exact matrix elements) and
delta = 2 pi (nu_beat - nu_ud - nu_t).  The first red sideband n -> n-1 is static;
the carrier rotates at -omega_t and the blue sideband at -2 omega_t, so they are
retained as off-resonant terms rather than dropped.

With raman.tone_layout = both_tones_both_beams,
the four pathways (low tone from beam i, high tone from beam j) drive the same
resonance and add coherently:

  D -> sqrt(p1l p2h) e^{-i theta} D(-eta) + sqrt(p2l p1h) e^{i(chi + theta)} D(eta)
       + [sqrt(p1l p1h) + sqrt(p2l p2h) e^{i chi}] 1,

where the co-propagating pathways have dk = 0 (carrier only), theta is the atom
position in the static beat interference pattern and chi the beam-2 minus beam-1
beat phase.

Lattice (trap.potential = lattice)
----------------------------------
The levels are the bound states of one site of V0 sin^2(k_L z) (``motion.LatticeSite``),
n = 0..N-1 with exact energies E_n.  The frame uses H0 = sum_n E_n |n><n| - omega_10 |up><up|
with omega_10 = (E_1 - E_0)/hbar, so |up,n><down,m| rotates at E_n - E_m - omega_10 and
delta = 2 pi (nu_beat - nu_ud - nu_10).  D_nm = <n|exp(i dk z)|m> and the recoil kernels are
overlaps of the site states; probability promoted above V0 (the atom leaves its site)
goes to the overflow bin, which is then physical loss.
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
    K_B,
    AtomicData,
    ground_label,
    load_atomic_data,
    zeeman_validity,
)
from .config import ConfigError, SimConfig, illustrative_parameters
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
    vk: list  # [(frequency, V_k)] with V_k (N,N) complex; harmonic: frequency = k*omega_t
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
    def __init__(self, cfg: SimConfig, atom: AtomicData | None = None, raman_path_coefficients=None):
        cfg.validate()
        if cfg.raman.calibration == "measured_carrier" and raman_path_coefficients is None:
            raise ConfigError("measured_carrier configs must run through runner.run (spatial calibration and 3D ensemble), not the legacy 1D Model")
        self.cfg = cfg
        self.atom = atom or load_atomic_data()
        at = self.atom
        self.b_dir = pol.unit(cfg.magnetic.direction)
        self.B = cfg.magnetic.magnitude_gauss * 1e-4
        self.u = pol.unit(cfg.trap.axis)
        self.cos_beta = float(np.dot(self.u, self.b_dir))
        self.nu_t = (mo.lattice_frequency_hz(at.mass_kg, cfg.trap.wavelength_nm, cfg.trap.depth_uk)
                     if cfg.trap.potential == "lattice" and cfg.trap.wavelength_nm is not None else cfg.trap.frequency_hz)
        self.omega_t = 2 * np.pi * self.nu_t
        self.x0 = mo.x0_m(at.mass_kg, self.nu_t)
        tc = cfg.trap
        if tc.potential == "lattice":
            self.motion = mo.lattice_site(at.mass_kg, self.nu_t, tc.depth_uk * 1e-6 * K_B / H, tc.n_max + 1)
            if self.motion.N < 2 and not cfg.ensemble.enabled:
                raise ConfigError(f"lattice of depth {tc.depth_uk} uK holds {self.motion.n_bound} bound level(s) at nu_t = {self.nu_t:g} Hz; need >= 2")
        else:
            self.motion = mo.HarmonicBasis(tc.n_max + 1, self.nu_t)
        self.N = N = self.motion.N
        self.lattice_full = self.motion.kind == "lattice" and N == self.motion.n_bound  # overflow = physical loss
        self.eps = 2 * np.pi * (self.motion.energies_hz - self.motion.energies_hz[0])  # level energies, rad/s
        # n=1 -> 0 spacing nu_10: nu_t (harmonic) or the exact level spacing (lattice)
        self.nu_ref = self.nu_t if self.motion.kind == "harmonic" or self.N < 2 else float(self.motion.energies_hz[1] - self.motion.energies_hz[0])
        self.omega_ref = 2 * np.pi * self.nu_ref
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
            kern = {cls: self.motion.recoil_kernel(self.eta_d2, proj, self.cos_beta, pat(cls), nq) for cls in ("sigma", "pi")}
            self.T[b], self.out[b], self.ovf[b] = self._transfer(self.rates[b].W, kern)

        # ---- Raman ------------------------------------------------------------
        rc = cfg.raman
        self.k_raman = 2 * np.pi / (rc.wavelength_nm * 1e-9)
        klo, khi = pol.unit(rc.beam_low_direction), pol.unit(rc.beam_high_direction)
        self.dk = self.k_raman * (klo - khi)  # k_abs - k_emit for up -> down (low from beam 1, high from beam 2)
        self.eta_R_signed = float(np.dot(self.dk, self.u) * self.x0)
        self.eta_R = abs(self.eta_R_signed)
        # pathway amplitudes a[i, j]: low tone from beam i, high tone from beam j (same beat frequency -> coherent sum)
        if rc.tone_layout == "both_tones_both_beams":
            pl, ph = np.asarray(rc.tone_powers_low, float), np.asarray(rc.tone_powers_high, float)
            th, chi = np.deg2rad(rc.lattice_phase_deg), np.deg2rad(rc.beam_beat_phase_deg)
        else:
            pl, ph, th, chi = np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.0
        a = np.sqrt(np.outer(pl, ph))
        if raman_path_coefficients is not None:
            a = a * np.asarray(raman_path_coefficients, complex)
        self.raman_pathways = a
        # This operator describes up -> down: absorb LOW, emit HIGH. Its
        # co-propagating beam-2 phase is -chi, because chi is HIGH-minus-LOW.
        # H's |up><down| block must contain its adjoint (the reverse process).
        self.raman_downward_D = (
            a[0, 1] * np.exp(1j * th) * self.motion.displacement(self.eta_R_signed)
            + a[1, 0] * np.exp(-1j * (chi + th)) * self.motion.displacement(-self.eta_R_signed)
            + (a[0, 0] + a[1, 1] * np.exp(-1j * chi)) * np.eye(N)
        )
        self.D = self.raman_downward_D.conj().T
        self.omega_c = 2 * np.pi * rc.carrier_rabi_hz if rc.enabled else 0.0
        order = rc.max_sideband_order
        n = np.arange(N)
        dn = n[:, None] - n[None, :]
        # interaction-picture frequency of |up,n><down,m|: E_n - E_m - omega_ref (harmonic: (n - m - 1) omega_t)
        if self.motion.kind == "harmonic":
            self.F_raman = (dn - 1) * self.omega_t
        else:
            self.F_raman = self.eps[:, None] - self.eps[None, :] - self.omega_ref
        keep = np.abs(dn) <= order
        self.vk_unit = []
        for f in np.unique(self.F_raman[keep]):
            Vk = np.where(keep & (self.F_raman == f), 0.5 * self.omega_c * self.D, 0.0)
            if np.any(Vk != 0):
                self.vk_unit.append((float(f), Vk))
        eg = at.ground_energy_hz(self.B)
        self.nu_ud_zeeman = float(eg[UP] - eg[DOWN])  # hyperfine + Breit-Rabi Zeeman
        self.nu_ud_ref = self.nu_ud_zeeman + rc.differential_light_shift_hz
        if rc.frequency_mode == "sideband_offset":
            self.nu_beat = self.nu_ud_ref + self.nu_ref + rc.red_sideband_offset_hz
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
        k1, o1 = self.motion.recoil_kernel(eta_rp, klo_p, self.cos_beta, "isotropic", nq)
        k2, o2 = self.motion.recoil_kernel(eta_rp, khi_p, self.cos_beta, "isotropic", nq)
        w1 = (pl[0] + ph[0]) / max(pl.sum() + ph.sum(), 1e-300)  # beam-1 share of the scattered photons
        kr = (w1 * k1 + (1 - w1) * k2, w1 * o1 + (1 - w1) * o2)
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
        if self.motion.kind == "lattice" and ic.motion_mode == "thermal_temperature":
            all_pn = mo.boltzmann_pn(self.motion.all_bound_energies_hz, ic.temperature_k)
            tail = float(all_pn[N:].sum())
            pn = all_pn[:N] / all_pn[:N].sum()
            notes.append(f"lattice: Boltzmann at T = {ic.temperature_k:g} K conditional on initially bound atoms; "
                         f"discarded bound-state weight = {tail:.3g}; the initially unbound fraction is unspecified")
        elif ic.motion_mode in ("thermal_nbar", "thermal_temperature"):
            nb = ic.nbar if ic.motion_mode == "thermal_nbar" else mo.nbar_from_temperature(ic.temperature_k, self.nu_t)
            pn, tail = mo.thermal_pn(nb, N)
            notes.append(f"geometric occupation distribution, input nbar={nb:.6g}; discarded weight = {tail:.3g}")
            if self.motion.kind == "lattice":
                notes.append("thermal_nbar specifies a geometric level distribution, not a canonical lattice temperature")
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
        """delta = 2 pi (nu_beat - nu_ud(t) - nu_10), nu_ud including all configured shifts."""
        ls_u, ls_d = self.light_shifts(seg)
        amp = seg.raman if self.cfg.raman.enabled else 0.0
        nu_ud = self.nu_ud_zeeman + amp * self.cfg.raman.differential_light_shift_hz + (ls_u - ls_d) / (2 * np.pi)
        return 2 * np.pi * (self.nu_beat - nu_ud - self.nu_ref)

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
                # Use actual level differences through the configured Raman
                # sideband order, not a hard-coded +/-2 harmonic approximation.
                for n, m2n in np.ndindex(self.N, self.N):
                    k = n - m2n
                    if abs(k) > self.cfg.raman.max_sideband_order:
                        continue
                    spacing = (self.eps[n] - self.eps[m2n]) / (2 * np.pi)
                    det = self.nu_beat - (nu + spacing)
                    if best is None or abs(det) < abs(best[0]):
                        best = (det, ground_label(i), ground_label(j), k, n, m2n)
        det, a, b, k, n, m = best
        amp = max((s.raman for s in self.schedule), default=0.0) if self.cfg.raman.enabled else 0.0
        scale = max(amp * self.cfg.raman.carrier_rabi_hz * float(np.abs(self.raman_pathways).sum()), 1e-30)
        return {"nearest_detuning_hz": det, "pair": f"{a}<->{b}", "sideband": k,
                "motional_levels": [n, m], "ratio_to_carrier_rabi": abs(det) / scale}

    def static_validity(self) -> Validity:
        cfg = self.cfg
        v = Validity()
        z = zeeman_validity(self.atom, self.B, self.nu_t)
        self.zeeman = z
        v.add("magnetic field", z["status"], "; ".join(z["notes"]) or
              f"Breit-Rabi ground energies (correction to linear nu_ud {z['raman_quadratic_zeeman_error_hz']:.3g} Hz included; "
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
        min_gap = float(np.min(np.diff(self.eps)))
        rmax = max_rate / min_gap
        st = "ok" if rmax < 0.1 else ("warning" if rmax < 0.5 else "invalid")
        v.add("motional secular", st, f"max optical scattering rate / minimum motional angular spacing = {rmax:.3g} (needs <<1)")
        if max_rate > 0:
            v.add("recoil coherence", "warning", "population recoil model resolves each |s,n> jump separately: "
                  "coherence transfer between degenerate motional transitions and interference of elastic spin amplitudes are omitted; "
                  "the secular rate criterion alone does not justify this extra dephasing approximation")
        iso = self.raman_isolation()
        if self.omega_c > 0 and any(s.raman > 0 for s in self.schedule):
            st = "ok" if iso["ratio_to_carrier_rabi"] > 20 else ("warning" if iso["ratio_to_carrier_rabi"] > 3 else "invalid")
            v.add("Raman isolation", st, f"only |3,3>-|2,2> Raman pair modeled; nearest unmodeled channel {iso['pair']} (sideband {iso['sideband']:+d}) "
                  f"is {iso['nearest_detuning_hz'] / 1e3:.4g} kHz from the drive = {iso['ratio_to_carrier_rabi']:.3g} x carrier Rabi "
                  "(conservative bound through max_sideband_order; unmodeled polarization strengths and light shifts unknown)")
            if cfg.raman.scattering_rate_s == 0:
                v.add("Raman scattering", "warning", "IDEALIZED ASSUMPTION: residual Raman photon scattering rate = 0 (unknown)")
            v.add("Raman polarization", "ok", "calibrated mode: Raman polarization imperfections enter only via the calibrated extra decay, shift, and scattering inputs")
            if cfg.raman.tone_layout == "both_tones_both_beams":
                a = self.raman_pathways
                d1 = abs(self.motion.displacement(self.eta_R)[1, 0])
                cross = a[0, 1] + a[1, 0]
                rsb = abs(self.D[1, 0]) / (cross * d1) if cross * d1 > 0 else 0.0
                st = "ok" if rsb > 0.5 else "warning"  # physics, not a model-validity failure
                v.add("Raman tone layout", st,
                      f"both tones in both beams: red sideband |D_10| = {abs(self.D[1, 0]):.3g} ({rsb:.2f} x its maximum over lattice phase), "
                      f"carrier |D_00| = {abs(self.D[0, 0]):.3g} (co-propagating pathways add {a[0, 0] + a[1, 1]:.3g}, no motional coupling); "
                      f"result depends on raman.lattice_phase_deg = {cfg.raman.lattice_phase_deg:g} deg (atom position in the beat pattern), scan it if not stabilized")
        for b in BEAMS:
            src = self.beams[b].fraction_source
            if "EFFECTIVE" in src:
                v.add(f"{b} polarization", "warning", src)
            pc = getattr(cfg, b)
            if pc.polarization.mode == "geometry" and not np.allclose(self.beams[b].direction, pol.unit(pc.direction), atol=1e-8):
                v.add(f"{b} direction", "warning",
                      f"geometry angles set lab direction {np.round(self.beams[b].direction, 6).tolist()}; "
                      f"stored direction {pc.direction} is inactive; recoil uses the geometry direction")
        if cfg.optical.excited_manifold != "all_D2":
            v.add("excited manifold", "warning", "IDEALIZED F'=3-only absorption (F'=1,2,4 excluded)")
        if cfg.optical.emission_pattern == "isotropic":
            v.add("recoil", "warning", "isotropic emission approximation (optional; dipole pattern is the reference)")
        ld = self.eta_R**2 * (2 * self._nbar0() + 1)
        v.add("Lamb-Dicke", "ok", f"eta_R = {self.eta_R:.4f}, eta_R^2(2 nbar0+1) = {ld:.3g}; exact displacement elements used (no LD expansion)")
        if self.motion.kind == "lattice":
            s = self.motion
            gaps = ", ".join(f"{g / 1e3:.1f}" for g in np.diff(s.energies_hz))
            cap = "" if self.lattice_full else f"; basis capped at n_max = {self.N - 1} below the {s.n_bound} bound levels (overflow partly numerical)"
            v.add("lattice", "ok" if self.lattice_full else "warning",
                  f"1D lattice V0 = {cfg.trap.depth_uk:g} uK = {s.depth_hz / 1e3:.1f} kHz, E_r = {s.recoil_hz / 1e3:.2f} kHz "
                  f"(lattice wavelength {s.wavelength_m * 1e9:.0f} nm implied by depth and nu_t), {s.n_bound} bound levels, spacings {gaps} kHz "
                  f"(harmonic {self.nu_t / 1e3:.1f}); isolated site: tunnelling (lowest band ~{s.tunneling_hz():.2g} Hz, larger near the top), "
                  f"coherent Raman coupling into unbound states and anharmonic heating elements neglected{cap}")
        if self.initial_tail > cfg.numerics.boundary_tolerance:
            if self.lattice_full:
                v.add("initial tail", "warning", f"{self.initial_tail:.3g} of the specified level distribution discarded; "
                      "this is not a prediction of the initially unbound fraction")
            else:
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
            "raman_tone_layout": self.cfg.raman.tone_layout,
            "raman_pathway_amplitudes_low_i_high_j": self.raman_pathways.tolist(),
            "raman_D10_red_sideband": float(abs(self.D[1, 0])),
            "raman_D00_carrier": float(abs(self.D[0, 0])),
            "nu_ud_zeeman_hz": self.nu_ud_zeeman,
            "ground_energy_model": "Breit-Rabi (weak-field dipoles)",
            "raman_isolation": self.raman_isolation(),
            "nu_ud_ref_hz": self.nu_ud_ref,
            "nu_beat_hz": self.nu_beat,
            "delta_beat_minus_nu_t_ref_hz": self.nu_beat - self.nu_ud_ref - self.nu_ref,
            "trap_potential": self.motion.kind,
            "nu_10_hz": self.nu_ref,
            "level_energies_hz": (self.eps / (2 * np.pi)).tolist(),
            "lattice_bound_levels": self.motion.n_bound,
            "lattice_recoil_hz": getattr(self.motion, "recoil_hz", None),
            "lattice_wavelength_implied_m": getattr(self.motion, "wavelength_m", None),
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
