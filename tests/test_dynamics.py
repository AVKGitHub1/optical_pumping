"""Checks 2-9: dynamics, Raman sign conventions, recoil, conservation, convergence, rate backend."""
import numpy as np
import pytest

from rb85rsc.atomic import DOWN, UP
from rb85rsc.dynamics import CoherentSolver, RateInvalid, _Progress, solve
from rb85rsc.model import Model
from rb85rsc import motion as mo
from rb85rsc.motion import mean_c2

from helpers import cfg, run

ONLY_UP = [0.0] * 11 + [1.0]
ONLY_DOWN = [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------- check 2 (dynamics)
def test_idealized_pumping_reaches_dark_state():
    c = cfg(timing__protocol="optical_pumping_only", timing__total_duration_ms=3, optical__excited_manifold="F3_only_IDEALIZED",
            numerics__solver="rate")
    _, r, a = run(c)
    assert a["final"]["P_up"] > 0.999
    assert a["final"]["target_state_scattering_rate_per_s"] == 0.0


def test_F4_and_impurity_add_target_scattering():
    base = cfg(timing__protocol="optical_pumping_only", timing__total_duration_ms=3, numerics__solver="rate")
    _, _, a0 = run(base)
    c = base.copy()
    c.spin_pump.polarization.pi_fraction = 0.01
    _, _, a1 = run(c)
    assert a0["final"]["target_state_scattering_rate_per_s"] > 0
    assert a1["final"]["target_state_scattering_rate_per_s"] > 3 * a0["final"]["target_state_scattering_rate_per_s"]
    assert a1["final"]["P_up"] < a0["final"]["P_up"]


# ---------------------------------------------------------------- check 3
def test_all_off_is_static():
    _, r, a = run(cfg(timing__protocol="all_off", timing__total_duration_ms=2))
    np.testing.assert_allclose(r.P[-1], r.P[0], atol=1e-14)


def test_raman_only_is_coherent_and_reversible():
    c = cfg(timing__protocol="raman_only", initial__spin_preset="custom", initial__custom_spin_populations=ONLY_UP,
            initial__motion_mode="fock", initial__fock_n=1, raman__carrier_rabi_hz=1e3, trap__n_max=10)
    m = Model(c)
    om = 2 * np.pi * 1e3 * abs(m.D[1, 0])
    c.timing.total_duration_ms = 1e3 * (2 * np.pi / om)  # 2 pi pulse
    _, r, a = run(c)
    assert r.P[-1, UP, 1] > 0.995  # returns: no fabricated relaxation
    mid = r.P[len(r.t) // 2]
    assert mid[DOWN, 0] > 0.99


# ---------------------------------------------------------------- both tones in both Raman beams
def test_both_tones_with_one_tone_each_matches_default_layout():
    ref = Model(cfg(trap__n_max=10))
    m = Model(cfg(trap__n_max=10, raman__tone_layout="both_tones_both_beams", raman__tone_powers_low=[1.0, 0.0],
                  raman__tone_powers_high=[0.0, 1.0], raman__lattice_phase_deg=0.0))
    assert np.allclose(m.D, ref.D, atol=1e-14)


def test_both_tones_coherent_pathway_sum():
    d1 = mo.displacement_matrix(Model(cfg(trap__n_max=10)).eta_R, 11)
    kw = dict(trap__n_max=10, raman__tone_layout="both_tones_both_beams")
    m = Model(cfg(raman__lattice_phase_deg=90.0, **kw))
    assert abs(m.D[1, 0]) == pytest.approx(2 * abs(d1[1, 0]), rel=1e-12)  # crossed pathways add on the sideband
    assert abs(m.D[0, 0] - 2.0) < 1e-12  # crossed carriers cancel; co-propagating ones remain
    m0 = Model(cfg(raman__lattice_phase_deg=0.0, **kw))
    assert abs(m0.D[1, 0]) < 1e-12  # node of the beat pattern: no sideband
    assert m0.D[0, 0].real == pytest.approx(2 + 2 * d1[0, 0].real, rel=1e-12)


def test_both_tones_red_sideband_pi_pulse():
    c = cfg(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=1,
            raman__carrier_rabi_hz=1e3, raman__tone_layout="both_tones_both_beams", trap__n_max=10, timing__n_samples=3)
    m = Model(c)
    c.timing.total_duration_ms = 1e3 * (np.pi / (2 * np.pi * 1e3 * abs(m.D[1, 0])))
    _, r, _ = run(c)
    assert r.P[-1, DOWN, 0] > 0.98


# ---------------------------------------------------------------- check 4
@pytest.mark.parametrize("offset,n_final,label", [(0.0, 1, "red"), (-70e3, 2, "carrier"), (-140e3, 3, "blue")])
def test_sideband_frequency_convention(offset, n_final, label):
    c = cfg(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=2,
            raman__carrier_rabi_hz=1e3, raman__red_sideband_offset_hz=offset, trap__n_max=10, timing__n_samples=3)
    m = Model(c)
    # the red sideband sits at delta_beat = nu_beat - nu_ud = +nu_t
    assert m.nu_beat - m.nu_ud_ref == pytest.approx(70e3 + offset)
    c.timing.total_duration_ms = 1e3 * (np.pi / (2 * np.pi * 1e3 * abs(m.D[2, n_final])))
    _, r, _ = run(c)
    assert r.P[-1, DOWN, n_final] > 0.99, label


def test_up_n0_has_no_red_partner_but_offresonant_excitation_exists():
    c = cfg(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=0,
            raman__carrier_rabi_hz=5e3, trap__n_max=10, timing__total_duration_ms=1, timing__n_samples=101)
    _, r, _ = run(c)
    pd = r.P[:, DOWN].sum(axis=1)
    assert pd.max() < 0.02
    assert pd.max() > 1e-5  # off-resonant carrier excitation retained in the full model


# ---------------------------------------------------------------- check 5
def test_red_pi_pulse_removes_one_quantum_then_reset_adds_predicted_recoil():
    c = cfg(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=3,
            raman__carrier_rabi_hz=1e3, trap__n_max=25, timing__n_samples=3)
    m = Model(c)
    c.timing.total_duration_ms = 1e3 * (np.pi / (2 * np.pi * 1e3 * abs(m.D[3, 2])))
    _, r, _ = run(c)
    assert r.P[-1, DOWN, 2] > 0.99
    n = np.arange(m.N)
    assert (r.P[-1].sum(0) * n).sum() == pytest.approx(2.0, abs=0.02)
    # optical reset from |down, n=2>: nbar increase = photons * eta^2 (proj^2 + <c^2>) for isotropic emission
    for pattern in ("isotropic", "dipole"):
        c2 = cfg(timing__protocol="custom", timing__custom_segments=[{"duration_ms": 0.3, "pump": 1.0, "repump": 1.0}],
                 initial__spin_preset="custom", initial__custom_spin_populations=ONLY_DOWN, initial__motion_mode="fock",
                 initial__fock_n=2, trap__n_max=30, optical__emission_pattern=pattern, numerics__solver="rate")
        m2, r2, a2 = run(c2)
        dn = a2["series"]["nbar"][-1] - a2["series"]["nbar"][0]
        assert dn > 0  # reset is not recoil-free
        if pattern == "isotropic":
            proj = np.dot(m2.beams["repump"].direction, m2.u)
            pred = r2.counts[-1, :2].sum() * m2.eta_d2**2 * (proj**2 + 1 / 3)
            assert dn == pytest.approx(pred, rel=1e-6)


# ---------------------------------------------------------------- check 6, 9
def test_net_cooling_conservation_positivity():
    _, r, a = run(cfg(timing__total_duration_ms=3))
    f = a["final"]
    assert f["nbar"] < 0.5 * f["nbar_initial"]
    assert f["max_conservation_error"] < 1e-9
    assert f["min_eigenvalue"] > -1e-10
    # photon-count consistency: counts non-decreasing
    assert np.all(np.diff(r.counts, axis=0) >= -1e-12)


def test_blue_sideband_heats():
    _, _, a = run(cfg(timing__total_duration_ms=3, raman__red_sideband_offset_hz=-140e3))
    assert a["final"]["nbar"] > a["final"]["nbar_initial"]
    assert a["final"]["fractional_energy_reduction"] < 0


# ---------------------------------------------------------------- check 7
def test_convergence_nmax_quadrature_tolerance():
    base = cfg(timing__total_duration_ms=1)
    ref = run(base)[2]["final"]
    for k, v in (("trap.n_max", 55), ("numerics.recoil_quadrature_points", 48), ("numerics.rtol", 1e-9)):
        c = base.copy()
        from rb85rsc.config import set_param

        set_param(c, k, v)
        f = run(c)[2]["final"]
        for key in ("P_up", "P_target"):
            assert f[key] == pytest.approx(ref[key], abs=2e-5), (k, key)
        # n_max=40 discards a 7.5e-6 thermal tail (n>40) carrying ~3e-4 quanta; allow for it
        assert f["nbar"] == pytest.approx(ref["nbar"], abs=1e-3 if k == "trap.n_max" else 2e-5), k


def test_overflow_is_flagged_not_reflected():
    c = cfg(trap__n_max=8, timing__protocol="optical_pumping_only", timing__total_duration_ms=2)
    m, r, a = run(c)
    assert a["validity"].status == "invalid"
    assert r.overflow[-1] > 0
    tr = r.P[-1].sum() + r.overflow[-1]
    assert tr == pytest.approx(1.0, abs=1e-9)


def test_max_sideband_order_converged():
    base = cfg(timing__total_duration_ms=1)
    ref = run(base)[2]["final"]
    c = base.copy()
    c.raman.max_sideband_order = 6
    f = run(c)[2]["final"]
    assert f["P_target"] == pytest.approx(ref["P_target"], abs=1e-4)


# ---------------------------------------------------------------- check 8
def test_rate_backend_matches_coherent_in_valid_regime():
    c = cfg(raman__carrier_rabi_hz=1e3, initial__nbar=1.0, trap__n_max=20, timing__total_duration_ms=5)
    fc = run(c)[2]["final"]
    c.numerics.solver = "rate"
    fr = run(c)[2]["final"]
    for k in ("P_up", "P_n0", "P_target", "nbar"):
        assert fr[k] == pytest.approx(fc[k], abs=3e-3), k


def test_rate_backend_rejected_when_invalid():
    for proto in ("raman_only", "continuous"):
        c = cfg(timing__protocol=proto, timing__total_duration_ms=1, numerics__solver="rate")
        with pytest.raises(RateInvalid):
            solve(Model(c))
    c.numerics.solver = "auto"
    r = solve(Model(c))
    assert r.solver == "coherent" and "switched" in r.notes[0]


def test_optical_scattering_damps_raman_coherence_consistently():
    c = cfg()
    m = Model(c)
    cs = CoherentSolver(m)
    seg = m.schedule[0]
    o = m.ops(seg)
    o.vk = []  # isolate the dissipator
    rhs = cs.make_rhs(o, _Progress(None, 1))
    y = cs.pack(m.P0 * 0)
    n, k = 2, 1
    y[cs.N2 + n * m.N + k] = 1e-3  # rho_{up n, down k}
    dy = rhs(0.0, y)
    got = dy[cs.N2 + n * m.N + k] / 1e-3
    expect = -0.5 * (o.gout[UP * m.N + n] + o.gout[DOWN * m.N + k]) - o.gamma_extra + 1j * o.delta
    assert got == pytest.approx(expect, rel=1e-12)


def test_strong_dephasing_suppression():
    """Resonant |up,1>-|down,0> with large extra dephasing: P_down = (1 - exp(-2 W t))/2, W = Omega^2/(2 gamma)."""
    g = 5e4
    c = cfg(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=1,
            raman__carrier_rabi_hz=1e3, raman__extra_coherence_decay_rate_s=g, trap__n_max=8, timing__total_duration_ms=5)
    m, r, _ = run(c)
    om = 2 * np.pi * 1e3 * abs(m.D[1, 0])
    W = om**2 / (2 * g)
    pred = 0.5 * (1 - np.exp(-2 * W * r.t[-1]))
    assert r.P[-1, DOWN, 0] == pytest.approx(pred, rel=0.03)
    # and the transfer is strongly suppressed relative to the undamped pi-pulse limit
    assert r.P[-1, DOWN, 0] < 0.5


def test_kramers_heisenberg_and_isotropic_are_small_perturbations():
    base = cfg(timing__total_duration_ms=2)
    ref = run(base)[2]["final"]
    for k, v in (("optical.path_model", "kramers_heisenberg"), ("optical.emission_pattern", "isotropic")):
        c = base.copy()
        from rb85rsc.config import set_param

        set_param(c, k, v)
        f = run(c)[2]["final"]
        assert abs(f["nbar"] - ref["nbar"]) < 0.05, k
