"""Checks 1, 2 (rates part) and 11: atomic data, selection rules, polarization bookkeeping."""
import numpy as np
import pytest

from rb85rsc import polarization as pol
from rb85rsc.atomic import (
    C_LIGHT, DOWN, EPS0, EXCITED_STATES, GROUND_STATES, HBAR, UP, load_atomic_data, steck_comparison,
)
from rb85rsc.config import SimConfig
from rb85rsc.motion import lamb_dicke
from rb85rsc.optical import beam_rates, beam_spec

AD = load_atomic_data()
Z = [0, 0, 1]
B = 0.5e-4


def rates(name, c, manifold="all_D2", path="independent"):
    return beam_rates(AD, beam_spec(name, getattr(c, name), AD, Z), B, 0.0, manifold, path)


# ---------------------------------------------------------------- check 1
def test_branching_normalized_and_nonnegative():
    br = AD.branching
    assert br.min() >= 0
    np.testing.assert_allclose(br.sum(axis=1), 1.0, atol=1e-12)


def test_forbidden_dipoles_vanish_and_sum_rule():
    for ie, (fe, me) in enumerate(EXCITED_STATES):
        for ig, (fg, mg) in enumerate(GROUND_STATES):
            if abs(me - mg) > 1 or abs(fe - fg) > 1:
                assert AD.dipole_au[ie, ig] == 0
    s = np.sum(AD.dipole_au**2, axis=1)
    np.testing.assert_allclose(s, s[0], rtol=1e-12)
    # branching equals |d|^2 / sum (consistency of ARC amplitudes and ARC branching)
    np.testing.assert_allclose(AD.branching, AD.dipole_au**2 / s[:, None], atol=1e-12)


def test_F3prime_m3_decay_destinations():
    ie = EXCITED_STATES.index((3, 3))
    dest = {GROUND_STATES[i]: AD.branching[ie, i] for i in range(12) if AD.branching[ie, i] > 1e-14}
    assert set(dest) == {(3, 3), (3, 2), (2, 2)}
    assert dest[(3, 3)] == pytest.approx(5 / 12, rel=1e-9)
    assert dest[(3, 2)] == pytest.approx(5 / 36, rel=1e-9)
    assert dest[(2, 2)] == pytest.approx(4 / 9, rel=1e-9)


def test_F4prime_m4_closed_cycle():
    ie = EXCITED_STATES.index((4, 4))
    assert AD.branching[ie, UP] == pytest.approx(1.0, abs=1e-12)


def test_steck_cross_check():
    for k, v in steck_comparison(AD).items():
        assert abs(v["rel_dev"]) < 1e-3, k


def test_single_photon_lamb_dicke():
    assert lamb_dicke(780.241e-9, AD.mass_kg, 70e3) == pytest.approx(0.235, abs=0.001)


# ---------------------------------------------------------------- check 2 (rate level)
def test_pure_sigma_plus_F3_only_up_is_dark_and_F4_opens_it():
    c = SimConfig()
    assert rates("spin_pump", c, "F3_only_IDEALIZED").gamma_out[UP] == 0.0
    r = rates("spin_pump", c)
    assert r.gamma_out[UP] > 0
    # all of it returns to |3,3> (closed F'=4 cycle): spin unchanged, recoil only
    assert r.W[UP].sum(axis=1)[UP] == pytest.approx(r.gamma_out[UP], rel=1e-9)


def test_repump_sigma_plus_excites_down_to_F3prime_m3():
    c = SimConfig()
    c.repump.detuning_mhz = 0.0
    c.magnetic.magnitude_gauss = 0.0
    r = beam_rates(AD, beam_spec("repump", c.repump, AD, Z), 0.0, 0.0, "F3_only_IDEALIZED")
    dest = r.W[DOWN].sum(axis=1)
    got = {GROUND_STATES[i] for i in np.nonzero(dest > 0)[0]}
    assert got == {(3, 3), (3, 2), (2, 2)}
    # isolated weak-excitation formula R = Gamma Omega^2 / (Gamma^2 + 4 Delta^2), Delta = 0
    e0 = np.sqrt(2 * c.repump.peak_intensity_w_m2 / (C_LIGHT * EPS0))
    om = e0 * AD.dipole_si[EXCITED_STATES.index((3, 3)), DOWN] / HBAR
    assert r.gamma_out[DOWN] == pytest.approx(AD.gamma * om**2 / AD.gamma**2, rel=1e-9)


def test_rate_lorentzian_and_light_shift_sign():
    c = SimConfig()
    c.magnetic.magnitude_gauss = 0.0
    g = AD.gamma
    for det_mhz in (3.0, -3.0):
        c.repump.detuning_mhz = det_mhz
        r = beam_rates(AD, beam_spec("repump", c.repump, AD, Z), 0.0, 0.0, "F3_only_IDEALIZED")
        e0 = np.sqrt(2 * c.repump.peak_intensity_w_m2 / (C_LIGHT * EPS0))
        om = e0 * AD.dipole_si[EXCITED_STATES.index((3, 3)), DOWN] / HBAR
        d = 2 * np.pi * det_mhz * 1e6
        assert r.gamma_out[DOWN] == pytest.approx(g * om**2 / (g**2 + 4 * d**2), rel=1e-9)
        assert np.sign(r.light_shift[DOWN]) == np.sign(det_mhz)  # blue detuning pushes ground up


def test_kramers_heisenberg_preserves_total_rates():
    c = SimConfig()
    c.spin_pump.polarization.pi_fraction = 0.05
    a = rates("spin_pump", c, path="independent")
    b = rates("spin_pump", c, path="kramers_heisenberg")
    np.testing.assert_allclose(a.gamma_out, b.gamma_out, rtol=1e-10)


# ---------------------------------------------------------------- check 11
def test_fraction_bookkeeping_and_power_conservation():
    c = SimConfig()
    base = rates("spin_pump", c)
    for tot in (0.0, 1e-3, 1e-2, 5e-2):
        for share in (0.0, 0.3, 1.0):
            c.spin_pump.polarization.set_impurity(tot, share)
            bs = beam_spec("spin_pump", c.spin_pump, AD, Z)
            assert sum(bs.fractions.values()) == pytest.approx(1.0, abs=1e-14)
            assert bs.intensity_w_m2 == c.spin_pump.peak_intensity_w_m2  # total intensity fixed
    c.spin_pump.polarization.set_impurity(0.0, 0.7)
    np.testing.assert_array_equal(rates("spin_pump", c).W, base.W)  # zero-impurity limit


def test_invalid_fractions_rejected():
    from rb85rsc.config import ConfigError

    c = SimConfig()
    c.spin_pump.polarization.pi_fraction = 0.7
    c.spin_pump.polarization.sigma_minus_fraction = 0.4
    with pytest.raises(ConfigError):
        c.validate()
    c.spin_pump.polarization.sigma_minus_fraction = -0.1
    with pytest.raises(ConfigError):
        c.validate()


def test_pi_and_sigma_minus_open_distinct_transitions_from_up():
    """F'=3-only: pi drives |3,3> -> |3',3>; sigma- drives |3,3> -> |3',2>, with different strengths."""
    c = SimConfig()
    c.magnetic.magnitude_gauss = 0.0
    out = {}
    for name, (pi_, sm) in {"pi": (0.01, 0.0), "sigma-": (0.0, 0.01)}.items():
        c.spin_pump.polarization.pi_fraction, c.spin_pump.polarization.sigma_minus_fraction = pi_, sm
        r = beam_rates(AD, beam_spec("spin_pump", c.spin_pump, AD, Z), 0.0, 0.0, "F3_only_IDEALIZED")
        e0 = np.sqrt(2 * c.spin_pump.peak_intensity_w_m2 / (C_LIGHT * EPS0))
        me = 3 if name == "pi" else 2
        om = e0 * np.sqrt(0.01) * AD.dipole_si[EXCITED_STATES.index((3, me)), UP] / HBAR
        assert r.gamma_out[UP] == pytest.approx(om**2 / AD.gamma, rel=1e-9)
        out[name] = r.W[UP].sum(axis=1)
    # sigma- path via |3',2> can reach |2,1>, the pi path via |3',3> cannot
    i21 = GROUND_STATES.index((2, 1))
    assert out["pi"][i21] == 0 and out["sigma-"][i21] > 0


def test_repump_impurity_does_not_touch_spin_pump():
    c = SimConfig()
    a = rates("spin_pump", c)
    c.repump.polarization.set_impurity(0.05, 0.5)
    b = rates("spin_pump", c)
    np.testing.assert_array_equal(a.W, b.W)
    assert c.spin_pump.polarization.total_impurity == 0


def test_geometry_limits():
    k = [0, 0, 1]
    f = pol.spherical_fractions(pol.jones_field_vector(k, 45, 0, Z), Z)
    assert f[1] == pytest.approx(1) and f[0] == pytest.approx(0, abs=1e-15)
    f = pol.spherical_fractions(pol.jones_field_vector(k, 0, 30, Z), Z)  # aligned linear
    assert f[0] == pytest.approx(0, abs=1e-15) and f[1] == pytest.approx(0.5) and f[-1] == pytest.approx(0.5)
    for chi in (45, 20, 0):  # ellipticity along B never creates pi
        assert pol.spherical_fractions(pol.jones_field_vector(k, chi, 10, Z), Z)[0] == pytest.approx(0, abs=1e-15)
    kt = pol.beam_direction_from_angles(10, 0, Z)
    ft = pol.spherical_fractions(pol.jones_field_vector(kt, 45, 0, Z), Z)
    assert ft[0] > 1e-3  # tilt creates pi
    assert ft[0] == pytest.approx(np.sin(np.radians(10)) ** 2 / 2, rel=1e-9)
    assert pol.realizable_along(ft, kt, Z)
    assert not pol.realizable_along({1: 0.99, 0: 0.01, -1: 0.0}, k, Z)  # pi along B: effective model only
    # geometry mode feeds derived fractions to the solver
    c = SimConfig()
    c.spin_pump.polarization.mode = "geometry"
    c.spin_pump.polarization.beam_angle_deg = 10
    bs = beam_spec("spin_pump", c.spin_pump, AD, Z)
    assert bs.fractions[0] == pytest.approx(ft[0], rel=1e-9)
