"""trap.potential = lattice: bound levels of one lattice site, anharmonic sidebands, loss above the depth."""
import numpy as np
import pytest
from scipy.special import mathieu_a, mathieu_b

from rb85rsc import motion as mo
from rb85rsc.atomic import DOWN, UP, H, K_B, load_atomic_data
from rb85rsc.config import ConfigError, SimConfig
from rb85rsc.model import Model

from helpers import cfg, run

MASS = load_atomic_data().mass_kg
NU, DEPTH_UK = 70e3, 15.0
DEPTH_HZ = DEPTH_UK * 1e-6 * K_B / H


def lattice(**kw):
    return cfg(trap__potential="lattice", trap__depth_uk=DEPTH_UK, trap__frequency_hz=NU, **kw)


def test_levels_match_mathieu_band_centres():
    s = mo.LatticeSite(MASS, NU, DEPTH_HZ, 50)
    er = s.recoil_hz
    assert er == pytest.approx(NU**2 / (4 * DEPTH_HZ))
    assert s.n_bound == 6
    q = DEPTH_HZ / (4 * er)  # V0 sin^2(k z) -> Mathieu y'' + (a - 2q cos 2x) y = 0, E = V0/2 + E_r a
    for n in range(4):  # deep bands: narrow, so the isolated-site level sits at the band centre
        centre = DEPTH_HZ / 2 + er * 0.5 * (mathieu_a(n, q) + mathieu_b(n + 1, q))
        assert s.energies_hz[n] == pytest.approx(centre, abs=100.0), n
    # anharmonic ladder: E_n - E_{n-1} ~ h nu - n E_r for low n
    assert s.energies_hz[1] - s.energies_hz[0] == pytest.approx(NU - er, abs=500.0)


def test_deep_lattice_reduces_to_harmonic():
    s = mo.LatticeSite(MASS, NU, 400 * NU, 8)
    gaps = np.diff(s.energies_hz[:5])
    assert np.allclose(gaps, NU - s.recoil_hz * np.arange(1, 5), atol=50.0)
    assert np.abs(s.displacement(0.47)[:6, :6] - mo.displacement_matrix(0.47, 6)).max() < 5e-3


def test_recoil_kernel_conserves_probability_and_loss_grows_with_n():
    s = mo.LatticeSite(MASS, NU, DEPTH_HZ, 50)
    K, ov = s.recoil_kernel(0.2348, 0.0, 0.0, "sigma", 24)
    assert np.allclose(K.sum(axis=0) + ov, 1.0, atol=1e-12)
    assert ov[0] < 1e-6 and ov[-1] > 0.05
    assert np.all(np.diff(ov) >= -1e-12)


def test_lattice_requires_depth_and_two_bound_levels():
    with pytest.raises(ConfigError):
        cfg(trap__potential="lattice")
    with pytest.raises(ConfigError):
        Model(cfg(trap__potential="lattice", trap__depth_uk=1.0, trap__frequency_hz=NU))  # < 2 bound levels


def test_harmonic_default_unchanged():
    m = Model(cfg(trap__n_max=10))
    assert m.motion.kind == "harmonic" and m.nu_ref == m.nu_t
    assert np.array_equal(m.F_raman, (np.arange(11)[:, None] - np.arange(11)[None, :] - 1) * m.omega_t)


def test_zero_offset_is_resonant_with_the_exact_1_to_0_spacing():
    c = lattice(timing__protocol="raman_only", initial__spin_preset="stretched", initial__motion_mode="fock", initial__fock_n=1,
                raman__carrier_rabi_hz=1e3, timing__n_samples=3)
    m = Model(c)
    assert m.N == 6 and m.lattice_full
    assert m.nu_ref == pytest.approx(65.8e3, abs=100.0)
    c.timing.total_duration_ms = 1e3 * (np.pi / (2 * np.pi * 1e3 * abs(m.D[1, 0])))
    _, r, _ = run(c)
    assert r.P[-1, DOWN, 0] > 0.98


def test_loss_above_depth_is_conserved_and_reported():
    _, r, a = run(lattice(timing__protocol="optical_pumping_only", timing__total_duration_ms=1, initial__nbar=1.0,
                          initial__motion_mode="thermal_nbar"))
    f = a["final"]
    assert f["max_conservation_error"] < 1e-6
    assert f["lattice_loss"] is not None and f["lattice_loss"] > 0
    names = [it[0] for it in a["validity"].items]
    assert "lattice loss" in names and "numerical overflow" not in names and "lattice" in names
    # reported populations are per trapped atom; *_absolute are fractions of all atoms
    pn = r.P[-1].sum(axis=0)
    assert f["P_trapped"] == pytest.approx(1 - f["lattice_loss"], abs=1e-9)
    assert f["P_trapped"] == pytest.approx(pn.sum(), rel=1e-12)
    assert f["P_n0"] == pytest.approx(pn[0] / pn.sum(), rel=1e-12)
    assert f["P_target"] == pytest.approx(r.P[-1, UP, 0] / pn.sum(), rel=1e-12)
    assert f["P_target_absolute"] == pytest.approx(r.P[-1, UP, 0], rel=1e-12)
    assert sum(f["spin_populations_final"].values()) == pytest.approx(1.0, abs=1e-12)
    assert a["P_final"].sum() == pytest.approx(1.0, abs=1e-12)
    assert f["nbar"] == pytest.approx(pn @ np.arange(pn.size) / pn.sum(), rel=1e-12)


def test_rate_backend_matches_coherent_in_lattice():
    c = lattice(raman__carrier_rabi_hz=1e3, initial__nbar=1.0, initial__motion_mode="thermal_nbar", timing__total_duration_ms=5)
    fc = run(c)[2]["final"]
    c.numerics.solver = "rate"
    fr = run(c)[2]["final"]
    for k in ("P_up", "P_n0", "P_target", "nbar", "lattice_loss"):
        assert fr[k] == pytest.approx(fc[k], abs=3e-3), k
