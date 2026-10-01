"""Independent limits and regressions from the experiment-config physics audit."""
from pathlib import Path

import numpy as np
import pytest

from rb85rsc.analysis import analyze
from rb85rsc.atomic import GROUND_STATES, H, K_B, MU_B, UP, DOWN, load_atomic_data, zeeman_validity
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import RunResult, solve
from rb85rsc.model import Model
from rb85rsc.motion import annihilation, boltzmann_pn, temperature_from_energy
from rb85rsc.optical import beam_rates, beam_spec

from helpers import cfg

ROOT = Path(__file__).resolve().parents[1]


def experiment():
    return SimConfig.from_json(ROOT / "tests/fixtures/audit_experiment_1d.json")


def populations_result(m, pn):
    """Synthetic, spin-polarized distributions for testing observables independently of ODEs."""
    P = np.zeros((len(pn), 12, m.N))
    P[:, UP] = pn
    return RunResult(np.linspace(0, 1e-3, len(pn)), P, 1 - P.sum(axis=(1, 2)),
                     np.zeros((len(pn), 3)), np.zeros(len(pn)), np.zeros(len(pn)),
                     "coherent", m.schedule, 0.0)


@pytest.mark.parametrize("temp", [0.0, 0.5e-6, 10e-6, 100e-6])
def test_lattice_canonical_temperature_round_trip(temp):
    c = experiment()
    c.initial.temperature_k = temp
    m = Model(c)
    pn = m.P0.sum(axis=0)
    e = m.motion.energies_hz - m.motion.energies_hz[0]
    assert temperature_from_energy(pn @ e, e) == pytest.approx(temp, rel=1e-8, abs=1e-15)
    a = analyze(m, populations_result(m, np.tile(pn, (5, 1))))
    assert a["final"]["T_equiv_K"] == pytest.approx(temp, rel=1e-8, abs=1e-15)
    assert a["final"]["final_pn_TVD_from_thermal"] < 1e-9
    assert m.initial_tail == 0  # not an invented unbound/continuum fraction
    assert a["final"]["initial_unbound_fraction"] is None


def test_lattice_cap_reports_exact_missing_bound_weight():
    c = experiment()
    c.trap.n_max = 2
    m = Model(c)
    full = boltzmann_pn(m.motion.all_bound_energies_hz, c.initial.temperature_k)
    assert m.initial_tail == pytest.approx(full[3:].sum(), abs=1e-14)
    np.testing.assert_allclose(m.P0.sum(axis=0), full[:3] / full[:3].sum())


def test_lattice_energy_reduction_and_power_use_actual_level_spacings():
    m = Model(experiment())
    pn = np.zeros((5, m.N))
    pn[:, 2] = np.linspace(1, 0, 5)
    pn[:, 1] = 1 - pn[:, 2]
    a = analyze(m, populations_result(m, pn))
    e = m.motion.energies_hz - m.motion.energies_hz[0]
    assert a["final"]["fractional_energy_reduction"] == pytest.approx(1 - e[1] / e[2])
    assert abs(a["final"]["fractional_energy_reduction"] - 0.5) > 0.01
    np.testing.assert_allclose(a["series"]["cooling_power_W"], H * (e[2] - e[1]) / 1e-3, rtol=1e-12)


def test_population_inversion_does_not_get_a_positive_thermal_temperature():
    m = Model(experiment())
    pn = np.zeros((5, m.N))
    pn[:, -1] = 1
    a = analyze(m, populations_result(m, pn))
    assert a["final"]["T_equiv_K"] is None
    assert a["final"]["final_pn_TVD_from_thermal"] is None
    assert "no positive-temperature" in a["final"]["T_equiv_note"]


def test_loss_selection_is_not_a_per_photon_cooling_efficiency():
    m = Model(experiment())
    pn = np.tile(m.P0.sum(axis=0), (5, 1))
    pn *= np.linspace(1, 0.5, 5)[:, None]
    r = populations_result(m, pn)
    r.counts[:, 0] = np.linspace(0, 10, 5)
    a = analyze(m, r)
    assert a["final"]["net_quanta_removed_per_photon"] is None
    o = m.ops(m.schedule[0])
    e = H * (m.motion.energies_hz - m.motion.energies_hz[0])
    p = pn[0]
    hazard = o.ovr.reshape(12, m.N)[UP]
    expected = (p * hazard) @ e - (p @ e) * (p @ hazard)
    assert a["series"]["selection_cooling_power_W"][0] == pytest.approx(expected, rel=1e-12, abs=1e-40)


def _spin_operators(j):
    ms = np.arange(-j, j + 1)
    plus = np.diag(np.sqrt(j * (j + 1) - ms[:-1] * (ms[:-1] + 1)), -1)
    return (plus + plus.T) / 2, (plus - plus.T) / (2j), np.diag(ms)


@pytest.mark.parametrize("field_gauss", [0.0, 1.0, 10.0])
def test_breit_rabi_matches_independent_uncoupled_spin_hamiltonian(field_gauss):
    a = load_atomic_data()
    gi = (a.g_ground[3] + a.g_ground[2]) / 2
    gj = gi + 3 * (a.g_ground[3] - a.g_ground[2])
    ix, iy, iz = _spin_operators(2.5)
    jx, jy, jz = _spin_operators(0.5)
    hf = a.hfs_ground_splitting_hz / 3 * (np.kron(ix, jx) + np.kron(iy, jy) + np.kron(iz, jz))
    hz = MU_B * field_gauss * 1e-4 / H * (gi * np.kron(iz, np.eye(2)) + gj * np.kron(np.eye(6), jz))
    mz = np.diag(np.kron(iz, np.eye(2)) + np.kron(np.eye(6), jz))
    expected = {}
    for m in range(-3, 4):
        indices = np.flatnonzero(mz == m)
        energies = np.linalg.eigvalsh((hf + hz)[np.ix_(indices, indices)])
        expected[3, m] = energies[-1]
        if len(energies) == 2:
            expected[2, m] = energies[0]
    np.testing.assert_allclose(a.ground_energy_hz(field_gauss * 1e-4),
                               [expected[state] for state in GROUND_STATES], rtol=0, atol=2e-6)


def test_zeeman_check_does_not_silently_report_zero_when_arc_database_unavailable(monkeypatch):
    import rb85rsc.atomic as atomic

    def fail(*args):
        raise RuntimeError("readonly database")

    monkeypatch.setattr(atomic, "_breit_rabi_ground", fail)
    v = zeeman_validity(load_atomic_data(), 1e-4, 70e3)
    assert 350 < v["raman_quadratic_zeeman_error_hz"] < 370
    assert "analytic" in v["breit_rabi_source"]


def test_optical_resonance_uses_same_ground_energy_as_raman():
    a = load_atomic_data()
    c = SimConfig()
    c.repump.polarization.pi_fraction = 1
    b = 1e-4
    g = GROUND_STATES.index((2, 0))
    beam = beam_spec("repump", c.repump, a, [0, 0, 1])
    beam.laser_hz = a.d2_centroid_hz + a.excited_hfs_hz[3] - a.ground_energy_hz(b)[g]
    rates = beam_rates(a, beam, b, excited_manifold="F3_only_IDEALIZED")
    assert rates.light_shift[g] == pytest.approx(0, abs=1e-10)


def test_up_to_down_raman_kick_has_absorbed_minus_emitted_momentum():
    m = Model(cfg(trap__n_max=18))
    # Apply the lowering block of H to a motional ground state. In units
    # hbar/(2*x0), p = i(a^dagger-a), and a kick gives <p> = 2*eta.
    psi = m.D.conj().T[:, 0]
    aa = annihilation(m.N)
    momentum = np.vdot(psi, 1j * (aa.T - aa) @ psi).real
    assert momentum == pytest.approx(2 * m.eta_R_signed, rel=1e-12)


def test_four_raman_pathways_factorize_into_physical_tone_fields():
    c = cfg(trap__n_max=10, raman__tone_layout="both_tones_both_beams",
            raman__tone_powers_low=[0.4, 1.3], raman__tone_powers_high=[1.1, 0.6],
            raman__lattice_phase_deg=31, raman__beam_beat_phase_deg=67)
    m = Model(c)
    pl, ph = np.sqrt(c.raman.tone_powers_low), np.sqrt(c.raman.tone_powers_high)
    theta, chi = np.deg2rad([31, 67])
    # Choose phi_l1=phi_h1=0, phi_h2=-theta, phi_l2=-theta-chi.
    expected = (pl[0] * ph[0] + pl[1] * ph[1] * np.exp(-1j * chi)) * np.eye(m.N)
    expected = expected + pl[0] * ph[1] * np.exp(1j * theta) * m.motion.displacement(m.eta_R_signed)
    expected = expected + pl[1] * ph[0] * np.exp(-1j * (theta + chi)) * m.motion.displacement(-m.eta_R_signed)
    np.testing.assert_allclose(m.D.conj().T, expected, atol=1e-14)


def test_lattice_interaction_picture_matches_static_hamiltonian_exponential():
    from scipy.linalg import expm

    c = experiment()
    c.timing.protocol = "raman_only"
    c.timing.total_duration_ms = 0.037
    c.timing.n_samples = 3
    c.initial.spin_preset = "stretched"
    c.initial.motion_mode = "fock"
    c.initial.fock_n = 2
    c.raman.lattice_phase_deg = 31
    c.raman.beam_beat_phase_deg = 67
    c.raman.max_sideband_order = 5
    c.raman.red_sideband_offset_hz = 2345
    c.numerics.rtol = 1e-10
    c.numerics.atol = 1e-12
    m = Model(c)
    o = m.ops(m.schedule[0])
    V = m.omega_c / 2 * m.D
    Hrot = np.block([[np.diag(m.eps - m.omega_ref - o.delta), V],
                     [V.conj().T, np.diag(m.eps)]])
    psi0 = np.zeros(2 * m.N, complex)
    psi0[2] = 1
    t = c.timing.total_duration_ms * 1e-3
    psi = expm(-1j * Hrot * t) @ psi0
    psi *= np.exp(1j * np.concatenate([m.eps - m.omega_ref, m.eps]) * t)
    expected = np.outer(psi, psi.conj())
    np.testing.assert_allclose(solve(m).rho_final, expected, rtol=0, atol=2e-9)


def test_experiment_geometry_conflict_and_recoil_approximation_are_visible():
    m = Model(experiment())
    np.testing.assert_allclose(m.beams["repump"].direction, [0, 1, 0], atol=1e-14)
    v = {category: (status, message) for category, status, message in m.static_validity().items}
    assert v["repump direction"][0] == "warning"
    assert v["recoil coherence"][0] == "warning"


def test_raman_isolation_checks_higher_anharmonic_sidebands_and_schedule_amplitude():
    c = experiment()
    m = Model(c)
    nearest = m.raman_isolation()
    assert abs(nearest["sideband"]) > 2
    n, k = nearest["motional_levels"]
    assert n < m.N and k < m.N
    c.timing.custom_segments[0]["raman"] = 2
    stronger = Model(c).raman_isolation()
    assert stronger["ratio_to_carrier_rabi"] == pytest.approx(nearest["ratio_to_carrier_rabi"] / 2)
