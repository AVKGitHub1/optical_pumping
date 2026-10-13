"""Independent spin Hamiltonians and optical sum rules for finite-field atoms."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from rb85rsc import atomic as at, far_detuned as fd
from rb85rsc.config import SimConfig
from rb85rsc.finite_field import d2_dipoles, d2_eigensystem, ground_eigensystem
from rb85rsc.optical import beam_rates, beam_spec
from rb85rsc.runner import run, metadata

ROOT = Path(__file__).resolve().parents[1]
ATOM = at.load_atomic_data()


def spin(j):
    m = np.arange(-j, j+1)
    plus = np.diag(np.sqrt(j*(j+1)-m[:-1]*(m[:-1]+1)), -1)
    return (plus+plus.T)/2, (plus-plus.T)/(2j), np.diag(m)


@lru_cache(maxsize=2)
def coupled_transform(j):
    # Independently generated Clebsch-Gordan coefficients, not the analytic
    # adjacent-F formula used in production. ARC uses the (J I) order.
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan
    states = at.GROUND_STATES if j == .5 else at.EXCITED_STATES
    pairs = [(mj, mi) for mj in np.arange(-j, j+1) for mi in np.arange(-2.5, 3.5)]
    transform = np.array([[float(clebsch_gordan(Rational(j), Rational(2.5), f,
                            Rational(mj), Rational(mi), m)) for f, m in states] for mj, mi in pairs])
    return transform, pairs


def independent_hamiltonian(j, b):
    # Build I.J directly in |m_J,m_I>, and recover the scalar hyperfine
    # polynomial from the cached zero-field spectrum (including quadrupole).
    js, ins = spin(j), spin(2.5)
    dot = sum(np.kron(a, i) for a, i in zip(js, ins))
    hfs = ATOM.ground_hfs_hz if j == .5 else ATOM.excited_hfs_hz
    x = np.array([(f*(f+1)-2.5*3.5-j*(j+1))/2 for f in hfs])
    coefficients = np.polynomial.polynomial.polyfit(x, list(hfs.values()), 1 if j == .5 else 2)
    hf = sum(c*np.linalg.matrix_power(dot, order) for order, c in enumerate(coefficients))
    gi = (ATOM.g_ground[2]+ATOM.g_ground[3])/2
    gj = gi+3*(ATOM.g_ground[3]-ATOM.g_ground[2]) if j == .5 else ATOM.gJ_excited
    return hf + at.MU_B*b/at.H*(gj*np.kron(js[2], np.eye(6))+gi*np.kron(np.eye(int(2*j+1)), ins[2]))


@pytest.mark.parametrize('gauss', [0., .001, 1., 3., 30., 100.])
@pytest.mark.parametrize('j', [.5, 1.5])
def test_eigenstates_match_independent_uncoupled_spin_hamiltonian(gauss, j):
    b = gauss*1e-4
    energies, vectors = ground_eigensystem(ATOM, b) if j == .5 else d2_eigensystem(ATOM, b)
    transform, _ = coupled_transform(j)
    psi = transform @ vectors
    hamiltonian = independent_hamiltonian(j, b)
    np.testing.assert_allclose(hamiltonian @ psi, psi*energies, rtol=0, atol=3e-6)
    np.testing.assert_allclose(psi.T @ psi, np.eye(len(energies)), atol=2e-15)
    if j == .5:
        np.testing.assert_allclose(energies, ATOM.ground_energy_hz(b), rtol=0, atol=1e-6)


def test_cached_dipole_phases_match_uncoupled_nuclear_spin_conservation():
    ug, ground = coupled_transform(.5)
    ue, excited = coupled_transform(1.5)
    uncoupled = ue @ ATOM.dipole_au @ ug.T
    for row, (_, mi_e) in enumerate(excited):
        for col, (_, mi_g) in enumerate(ground):
            if mi_e != mi_g:
                assert abs(uncoupled[row, col]) < 2e-15
    # Electronic amplitudes are identical for every spectator nuclear spin.
    blocks = [uncoupled[np.ix_([i for i, (_, mi) in enumerate(excited) if mi == m],
                              [i for i, (_, mi) in enumerate(ground) if mi == m])]
              for m in np.arange(-2.5, 3.5)]
    for block in blocks[1:]:
        np.testing.assert_allclose(block, blocks[0], atol=2e-15)


@pytest.mark.parametrize('gauss', [0., 3., 100.])
def test_mixed_dipoles_branching_cycling_and_rate_sum_rules(gauss):
    b = gauss*1e-4
    d = d2_dipoles(ATOM, b)
    np.testing.assert_allclose((d*d).sum(axis=1), ATOM.emission_norm(), rtol=2e-15)
    q = np.array([m for _, m in at.EXCITED_STATES])[:, None]-np.array([m for _, m in at.GROUND_STATES])
    assert np.max(abs(d[abs(q) > 1])) == 0
    stretched = at.EXCITED_STATES.index((4, 4))
    assert d[stretched, at.UP] == ATOM.dipole_au[stretched, at.UP]
    c = SimConfig()
    c.spin_pump.polarization.pi_fraction = .2
    beam = beam_spec('spin_pump', c.spin_pump, ATOM, [0, 0, 1])
    independent = beam_rates(ATOM, beam, b, zeeman_model='diagonalized')
    coherent = beam_rates(ATOM, beam, b, path_model='kramers_heisenberg', zeeman_model='diagonalized')
    np.testing.assert_allclose(independent.gamma_out, ATOM.gamma*independent.p_exc, rtol=3e-15)
    np.testing.assert_allclose(coherent.gamma_out, independent.gamma_out, rtol=3e-15)
    assert coherent.W.min() >= 0


def test_zero_field_recovers_legacy_optical_and_dline_response():
    c = SimConfig()
    beam = beam_spec('repump', c.repump, ATOM, [0, 0, 1])
    for path in ('independent', 'kramers_heisenberg'):
        old = beam_rates(ATOM, beam, 0., path_model=path)
        new = beam_rates(ATOM, beam, 0., path_model=path, zeeman_model='diagonalized')
        np.testing.assert_allclose(new.W, old.W, rtol=2e-15, atol=1e-15)
    old = fd.DLineResponse(ATOM, 0., [1, 0, 0])
    new = fd.DLineResponse(ATOM, 0., [1, 0, 0], 'diagonalized')
    np.testing.assert_array_equal(new.ee, old.ee)
    np.testing.assert_array_equal(new.d, old.d)
    freq = at.C_LIGHT/783e-9
    eps = np.array([0, 1, 1j])/np.sqrt(2)
    np.testing.assert_array_equal(new.shift_hz(freq, eps), old.shift_hz(freq, eps))
    np.testing.assert_array_equal(new.scattering_amplitudes(freq, eps), old.scattering_amplitudes(freq, eps))


def test_far_detuned_and_pump_models_share_d2_eigenstates_and_dipoles():
    response = fd.DLineResponse(ATOM, 3., [1, 0, 0], 'diagonalized')
    np.testing.assert_allclose(response.d[12:]/at.EA0, d2_dipoles(ATOM, 3e-4), rtol=0, atol=2e-15)
    np.testing.assert_allclose(response.ee[12:], ATOM.d2_centroid_hz+d2_eigensystem(ATOM, 3e-4)[0], rtol=0, atol=.1)
    assert response.source['zeeman_model'] == 'diagonalized'


@pytest.mark.parametrize('gauss', [3., 100.])
def test_d1_energies_follow_independent_breit_rabi_formula(gauss):
    rows = fd.dline_data()['rows'][:12]
    hfs = {r['f']: r['energy_hz'] for r in rows}
    gf = {r['f']: r['g'] for r in rows}
    split = hfs[3]-hfs[2]
    gi = (gf[3]+gf[2])/2
    zeeman = at.MU_B*gauss*1e-4/at.H
    x = 3*(gf[3]-gf[2])*zeeman/split
    expected = []
    for r in rows:
        f, m = r['f'], r['m']
        shift = gf[f]*m*zeeman if abs(m) == 3 else (
            gi*m*zeeman + (1 if f == 3 else -1)*split/2*(np.sqrt(1+2*m*x/3+x*x)-1))
        expected.append(hfs[f]+shift)
    response = fd.DLineResponse(ATOM, gauss, [0, 0, 1], 'diagonalized')
    np.testing.assert_allclose(response.ee[:12], expected, rtol=0, atol=.125)  # absolute optical frequencies


def test_scan_preflight_uses_the_selected_atomic_rates():
    from rb85rsc.scan import _reject_weak_excitation
    c = SimConfig.from_json(ROOT/'examples/onlyOP.json')
    peaks = {}
    for mode in ('weak_field', 'diagonalized'):
        rates = [beam_rates(ATOM, beam_spec(name, getattr(c, name), ATOM, c.magnetic.direction),
                            c.magnetic.magnitude_gauss*1e-4, zeeman_model=mode)
                 for name in ('spin_pump', 'repump')]
        peaks[mode] = float(sum(r.p_exc for r in rates).max())
    assert abs(peaks['weak_field']-peaks['diagonalized']) > 1e-6
    c.optical.weak_excitation_invalid = np.mean(list(peaks.values()))
    for mode, peak in peaks.items():
        c.optical.zeeman_model = mode
        assert (_reject_weak_excitation(c) is not None) == (peak > c.optical.weak_excitation_invalid)


def test_magnetic_diagnostic_changes_only_for_included_physics():
    legacy = at.zeeman_validity(ATOM, 3e-4, 70e3)
    corrected = at.zeeman_validity(ATOM, 3e-4, 70e3, zeeman_model='diagonalized')
    assert legacy['status'] == 'warning'
    assert corrected['status'] == 'ok'
    assert corrected['max_included_excited_correction_hz'] > 1e6
    assert at.zeeman_validity(ATOM, 1., 70e3, zeeman_model='diagonalized')['status'] == 'invalid'


def test_calibration_cache_separates_basis_models_and_rejects_mismatch(tmp_path, monkeypatch):
    from rb85rsc.dynamics import _single_thread

    # Read portable atomic inputs before redirecting calibration writes.
    fd.dline_data()
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    c = SimConfig.from_json(ROOT/'examples/experiment.json')
    c.ensemble.calibration_samples = 8
    responses = [fd.DLineResponse(ATOM, c.magnetic.magnitude_gauss, c.magnetic.direction, mode)
                 for mode in ('weak_field', 'diagonalized')]
    # Match run_ensemble's numerical context: calibration also warms shared
    # motional matrix caches used by later serial and spawned-worker runs.
    with _single_thread():
        fd.calibrate_carrier(c, responses[0])
        c.optical.zeeman_model = 'diagonalized'
        with pytest.raises(ValueError, match='same optical zeeman_model'):
            fd.calibrate_carrier(c, responses[0])
        result = fd.calibrate_carrier(c, responses[1])
    assert result['intensity_per_unit_tone_w_m2'] > 0
    assert len(list(tmp_path.glob('carrier_*.json'))) == 2


def test_ensemble_uses_selected_model_and_retains_other_validity_checks():
    c = SimConfig.from_json(ROOT/'examples/onlyOP.json')
    c.ensemble.samples = c.ensemble.workers = 1
    c.timing.custom_segments[0]['duration_ms'] = .02
    c.timing.n_samples = 5
    c.optical.zeeman_model = 'diagonalized'
    result = run(c)
    meta = metadata(result)
    checks = {v['category']: v['status'] for v in meta['validity']}
    assert checks['magnetic field'] == 'ok'
    assert checks['motional secular'] == 'invalid'
    assert meta['model_summary']['optical_zeeman_model'] == 'diagonalized'
    assert result['final']['max_conservation_error'] < 1e-12
    assert any('mixed optical dipoles' in line for line in meta['assumptions'])
