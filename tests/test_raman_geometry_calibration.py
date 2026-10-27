"""Carrier calibration retains momentum transfer along every lattice axis."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rb85rsc import atomic as at, far_detuned as fd, motion as mo, polarization as pol
from rb85rsc.config import SimConfig


def config():
    return SimConfig.from_json(Path(__file__).resolve().parents[1] / 'examples/experiment.json')


@pytest.mark.parametrize('low,high', [([1, 0, 1], [-1, 0, -1]),
                                     ([1, 0, 1], [0, 1, -1]),
                                     ([1, 1, 0], [2, 2, 0])])
@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
def test_accepts_arbitrary_beams_and_both_tone_layouts(low, high, layout):
    cfg = config()
    cfg.raman.beam_low_direction, cfg.raman.beam_high_direction = low, high
    cfg.raman.tone_layout = layout
    cfg.validate()
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


def test_geometry_uses_momentum_transfer_and_is_rotation_covariant():
    low, high = np.array([1., 0., 1.]), np.array([-1., 0., -1.])
    rotation = Rotation.from_rotvec([.32, -.54, .71]).as_matrix()
    reference, driven, delta = pol.raman_geometry(low, high, np.eye(3))
    other_reference, other_driven, other_delta = pol.raman_geometry(rotation @ low, rotation @ high, rotation.T)
    assert reference == other_reference == 0  # Deterministic tie between x and z.
    assert driven == other_driven == (0, 2)
    np.testing.assert_allclose(delta, other_delta, atol=1e-15)
    # Neither beam follows a lattice axis, but their difference follows x.
    assert pol.raman_geometry([1, 1, 0], [-1, 1, 0], np.eye(3))[:2] == (0, (0,))


def test_copropagating_beams_have_constant_registration_phase():
    cfg = config()
    cfg.raman.beam_low_direction, cfg.raman.beam_high_direction = [1, 2, 3], [2, 4, 6]
    cfg.ensemble.phase_offset_deg = 37.
    _, phases = fd.spatial_samples(cfg, 8)
    np.testing.assert_allclose(phases, np.deg2rad(37.))
    assert pol.raman_geometry(cfg.raman.beam_low_direction, cfg.raman.beam_high_direction, np.eye(3))[:2] == (2, ())


@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
def test_oblique_carrier_matches_independent_product_and_beam_envelopes(tmp_path, monkeypatch, layout):
    cfg = config()
    cfg.ensemble.calibration_samples = 1
    cfg.raman.tone_layout = layout
    cfg.raman.beam_low_direction, cfg.raman.beam_high_direction = [1, 0, 1], [0, 1, -1]
    cfg.raman.tone_powers_low, cfg.raman.tone_powers_high = [.2, .8], [1.3, .7]
    cfg.raman.beam_beat_phase_deg = 28.
    position, phase = np.array([60., 10., 35.]), .37
    paths = np.array([[.2 + .1j, .7 - .3j], [-.4 + .2j, .5 + .4j]])
    response = SimpleNamespace(
        atom=at.load_atomic_data(), zeeman_model='weak_field', source={},
        scattering_amplitudes=lambda frequency, polarization: np.ones((12, 12, 3)),
        shift_hz=lambda frequency, polarization: np.ones(12),
    )
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    monkeypatch.setattr(fd, 'spatial_samples', lambda cfg, count: (position[None, :], np.array([phase])))
    monkeypatch.setattr(fd, 'raman_fields', lambda cfg, response: (1., 2., [np.eye(3)[0], np.eye(3)[1]], paths))
    actual = fd.calibrate_carrier(cfg, response)
    delta = pol.unit(cfg.raman.beam_low_direction) - pol.unit(cfg.raman.beam_high_direction)
    diagonal, thermal = np.ones(1, complex), np.ones(1)
    for i in range(3):
        lattice = cfg.lattice_parameters(i, response.atom.mass_kg)
        depth = lattice.depth_uk * fd.envelope(position, np.eye(3)[i], lattice.waist_um)
        nu = mo.lattice_frequency_hz(response.atom.mass_kg, lattice.wavelength_nm, depth)
        site = mo.lattice_site(response.atom.mass_kg, nu, depth * 1e-6 * at.K_B / at.H, cfg.trap.n_max + 1)
        diagonal = np.kron(diagonal, np.diag(site.displacement(2 * np.pi / (cfg.raman.wavelength_nm * 1e-9) * delta[i] * site.x0)))
        temperature = cfg.initial.temperature_k if i == 2 else cfg.ensemble.transverse_temperature_k
        thermal = np.kron(thermal, mo.boltzmann_pn(site.energies_hz, temperature))
    envelopes = [fd.envelope(position, direction, cfg.raman.waist_um)
                 for direction in (cfg.raman.beam_low_direction, cfg.raman.beam_high_direction)]
    if layout == 'one_tone_per_beam':
        low, high = [1., 0.], [0., 1.]
        theta = chi = 0.
    else:
        low, high = cfg.raman.tone_powers_low, cfg.raman.tone_powers_high
        theta, chi = phase, np.deg2rad(cfg.raman.beam_beat_phase_deg)
    amplitudes = paths * np.sqrt(np.outer(low, high)) * np.sqrt(np.outer(envelopes, envelopes))
    carrier = (amplitudes[0, 1] * np.exp(1j * theta) * diagonal
               + amplitudes[1, 0] * np.exp(-1j * (theta + chi)) * diagonal.conj()
               + amplitudes[0, 0] + amplitudes[1, 1] * np.exp(-1j * chi))
    intensity = cfg.raman.carrier_rabi_hz / np.sqrt(thermal @ np.abs(carrier)**2)
    assert actual['intensity_per_unit_tone_w_m2'] == pytest.approx(intensity, rel=2e-14)
    assert actual['driven_lattice_axes'] == ['x', 'y', 'z']
    assert actual['lattice_axis'] == 'z'
    assert actual['phase_averaged_central_raman_scattering_by_spin_s'][at.ground_label(0)] == pytest.approx(
        intensity * (sum(low) + sum(high)) * 36)


@pytest.mark.parametrize('layout', ['one_tone_per_beam', 'both_tones_both_beams'])
@pytest.mark.parametrize('copropagating', [False, True])
def test_local_raman_shift_matches_total_fields_with_beam_envelopes(layout, copropagating):
    from rb85rsc.ensemble import LocalExperiment
    cfg = config()
    cfg.raman.tone_layout = layout
    cfg.raman.beam_low_direction = [1, 0, 1]
    cfg.raman.beam_high_direction = [2, 0, 2] if copropagating else [0, 1, -1]
    cfg.raman.polarization_mode = 'jones'
    cfg.raman.beam_low_ellipticity_deg, cfg.raman.beam_high_ellipticity_deg = 20., -30.
    cfg.raman.beam_low_orientation_deg, cfg.raman.beam_high_orientation_deg = 37., -12.
    cfg.raman.tone_powers_low, cfg.raman.tone_powers_high = [.2, .8], [1.3, .7]
    cfg.raman.beam_beat_phase_deg = 28.
    response = fd.DLineResponse(at.load_atomic_data(), cfg.magnetic.magnitude_gauss, cfg.magnetic.direction)
    position, phase, intensity = np.array([60., 10., 35.]), .37, 12000.
    calibration = dict(intensity_per_unit_tone_w_m2=intensity, residual_coherence_decay_rate_s=0.)
    local = LocalExperiment(cfg, response, calibration, position, phase, population_recoil=False)
    low, high, eps, _ = fd.raman_fields(cfg, response)
    beam_i = intensity * np.array([fd.envelope(position, direction, cfg.raman.waist_um)
                                  for direction in (cfg.raman.beam_low_direction, cfg.raman.beam_high_direction)])
    np.testing.assert_allclose(local.beam_intensities, beam_i)
    expected = np.zeros(local.shape)
    # Fix beam 1's phase to zero. The low/high beam-2 phases independently
    # give crossed Raman phases theta, -(theta+chi), and same-beam phase -chi.
    chi = np.deg2rad(cfg.raman.beam_beat_phase_deg)
    for freq, powers, relative in zip((low, high), cfg.raman.tone_powers(), (-phase - chi, -phase)):
        field0 = np.sqrt(beam_i[0] * powers[0]) * eps[0]
        field1 = np.sqrt(beam_i[1] * powers[1]) * eps[1] * np.exp(1j * relative)
        self_terms = response.shift_hz(freq, field0) + response.shift_hz(freq, field1)
        interference = response.shift_hz(freq, field0 + field1) - self_terms
        delta = local.axes_lab @ (pol.unit(cfg.raman.beam_low_direction) - pol.unit(cfg.raman.beam_high_direction))
        overlap = np.ones(1)
        for i, site in enumerate(local.sites):
            # Direct quadrature of |psi_n(x)|^2 exp(i delta_k x); the isolated
            # wells have definite parity, making this diagonal overlap real.
            diagonal = (np.abs(site._vec)**2).T @ np.exp(2j * np.pi * freq / at.C_LIGHT * delta[i] * site.z)
            overlap = np.kron(overlap, diagonal.real)
        expected += self_terms[:, None, None, None] + interference[:, None, None, None] * overlap.reshape(local.shape[1:])
    np.testing.assert_allclose(local.raman_shifts_hz, expected, rtol=2e-12, atol=1e-10)
