"""Raman amplitudes retain joint motional transitions for oblique beams."""
from types import SimpleNamespace

import numpy as np
import pytest

from rb85rsc.config import SimConfig
from rb85rsc.model import raman_product_operator, raman_sideband_mask
from rb85rsc.motion import displacement_matrix


def bases():
    return [SimpleNamespace(N=n, x0=x0,
                            displacement=lambda eta, n=n: displacement_matrix(eta, n))
            for n, x0 in ((2, 21e-9), (3, 27e-9), (2, 35e-9))]


def raman():
    rc = SimConfig().raman
    rc.beam_low_direction = [1., 2., 3.]
    rc.beam_high_direction = [-2., 1., -1.]
    return rc


def test_oblique_single_path_has_joint_sidebands_and_spectator_factors():
    rc, sites = raman(), bases()
    actual = raman_product_operator(rc, sites, np.eye(3))
    klo, khi = (np.asarray(k) / np.linalg.norm(k)
                for k in (rc.beam_low_direction, rc.beam_high_direction))
    eta = 2 * np.pi / (rc.wavelength_nm * 1e-9) * (klo - khi) * [s.x0 for s in sites]
    factors = [displacement_matrix(e, s.N) for e, s in zip(eta, sites)]
    # Explicit matrix-element products are independent of tensor flattening.
    shape = tuple(s.N for s in sites)
    for n in np.ndindex(shape):
        for m in np.ndindex(shape):
            expected = np.prod([f[nj, mj] for f, nj, mj in zip(factors, n, m)])
            assert actual[np.ravel_multi_index(n, shape), np.ravel_multi_index(m, shape)] == pytest.approx(expected)
    # A single photon pair can change two or three occupations coherently.
    assert abs(actual[np.ravel_multi_index((1, 1, 1), shape), 0]) > 1e-4


def test_path_phases_and_distinct_beam_envelopes_add_as_amplitudes():
    rc, sites = raman(), bases()[:2]
    rc.tone_layout = 'both_tones_both_beams'
    rc.tone_powers_low = [2., .3]
    rc.tone_powers_high = [.6, 1.7]
    rc.beam_beat_phase_deg = 37.
    phase, intensities = .71, [.25, .81]
    paths = np.array([[.3 + .2j, -.7 + .5j], [1.1 - .6j, .9 + .3j]])
    actual = raman_product_operator(rc, sites, np.eye(3)[:2], paths,
                                    phase_rad=phase, beam_intensities=intensities)
    single = raman()
    forward = raman_product_operator(single, sites, np.eye(3)[:2])
    # theta is the 1-low -> 2-high path phase; chi is beam 2's
    # high-minus-low phase relative to beam 1, matching the config convention.
    low_phase = [phase / 2, -phase / 2 - np.deg2rad(rc.beam_beat_phase_deg)]
    high_phase = [phase / 2, -phase / 2]
    expected = np.zeros_like(actual)
    for i in range(2):
        for j in range(2):
            displacement = np.eye(len(actual)) if i == j else (forward if i == 0 else forward.conj().T)
            coefficient = (paths[i, j] * np.sqrt(rc.tone_powers_low[i] * rc.tone_powers_high[j]
                                               * intensities[i] * intensities[j])
                           * np.exp(1j * (low_phase[i] - high_phase[j])))
            expected += coefficient * displacement
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    diagonal = raman_product_operator(rc, sites, np.eye(3)[:2], paths,
                                      phase_rad=phase, beam_intensities=intensities, diagonal=True)
    np.testing.assert_allclose(diagonal, np.diag(actual), atol=1e-15)


def test_common_lab_rotation_leaves_product_operator_unchanged():
    rc, sites = raman(), bases()
    expected = raman_product_operator(rc, sites, np.eye(3))
    angle = .63
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rc.beam_low_direction = rotation @ rc.beam_low_direction
    rc.beam_high_direction = rotation @ rc.beam_high_direction
    actual = raman_product_operator(rc, sites, rotation.T)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_axis_aligned_product_is_legacy_displacement_times_identity():
    rc, sites = raman(), bases()
    rc.beam_low_direction, rc.beam_high_direction = [0., 0., 1.], [0., 0., -1.]
    actual = raman_product_operator(rc, sites, np.eye(3))
    axial = displacement_matrix(4 * np.pi / (rc.wavelength_nm * 1e-9) * sites[2].x0, sites[2].N)
    np.testing.assert_allclose(actual, np.kron(np.eye(6), axial), atol=1e-15)


def test_one_tone_layout_ignores_inactive_powers_and_phases():
    rc, sites = raman(), bases()
    expected = raman_product_operator(rc, sites, np.eye(3))
    rc.tone_powers_low = [3., 0.]
    rc.tone_powers_high = [5., 0.]
    rc.beam_beat_phase_deg = 127.
    actual = raman_product_operator(rc, sites, np.eye(3), phase_rad=1.9)
    np.testing.assert_array_equal(actual, expected)


def test_sideband_cutoff_counts_total_changes_across_axes():
    sites = bases()
    mask = raman_sideband_mask(sites, 2)
    shape = tuple(s.N for s in sites)
    for n in np.ndindex(shape):
        for m in np.ndindex(shape):
            assert mask[np.ravel_multi_index(n, shape), np.ravel_multi_index(m, shape)] == (sum(abs(a - b) for a, b in zip(n, m)) <= 2)
    single = raman_sideband_mask(sites[1:2], 1)
    n = np.arange(sites[1].N)
    np.testing.assert_array_equal(single, abs(n[:, None] - n[None, :]) <= 1)
