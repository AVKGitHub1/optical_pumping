"""Independent scalar/vector limits for the counter-rotating light shift."""
from pathlib import Path

import numpy as np
import pytest

from rb85rsc import atomic as at, far_detuned as fd, polarization as pol
from rb85rsc.config import SimConfig
from rb85rsc.dynamics import _single_thread


@pytest.fixture(params=['weak_field', 'diagonalized'])
def response(request):
    return fd.DLineResponse(at.load_atomic_data(), 0., [0., 0., 1.], request.param)


@pytest.mark.parametrize('wavelength_nm', [785., 1188.])
@pytest.mark.parametrize('ellipticity_deg', [0., 22.5, 45.])
def test_unresolved_hyperfine_scalar_vector_limit(response, wavelength_nm, ellipticity_deg):
    # Suppress hyperfine splittings to recover the electronic J=1/2 limit.
    # The F=3 stretched states have m_J=+/-1/2. Electronic Clebsch-Gordan
    # weights give scalar strengths S1/3, S2/3 and vector strengths -S1/3,
    # S2/6. Scalar denominators subtract; vector denominators ADD.
    rows = fd.dline_data()['rows']
    response.eg[:] = 0.
    frequency = at.C_LIGHT / (wavelength_nm * 1e-9)
    scalar, vector = 0., 0.
    for j, vector_weight in ((.5, -1/3), (1.5, 1/6)):
        selected = np.array([row['j'] == j for row in rows])
        centroid = response.ee[selected].mean()
        response.ee[selected] = centroid
        strength = np.sum(response.d[selected]**2, axis=0).mean()
        scalar += strength/3 * (1/(frequency-centroid) - 1/(frequency+centroid))
        vector += strength*vector_weight * (1/(frequency-centroid) + 1/(frequency+centroid))
    scale = response.eunit**2 / (4*at.H**2)
    chi = np.deg2rad(ellipticity_deg)
    eps = np.array([np.cos(chi), 1j*np.sin(chi), 0.]) * np.exp(.73j)
    helicity = np.sin(2*chi)
    stretched = [at.GROUND_STATES.index((3, m)) for m in (3, -3)]
    expected = scale * (scalar + vector*helicity*np.array([1., -1.]))
    np.testing.assert_allclose(response.shift_hz(frequency, eps)[stretched], expected,
                               rtol=3e-13, atol=1e-16)
    # Reversing helicity reverses the vector contribution, not the scalar.
    np.testing.assert_allclose(response.shift_hz(frequency, eps.conj())[stretched],
                               expected[::-1], rtol=3e-13, atol=1e-16)


def test_vector_shift_vanishes_in_static_limit(response):
    eps = np.array([1., 1j, 0.]) / np.sqrt(2)
    shifts = response.shift_hz(0., eps)
    reversed_m = [at.GROUND_STATES.index((f, -m)) for f, m in at.GROUND_STATES]
    np.testing.assert_allclose(shifts, shifts[reversed_m], rtol=3e-14, atol=1e-18)


@pytest.mark.parametrize('wavelength_nm', [785., 1188.])
def test_linear_polarization_preserves_previous_shift_at_finite_field(response, wavelength_nm):
    response = fd.DLineResponse(response.atom, 3., [1., 2., 3.], response.zeeman_model)
    frequency = at.C_LIGHT / (wavelength_nm * 1e-9)
    linear = pol.unit([1., -2., .3])
    # The previous expression is valid for a real Jones vector.
    nu = response.ee[:, None] - response.eg
    denom = 1/(2*np.pi*(frequency-nu)) - 1/(2*np.pi*(frequency+nu))
    previous = np.sum(abs(response.absorption(linear)*response.eunit/at.HBAR)**2
                      * denom, axis=0) / (8*np.pi)
    for eps in (linear.tolist(), linear*np.exp(.73j)):
        np.testing.assert_allclose(response.shift_hz(frequency, eps), previous,
                                   rtol=3e-14, atol=1e-18)


def test_carrier_cache_refreshes_light_shifts_after_model_correction(tmp_path, monkeypatch):
    cfg = SimConfig.from_json(Path(__file__).resolve().parents[1] / 'examples/experiment.json')
    cfg.ensemble.calibration_samples = 2
    cfg.raman.carrier_decay_mode = 'modeled'
    atom = at.load_atomic_data()
    fd.dline_data()  # Load atomic input before redirecting calibration writes.
    monkeypatch.setattr(fd, 'CACHE', tmp_path)
    response = fd.DLineResponse(atom, cfg.magnetic.magnitude_gauss,
                               cfg.magnetic.direction, cfg.optical.zeeman_model)
    current_source = response.source.copy()
    # Previously written carrier files have no light-shift model version.
    response.source.pop('light_shift_model_version', None)
    with _single_thread():
        legacy = fd.calibrate_carrier(cfg, response)
        path = tmp_path / f"carrier_{legacy['cache_key']}.json"
        fd._write_cache(path, dict(legacy, phase_averaged_central_differential_shift_hz=1e30))
        response.source = current_source
        corrected = fd.calibrate_carrier(cfg, response)
    assert corrected['cache_key'] != legacy['cache_key']
    assert abs(corrected['phase_averaged_central_differential_shift_hz']) < 1e30
    assert fd.calibrate_carrier(cfg, response) == corrected
    assert len(list(tmp_path.glob('carrier_*.json'))) == 2
