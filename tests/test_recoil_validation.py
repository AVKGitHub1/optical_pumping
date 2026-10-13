"""Cloud reference checks must match local inputs and expose incomplete coverage."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from rb85rsc import atomic as at
from rb85rsc.config import LatticeAxisConfig, SimConfig
from rb85rsc.dynamics import solve
from rb85rsc.far_detuned import spatial_samples
from rb85rsc.recoil_reference import _site_model, compare_recoil, validation_positions
from test_ensemble_geometry import rotate, ROTATIONS

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = SimConfig.from_json(ROOT/'examples/onlyOP.json')
    c.timing.custom_segments = [dict(duration_ms=.006, raman=0., pump=1., repump=.8),
                                dict(duration_ms=.004, raman=0., pump=.2, repump=1.)]
    c.timing.n_samples = 5
    c.numerics.rtol, c.numerics.atol = 1e-10, 1e-12
    c.trap.lattice_x = LatticeAxisConfig(wavelength_nm=1064., depth_uk=12., waist_um=140.)
    c.trap.lattice_y = LatticeAxisConfig(wavelength_nm=1188., depth_uk=10., waist_um=210.)
    c.trap.lattice_z = LatticeAxisConfig(wavelength_nm=785., depth_uk=15., waist_um=270.)
    return c


@pytest.mark.parametrize('axis', list('xyz'))
def test_local_population_reference_matches_production_1d(axis):
    c = config()
    original = c.to_dict()
    position = [60., 70., 80.]
    result = compare_recoil(c, axis=axis, position_um=position)
    m = _site_model(c, axis, position)
    production = solve(m)
    np.testing.assert_allclose(result['population']['P_absolute'], production.P, atol=2e-10)
    np.testing.assert_allclose(result['population']['well_escape_no_recapture'], production.overflow, atol=2e-10)
    np.testing.assert_allclose(result['population']['photons_spin_pump'], production.counts[:, 0], atol=2e-10)
    ia = 'xyz'.index(axis)
    spec = c.lattice_parameters(ia, at.load_atomic_data().mass_kg)
    p = np.array(position)
    expected = spec.depth_uk*np.exp(-2*(p @ p-p[ia]**2)/spec.waist_um**2)
    assert result['local_well']['depth_uk'] == pytest.approx(expected)
    assert c.to_dict() == original


def test_cloud_positions_use_reproducible_production_prefix():
    c = config()
    points = validation_positions(c, 4, [1., 2., 3.])
    np.testing.assert_array_equal([p['position_um'] for p in points[1:]], spatial_samples(c, 4)[0])
    assert points[0] == dict(site_id='reference', position_um=[1., 2., 3.])
    assert validation_positions(c, 8)[1:5] == points[1:]
    assert len(validation_positions(c)) == 1
    with pytest.raises(ValueError, match='nonnegative integer'):
        validation_positions(c, -1)


def test_rotated_reference_preserves_observables():
    c = config()
    c.spin_pump.polarization.mode = c.repump.polarization.mode = 'spherical'
    position = np.array([60., 70., 80.])
    rotation = ROTATIONS[-1]
    a = compare_recoil(c, axis='x', position_um=position)
    b = compare_recoil(rotate(c, rotation), axis='x', position_um=rotation @ position)
    for name in ('coherent', 'population'):
        np.testing.assert_allclose(a[name]['P_absolute'], b[name]['P_absolute'], atol=2e-11)
    np.testing.assert_allclose(b['local_well']['direction_lab'], rotation @ [1., 0., 0.], atol=1e-14)


def workflow(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module(name)


def test_single_photon_check_uses_overrides_preparation_and_rotated_lattices(monkeypatch):
    script = workflow(monkeypatch, 'check_recoil_factorization')
    c = config()
    c.initial.motion_mode, c.initial.fock_n = 'fock', 1
    c.spin_pump.polarization.mode = c.repump.polarization.mode = 'spherical'
    position = np.array([60., 70., 80.])
    rotation = ROTATIONS[-1]
    a = script.compare_single_photon(c, position, quadratures=(8, 16))
    b = script.compare_single_photon(rotate(c, rotation), rotation @ position, quadratures=(8, 16))
    np.testing.assert_allclose(a['wavelengths_nm'], [1064., 1188., 785.])
    assert a['initial_motional_populations'][2][1] == pytest.approx(1.)
    np.testing.assert_allclose(a['depths_uk'], [d*np.exp(-2*(position @ position-position[i]**2)/w**2)
                                              for i, (d, w) in enumerate(zip([12., 10., 15.], [140., 210., 270.]))])
    for first, second in zip(a['cases'], b['cases']):
        for key in ('correlated_3d_survival', 'marginal_product_survival'):
            assert first[key] == pytest.approx(second[key], abs=1e-12)
            assert 0 <= first[key] <= 1


def test_cloud_cli_exports_every_case_and_quadrature_check(monkeypatch, tmp_path):
    script = workflow(monkeypatch, 'check_coherent_recoil')
    c = config()
    source = tmp_path/'input.json'
    c.to_json(source)
    output = tmp_path/'results'
    assert script.main(['--config', str(source), '--output', str(output), '--axes', 'x', 'y', 'z',
                        '--cloud-samples', '1', '--power-scales', '.1', '--check-quadrature']) == 0
    report = json.loads((output/'survey.json').read_text())
    cases = json.loads((output/'comparison.json').read_text())
    assert report['status'] == 'complete'
    assert report['completed_cases'] == report['expected_cases'] == len(cases) == 6
    assert report['config'] == c.to_dict()
    for case in cases:
        assert case['numerics']['status'] == case['quadrature']['status'] == 'passed'
        assert case['quadrature']['points'] == [24, 48]
        with np.load(output/f"{case['case_id']}.npz") as arrays:
            delta = arrays['coherent_P_target_absolute']-arrays['population_P_target_absolute']
            assert case['differences']['P_target_absolute']['max_absolute'] == pytest.approx(abs(delta).max())
        refined = SimConfig.from_json(output/f"{case['case_id']}_refined_config.json")
        assert refined.numerics.recoil_quadrature_points == 48


def test_unsupported_cases_are_recorded_and_return_failure(monkeypatch, tmp_path):
    script = workflow(monkeypatch, 'check_coherent_recoil')
    source = tmp_path/'input.json'
    config().to_json(source)
    original = script.compare_recoil

    def reject_y(cfg, **kw):
        if kw['axis'] == 'y':
            raise ValueError('reference requires 2 to 16 bound levels')
        return original(cfg, **kw)

    monkeypatch.setattr(script, 'compare_recoil', reject_y)
    output = tmp_path/'results'
    assert script.main(['--config', str(source), '--output', str(output), '--axes', 'x', 'y', 'z',
                        '--power-scales', '1']) == 1
    report = json.loads((output/'survey.json').read_text())
    assert report['completed_cases'] == 2 and report['expected_cases'] == 3
    assert report['failed_case_ids'] == ['case_1']
    assert 'bound levels' in json.loads((output/'comparison.json').read_text())[1]['error']


def test_failed_quadrature_cannot_be_reported_as_passed(monkeypatch):
    script = workflow(monkeypatch, 'check_coherent_recoil')
    coarse = {name: {key: np.array([.5]) for key in script.PROBABILITIES} for name in ('coherent', 'population')}
    fine = {name: {key: val.copy() for key, val in values.items()} for name, values in coarse.items()}
    fine['coherent']['survival'] += 2e-4
    assert script.quadrature_check(coarse, fine, 1e-6)['status'] == 'failed'
    fine['coherent']['survival'][:] = np.nan
    assert script.quadrature_check(coarse, fine, 1e-6)['status'] == 'failed'
