"""Selected spin pairs must agree across dynamics, atomic response, and GUI I/O."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rb85rsc import atomic as at, far_detuned as fd, polarization as pol
from rb85rsc.config import ConfigError, SimConfig
from rb85rsc.dynamics import _single_thread
from rb85rsc.ensemble import LocalExperiment, solve_local
from rb85rsc.model import Model
from rb85rsc.runner import export, run

ROOT = Path(__file__).resolve().parents[1]


def small_config(up=(3, 2), down=(2, 0)):
    c = SimConfig()
    c.raman.state_up, c.raman.state_down = list(up), list(down)
    c.initial.spin_preset = "raman_up"
    c.initial.motion_mode, c.initial.fock_n = "fock", 1
    c.timing.protocol, c.timing.n_samples = "raman_only", 5
    c.trap.n_max = 6
    c.raman.carrier_rabi_hz = 1000.
    return c


def ensemble_config(up=(3, 3), down=(2, 1)):
    c = SimConfig.from_json(ROOT / "examples/experiment.json")
    c.raman.state_up, c.raman.state_down = list(up), list(down)
    c.raman.polarization_mode = "jones"
    c.raman.beam_low_ellipticity_deg = c.raman.beam_high_ellipticity_deg = 0.
    c.raman.beam_low_direction, c.raman.beam_high_direction = [1, 0, 0], [-1, 0, 0]
    c.magnetic.direction = [1, 0, 0]
    c.raman.wavelength_nm = 797.
    c.raman.carrier_rabi_hz = 20.
    c.ensemble.samples = c.ensemble.workers = 1
    c.ensemble.calibration_samples = 8
    c.timing.custom_segments = [dict(duration_ms=.01, raman=1., pump=1., repump=1.)]
    c.timing.n_samples = 3
    return c


@pytest.mark.parametrize("state", [[1, 0], [2, 3], [3, -4], [3], [3, 1, 0], [3., 1], [3, True], None, "3,1"])
def test_invalid_ground_states_are_rejected(state):
    c = SimConfig()
    c.raman.state_down = state
    with pytest.raises(ConfigError, match="raman.state_down"):
        c.validate()


@pytest.mark.parametrize("down, message", [([3, 3], "must be different"), ([2, -2], "delta mF")])
def test_invalid_pairs_are_rejected(down, message):
    c = SimConfig()
    c.raman.state_down = down
    with pytest.raises(ConfigError, match=message):
        c.validate()


@pytest.mark.parametrize("up,down", [((3, 2), (2, 0)), ((2, 0), (3, 2)), ((3, 1), (3, 0))])
def test_red_pi_pulse_transfers_only_the_selected_pair(up, down):
    c = small_config(up, down)
    m = Model(c)
    energies = m.atom.ground_energy_hz(m.B)
    assert m.nu_ud_zeeman == pytest.approx(energies[m.up] - energies[m.down])
    assert m.nu_beat == pytest.approx(energies[m.up] - energies[m.down] + m.nu_ref)
    c.timing.total_duration_ms = 1000 / (2 * c.raman.carrier_rabi_hz * abs(m.D[1, 0]))
    a = run(c)
    result = a["result"]
    assert result.P[-1, m.down, 0] > .99
    np.testing.assert_allclose(result.P[:, m.spectators], 0, atol=1e-12)
    np.testing.assert_allclose(result.P.sum(axis=(1, 2)) + result.overflow, 1, atol=1e-10)
    np.testing.assert_allclose(a["series"]["P_up"], result.P[:, m.up].sum(axis=1))
    np.testing.assert_allclose(a["series"]["P_down"], result.P[:, m.down].sum(axis=1))
    np.testing.assert_allclose(a["series"]["P_target_absolute"], result.P[:, m.up, 0])
    assert a["model_summary"]["raman_state_indices"] == [m.up, m.down]
    assert a["model_summary"]["raman_isolation"]["pair"] != f"{at.ground_label(m.up)}<->{at.ground_label(m.down)}"


@pytest.mark.parametrize("solver", ["coherent", "rate"])
def test_selected_pair_dephasing_matches_two_level_limit(solver):
    c = small_config()
    c.initial.fock_n = 0
    c.raman.beam_low_direction = c.raman.beam_high_direction = [1, 0, 0]
    c.raman.red_sideband_offset_hz = -c.trap.frequency_hz  # carrier
    c.raman.extra_coherence_decay_rate_s = 200000.
    c.timing.total_duration_ms = 1.
    c.numerics.solver = solver
    a = run(c)
    gamma, omega = c.raman.extra_coherence_decay_rate_s, 2 * np.pi * c.raman.carrier_rabi_hz
    t = a["series"]["t_s"]
    if solver == "coherent":
        rate = np.sqrt(gamma**2 / 4 - omega**2)
        z = np.exp(-gamma * t / 2) * (np.cosh(rate*t) + gamma / (2 * rate) * np.sinh(rate*t))
    else:
        z = np.exp(-omega**2 / gamma * t)
    np.testing.assert_allclose(a["series"]["P_down"], (1 - z) / 2, atol=2e-8)


def test_stretched_preparation_stays_literal_and_pump_shifts_follow_pair():
    c = small_config()
    c.initial.spin_preset = "stretched"
    c.timing.protocol = "continuous"
    m = Model(c)
    assert m.P0[at.UP].sum() == 1.
    assert m.P0[m.up].sum() == 0.
    u, d = m.light_shifts(m.schedule[0])
    assert u == pytest.approx(sum(r.light_shift[m.up] for r in m.rates.values()))
    assert d == pytest.approx(sum(r.light_shift[m.down] for r in m.rates.values()))
    # Constructing another pair never changes the first model or atomic defaults.
    other = Model(small_config((2, 1), (3, 1)))
    assert (m.up, m.down) != (other.up, other.down)
    assert (at.GROUND_STATES[at.UP], at.GROUND_STATES[at.DOWN]) == ((3, 3), (2, 2))


def test_linear_parallel_fields_drive_delta_m_two_but_not_one():
    c = ensemble_config()
    response = fd.DLineResponse(at.load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)
    low, high, fields, paths = fd.raman_fields(c, response)
    up, down = c.raman.state_indices()
    assert high - low == pytest.approx(response.eg[up] - response.eg[down], abs=.1)
    for field in fields:
        assert pol.spherical_fractions(field, c.magnetic.direction) == pytest.approx({1: .5, 0: 0., -1: .5})
    assert np.all(np.abs(paths) > 1e-6)
    # Linear fields have half the two-photon amplitude of the pure sigma-/sigma+ path.
    absorbed = pol.jones_field_vector([1, 0, 0], -45, 0, c.magnetic.direction)
    emitted = pol.jones_field_vector([1, 0, 0], 45, 0, c.magnetic.direction)
    pure = response.raman_hz(low, high, absorbed, emitted, up, down)
    assert abs(paths[0, 1]) == pytest.approx(abs(pure) / 2, rel=1e-8)
    c.raman.state_down = [2, 2]
    np.testing.assert_allclose(fd.raman_fields(c, response)[3], 0, atol=1e-15)


def test_calibration_cache_tracks_states_and_polarization(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "CACHE", tmp_path)
    c = ensemble_config()
    response = fd.DLineResponse(at.load_atomic_data(), c.magnetic.magnitude_gauss, c.magnetic.direction)
    first = fd.calibrate_carrier(c, response)
    c.raman.state_up, c.raman.state_down = [3, 2], [2, 0]
    second = fd.calibrate_carrier(c, response)
    c.raman.beam_low_ellipticity_deg = 15.
    third = fd.calibrate_carrier(c, response)
    assert len({r["cache_key"] for r in (first, second, third)}) == 3
    assert first["intensity_per_unit_tone_w_m2"] != second["intensity_per_unit_tone_w_m2"]
    assert second["intensity_per_unit_tone_w_m2"] != third["intensity_per_unit_tone_w_m2"]
    assert third["raman_state_up"] == [3, 2]
    shifts = third["phase_averaged_central_raman_light_shifts_by_spin_hz"]
    up, down = c.raman.state_indices()
    assert third["phase_averaged_central_differential_shift_hz"] == pytest.approx(shifts[at.ground_label(up)] - shifts[at.ground_label(down)])


def test_ensemble_selected_pair_dynamics_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "CACHE", tmp_path / "cache")
    c = ensemble_config((3, 2), (2, 0))
    c.initial.spin_preset = "raman_up"
    a = run(c)
    m, result = a["model"], a["result"]
    up, down = m.up, m.down
    np.testing.assert_allclose(a["series"]["P_down"] * result.P.sum(axis=(1, 2)), result.P[:, down].sum(axis=1))
    np.testing.assert_allclose(a["series"]["P_target_absolute"], result.P[:, up, 0])
    local, node = a["ensemble_nodes"][0]
    assert a["final"]["P_target_3d_absolute"] == pytest.approx(node["P_final_3d"][up, 0, 0, 0])
    # Isolate the coherent term: only the chosen pair may exchange population.
    local.terms = []
    local.model.cfg.spin_pump.enabled = local.model.cfg.repump.enabled = False
    with _single_thread():
        coherent = solve_local(local, c)
    np.testing.assert_allclose(coherent["P_final_3d"][m.spectators], 0, atol=1e-12)
    assert coherent["P_final_3d"][down].sum() > 1e-10
    export(a, tmp_path / "out", figures=False)
    with np.load(tmp_path / "out/run_joint.npz") as saved:
        np.testing.assert_array_equal(saved["raman_state_indices"], [up, down])
    metadata = json.loads((tmp_path / "out/run_metadata.json").read_text())
    assert metadata["model_summary"]["raman_state_down"] == [2, 0]
    from rb85rsc.plotting import diagnostics_text
    assert f"A={at.ground_label(up)} <-> B={at.ground_label(down)}" in diagnostics_text(a)


def test_gui_selects_pair_linear_polarization_and_roundtrips(tmp_path):
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QComboBox
    from rb85rsc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow(ensemble_config())
    try:
        for key, value in (("state_up", [3, 2]), ("state_down", [2, 0])):
            widget = window.fields[f"raman.{key}"].w
            assert isinstance(widget, QComboBox) and widget.count() == 12
            widget.setCurrentIndex(widget.findData(value))
        window.fields["raman.polarization_mode"].w.setCurrentText("jones")
        window.fields["raman.beam_low_ellipticity_deg"].w.setText("0")
        window.fields["raman.beam_high_ellipticity_deg"].w.setText("0")
        c = window.current_config()
        assert c.raman.state_up == [3, 2] and c.raman.state_down == [2, 0]
        c.to_json(tmp_path / "pair.json")
        window.load_config(SimConfig.from_json(tmp_path / "pair.json"))
        assert window.current_config().raman == c.raman
        assert window.fields["raman.beam_helicities"].w.isHidden()
        assert not window.fields["raman.beam_low_ellipticity_deg"].w.isHidden()
        # Older JSON still selects the original pair and circular input convention.
        window.load_config(SimConfig.from_json(ROOT / "examples/experiment.json"))
        old = window.current_config()
        assert old.raman.state_up == [3, 3] and old.raman.state_down == [2, 2]
        assert old.raman.polarization_mode == "helicity"
    finally:
        window.close()
