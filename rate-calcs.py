"""
mF-resolved optical pumping for 85Rb D2 using population rate equations.

This script models the 85Rb D2 manifolds
    ground: F=3, mF=-3..+3
    ground: F=2, mF=-2..+2
    excited: F'=3, mF'=-3..+3
driven by
    1) an optical pumping laser on F=3 -> F'=3
    2) a repump laser on F=2 -> F'=3

Important limitation:
This is a pure population-rate model. It does not include optical coherences,
ground-state coherences, or excited-state coherences, so coherent dark states,
CPT, EIT, and other interference effects are not captured here.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sympy import S
from sympy.physics.wigner import wigner_3j, wigner_6j


# -----------------------------------------------------------------------------
# User-editable example inputs
# -----------------------------------------------------------------------------

MHz = 2.0 * np.pi * 1.0e6  # angular frequency units
Gauss = 1.0e-4  # Tesla

SIMULATION_CONFIG = {
    "detuning_op": 0 * MHz,
    "detuning_rp": 0.0 * MHz,
    "intensity_op": 1.0,
    "intensity_rp": 1.0,
    "B_field": 2.35 * Gauss,
    "total_time": 120.0e-6,
    "num_time_points": 1200,
    "Gamma": 2.0 * np.pi * 6.065e6,
    "saturation_intensity": 1.0,
    "include_saturation_broadening": True,
    "population_drift_warning": 5.0e-7,
    "show_plots": False,
    "save_plots": True,
    "output_filename": "output.png",
    "detailed_output_filename": "detailed_output.md",
}

# Polarization amplitudes may be real or complex. The code uses |epsilon_q|^2
# and normalizes the three components so the listed laser intensity is the total
# intensity for that beam.
OPTICAL_PUMP_POLARIZATION = {
    "epsilon_plus": 1 + 0.0j,
    "epsilon_pi": 0.0 + 0.0j,
    "epsilon_minus": 0.0 + 0.0j,
}

REPUMP_POLARIZATION = {
    "epsilon_plus": 0.0 + 0.0j,
    "epsilon_pi": 1 + 0.0j,
    "epsilon_minus": 0.0 + 0.0j,
}

# Example initial condition: uniform across F=3, zero elsewhere.
# Edit this dictionary directly if you want a custom initial state. If left
# empty, the script uses the default uniform-F=3 preparation.
INITIAL_POPULATIONS: Dict[str, float] = {}

# Optional parameter sweep helper.
RUN_DETUNING_SWEEP = False
DETUNING_SWEEP = {
    "parameter": "detuning_op",
    "values": np.linspace(-30.0, 30.0, 121) * MHz,
    "tracked_states": ["|g, F=3, mF=+3>", "|g, F=3, mF=+2>", "|g, F=2, mF=+2>"],
}


# -----------------------------------------------------------------------------
# Atomic constants and data containers
# -----------------------------------------------------------------------------

I_RB85 = S(5) / 2
J_GROUND = S(1) / 2
J_EXCITED = S(3) / 2
F_EXCITED = 3
GROUND_F_VALUES = (2, 3)

GJ_5S12 = 2.00233113
GJ_5P32 = 1.3362
GI_RB85 = 0.54136  # nuclear g factor in nuclear-magneton units
MU_N_OVER_MU_B = 5.446170214e-4
MU_B_OVER_HBAR = 2.0 * np.pi * 13.99624555e9  # rad / (s*T)


@dataclass(frozen=True)
class State:
    """A single Zeeman sublevel in the model."""

    manifold: str  # "g" or "e"
    F: int
    m: int
    index: int

    @property
    def label(self) -> str:
        return state_label(self.manifold, self.F, self.m)

    @property
    def key(self) -> str:
        return state_key(self.manifold, self.F, self.m)


@dataclass(frozen=True)
class Laser:
    """Laser parameters for one hyperfine pumping beam."""

    name: str
    ground_F: int
    detuning: float
    intensity: float
    epsilon_plus: complex
    epsilon_pi: complex
    epsilon_minus: complex

    def polarization_weights(self) -> Dict[int, float]:
        """Return normalized power fractions for q=+1,0,-1."""
        raw = {
            +1: abs(self.epsilon_plus) ** 2,
            0: abs(self.epsilon_pi) ** 2,
            -1: abs(self.epsilon_minus) ** 2,
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {+1: 0.0, 0: 0.0, -1: 0.0}
        return {q: value / total for q, value in raw.items()}


@dataclass(frozen=True)
class Channel:
    """One laser-driven absorption channel g -> e."""

    laser_name: str
    ground_index: int
    excited_index: int
    q: int
    bare_strength: float
    weighted_strength: float
    saturation_parameter: float
    detuning: float
    rate: float


@dataclass(frozen=True)
class DecayChannel:
    """One spontaneous decay channel e -> g."""

    excited_index: int
    ground_index: int
    bare_strength: float
    branching_ratio: float
    rate: float


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def state_label(manifold: str, F: int, m: int) -> str:
    """Return a user-facing ket-style label such as |g, F=3, mF=+2>."""
    return f"|{manifold}, F={F}, mF={m:+d}>"


def state_key(manifold: str, F: int, m: int) -> str:
    """Return a compact machine-friendly key such as g_F3_m+2."""
    return f"{manifold}_F{F}_m{m:+d}"


def clean_small(value: float, threshold: float = 1.0e-12) -> float:
    """Suppress tiny numerical roundoff in printed output."""
    return 0.0 if abs(value) < threshold else value


def linear_gF(F: int, J: S, gJ: float, gI_nuclear: float) -> float:
    """Hyperfine g_F including the small nuclear term."""
    Ff = float(F)
    Jf = float(J)
    If = float(I_RB85)
    numerator_e = Ff * (Ff + 1.0) + Jf * (Jf + 1.0) - If * (If + 1.0)
    numerator_n = Ff * (Ff + 1.0) + If * (If + 1.0) - Jf * (Jf + 1.0)
    denominator = 2.0 * Ff * (Ff + 1.0)
    return gJ * numerator_e / denominator + gI_nuclear * MU_N_OVER_MU_B * numerator_n / denominator


def zeeman_shift_rad_s(state: State, B_field: float, gF_map: Dict[Tuple[str, int], float]) -> float:
    """Linear Zeeman shift in angular-frequency units."""
    return MU_B_OVER_HBAR * gF_map[(state.manifold, state.F)] * state.m * B_field


@lru_cache(maxsize=None)
def dipole_strength(F_ground: int, m_ground: int, F_excited: int, m_excited: int) -> float:
    """
    Relative dipole line strength |<F',m'|d_q|F,m>|^2 up to an overall constant.

    The common electronic reduced matrix element is omitted because it cancels in
    branching-ratio normalization and is absorbed into the saturation scale.
    """
    q = m_excited - m_ground
    if abs(q) > 1 or abs(m_ground) > F_ground or abs(m_excited) > F_excited:
        return 0.0

    angular = (
        (2 * F_excited + 1)
        * (2 * F_ground + 1)
        * wigner_6j(J_EXCITED, F_excited, I_RB85, F_ground, J_GROUND, 1) ** 2
        * wigner_3j(F_excited, 1, F_ground, -m_excited, q, m_ground) ** 2
    )
    return float(angular)


def build_states() -> Tuple[List[State], List[State], List[State]]:
    """Create all ground and excited Zeeman sublevels."""
    states: List[State] = []
    ground_states: List[State] = []
    excited_states: List[State] = []

    index = 0
    for F in (3, 2):
        for m in range(-F, F + 1):
            state = State("g", F, m, index)
            states.append(state)
            ground_states.append(state)
            index += 1

    for m in range(-F_EXCITED, F_EXCITED + 1):
        state = State("e", F_EXCITED, m, index)
        states.append(state)
        excited_states.append(state)
        index += 1

    return states, ground_states, excited_states


def default_initial_populations(states: Sequence[State]) -> np.ndarray:
    """Uniform population over the F=3 ground manifold."""
    initial = np.zeros(len(states))
    f3_indices = [state.index for state in states if state.manifold == "g" and state.F == 3]
    for index in f3_indices:
        initial[index] = 1.0 / len(f3_indices)
    return initial


def parse_initial_populations(states: Sequence[State], user_values: Dict[str, float]) -> np.ndarray:
    """Create a normalized initial population vector from a label:value map."""
    if not user_values:
        return default_initial_populations(states)

    label_to_index = {state.label: state.index for state in states}
    label_to_index.update({state.key: state.index for state in states})
    initial = np.zeros(len(states))
    for label, value in user_values.items():
        if label not in label_to_index:
            raise KeyError(f"Unknown state label in INITIAL_POPULATIONS: {label}")
        initial[label_to_index[label]] = float(value)

    total = np.sum(initial)
    if total <= 0.0:
        raise ValueError("INITIAL_POPULATIONS must sum to a positive value.")
    return initial / total


def build_lasers(config: Dict[str, float]) -> List[Laser]:
    """Construct the two lasers from the user-editable inputs."""
    return [
        Laser(
            name="optical_pump",
            ground_F=3,
            detuning=config["detuning_op"],
            intensity=config["intensity_op"],
            epsilon_plus=OPTICAL_PUMP_POLARIZATION["epsilon_plus"],
            epsilon_pi=OPTICAL_PUMP_POLARIZATION["epsilon_pi"],
            epsilon_minus=OPTICAL_PUMP_POLARIZATION["epsilon_minus"],
        ),
        Laser(
            name="repump",
            ground_F=2,
            detuning=config["detuning_rp"],
            intensity=config["intensity_rp"],
            epsilon_plus=REPUMP_POLARIZATION["epsilon_plus"],
            epsilon_pi=REPUMP_POLARIZATION["epsilon_pi"],
            epsilon_minus=REPUMP_POLARIZATION["epsilon_minus"],
        ),
    ]


def max_absorption_strength() -> float:
    """Reference dipole strength used to map intensity to an effective s parameter."""
    strengths = []
    for F_ground in GROUND_F_VALUES:
        for m_ground in range(-F_ground, F_ground + 1):
            for q in (-1, 0, +1):
                m_excited = m_ground + q
                if abs(m_excited) <= F_EXCITED:
                    strengths.append(dipole_strength(F_ground, m_ground, F_EXCITED, m_excited))
    return max(strengths)


def build_absorption_channels(
    states: Sequence[State],
    lasers: Sequence[Laser],
    gF_map: Dict[Tuple[str, int], float],
    config: Dict[str, float],
) -> Tuple[List[Channel], Dict[int, float]]:
    """
    Build all laser-driven excitation channels.

    The rate model is a saturation-broadened Lorentzian approximation:
        W_ge = (Gamma / 2) * s_ge / (1 + s_total(g) + (2*Delta_ge/Gamma)^2)
    where s_ge is channel-specific and s_total(g) is the summed saturation
    parameter out of that ground state for the addressed laser.
    """
    key_to_state = {state.key: state for state in states}
    reference_strength = max_absorption_strength()
    saturation_intensity = config["saturation_intensity"]
    Gamma = config["Gamma"]
    include_sat = bool(config["include_saturation_broadening"])

    channels: List[Channel] = []
    total_excitation_out: Dict[int, float] = {state.index: 0.0 for state in states if state.manifold == "g"}

    for laser in lasers:
        if laser.intensity <= 0.0:
            continue

        polarization_weights = laser.polarization_weights()
        ground_states = [state for state in states if state.manifold == "g" and state.F == laser.ground_F]
        s_total_ground: Dict[int, float] = {}

        for ground in ground_states:
            s_total = 0.0
            for q, pol_fraction in polarization_weights.items():
                if pol_fraction <= 0.0:
                    continue
                m_excited = ground.m + q
                if abs(m_excited) > F_EXCITED:
                    continue
                bare_strength = dipole_strength(ground.F, ground.m, F_EXCITED, m_excited)
                if bare_strength <= 0.0:
                    continue
                s_total += (laser.intensity / saturation_intensity) * pol_fraction * (bare_strength / reference_strength)
            s_total_ground[ground.index] = s_total

        for ground in ground_states:
            for q, pol_fraction in polarization_weights.items():
                if pol_fraction <= 0.0:
                    continue
                m_excited = ground.m + q
                if abs(m_excited) > F_EXCITED:
                    continue

                excited = key_to_state[state_key("e", F_EXCITED, m_excited)]
                bare_strength = dipole_strength(ground.F, ground.m, F_EXCITED, excited.m)
                if bare_strength <= 0.0:
                    continue

                weighted_strength = pol_fraction * bare_strength
                s_channel = (laser.intensity / saturation_intensity) * (weighted_strength / reference_strength)
                detuning = (
                    laser.detuning
                    + zeeman_shift_rad_s(ground, config["B_field"], gF_map)
                    - zeeman_shift_rad_s(excited, config["B_field"], gF_map)
                )
                s_total = s_total_ground[ground.index] if include_sat else 0.0
                denominator = 1.0 + s_total + (2.0 * detuning / Gamma) ** 2
                rate = 0.5 * Gamma * s_channel / denominator

                channel = Channel(
                    laser_name=laser.name,
                    ground_index=ground.index,
                    excited_index=excited.index,
                    q=q,
                    bare_strength=bare_strength,
                    weighted_strength=weighted_strength,
                    saturation_parameter=s_channel,
                    detuning=detuning,
                    rate=rate,
                )
                channels.append(channel)
                total_excitation_out[ground.index] += rate

    return channels, total_excitation_out


def build_decay_channels(states: Sequence[State], Gamma: float) -> Tuple[List[DecayChannel], Dict[int, float]]:
    """Build spontaneous-emission channels and branching ratios."""
    ground_states = [state for state in states if state.manifold == "g"]
    excited_states = [state for state in states if state.manifold == "e"]

    channels: List[DecayChannel] = []
    branching_sums: Dict[int, float] = {}

    for excited in excited_states:
        raw_channels: List[Tuple[State, float]] = []
        for ground in ground_states:
            raw_strength = dipole_strength(ground.F, ground.m, excited.F, excited.m)
            if raw_strength > 0.0:
                raw_channels.append((ground, raw_strength))

        total_strength = sum(strength for _, strength in raw_channels)
        branching_sums[excited.index] = 0.0

        if total_strength <= 0.0:
            raise RuntimeError(f"No decay channels found for excited state {excited.label}")

        for ground, raw_strength in raw_channels:
            branching_ratio = raw_strength / total_strength
            channel = DecayChannel(
                excited_index=excited.index,
                ground_index=ground.index,
                bare_strength=raw_strength,
                branching_ratio=branching_ratio,
                rate=Gamma * branching_ratio,
            )
            channels.append(channel)
            branching_sums[excited.index] += branching_ratio

    return channels, branching_sums


def build_rate_matrix(
    states: Sequence[State],
    absorption_channels: Sequence[Channel],
    decay_channels: Sequence[DecayChannel],
) -> np.ndarray:
    """Assemble the linear rate matrix dp/dt = M p."""
    matrix = np.zeros((len(states), len(states)))

    for channel in absorption_channels:
        matrix[channel.excited_index, channel.ground_index] += channel.rate
        matrix[channel.ground_index, channel.ground_index] -= channel.rate

    for channel in decay_channels:
        matrix[channel.ground_index, channel.excited_index] += channel.rate
        matrix[channel.excited_index, channel.excited_index] -= channel.rate

    return matrix


def solve_populations(
    rate_matrix: np.ndarray,
    initial_populations: np.ndarray,
    total_time: float,
    num_time_points: int,
) -> solve_ivp:
    """Integrate the linear population ODE system."""
    times = np.linspace(0.0, total_time, num_time_points)

    solution = solve_ivp(
        fun=lambda _t, populations: rate_matrix @ populations,
        t_span=(0.0, total_time),
        y0=initial_populations,
        t_eval=times,
        method="DOP853",
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solve failed: {solution.message}")
    return solution


def fluorescence_rate(solution_y: np.ndarray, excited_indices: Sequence[int], Gamma: float) -> np.ndarray:
    """Total spontaneous emission rate Gamma * sum_e p_e(t)."""
    return Gamma * np.sum(solution_y[list(excited_indices), :], axis=0)


def print_configuration_summary(lasers: Sequence[Laser], gF_map: Dict[Tuple[str, int], float], config: Dict[str, float]) -> None:
    """Print the key physical inputs in a readable form."""
    print("\n=== Configuration Summary ===")
    print(f"Gamma / (2 pi)                  = {config['Gamma'] / (2.0 * np.pi * 1e6): .6f} MHz")
    print(f"Optical-pump detuning / (2 pi)  = {config['detuning_op'] / (2.0 * np.pi * 1e6): .6f} MHz")
    print(f"Repump detuning / (2 pi)        = {config['detuning_rp'] / (2.0 * np.pi * 1e6): .6f} MHz")
    print(f"B field                         = {config['B_field'] / Gauss: .6f} G")
    print(f"Saturation intensity scale      = {config['saturation_intensity']:.6g} (arbitrary units)")
    print(f"Include sat broadening          = {config['include_saturation_broadening']}")
    print(f"gF(F=3, ground)                 = {gF_map[('g', 3)]: .8f}")
    print(f"gF(F=2, ground)                 = {gF_map[('g', 2)]: .8f}")
    print(f"gF(F'=3, excited)               = {gF_map[('e', 3)]: .8f}")

    for laser in lasers:
        weights = laser.polarization_weights()
        print(f"\nLaser: {laser.name}")
        print(f"  addressed manifold           = F={laser.ground_F} -> F'=3")
        print(f"  intensity                    = {laser.intensity:.6g} (same units as I_sat)")
        print(f"  sigma+ power fraction        = {weights[+1]:.6f}")
        print(f"  pi power fraction            = {weights[0]:.6f}")
        print(f"  sigma- power fraction        = {weights[-1]:.6f}")


def print_branching_checks(states: Sequence[State], branching_sums: Dict[int, float]) -> None:
    """Sanity-check the spontaneous branching normalization."""
    print("\n=== Branching-Ratio Sanity Check ===")
    max_error = 0.0
    for excited in [state for state in states if state.manifold == "e"]:
        total = branching_sums[excited.index]
        error = abs(total - 1.0)
        max_error = max(max_error, error)
        print(f"{excited.label}: sum(branching ratios) = {total:.12f}")
    print(f"Maximum branching-ratio normalization error = {max_error:.3e}")


def print_dark_states(states: Sequence[State], excitation_out_rates: Dict[int, float]) -> None:
    """
    Identify ground states with zero excitation out of them for the chosen
    laser intensities and polarization mix.
    """
    print("\n=== Dark / Unaddressed Ground States ===")
    dark_labels = [
        state.label
        for state in states
        if state.manifold == "g" and excitation_out_rates.get(state.index, 0.0) <= 1.0e-18
    ]
    if dark_labels:
        for label in dark_labels:
            print(f"{label}: zero excitation rate in this rate model")
    else:
        print("No exactly dark ground states found from the chosen laser couplings.")


def print_final_populations(states: Sequence[State], final_populations: np.ndarray) -> None:
    """Print all final populations and manifold totals."""
    print("\n=== Final Populations ===")
    for state in states:
        print(f"{state.label}: {clean_small(final_populations[state.index]):.10f}")

    total_f3 = sum(final_populations[state.index] for state in states if state.manifold == "g" and state.F == 3)
    total_f2 = sum(final_populations[state.index] for state in states if state.manifold == "g" and state.F == 2)
    total_excited = sum(final_populations[state.index] for state in states if state.manifold == "e")
    print("\n=== Final Manifold Totals ===")
    print(f"Total F=3 ground   = {clean_small(total_f3):.10f}")
    print(f"Total F=2 ground   = {clean_small(total_f2):.10f}")
    print(f"Total excited      = {clean_small(total_excited):.10f}")
    print(f"Grand total        = {clean_small(total_f3 + total_f2 + total_excited):.10f}")


def print_population_conservation(times: np.ndarray, populations: np.ndarray, warning_threshold: float) -> None:
    """Report the numerical drift in the total population."""
    totals = np.sum(populations, axis=0)
    max_drift = float(np.max(np.abs(totals - 1.0)))
    print("\n=== Population-Conservation Check ===")
    print(f"Initial total population       = {totals[0]:.12f}")
    print(f"Final total population         = {totals[-1]:.12f}")
    print(f"Maximum |sum(p)-1| over time   = {max_drift:.3e}")
    if max_drift > warning_threshold:
        print("WARNING: total population drift exceeded the chosen warning threshold.")


def describe_transition_network(states: Sequence[State], absorption_channels: Sequence[Channel]) -> None:
    """Print the allowed laser-driven transitions that have nonzero rate."""
    q_name = {+1: "sigma+", 0: "pi", -1: "sigma-"}
    print("\n=== Laser-Driven Transition Network ===")
    for channel in absorption_channels:
        ground = states[channel.ground_index]
        excited = states[channel.excited_index]
        print(
            f"{channel.laser_name}: {ground.label} -> {excited.label}  "
            f"{q_name[channel.q]}  rate = {channel.rate: .6e} s^-1  "
            f"Delta/(2pi) = {channel.detuning / (2.0 * np.pi * 1e6): .6f} MHz"
        )


def write_detailed_output(
    filename: str,
    states: Sequence[State],
    lasers: Sequence[Laser],
    gF_map: Dict[Tuple[str, int], float],
    config: Dict[str, float],
    branching_sums: Dict[int, float],
    excitation_out_rates: Dict[int, float],
    absorption_channels: Sequence[Channel],
    solution: solve_ivp,
) -> None:
    """Write the detailed textual diagnostics to a markdown file."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_configuration_summary(lasers, gF_map, config)
        print_branching_checks(states, branching_sums)
        print_dark_states(states, excitation_out_rates)
        describe_transition_network(states, absorption_channels)
        print_population_conservation(solution.t, solution.y, config["population_drift_warning"])
        print_final_populations(states, solution.y[:, -1])

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("# Detailed Output\n\n")
        handle.write("```text\n")
        handle.write(buffer.getvalue().lstrip("\n"))
        handle.write("\n```\n")


def plot_level_structure(
    states: Sequence[State],
    absorption_channels: Sequence[Channel],
    gF_map: Dict[Tuple[str, int], float],
    config: Dict[str, float],
) -> None:
    """Draw a level-structure diagram with x as mF and y as energy."""
    laser_colors = {
        "optical_pump": "#c81d25",
        "repump": "#2f9e44",
    }
    B_field = config["B_field"]
    manifold_offsets_mhz = {
        ("g", 2): -220.0,
        ("g", 3): 0.0,
        ("e", 3): 320.0,
    }
    x_offsets = {
        ("g", 2): -0.14,
        ("g", 3): +0.14,
        ("e", 3): 0.0,
    }
    manifold_colors = {
        ("g", 2): "tab:orange",
        ("g", 3): "tab:blue",
        ("e", 3): "tab:green",
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for state in states:
        x_center = state.m + x_offsets[(state.manifold, state.F)]
        y = manifold_offsets_mhz[(state.manifold, state.F)] + zeeman_shift_rad_s(state, B_field, gF_map) / MHz
        ax.plot(
            [x_center - 0.12, x_center + 0.12],
            [y, y],
            color=manifold_colors[(state.manifold, state.F)],
            lw=2.0,
        )
        ax.text(x_center, y - 18.0, rf"$m_F={state.m:+d}$", ha="center", va="top", fontsize=10)

    max_rate = max((channel.rate for channel in absorption_channels), default=1.0)
    laser_legend_drawn = set()
    for channel in absorption_channels:
        ground = states[channel.ground_index]
        excited = states[channel.excited_index]
        x_ground = ground.m + x_offsets[(ground.manifold, ground.F)]
        x_excited = excited.m + x_offsets[(excited.manifold, excited.F)]
        y_ground = manifold_offsets_mhz[(ground.manifold, ground.F)] + zeeman_shift_rad_s(ground, B_field, gF_map) / MHz
        y_excited = manifold_offsets_mhz[(excited.manifold, excited.F)] + zeeman_shift_rad_s(excited, B_field, gF_map) / MHz
        linewidth = 0.8 + 3.2 * np.sqrt(channel.rate / max_rate) if max_rate > 0.0 else 1.0
        polarization_fraction = channel.weighted_strength / channel.bare_strength if channel.bare_strength > 0.0 else 0.0
        alpha = 0.15 + 0.85 * polarization_fraction
        label = channel.laser_name.replace("_", " ") if channel.laser_name not in laser_legend_drawn else None
        ax.plot(
            [x_ground, x_excited],
            [y_ground, y_excited],
            color=laser_colors[channel.laser_name],
            lw=linewidth,
            alpha=alpha,
            linestyle="-",
            label=label,
        )
        laser_legend_drawn.add(channel.laser_name)

    y_values = [
        manifold_offsets_mhz[(state.manifold, state.F)] + zeeman_shift_rad_s(state, B_field, gF_map) / MHz
        for state in states
    ]
    ax.set_xlim(-3.85, 3.85)
    ax.set_ylim(min(y_values) - 45.0, max(y_values) + 55.0)
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Laser coupling diagram in the |g/e, F, mF> basis")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.text(-3.55, manifold_offsets_mhz[("g", 2)], r"$F=2$", color=manifold_colors[("g", 2)], fontsize=11, va="center")
    ax.text(-3.55, manifold_offsets_mhz[("g", 3)], r"$F=3$", color=manifold_colors[("g", 3)], fontsize=11, va="center")
    ax.text(-3.55, manifold_offsets_mhz[("e", 3)], r"$F'=3$", color=manifold_colors[("e", 3)], fontsize=11, va="center")

    plt.tight_layout()


def plot_population_dynamics(states: Sequence[State], solution: solve_ivp, config: Dict[str, float]) -> None:
    """Generate the ground-state population plots."""
    times_us = solution.t * 1.0e6
    populations = solution.y

    ground_f3 = [state for state in states if state.manifold == "g" and state.F == 3]
    ground_f2 = [state for state in states if state.manifold == "g" and state.F == 2]

    fig_ground, axes_ground = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for state in ground_f3:
        axes_ground[0].plot(times_us, populations[state.index], label=state.label)
    for state in ground_f2:
        axes_ground[1].plot(times_us, populations[state.index], label=state.label)
    axes_ground[0].set_ylabel("Population")
    axes_ground[0].set_xlabel("Time (us)")
    axes_ground[1].set_xlabel("Time (us)")
    axes_ground[0].set_title(r"Ground $F=3$ populations vs time")
    axes_ground[1].set_title(r"Ground $F=2$ populations vs time")
    axes_ground[0].legend(ncol=2, fontsize=8)
    axes_ground[1].legend(ncol=2, fontsize=8)
    axes_ground[0].grid(alpha=0.25)
    axes_ground[1].grid(alpha=0.25)
    fig_ground.tight_layout()

def run_detuning_sweep(
    states: Sequence[State],
    gF_map: Dict[Tuple[str, int], float],
    base_config: Dict[str, float],
    tracked_states: Sequence[str],
    sweep_values: Sequence[float],
    parameter_name: str,
) -> None:
    """Optional helper: sweep one parameter and plot final populations."""
    label_to_index = {state.label: state.index for state in states}
    label_to_index.update({state.key: state.index for state in states})
    missing = [label for label in tracked_states if label not in label_to_index]
    if missing:
        raise KeyError(f"Tracked states not found in sweep configuration: {missing}")

    final_values = {label: [] for label in tracked_states}
    sweep_axis_mhz = np.asarray(sweep_values) / MHz

    for value in sweep_values:
        config = dict(base_config)
        config[parameter_name] = float(value)
        lasers = build_lasers(config)
        absorption_channels, _ = build_absorption_channels(states, lasers, gF_map, config)
        decay_channels, _ = build_decay_channels(states, config["Gamma"])
        rate_matrix = build_rate_matrix(states, absorption_channels, decay_channels)
        initial = parse_initial_populations(states, INITIAL_POPULATIONS)
        solution = solve_populations(rate_matrix, initial, config["total_time"], config["num_time_points"])
        final = solution.y[:, -1]
        for label in tracked_states:
            final_values[label].append(final[label_to_index[label]])

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in final_values.items():
        ax.plot(sweep_axis_mhz, values, label=label)
    ax.set_xlabel(f"{parameter_name} / (2 pi MHz)")
    ax.set_ylabel("Final population")
    ax.set_title("Optional detuning sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

def save_open_figures_as_single_png(filename: str, dpi: int = 180, padding_px: int = 24) -> None:
    """Render all currently open matplotlib figures into one stacked PNG."""
    figure_numbers = plt.get_fignums()
    if not figure_numbers:
        return

    rendered_images = []
    max_width = 0
    total_height = 0

    for figure_number in figure_numbers:
        figure = plt.figure(figure_number)
        figure.set_dpi(dpi)
        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        rendered_images.append(image)
        height, width = image.shape[:2]
        max_width = max(max_width, width)
        total_height += height

    total_height += padding_px * (len(rendered_images) - 1)
    combined = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    y_offset = 0
    for image in rendered_images:
        height, width = image.shape[:2]
        x_offset = (max_width - width) // 2
        combined[y_offset : y_offset + height, x_offset : x_offset + width] = image
        y_offset += height + padding_px

    plt.imsave(filename, combined)


def main() -> None:
    """Run the optical-pumping simulation with the example configuration."""
    states, _, excited_states = build_states()

    gF_map = {
        ("g", 3): linear_gF(3, J_GROUND, GJ_5S12, GI_RB85),
        ("g", 2): linear_gF(2, J_GROUND, GJ_5S12, GI_RB85),
        ("e", 3): linear_gF(3, J_EXCITED, GJ_5P32, GI_RB85),
    }

    config = dict(SIMULATION_CONFIG)
    lasers = build_lasers(config)
    initial = parse_initial_populations(states, INITIAL_POPULATIONS)

    absorption_channels, excitation_out_rates = build_absorption_channels(states, lasers, gF_map, config)
    decay_channels, branching_sums = build_decay_channels(states, config["Gamma"])
    rate_matrix = build_rate_matrix(states, absorption_channels, decay_channels)
    solution = solve_populations(rate_matrix, initial, config["total_time"], config["num_time_points"])

    write_detailed_output(
        filename=config["detailed_output_filename"],
        states=states,
        lasers=lasers,
        gF_map=gF_map,
        config=config,
        branching_sums=branching_sums,
        excitation_out_rates=excitation_out_rates,
        absorption_channels=absorption_channels,
        solution=solution,
    )

    plot_level_structure(states, absorption_channels, gF_map, config)
    plot_population_dynamics(states, solution, config)

    if RUN_DETUNING_SWEEP:
        run_detuning_sweep(
            states=states,
            gF_map=gF_map,
            base_config=config,
            tracked_states=DETUNING_SWEEP["tracked_states"],
            sweep_values=DETUNING_SWEEP["values"],
            parameter_name=DETUNING_SWEEP["parameter"],
        )

    if config["save_plots"]:
        save_open_figures_as_single_png(config["output_filename"])
        print(f"Saved plots to {config['output_filename']}")

    print(f"Saved detailed report to {config['detailed_output_filename']}")

    if config["show_plots"]:
        if "agg" in plt.get_backend().lower():
            print("\nPlots were generated with a non-interactive matplotlib backend, so no GUI window was opened.")
        else:
            plt.show()


if __name__ == "__main__":
    main()
