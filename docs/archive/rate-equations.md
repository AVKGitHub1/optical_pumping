# Optical Pumping Rate-Equation Model

> Archived documentation for the separate population-rate model. For the current RSC experiment, start with the [repository README](../../README.md) and [experiment workflow](../experiment-workflow.md). Commands below run from the repository root.

This repository contains a single self-contained Python script, [rate-calcs.py](../../scripts/archive/rate-calcs.py), that simulates mF-resolved optical pumping of `85Rb` on the D2 line using population rate equations.

The model includes:

- Ground manifolds `F=3` and `F=2`
- Excited D2 hyperfine manifolds `F'=1,2,3,4`
- A nominal optical-pump laser on `F=3 -> F'=3`
- A nominal repump laser on `F=2 -> F'=3`
- Off-resonant excitation by both lasers into other allowed hyperfine states
- Spontaneous decay back into both ground hyperfine manifolds
- Optional Zeeman shifts from a bias magnetic field along the quantization axis

The script saves:

- [output.png](../../results/archive/rate-calcs/output.png): figure dashboard
- [detailed_output.md](rate-equations-output.md): detailed text report

For the physics background, see [physics.md](rate-equations-physics.md).

## Requirements

Install:

- `numpy`
- `scipy`
- `sympy`
- `matplotlib`

Example:

```bash
conda run --no-capture-output -n scripts python -m pip install numpy scipy sympy matplotlib
```

## Running

Run:

```bash
conda run --no-capture-output -n scripts python scripts/archive/rate-calcs.py
```

By default, the script writes `output.png` and `detailed_output.md` in the working directory instead of opening interactive plot windows. The links above refer to preserved historical output. The root `rate-calcs.py` entry point remains available for compatibility.

## What The Script Does

At a high level, the script:

1. Builds the full state list of ground and excited Zeeman sublevels.
2. Computes dipole line strengths from Wigner `3j` and `6j` symbols.
3. Builds laser-driven absorption channels for all allowed `Delta mF = q` transitions.
4. Includes off-resonant excitation to non-target D2 hyperfine states.
5. Computes spontaneous-emission branching ratios from the same angular factors.
6. Assembles a linear ODE system `dp/dt = M p`.
7. Solves the ODE with `scipy.integrate.solve_ivp`.
8. Writes a detailed report and saves a summary figure.

## Main User Inputs

Most settings live near the top of [rate-calcs.py](../../scripts/archive/rate-calcs.py).

### 1. Hyperfine offsets

- `GROUND_HYPERFINE_OFFSETS_MHZ`
- `EXCITED_HYPERFINE_OFFSETS_MHZ`

These define the zero-field hyperfine splittings used for off-resonant detunings.

### 2. Included vs displayed excited manifolds

- `INCLUDED_EXCITED_F_VALUES`
- `DISPLAYED_EXCITED_F_VALUES`

`INCLUDED_EXCITED_F_VALUES` controls the actual physics model.

`DISPLAYED_EXCITED_F_VALUES` controls what is shown in the level-structure diagram.

At the moment, the model includes `F'=1,2,3,4`, but the diagram only shows `F'=3`.

### 3. Simulation config

`SIMULATION_CONFIG` contains:

- laser detunings
- laser intensities
- `B_field`
- total simulation time
- number of time points
- natural linewidth `Gamma`
- saturation scale
- plotting/output options

The script uses angular-frequency units internally, so the convenient constant `MHz = 2*pi*1e6` is defined at the top.

### 4. Polarizations

- `OPTICAL_PUMP_POLARIZATION`
- `REPUMP_POLARIZATION`

Each beam is specified by spherical polarization amplitudes:

- `epsilon_plus`
- `epsilon_pi`
- `epsilon_minus`

The code converts them into normalized power fractions using `|epsilon_q|^2`.

### 5. Initial populations

`INITIAL_POPULATIONS` can be left empty for the default uniform `F=3` preparation, or filled with state labels such as:

```python
INITIAL_POPULATIONS = {
    "|g, F=3, mF=+3>": 1.0,
}
```

Compact keys like `g_F3_m+3` are also accepted.

### 6. Sweep settings

The optional parameter scan is controlled by:

- `RUN_SWEEP`
- `OP_DETUNING_PARAMS`
- `RP_POWER_PARAMS`
- `SWEEP_SCAN_PARAMS`

If `RUN_SWEEP = False`, the saved dashboard contains:

- left: level structure
- right top: `F=3` populations vs time
- right bottom: `F=2` populations vs time

If `RUN_SWEEP = True`, the right column is split into:

- top: `F=3`
- middle: `F=2`
- bottom: sweep scan

## Outputs

### `output.png`

This is the main visual summary.

It includes:

- a level-structure diagram for the displayed nominal couplings
- time traces of `F=3` populations
- time traces of `F=2` populations
- optionally a sweep plot

Important:

- The physics model includes off-resonant couplings.
- The level diagram intentionally hides those off-resonant couplings for clarity.

### `detailed_output.md`

This report contains:

- configuration summary
- branching-ratio checks
- dark-state/unaddressed-state checks
- displayed transition list
- population-conservation diagnostics
- final populations

## Code Structure

The script is organized in a fairly direct way.

### State and laser representation

- `State`
- `Laser`
- `Channel`
- `DecayChannel`

### Physics helpers

- `linear_gF`
- `dipole_strength`
- `transition_offset_rad_s`
- `zeeman_shift_rad_s`

### Model construction

- `build_states`
- `build_lasers`
- `build_absorption_channels`
- `build_decay_channels`
- `build_rate_matrix`

### Solving and diagnostics

- `solve_populations`
- `write_detailed_output`

### Plotting

- `plot_level_structure`
- `plot_population_dynamics`
- `run_sweep`
- `save_dashboard_png`

## Important Limitations

This is a population-only rate model.

It does not include:

- optical coherences
- ground-state coherences
- excited-state coherences
- coherent dark states
- CPT / EIT physics
- optical Bloch dynamics

So it is a good tool for incoherent pumping and leakage estimates, but not for coherence-driven phenomena.

## Typical Edits

Common things to change:

- detunings: `SIMULATION_CONFIG["detuning_op"]`, `SIMULATION_CONFIG["detuning_rp"]`
- powers: `SIMULATION_CONFIG["intensity_op"]`, `SIMULATION_CONFIG["intensity_rp"]`
- field: `SIMULATION_CONFIG["B_field"]`
- polarization impurity: `OPTICAL_PUMP_POLARIZATION`, `REPUMP_POLARIZATION`
- simulation duration: `SIMULATION_CONFIG["total_time"]`
- whether to scan: `RUN_SWEEP`

## Suggested Workflow

1. Start with `RUN_SWEEP = False` and verify the time evolution.
2. Adjust beam polarizations and detunings.
3. Check `detailed_output.md` for dark states and branching normalization.
4. Turn on `RUN_SWEEP` once the single-run behavior looks sensible.

