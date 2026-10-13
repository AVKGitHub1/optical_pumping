# Three-dimensional coherent optical pumping

The separate [trajectory solver](../rb85rsc/coherent_pumping.py) retains motional
coherences throughout pumping on all three lattice axes. It supports the same
local bound spectra, optical atomic model, cloud positions, dipole patterns,
standing-wave lattice scattering, and absorbing escape as the correlated
population solver. **Raman light must be off for the entire schedule.** Background
heating must also be zero. It is available in the GUI, through the normal headless
entry point, and through the comparison command below. The default population
backend continues to support Raman with population-resolving recoil.

## Why this addresses the motional-secular warning

The population model discards off-diagonal motional density-matrix elements
after every photon. The diagnostic `max optical rate / smallest local motional
angular spacing` checks whether this reduction has a clear separation of time
scales. Its value is about 2.84 for the corrected `onlyOP` example.

Here, a trajectory contains a resolved ground spin and three complex motional
wavefunctions. A photon applies its combined absorption/emission displacement
to those wavefunctions; subsequent evolution retains their phases. Each emitted
photon has one shared direction for all axes. Averaging trajectories retains
correlations among spin, motion, and loss. The ensemble density matrix is not
assumed to factorize.

The product form of an individual trajectory is preserved for the supported
model: the Hamiltonian separates by axis, a traveling-wave jump is a product of
three kicks, and interference between the two beams of a lattice occurs on one
axis. Resolving a final spin leaves a product motional state. Coherent Raman
coupling would generally break this representation, so it is explicitly rejected.

## Jump and escape accounting

For traveling-wave pumping, a bound recoil operator on axis `a` is
`P_a exp[i(k_abs,a - k_emit u_a)x_a] P_a`, with `P_a` the projector onto that
axis's bound states. Its missing norm contributes to escape.

For a lattice with equal counterpropagating amplitudes, absorption is
`A = exp(ikx) + exp(-ikx)`. The full absorption hazard is

```text
F = P_bound A† A P_bound = 2 I + D(2k) + D(-2k).
```

This includes off-diagonal elements of `F`. It is not replaced by the norm of
the retained bound jump, which would incorrectly discard escape channels.
The conditional Hamiltonian contains `-i Gamma/2`, where `Gamma` is the full
photon-rate operator. Each retained jump uses the sum
`D(k - k_emit u) + D(-k - k_emit u)` on the absorption axis, and the emission
kick on the other axes. The ratio of retained jump norm to full absorption norm
sets the survival probability. A photon causing escape is counted once.

Poisson thinning samples candidate times using a spectral upper bound on the
instantaneous rate. Exact matrix exponentials, evaluated by eigendecomposition,
propagate the conditional states between events. Rejected proposals preserve
the conditionally evolved state. There is no integration timestep; output
sampling sets recorded times. Changing output times can reassign random draws
among trajectories, so an identical seed does not imply an identical Monte
Carlo realization across different output grids.

The effective-jump reduction follows [Reiter and Sorensen](https://arxiv.org/abs/1112.2806);
the conditional evolution and jump framework is described in
[QuTiP's Monte Carlo guide](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html).
The implementation uses NumPy/SciPy and an independent unravelling of the
repository's reduced optical model.

## Run and inspect

In the GUI, load `examples/onlyOP_correlated.json`, open **3D spatial ensemble**,
and set **Recoil dynamics** to **coherent_pumping**. Set **Trajectories per site**
(default 16,384), **Trajectory seed** (default 1729), and **workers**, then Run.
The GUI selects correlated recoil for this mode and disables inactive split-step
and legacy ODE settings. It explains incompatible Raman schedules before starting.
The actual schedule matters: an enabled Raman beam with zero amplitude in every
segment is accepted. Disabling Raman also makes its scheduled multipliers inactive.

The **Coherent pumping** tab shows survival, spin/axis targets, 3D ground-state
fractions, and photon counts with pointwise 95% Monte Carlo bands. **Diagnostics /
validity** includes one-sigma errors, weak-excitation and ground-spin checks,
positivity, conditioning, and remaining assumptions. It does not apply the
population motional-secular test to this solver. Population/thermal result tabs
are disabled for this result type to avoid showing stale or unavailable outputs.
Protocol comparisons and scans remain available with population recoil dynamics.

Save config preserves `ensemble.recoil_dynamics`, `trajectories_per_site`, and
`trajectory_seed`. Export results writes `run_config.json`, `run_metadata.json`,
`run_diagnostics.txt`, `run_timeseries.csv` with standard-error columns, the
trajectory NPZ files below, and a PNG/PDF dashboard. The normal headless entry
point runs the same saved configuration:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config saved_coherent_config.json --headless --output results/coherent_gui_config
```

Stop and closing the window signal every spawned worker and wait for cleanup;
cancelled runs preserve the last successful result. The GUI runs only the selected
coherent model. Use the comparison command to add the population controls.

The [GUI preview](../results/coherent_gui_validation/gui_preview.png) and its
exported settings demonstrate a four-site, 4096-trajectory run. These preview
settings are for checking the interface; the 64-site scientific comparison below
uses its separately recorded sampling settings. GUI tests cover actual worker
runs, saved settings, export, Stop/restart, window-close cleanup, and switching
back to the population backend.

From the repository root, in the `scripts` conda environment:

```powershell
conda run --no-capture-output -n scripts python scripts/check_coherent_pumping.py --config examples/onlyOP_correlated.json --workers 8 --trajectories 16384 --seed 1729 --output results/onlyOP_coherent_3d
```

The command runs the deterministic correlated population solver, coherent
trajectories, and independent population trajectories. The latter collapse the
motional state after each photon and provide a check against the deterministic
generator. All three use the selected atomic model and the same spatial and
angular grids. The command explicitly selects correlated recoil in its exported
config even if the supplied config uses Cartesian marginals.

`--trajectories` applies per site and per trajectory model. Trajectory seeds are
derived from `(seed, site index, mode)` and are independent of worker count,
completion order, or which mode is run first. `numerics.seed` continues to
determine the Sobol cloud. Each spatial site has equal weight.

For observable `O`, the cloud Monte Carlo variance is
`sum_j Var(mean O at site j) / number_of_sites²`. Differences between spatial
sites are not folded into these errors. These are errors in the estimated mean
for a fixed numerical cloud, not the experimental shot-to-shot spread or a
bound on model error. Very rare events can have zero empirical variance when
none are sampled; increase the trajectory count before interpreting such a zero.

Outputs include:

- `comparison.json`: final means, one-sigma Monte Carlo errors, population
  control residuals, population-model validity diagnostics, assumptions, and
  dependency versions. The command exits nonzero if the final population control
  disagrees beyond seven standard errors plus `2/N + 1e-8`.
- `comparison.png` and `.pdf`: absolute observables and coherent-minus-population
  changes with pointwise 95% Monte Carlo intervals. The legend is above the plots.
- `cloud_trajectories.npz`: time-dependent means and standard errors.
- `nodes.json` and `nodes/site_*.npz`: positions, local spectra sizes, seed
  information, per-site observables, and final reduced density matrices on x/y/z,
  resolved by spin and normalized to initially loaded atoms.
- `run_config.json` and `population_deterministic/`: exact inputs and the
  matched deterministic calculation.

`P_target_absolute` selects state A and the ground state of the configured Raman
axis, even though Raman light is off. `P_3d_ground_absolute` sums all spin states
in the 3D motional ground state; `P_target_3d_absolute` additionally selects state
A. All probabilities are relative to the initially loaded bound ensemble.

## Saved 64-site comparison (2026-10-13)

The same 1 ms `onlyOP_correlated.json` configuration was run with 24 polar nodes,
48 azimuth points, 201 output times, and 64 frozen cloud sites (spatial seed 0).
The high-statistics run uses **65,536 trajectories per site per model**, or
4,194,304 trajectories for each model, with trajectory seed 1730.

| Final observable | Deterministic population | Coherent trajectories, mean +/- one-sigma MC error |
| --- | ---: | ---: |
| Absolute target yield | 19.9356% | 19.8077% +/- 0.0167% |
| Remaining in all modeled wells | 47.9720% | 47.6690% +/- 0.0240% |
| Photons per initially loaded atom | 5.31272 | 5.28484 +/- 0.00244 |

The inferred changes are **-0.1279 +/- 0.0167 percentage points** in target yield
and **-0.3031 +/- 0.0240 percentage points** in well survival. These uncertainty
estimates describe Monte Carlo integration of this reduced model only.

A separate seed-1729 run with 16,384 trajectories per site agrees with the
high-statistics run: the largest difference among the checked final observables
in either trajectory mode is 1.71 combined standard errors. A seed-1731 run with
16,384 trajectories per site and **48 polar / 96 azimuth points** agrees within
1.59 combined standard errors. This tests angular refinement at the available
Monte Carlo precision; it does not provide a deterministic angular error bound.

All three population-trajectory controls agree with their deterministic cloud
averages within 1.43 standard errors. Six final probability observables were
also checked at every site in every run; the largest absolute standardized
residual was 3.67, within the seven-standard-error regression threshold.

- [High-statistics comparison](../results/onlyOP_coherent_3d/high_statistics/comparison.json)
  and [plot](../results/onlyOP_coherent_3d/high_statistics/comparison.png).
- [Initial independent run](../results/onlyOP_coherent_3d/comparison.json),
  [refined-angle run](../results/onlyOP_coherent_3d/refined_angles/comparison.json),
  and [all repeat/per-site checks](../results/onlyOP_coherent_3d/convergence.json).

After the base command above, reproduce the additional runs with:

```powershell
conda run --no-capture-output -n scripts python scripts/check_coherent_pumping.py --workers 8 --trajectories 65536 --seed 1730 --output results/onlyOP_coherent_3d/high_statistics --compare-to results/onlyOP_coherent_3d/comparison.json
conda run --no-capture-output -n scripts python scripts/check_coherent_pumping.py --workers 8 --trajectories 16384 --seed 1731 --quadrature 48 --output results/onlyOP_coherent_3d/refined_angles --compare-to results/onlyOP_coherent_3d/high_statistics/comparison.json
```

`--compare-to` verifies matching inputs apart from angular resolution and requires
distinct trajectory seeds. It writes the statistical comparison into the new
report and returns a nonzero status on inconsistency. Spatial quadrature is fixed
in these runs; spatial convergence and apparatus-level agreement remain untested.

## Validation and remaining assumptions

Tests compare both trajectory modes against an independently assembled full
3D density-matrix generator for two spins and three two-level axes. They include
correlated kicks, off-diagonal absorption hazards, noncommuting conditional
evolution, absorbing escape, photon counters, and a schedule boundary between
output times. An additional test checks the actual photon operators against
the production population generator. Tests cover seed reproducibility across
worker counts, cloud error propagation, cancellation, unsupported inputs, and
exported arrays.

Repeat with more trajectories and another seed to check statistical stability;
double `--quadrature` to test angular integration. Refine `--samples` separately
for spatial integration. Agreement between stochastic and deterministic
population propagation checks implementation, not their shared physical model.

The motional-secular criterion is unnecessary for this solver's treatment of
recoil, but **weak-excitation optical elimination, resolved spin and spherical
emission channels, diagonal Stark shifts, fixed separable wells, and no recapture
remain assumptions**. Ground-spin coherences and interference between distinct
spherical emission amplitudes remain absent. This does not validate Raman
cooling or apparatus-level atom loss. The initial distribution at every site is
conditioned on local bound states; loading survival remains unspecified.
