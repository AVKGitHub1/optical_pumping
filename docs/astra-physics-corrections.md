# RSC physics corrections and current results

Reviewed 2026-10-13. All execution and checks use the **`scripts` conda environment**. The active configuration is [examples/experiment.json](../examples/experiment.json), with **modeled carrier decay**. Its saved 5 ms results below retain the default weak-field optical model and Cartesian recoil marginals; the finite-field and correlated-recoil comparisons are reported separately. The [historical audit](astra-physics-history.md) preserves the original 1D audit, intermediate results, convergence studies and resolved questions.

## Current result

The sequence is 4 ms of simultaneous Raman cooling and pumping followed by 1 ms of pumping. Absolute yields refer to **initially loaded, locally bound atoms**; loading efficiency from the MOT is unknown.

| Observable after 5 ms | Modeled-decay prediction |
| --- | ---: |
| Joint `F=3,mF=3,nz=0` / initially loaded | **27.09%** |
| Remaining bound in all three axes / initially loaded | **52.11%** |
| Well loss / initially loaded, assuming no recapture | **47.89%** |
| Spin `F=3,mF=3` / survivors | **98.77%** |
| Spin `F=3,mF=3` / initially loaded | 51.46% |
| Axial ground state / survivors | 52.80% |
| Joint `F=3,mF=3,nz=0` / survivors | 52.00% |
| Temperature proxies x / y / z | **6.53 / 7.01 / 4.10 uK** |
| Spin target and all three motional ground states / initially loaded | 1.74% |
| Scattered photons / initially loaded atom | 4.412 |

**27.09% is an axial target yield, not a 3D ground-state yield.** Temperatures match the energy of the surviving distribution; that distribution need not be thermal. Transverse temperatures can fall through preferential loss of hotter atoms without transverse Raman cooling.

Relative to the previous envelope-fit run, absolute target yield changes by -0.089 percentage points and loss by -0.086 percentage points. Pump settings and detuning were held fixed; this was not a new optimization. These are rough model predictions, not experimental error bars.

## Configuration and selected polarizations

| Input | Implemented value |
| --- | --- |
| Bias field | 1 G along lab x |
| Raman beams | 783 nm along +z/-z, same laser; both tones in both beams at equal relative powers |
| Raman polarizations | Helicity +1 for +z, -1 for -z, defined along each beam's own propagation direction; equal lab Jones vectors |
| Raman phases | Stabilized relative beat phase; uniform static spatial registration relative to the lattice |
| Axial lattice | 785 nm, central depth 15 uK, derived bottom frequency 69.04 kHz |
| Transverse lattices | 1188 nm in x/y, central depth 15 uK each, derived bottom frequencies 45.62 kHz |
| Transverse optical offset | 160 MHz between the x and y lattice carriers |
| Beam intensity waists | 200 um, 1/e^2 radius |
| Cloud | Spherical Gaussian, 150 um 1/e^2 density radius; coordinate standard deviation 75 um |
| Initial motion and spin | 10 uK in all axes, conditioned on local bound states; uniform F=3 population; custom spin populations supported |
| Spin pump | Along x, **3 nW, pure sigma+** relative to B |
| Repump | Along z, **7.5 nW, linear E along y** |
| Lattice electric fields | x/y/z beam pairs: E along y/x/y respectively; same linear polarization within each pair |
| Raman detuning | -5 kHz from the central red-sideband reference; common beat approximately 3.038127556 GHz |

Depths and wavelengths are authoritative. The approximately 70/50 kHz frequencies are not imposed as independent constraints on the sinusoidal wells.

Pure sigma+ pumping favors the stretched state and avoids the tested sigma-minus leakage; achievable purity remains unmeasured. Repump E along y gives equal sigma+/sigma-minus components and outperformed E along x (pi) in the earlier admissible scan. The lattice-polarization preference was tiny: z-lattice E along y improved coarse-grid yield by only 0.027 percentage points, and the tested y-lattice choices were effectively tied. The settings were retained after switching to modeled decay and have not been reoptimized under that convention.

The earlier power screen covered 3 nW to 3 uW. Invalid points were excluded from ranking. The current maximum eliminated excited fraction is 0.000617, but the maximum pumping-rate / motional-angular-spacing ratio is **0.480**, near the configured invalid threshold of 0.5. A 10 nW repump reached 0.641 on the 64-node cloud. Selecting 7.5 nW reflects this reduced-model boundary; it does not establish that stronger experimental pumping performs worse.

## Modeled carrier decay

`raman.calibration = "measured_carrier"` and `raman.carrier_decay_mode = "modeled"` retain 5 kHz as the intensity reference. Without a raw trace, the explicit convention is **RMS local carrier frequency**, fixing the initial quadratic rise:

```text
f_RMS = sqrt(sum_j w_j * f_j^2) = 5000 Hz
P_transfer(t) = visibility * sum_j w_j * sin^2(pi * f_j * t)
P_transfer(t) = visibility * pi^2 * f_RMS^2 * t^2 + O(t^4)
```

The index j spans thermal states and spatial samples. Calibration uses the stated measurement condition: the 785 nm lattice only, pumping lights off. Thermal carrier overlaps, Gaussian intensities and static registration are included. This convention does not claim that an inhomogeneous trace has a unique fitted oscillation frequency.

- Intensity: **21692.6 W/m^2 per tone per beam**, or **1.363 mW per tone**, **2.726 mW total per Raman beam**.
- Modeled envelope: `abs(sum_j w_j * exp(2*pi*i*f_j*t))`. The **first 1/e crossing is 89.0 us**, or 0.445 reference periods. It is nonexponential and can revive.
- Fitted homogeneous damping: **zero**. Spatial dephasing is already present in the ensemble and is not added again as a Lindblad rate. Independently configured technical dephasing remains available and is zero here.
- Reported 3-4-flop decay: comparison-only. `flop_decay_periods` does not affect modeled-mode intensity or dynamics. `calibration_compatible = null` means measurement agreement was not assessed.
- The 90% contrast is preparation/readout visibility and never removes atoms.
- Cached central, phase-averaged Raman scattering: **0.3385/s** for `|3,3>` and **0.3370/s** for `|2,2>`; differential Raman light shift **-26.18 Hz**.

The [calibration cache](../rb85rsc/cache/carrier_97d9792bf4c62571cc17e221.json) includes traces, optical estimates and source metadata. ARC D1+D2 scattering amplitudes interfere before squaring; ground light shifts include the D-line counter-rotating term. ARC's working database stays in memory. Atomic estimates do not determine technical laser or magnetic noise.

For older configs, `measured_envelope` remains available and is the default when the new mode is omitted. It fits a synthetic trace with nonnegative residual damping and flags poor fits. That earlier fit could not reproduce the reported envelope. Choosing modeled decay removes that fit constraint; it does not establish experimental agreement.

## Corrections implemented

The ensemble backend is in [ensemble.py](../rb85rsc/ensemble.py), with cached optical estimates in [far_detuned.py](../rb85rsc/far_detuned.py). The legacy 1D backend remains available. Measured-carrier configs cannot silently reinterpret 5 kHz as a single-path Rabi frequency.

- **Energy and temperature:** actual anharmonic excitation energies and survivor-weighted mixtures of finite local spectra replace harmonic energy assignments. `nbar` is a mean level index in a lattice.
- **Magnetic energies:** analytic Breit-Rabi ground energies replace the linear approximation and a failed ARC check that previously reported zero error. The legacy default retains weak-field optical states; the optional finite-field model below includes their magnetic mixing.
- **Raman momentum and phases:** the lowering operator uses absorption minus emission momentum; the raising block is its adjoint. All four pathways retain their complex relative phases.
- **Three-axis motion:** joint `P(spin,nx,ny,nz)` and a coherent axial Raman-pair block at each transverse occupation. Recoil acts in all axes. Every spatial sample shares one fixed beat, with its local depths and diagonal light shifts.
- **Loss and conditioning:** departure from any local bound manifold is absorbing. Absolute yields and survivor fractions are distinct. Initial discarded basis weight is separate from unknown loading survival; custom motional inputs now report discarded weight correctly.
- **Scattering and integration:** equal-frequency absorption paths interfere; distinct optical tones are averaged in spontaneous rates. Exact static Raman unitaries and positive uniformized scattering updates are combined by Strang splitting.
- **Diagnostics:** loss-selection cooling, validity limits, conservation and positivity are explicit. Unknown quantities are not substituted with reassuring zeros.

The Raman schedule multiplier scales both beam intensities together, so two-photon Rabi frequency, light shifts and scattering all scale linearly with it. This convention was checked and retained.

Atomic conventions follow [ARC's hyperfine dipole documentation](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFS.html) and [Steck's Rb-85 data](https://steck.us/alkalidata/rubidium85numbers.pdf).

## Remaining uncertainties

1. **Calibration:** the RMS/initial-curvature convention is an assumption without a raw carrier trace. It does not validate frequency and the reported decay simultaneously.
2. **Spatial inputs:** microscopic registration is uniformly averaged and fixed during each sequence. Separate lasers do not specify a phase-diffusion rate. Experimental uncertainties in cloud size, temperature, depth and their correlations are unspecified.
3. **Recoil coherence:** population-resolving jumps discard coherence transfer, including indistinguishable elastic channels. The separate [3D pumping-only trajectory solver](coherent-pumping.md) now retains motional coherence on all axes; the Raman backend still uses population-resolving recoil. Resolved spherical emission channels omit mutual interference. The default Cartesian emission marginals also omit correlations within one photon's momentum; the optional correlated model below includes those Cartesian correlations. Passing numerical tests does not bound the remaining physical errors. See [Ozeri et al.](https://arxiv.org/abs/quant-ph/0502063) and [Reiter and Sorensen](https://arxiv.org/abs/1112.2806).
4. **Trap and Stark model:** isolated separable scalar wells, diagonal shifts, no off-diagonal static dressing or changes of motional basis. Additional Raman standing-wave depth can reach about 6.6% of the central axial depth. Excited-state lattice shifts remain a configurable common shift; state-resolved excited polarizability is absent. D-line estimates omit higher excited states and hyperfine-mediated polarizability corrections.
5. **Loss:** no recapture, tunnelling, intersite transport, coherent Raman escape to the continuum or collisions. Crossing one site barrier is not necessarily apparatus-level escape. The selected absorbing model estimates well loss; cooling-assisted recapture has not been calculated.
6. **Optical reduction:** the motional-secular ratio of 0.480 is marginal. Stronger pumping requires a more complete optical master equation. The finite earlier search does not establish a global optimum.

Transverse confinement therefore matters even for axial Raman momentum: pumping/recoil heating, lattice optical effects and loss act in x/y. Lower survivor temperature alone is not evidence of transverse Raman cooling.

## Finite-field optical states (2026-10-12)

`optical.zeeman_model = "diagonalized"` now includes hyperfine + Zeeman mixing
within the ground, D1, and D2 fine-structure manifolds. The existing
`"weak_field"` default remains available for reproducing saved results.
[onlyOP_diagonalized.json](../examples/onlyOP_diagonalized.json) selects the new
model with the same physical inputs as `onlyOP.json`.

The calculation diagonalizes
`H/h = E_hfs/h + (mu_B B/h)[g_I F_z + (g_J - g_I) J_z]` separately in each
fixed-m block, using cached ARC hyperfine energies and g factors. Both sides of
each optical dipole are transformed into the resulting basis. This includes
ground-state mixing as well as excited-state mixing and changes absorption,
branching, optical light shifts, and the D1+D2 Raman/trap response consistently.
The construction follows [Steck's atomic conventions](https://steck.us/alkalidata/rubidium85numbers.pdf)
and [ARC's Zeeman Hamiltonian](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.breitRabi.html).

Labels `F,mF` identify branches followed adiabatically from zero field; F is not
an exact quantum number at finite B. Laser detunings remain referenced to the
named zero-field transitions. The magnetic model enters the carrier-cache
identity, so calibrations from the two models cannot be reused interchangeably.
Weak-field caches and previous numerical predictions are retained.

At 3 G, the largest included D2 excited-energy correction relative to the linear
model is **1.093 MHz**. Matched 64-site, 1 ms `onlyOP` simulations give:

| Observable | Weak-field optical states | Diagonalized optical states |
| --- | ---: | ---: |
| Joint spin target and nz=0 / initially loaded | 19.31162% | 19.95359% |
| Well survival / initially loaded | 46.59315% | 48.01592% |
| Spin target / survivors | 93.84729% | 94.11094% |
| Photons / initially loaded atom | 5.54955 | 5.31617 |
| Maximum motional-secular ratio | 3.03675 | 2.83691 |

Absolute target yield changes by **+0.64197 percentage points** and survival by
**+1.42277 percentage points**. The corrected run still exceeds the
motional-secular limit and is marked **INVALID**. Spin coherences, correlated
photon recoil, recapture, and trap-basis changes are not added by this correction.
Mixing between different J manifolds and diamagnetism remain omitted; the
magnetic diagnostic now estimates the residual inter-J scale instead of
warning about the included hyperfine mixing.

Validation covers independent uncoupled-spin Hamiltonians from 0 to 100 G,
Clebsch-Gordan phase conventions, D1 Breit-Rabi energies, branching and optical
rate sum rules, the exact zero-field limit, cache separation, scan rejection,
and ensemble propagation. The matched weak-field run reproduces the earlier
population and photon-count arrays exactly. Halving the corrected run's step
from 2 us to 1 us leaves final target yield, survival, and photon count unchanged
at the saved precision. The Raman-off problem has a time-independent population
generator, so this is a consistency check, not a Raman splitting-error bound.
A separate short Raman-on run checks calibrated coherent propagation with the
new atomic model.

The complete suite passes **227 tests in 173.36 s**, including 24 new atomic
and integration tests. The new calibration test uses the production
single-thread BLAS context when warming shared motional caches; the GUI's
serial-versus-eight-worker comparisons retain exact equality assertions.
The [validation record](../results/onlyOP_finite_field/validation.json) includes
the earlier test-order mismatch and its resolution.

Run `scripts/check_finite_field.py` as shown in the README. Full outputs and
the matched comparison are in
[results/onlyOP_finite_field](../results/onlyOP_finite_field/), including
[comparison.json](../results/onlyOP_finite_field/comparison.json).

## Correlated photon recoil (2026-10-13)

`ensemble.recoil_model = "correlated"` integrates each photon's emission direction
once for all three local lattice axes. This replaces the product of three
independently averaged kernels with the angular average of their product.
Pumping, Raman scattering, and lattice scattering all use this construction,
including interference between equal-frequency absorption paths. The original
`"cartesian_marginals"` mode remains the default.

The emitted vector has unit norm; its Cartesian components are not independent.
For example, isotropic emission has `<ux^2 uy^2> = 1/15`, whereas multiplying
independent marginals gives `1/9`. The new quadrature reproduces this joint
moment and the analytic pi/sigma dipole covariance tensors. The optical-state
model is independent of this recoil choice. The new
[onlyOP_correlated.json](../examples/onlyOP_correlated.json) uses both
diagonalized atomic states and correlated recoil.

Matched 64-site, 1 ms runs with diagonalized atomic states give:

| Observable | Cartesian marginals | Correlated recoil |
| --- | ---: | ---: |
| Joint spin target and nz=0 / initially loaded | 19.95359% | 19.93560% |
| Well survival / initially loaded | 48.01592% | 47.97203% |

These are changes of **-0.01799 percentage points** in target yield and
**-0.04389 percentage points** in survival. This full-sequence comparison
supersedes extrapolating the earlier single-photon difference into a yield
correction. It retains all other model approximations. In particular, the
motional-secular limit is still exceeded and both runs remain **INVALID**.

Angular integration uses 24 polar nodes and 48 azimuth points; the repeat uses
48 and 96. The largest checked difference across every site's saved population
trajectory is **5.22e-15**, and photon-count differences are below **2.49e-14**.
Cloud averages agree at similarly small levels. These are quadrature checks,
not physical error bars or a test of continuous spatial convergence.

Dense kernels scale quadratically in the product of bound-level counts. The
current implementation supports up to 1024 joint motional states per site,
with no silent truncation. The example's central site has 9 x 9 x 6 = 486.
With eight workers, the measured full-cloud runtimes were 17.18 s for marginals,
59.53 s for correlated recoil, and 83.55 s for the refined angular grid.
Runtime depends on the machine and workload. Construction checks the Stop
signal between bounded batches; both GUI and CLI use the same model option.

Independent tests cover angular moments, bulk lattice overlaps, direct
Kronecker photon jumps, coherent absorption amplitudes, an augmented Markov
exponential with loss and photon counters, rigid apparatus rotations,
serial/parallel equality, active Raman/lattice scattering, cancellation, and
per-site convergence failures that cancel in the cloud average.

The committed baseline passed 227 tests before this change. The final expanded
suite passes **244 tests in 208.51 s**, including 17 new checks. The existing
scheduling mock was updated to accept the Stop argument now forwarded into
recoil construction; its ordering and configuration assertions are unchanged.

The marginal-mode run reproduces the preceding finite-field comparison's
population, photon-count, and loss arrays exactly. A separate calibrated
Raman-on check (one site, 20 us) halves the timestep from 2 us to 1 us: saved
population entries change by at most 6.14e-9, photon counts by 1.04e-8, and
loss by 4.36e-9. Probability is conserved within 7.78e-16 and sampled density
matrices remain positive. This short check does not establish timestep
convergence for a complete cooling sequence. Exact settings and checks are in
the [validation record](../results/onlyOP_correlated_recoil/validation.json).

Reproduce with `scripts/check_correlated_recoil.py` using the README command.
The [comparison](../results/onlyOP_correlated_recoil/comparison.json) and full
outputs record both models and the refined run. The command returns nonzero
if probability or photon-count convergence fails. Motional coherence transfer,
interference between emitted spherical channels, and recapture remain omitted.

## Coherent 3D optical pumping (2026-10-13)

A separate event-driven trajectory solver now retains motional phases on x/y/z
during pumping, with correlated emission, standing-wave lattice scattering, and
absorbing escape. The full absorption hazard drives conditional evolution;
photons that cause escape remain in the counters. Raman light must be off.

The GUI now exposes this as **3D spatial ensemble → Recoil dynamics →
coherent_pumping**, with saved trajectory-count and seed controls, a dedicated
results tab, uncertainty bands, diagnostics, export, and worker cancellation.
The same config selects this solver in headless runs. Raman is excluded because
the trajectory representation stores a definite spin and separate axis
wavefunctions; coherent Raman transfer requires spin-motion superpositions.

For `onlyOP_correlated.json`, 64 cloud sites and 65,536 trajectories per site give
**19.8077% +/- 0.0167% absolute target yield** and **47.6690% +/- 0.0240% well
survival**, compared with 19.9356% and 47.9720% in the matched deterministic
population model. Errors are one-sigma Monte Carlo uncertainties for the fixed
cloud. The changes are -0.1279 and -0.3031 percentage points, respectively.

The population-trajectory controls agree with deterministic propagation, both
in cloud averages and at individual sites. Independent seeds and doubling the
angular grid give consistent final results at the available statistical
precision. Tests compare the trajectory engine with independent full 3D master
equations, including noncommuting conditional dynamics and escape.

The population model's motional-secular warning remains applicable to that
backend. The new solver avoids that recoil approximation for pumping-only
sequences; optical elimination, resolved spin/emission channels, diagonal Stark
shifts, frozen isolated wells, and no recapture remain assumptions. It does not
validate the Raman calculation or apparatus-level loss. See the
[model and reproduction commands](coherent-pumping.md),
[high-statistics results](../results/onlyOP_coherent_3d/high_statistics/comparison.json),
and [repeat/per-site checks](../results/onlyOP_coherent_3d/convergence.json).

## Validation and cleanup

The full expanded suite passed **88 tests in 113.89 s** in the `scripts` environment. Tests include independent Hamiltonian and augmented Markov exponentials, cycling-transition normalization, Breit-Rabi diagonalization, Raman phases/momentum, helicities, canonical temperature inversion, custom populations and an analytic two-frequency carrier ensemble. Workflow regressions check stale-scan reuse, control matching, failed cache writes and early rejection of invalid scan objectives.

The complete 64-node experiment was rerun after cleanup. It reproduced the previous target yield, survival and temperature proxies to the saved precision; total probability is conserved to 3.7e-14 and sampled density matrices remain positive. The active example config matches the exported run config exactly.

| Numerical check | Change in absolute target fraction |
| --- | ---: |
| Timestep 2 us to 0.5 us, earlier matched four-node reference | -5.90e-7 |
| Emission quadrature 24 to 48, same reference | Below 1e-14 |
| Sideband cutoff 3 to all bound sidebands, same reference | -4.98e-7 |
| Spatial quadrature 32 to 64 nodes, current modeled-decay run | **+0.00705** (+0.705 percentage points) |

Earlier timestep/recoil checks used the previous calibration and z-lattice E along x. They establish numerical convergence for that reference, not a physical error bound. Spatial quadrature still has visible residual dependence; no statistical confidence intervals are claimed. A central single-photon angular check found Cartesian-factorization retention differences of 0.000042 to 0.000209, which is not a sequence-wide or coherence-error bound.

The [single-photon recoil check](../results/astra_modeled_decay/one_photon_recoil_check.json) was rerun in the current output directory and reproduced those values. It now uses the configured magnetic-field direction rather than hard-coding lab x.

Cleanup fixes: scan reuse now requires matching full configs; controls must match in physics and numerics; invalid objectives fail before propagation; unknown scan runtime is described without `nan`; cache publication is atomic; initial-state metadata reflects actual inputs. `--stage final` reruns the current config and needs explicit `--use-selected` to restore an optimization winner. Temporary test/smoke outputs are removed, while historical scientific results are preserved.

## Performance validation (2026-10-01)

The computational optimization pass retains the current physics, config, numerical settings, and precision. The full 64-node experiment took 268.83 s before the changes and 179.38 s afterward with four workers, a 33.3% reduction. An eight-worker execution override took 92.34 s; the config still defaults to four workers.

Both comparisons reproduced all 39,735 compared arrays bit-for-bit and all 8,869 non-timing scalar entries exactly, including complete node trajectories, final density matrices, loss, photon counts, and temperature proxies. Existing scientific outputs were preserved. See [performance implementation and validation](performance.md) and the [benchmark record](../results/performance/experiment_speedup.json) for the measured environment and details.

After the final performance edits, the complete suite passed **125 tests in 140.50 s** in `scripts`.

The GUI now honors `ensemble.workers`, including eight workers, with cancellation between integration steps and cleanup before restarting or closing. Tests with eight real spawned workers reproduce sequential results exactly and cover Stop, restart, window close, sparse output times, and worker failures. The expanded suite passed **130 tests in 122.59 s** in `scripts`. Select **3D spatial ensemble → workers → 8** in the GUI; the config's default remains four.

## Optical-pumping recoil checks (2026-10-12)

The `onlyOP.json` reference check now covers each local x/y/z axis at the cloud
center and the first four production Sobol positions (seed 0), with both pumping
beams scaled by 0.03, 0.1, and 1. Each run retains the full 1 ms schedule and 201
output times. All 45 cases and their 45 angular-quadrature repeats completed.
Doubling the emission quadrature from 24 to 48 changed checked probabilities by
at most 1.7e-14.

The table gives the largest absolute coherent-minus-population difference over
these five positions and saved times at **full pump power**, in percentage points:

| Independently modeled local axis | Well survival | Spin target and n=0 / initially loaded |
| --- | ---: | ---: |
| x | 0.99872 | 0.13669 |
| y | 0.03600 | 0.00752 |
| z | 0.05673 | 0.03128 |

The largest survivor-energy difference is 0.01541 uK. The original central
z-only check therefore understates the largest discrepancy found by looking at
other axes and positions. These are independent one-axis pumping calculations,
not a correction to the production 3D yield. They omit lattice scattering,
transverse escape, spin coherences, finite-field excited-state mixing, and
recapture. The magnetic-field and other shared validity warnings still apply.
The population model's motional-secular warning remains separate from numerical
convergence; a small measured difference does not justify removing it.

The single-photon factorization check now honors per-axis lattice parameters,
Gaussian attenuation, rotated lattice directions, and nonthermal initial axial
states. At the same five sites, its largest Cartesian-marginal versus correlated
3D survival difference was **0.38744 percentage points** (cloud_2, absorption
along local x, pi emission). Refining its polar/azimuth quadrature from 16/32 to
32/64 changed survival by at most 2.2e-9. This is a one-event diagnostic, not a
bound on accumulated sequence error.

Reproduction commands are in the [README](../README.md#check-the-motional-population-approximation).
The [survey](../results/onlyOP_recoil_cloud/survey.json) records exact inputs,
positions, and worst cases; [case comparisons](../results/onlyOP_recoil_cloud/comparison.json)
retain physical and numerical diagnostics. The
[single-photon report](../results/onlyOP_recoil_cloud/one_photon_recoil_check.json)
contains both angular resolutions. Failed or unsupported cases remain visible
and cause the validation commands to return a nonzero exit status. No local-axis
results are averaged or multiplied into a 3D prediction, and these sampled
maxima are not experimental uncertainty intervals or continuous-time bounds.

The expanded suite passes **203 tests**. New checks compare each axis's
population reference with the production 1D solver, verify whole-apparatus
rotation invariance and cloud sampling, and exercise missing-case and failed
quadrature reporting. The corrected single-photon checker reproduces the
archived central default-geometry results within 2.3e-16.

## Outputs and reproduction

- [Current metadata and all 12 absolute/conditional spin populations](../results/astra_modeled_decay/final/run_metadata.json), [time series](../results/astra_modeled_decay/final/run_timeseries.csv), [3D populations and local spectra](../results/astra_modeled_decay/final/run_ensemble.npz), and [exact run config](../results/astra_modeled_decay/final/run_config.json).
- [Dashboard](../results/astra_modeled_decay/final/run_dashboard.png), [modeled carrier trace](../results/astra_modeled_decay/final/run_carrier_calibration.png), and [spatial convergence](../results/astra_modeled_decay/spatial_convergence.json). PDF copies accompany the figures.
- Earlier scans and measured-envelope results remain in [results/astra_clarified](../results/astra_clarified/); the original 1D audit remains in [results/astra_audit](../results/astra_audit/). Their interpretation and reproduction details are preserved in the [historical audit](astra-physics-history.md).

From the repository root:

```powershell
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage final --workers 4
conda run -n scripts python -c "from pathlib import Path; Path('.test_tmp').mkdir(exist_ok=True)"
conda run --no-capture-output -n scripts python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp/pytest
```

Optional matched controls and numerical checks:

```powershell
conda run --no-capture-output -n scripts python scripts/check_experiment_controls.py
conda run --no-capture-output -n scripts python scripts/finalize_experiment.py
conda run --no-capture-output -n scripts python scripts/check_experiment_convergence.py
conda run --no-capture-output -n scripts python scripts/check_recoil_factorization.py
```

Control selection requires fresh matching outputs. The current modeled-decay settings have not been reoptimized; to repeat the search, run the stages in the [experiment workflow](experiment-workflow.md) and explicitly add `--use-selected` to the final stage. Outputs are separated by carrier-decay mode and include full configs and assumptions.
