# Historical physics audit and intermediate results

This is the preserved audit history before the final cleanup. Current settings, results and remaining limitations are in [astra-physics-corrections.md](astra-physics-corrections.md). Historical questions and numerical results below are superseded where the current report says so. Commands run from the repository root.

# RSC physics audit and corrections

Audit and implementation date: 2026-09-30. All commands and test runs use **`conda run -n scripts`**. [`examples/experiment.json`](../examples/experiment.json) now uses the clarified experimental model. The original 1D audit configuration is preserved in [`tests/fixtures/audit_experiment_1d.json`](../tests/fixtures/audit_experiment_1d.json); the historical audit results below must not be read as predictions for the revised experiment.

## Current revision: use modeled carrier decay

Following the instruction to **use modeled carrier decay**, [`examples/experiment.json`](../examples/experiment.json) now sets `raman.carrier_decay_mode = "modeled"`. The simulation predicts dephasing from the distribution of thermal carrier overlaps, Gaussian intensities and static spatial registration. It does not fit or impose the reported 3-4-flop envelope. The fitted homogeneous damping is exactly zero; independently configured technical dephasing remains available and is also zero in this example. The spatial dephasing is already present in the ensemble evolution and is not added again as a Lindblad decay rate.

The 5 kHz measurement remains the intensity reference. Because no raw trace is available, it is now assigned the explicit **RMS local carrier / initial-curvature convention**:

```text
f_RMS = sqrt(sum_j w_j * f_j^2) = 5000 Hz
P_transfer(t) = visibility * sum_j w_j * sin^2(pi * f_j * t)
P_transfer(t) = visibility * pi^2 * f_RMS^2 * t^2 + O(t^4)
```

Here j indexes thermal states and spatial samples, and f_j includes the carrier matrix element. This convention does not claim that an inhomogeneously damped trace has a unique fitted oscillation frequency. The measured 90% contrast remains a readout/preparation visibility only.

- Inferred intensity per tone per beam: **21692.6 W/m^2**, or **1.363 mW** per tone and **2.726 mW** total per beam, about 2.22% below the earlier envelope-fit estimate.
- Modeled inhomogeneous envelope: `abs(sum_j w_j * exp(2*pi*i*f_j*t))`; its **first 1/e crossing is 89.0 us**, or 0.445 periods of the 5 kHz reference. It is not exponential, may revive, and is not a homogeneous T2 or a photon-scattering lifetime.
- The reported 0.7 ms envelope is comparison-only. `calibration_compatible` is exported as `null` (agreement not assessed), rather than claiming agreement or failing a fit that was not performed.
- Cached ARC differential Raman shift at the new intensity: **-26.18 Hz**; phase-averaged central Raman scattering is **0.3385/s** for `|3,3>` and **0.3370/s** for `|2,2>`. The revised atomic/intensity/trace cache is [`carrier_97d9792bf4c62571cc17e221.json`](../rb85rsc/cache/carrier_97d9792bf4c62571cc17e221.json).
- Pump/repump powers, polarizations, detuning, geometry and schedule retain the earlier selected settings. This revision changes the carrier-calibration assumption; those settings have not been reoptimized under the new convention.

The full 64-sample rerun and revised results are saved under [`results/astra_modeled_decay/`](../results/astra_modeled_decay/), separate from the earlier measured-envelope calculation. All **18 ensemble and IO/GUI checks passed in 8.41 s**. New tests compare the modeled trace and envelope with an independent analytic two-frequency ensemble and verify that changing the reported decay or contrast does not change inferred intensity or add damping.

The completed 64-sample rerun gives:

| Observable after 5 ms | Modeled-decay result |
| --- | ---: |
| Joint `F=3,mF=3,nz=0` / initially loaded | **27.09%** |
| Remaining in all three local wells / initially loaded | **52.11%** |
| Well loss / initially loaded, assuming no recapture | **47.89%** |
| Spin `F=3,mF=3` / survivors | **98.77%** |
| Spin `F=3,mF=3` / initially loaded | 51.46% |
| Axial ground state / survivors | 52.80% |
| Joint `F=3,mF=3,nz=0` / survivors | 52.00% |
| x / y / z temperature proxies | **6.53 / 7.01 / 4.10 uK** |
| Spin target and all three motional ground states / initially loaded | 1.74% |
| Scattered photons / initially loaded atom | 4.412 |

Relative to the previous envelope-fit run, absolute axial target yield changes by -0.089 percentage points and well loss by -0.086 percentage points. The rough predictions are nearly unchanged. This comparison keeps pump powers, polarizations and the -5 kHz sideband offset fixed; it is not a repeat of the optimization or the Raman-off control.

See the revised [metadata, including all 12 absolute and conditional spin populations](../results/astra_modeled_decay/final/run_metadata.json), [time series](../results/astra_modeled_decay/final/run_timeseries.csv), [3D population data](../results/astra_modeled_decay/final/run_ensemble.npz), [dashboard](../results/astra_modeled_decay/final/run_dashboard.png), and [modeled carrier trace](../results/astra_modeled_decay/final/run_carrier_calibration.png). The example config exactly matches the exported run config. The headless run completed in 208 s, conserved total probability to 3.7e-14, and retained positive sampled density matrices. `git diff --check`, compilation of changed Python files, and a numerical regression of the legacy envelope-fit calibration also passed.

The remaining physical approximation warnings are retained, including the motional-secular ratio of 0.480 and population-resolving recoil. [Spatial quadrature](../results/astra_modeled_decay/spatial_convergence.json) gives 26.39% versus 27.09% target yield at 32 versus 64 samples, so sub-percent numerical changes should not be presented as experimental precision. The earlier timestep/recoil convergence checks below describe the prior calibration; the new run does not turn those checks into a bound on physical model error.

## Previous result using measured-envelope fitting (superseded)

Implementation and the final 64-sample run are complete. **These are provisional model predictions:** the carrier calibration does not reproduce the measured decay envelope, and the optical-pumping motional-secular approximation is marginal. Neither numerical convergence nor passing regression tests resolves those physical limitations.

The previous [saved configuration](../results/astra_clarified/final/run_config.json) retains the 4 ms simultaneous-cooling and 1 ms final-pumping sequence. Powers below are total power in each pumping beam; both waists remain 200 um. These power/polarization settings are also used in the current modeled-decay revision.

| Setting | Selected value | Reason / qualification |
| --- | --- | --- |
| Spin pump along x | 3 nW, pure sigma+ relative to B | Favors the stretched target and avoids the tested sigma-minus leakage; ideal purity is an assumption, not a measured extinction ratio. |
| Repump along z | 7.5 nW, linear E along y | Equal sigma+/sigma-minus components relative to B along x; this choice gave better absolute target yield than linear E along x (pi) in the admissible scan. |
| Raman offset | -5 kHz from the configured red-sideband reference | Beat frequency 3.038127556 GHz; the matched 64-sample zero-offset control gives a lower target yield. |
| Raman circular polarizations | Helicity +1 for +z, -1 for -z | Implements the supplied opposite helicities defined along each beam's own propagation vector, with equal laboratory Jones vectors. |
| x/y/z lattice linear polarizations | E along y / x / y, respectively | Each is transverse to its beam. The small coarse-grid preference for z-lattice E along y is only 0.027 percentage points in absolute yield; the y-lattice choices are effectively tied. This is not an experimentally established unique polarization optimum. |

The power screen covers 3 nW to 3 uW and rejects high-power points outside the reduced model's validity before ranking them. The final maximum eliminated excited fraction is 0.000617, but the maximum scattering-to-motional-spacing ratio is **0.480**. A 10 nW repump reaches 0.641 in the 64-sample cloud and fails the configured 0.5 cutoff. Thus 7.5 nW is a choice within the tested model boundary, not evidence that stronger experimental pumping is worse. A more complete optical master equation is needed to optimize beyond this boundary.

| Final observable, after 5 ms | Prediction |
| --- | ---: |
| Remaining in all three local wells / initially loaded | 52.02% |
| Well loss / initially loaded, assuming no recapture | 47.98% |
| Spin `F=3,mF=3` / survivors | 98.76% |
| Spin `F=3,mF=3` / initially loaded | 51.38% |
| Joint `F=3,mF=3,nz=0` / initially loaded (optimization objective) | **27.18%** |
| Joint `F=3,mF=3,nz=0` / survivors | 52.26% |
| Axial ground state / survivors | 53.06% |
| Energy-matched temperature proxies x / y / z | **6.52 / 7.00 / 4.06 uK** |
| Spin target and all three motional ground states / initially loaded | 1.75% |
| Scattered photons / initially loaded atom | 4.43 |

The useful rough predictions are therefore **27% absolute axial target yield and 48% well loss**, not a 27% three-dimensional ground-state yield. Transverse confinement cannot be ignored: optical recoil can eject atoms in x/y, and the transverse temperature changes partly reflect loss selection.

The full final spin distribution is below. Extra digits identify the numerical output; they are not experimental accuracy.

| Spin state `(F,mF)` | % of survivors | % of initially loaded |
| --- | ---: | ---: |
| (2,-2) | 0.0193 | 0.0100 |
| (2,-1) | 0.0410 | 0.0213 |
| (2,0) | 0.0406 | 0.0211 |
| (2,+1) | 0.0540 | 0.0281 |
| (2,+2) | 0.1291 | 0.0671 |
| (3,-3) | 0.1399 | 0.0728 |
| (3,-2) | 0.0776 | 0.0404 |
| (3,-1) | 0.0719 | 0.0374 |
| (3,0) | 0.0808 | 0.0421 |
| (3,+1) | 0.1259 | 0.0655 |
| (3,+2) | 0.4589 | 0.2387 |
| (3,+3) | 98.7610 | 51.3760 |

Matched controls use the same 64 spatial samples, pumping settings and lattice polarizations:

| Control | Absolute axial target | Survival | x / y / z temperature proxies (uK) |
| --- | ---: | ---: | --- |
| Raman off | 19.51% | 55.68% | 6.81 / 7.35 / 7.05 |
| Raman on, zero offset | 24.80% | 52.98% | 6.59 / 7.09 / 4.78 |
| Selected -5 kHz offset | 27.18% | 52.02% | 6.52 / 7.00 / 4.06 |

Raman driving improves absolute target yield by 7.67 percentage points over pumping alone and reduces the axial temperature proxy from 7.05 to 4.06 uK, while increasing well loss by 3.66 percentage points. The Raman-off control also cools the *surviving* distribution relative to the loaded 10 uK state: this illustrates why temperature reduction alone cannot establish cooling efficiency.

### Validation of the new implementation

The full suite passed **80 tests in 116.24 s**, including the legacy audit and ten new ensemble checks in [`tests/test_ensemble_physics.py`](../tests/test_ensemble_physics.py). After the final implementation adjustments, all **16 ensemble and IO/GUI tests passed again in 8.42 s**. `git diff --check` passed, and the example config exactly matches the saved final-run config. The new checks compare photon propagation with an independent augmented matrix exponential, coherent propagation with a static Hamiltonian exponential, and dipole normalization with an isolated cycling-transition limit. They also check helicity conventions, wavelength/depth frequencies, population conservation, custom spin mixtures, temperature inversion and invalid-power rejection. Headless exports, parallel spatial evaluation and GUI config handling were exercised.

| Check | Effect on absolute target yield |
| --- | ---: |
| Split timestep 2 us to 0.5 us, matched four-sample reference | -5.90e-7 |
| Emission quadrature 24 to 48 | Below 1e-14 |
| Raman sideband cutoff 3 to all bound sidebands | -4.98e-7 |
| Spatial quadrature 32 to 64 samples, final settings | **+0.00713** (+0.71 percentage points) |

The timestep/recoil checks use the same reference settings as one another, with z-lattice E along x, preceding the small final polarization adjustment. Final-run probability conservation is better than 4.3e-14 and sampled density matrices remain positive. These checks establish numerical consistency of the implemented equations, not completeness of the physics.

Spatial quadrature has visible residual dependence: absolute target yield at 8 / 16 / 32 / 64 samples is 25.28 / 25.96 / 26.47 / 27.18%. Survival changes by only 0.085 percentage points from 32 to 64 samples. The quoted rough results are not precision-converged predictions or statistical confidence intervals.

An independent single-photon angular integration at the cloud center checks the Cartesian emission factorization: for the initial 10 uK thermal state, retained probabilities differ by 0.000042 to 0.000209 across the tested absorption directions and dipole channels. This is a useful local check, **not** an accumulated sequence error bound or a check of omitted coherence transfer.

### Saved outputs and reproduction

- [Final metadata and all spin populations](../results/astra_clarified/final/run_metadata.json), [time series](../results/astra_clarified/final/run_timeseries.csv), and [node-resolved 3D populations and spectra](../results/astra_clarified/final/run_ensemble.npz).
- [Final dashboard](../results/astra_clarified/final/run_dashboard.png) and [carrier calibration discrepancy](../results/astra_clarified/final/run_carrier_calibration.png); PDF copies accompany both figures.
- [Matched controls](../results/astra_clarified/control_summary.json), [final detuning comparison](../results/astra_clarified/final_selection.json), [spatial convergence](../results/astra_clarified/spatial_convergence.json), [numerical convergence](../results/astra_clarified/numerical_convergence.json), and [single-photon recoil check](../results/astra_clarified/one_photon_recoil_check.json).
- Power, detuning and polarization scan records are retained under [`results/astra_clarified/`](../results/astra_clarified/). Some earlier exploratory runs fail the final validity checks; they are not the selected prediction.

To rerun the currently selected configuration and its checks (the scripts select the output directory from `carrier_decay_mode`):

```powershell
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage final --workers 4
conda run --no-capture-output -n scripts python scripts/check_experiment_controls.py
conda run --no-capture-output -n scripts python scripts/finalize_experiment.py
conda run --no-capture-output -n scripts python scripts/check_experiment_convergence.py
conda run --no-capture-output -n scripts python scripts/check_recoil_factorization.py
conda run --no-capture-output -n scripts python -m pytest -q -p no:cacheprovider --basetemp=results/test_tmp
```

The README gives the earlier optimization stages. The selected 64-sample run took about 294 seconds with four workers on this machine. Unmeasured quantities and follow-ups are documented below; no further clarification is required to run the implemented model.

## Implementation of the clarified experiment

The clarified model is implemented in [`rb85rsc/ensemble.py`](../rb85rsc/ensemble.py), with cached ARC estimates in [`rb85rsc/far_detuned.py`](../rb85rsc/far_detuned.py). The existing 1D backend remains available for older configs. A measured-carrier config cannot silently pass through that backend and reinterpret 5 kHz as a single-path Rabi frequency.

- Geometry: B along x, pump along x, repump along z, Raman beams along +z/-z. Helicity is explicitly defined along each beam's **own** propagation direction. Both beams carry both tones with equal relative powers. Opposite helicities for these counterpropagating beams give equal laboratory Jones vectors.
- All central lattice depths are 15 uK. The optical wavelengths are 785 nm along z and 1188 nm along x/y. Depth and wavelength determine bottom frequencies of 69.04 and 45.62 kHz; the old independently supplied 70/50 kHz values are not imposed as additional constraints. The 160 MHz shift is between the two transverse lattice axes, suppressing their cross-interference.
- Spatial averaging uses a Gaussian cloud of 150 um **1/e² density radius**, hence coordinate standard deviation 75 um. All beam intensity waists are 200 um. Each sample uses its own Gaussian beam intensities and local trap depths. Raman spatial registration is averaged uniformly and remains fixed during each simulated sequence; this is not rapid temporal phase noise.
- Loaded atoms start at 10 uK in all axes, conditional on locally bound states, and uniformly over F=3. Custom spin populations still work. There is no inferred survival fraction from the preloading MOT cloud.
- The solver carries joint `P(spin,nx,ny,nz)` and a full axial Raman-pair density matrix at each transverse occupation. Optical recoil heats x/y as well as z, so transverse confinement affects survival even though the coherent Raman momentum is axial. The full 4 ms cooling + 1 ms final-pumping schedule is retained.
- Departure from **any** local bound-state manifold is absorbing. The output explicitly calls this **“well-loss fraction, assuming no recapture.”** There is no claim to predict cooling-assisted recapture or apparatus-level escape.
- ARC D1 and D2 amplitudes interfere before spontaneous Raman-scattering probabilities are formed. Equal-frequency incident Raman beams interfere in absorption. Different optical tones are averaged incoherently in spontaneous rates. Ground trap shifts include the D-line counter-rotating term. ARC's temporary database stays in memory; the portable atomic data and inferred calibration are cached in the repository.
- Outputs include all spin populations, both absolute and conditional on survival; axis-resolved excitation energies and canonical-mixture temperature proxies; scattered photons including lattice photons; and `P_target_absolute`, the absolute yield of `|3,3,nz=0>` while remaining bound in all axes. This objective is not a 3D ground-state fraction; separately named 3D ground-state yields are also exported.

### Earlier carrier calibration discrepancy (no longer a fit constraint)

This subsection records the earlier `measured_envelope` fit. The current `modeled` choice above supersedes its intensity calibration and treats the reported decay as comparison-only. The discrepancy is not an active failed-fit diagnostic in that mode; it also has not been experimentally resolved.

The measured 5 kHz frequency, 90% contrast and 1/e decay after 3–4 periods are represented by a **synthetic exponential-envelope trace**, with nominal decay after 3.5 periods (0.7 ms). This is a modeling choice based on a measurement summary, not a fit to raw experimental data. Contrast is treated as preparation/readout visibility and does not discard 10% of initially loaded atoms.

Thermal carrier overlaps, Gaussian intensity variation and uniform registration averaging are included for the stated calibration condition: the 785 nm lattice only, pumping lights off. The fitted extra homogeneous dephasing is constrained to be nonnegative. The nominal best fit gives approximately **1.394 mW per tone per Raman beam** (2.788 mW total per beam), but has **RMS population mismatch 0.174** and effectively zero extra dephasing. This model dephases too rapidly already; it cannot reproduce the reported long-lived carrier oscillations by adding technical noise.

The intensity is therefore **provisional**, and the final cooling numbers inherit this unresolved inconsistency. The synthetic reference and predicted flop trace are exported for review. A later follow-up should inspect the raw carrier trace, its addressed spatial/thermal subset, and the beam/tone geometry used for that calibration. No negative dephasing rate, artificial contrast renormalization of atom number, or presumed ARC estimate of technical laser noise is used to hide the mismatch.

At that provisional intensity, the cached phase-averaged central Raman scattering estimates are 0.346/s for `|3,3>` and 0.345/s for `|2,2>`, with a -26.8 Hz differential Raman light shift. These are ARC D1+D2 optical estimates, not measurements of technical laser noise. The final cache is [`carrier_f59ce267c8b6c120ec5faf14.json`](../rb85rsc/cache/carrier_f59ce267c8b6c120ec5faf14.json).

### Approximations that remain explicit

The 3D recoil update preserves spin/motion and inter-axis population correlations, but uses the product of one-axis emission marginals for resolved spherical dipole channels. It omits interference between different emitted spherical channels and the correlation between Cartesian recoil components of a single emitted photon. Scattering still resolves individual motional populations and therefore discards coherence transfer, including indistinguishable elastic channels. These are physical approximations, not errors bounded by the timestep test.

The trap basis uses separable isolated scalar wells. Ground-state and Raman Stark shifts are included diagonally; changes of the motional basis and off-diagonal static Stark dressing are omitted. At the inferred Raman power, the maximum additional scalar Raman standing-wave depth is about **6.8% of the central z lattice depth**, so that omission is not identically zero. The near-resonant pumping transitions' excited-state lattice shift remains the configurable common `optical.excited_state_shift_hz`; a state-resolved excited polarizability calculation has not been implemented. D-line-only optical estimates omit higher excited states and hyperfine-mediated polarizability corrections. Tunnelling, coherent Raman coupling to continuum states and collisions are absent.

Temperatures match the survivor-weighted mixture of actual local bound spectra; averaging local temperatures is avoided. A nonthermal distribution still has only a temperature **proxy**. In particular, transverse survivor cooling can be preferential loss of hotter atoms rather than transverse Raman cooling. The energy-selection contribution is exported separately. Loading survival and experimental error bars are unspecified.

The new splitting integrator uses exact static Raman unitaries and a positive uniformized photon-scattering update with an explicit loss bin. Its accuracy is controlled by `ensemble.time_step_us`, not the legacy ODE tolerances. The search covers nanowatt-to-microwatt pump powers, excludes invalid weak-excitation/secular points, and uses finite grids; it is not proof of a global optimum in the saturated regime.

Atomic conventions follow the [ARC hyperfine dipole documentation](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFS.html). The retained population-resolving elastic-decoherence approximation should be distinguished from the amplitude-level coherence treatment discussed by [Ozeri et al.](https://arxiv.org/abs/quant-ph/0502063).

## Historical 1D audit: result and interpretation

The simulator's atomic branching, weak-excitation normalization, red-sideband frequency sign, recoil probabilities and lattice spectrum pass the checks described below. I corrected several energy/temperature, magnetic, Raman-phase and diagnostic errors. The experiment remains a **model with warnings**, not a quantitatively validated prediction for the apparatus. In particular, its recoil approximation is stronger than its previous documentation admitted.

The 5 ms schedule is 4 ms of simultaneous Raman/pumping followed by 1 ms of pumping. The corrected result is:

| Quantity | Before | Corrected |
| --- | ---: | ---: |
| Survival, relative to initially trapped atoms | 0.812376682 | 0.812377333 |
| Spin-up fraction among survivors | 0.973524618 | 0.973524142 |
| Motional ground-state fraction among survivors | 0.906762473 | 0.906759692 |
| Joint target fraction among survivors | 0.885373289 | 0.885370153 |
| Joint target fraction of initially trapped atoms | 0.719256615 | 0.719254644 |
| Mean vibrational level index among survivors | 0.182609718 | 0.182620390 |
| Reported temperature proxy | 1.79831 µK | **1.59605 µK** |
| Fractional reduction of mean excitation energy | 0.895390504 | **0.893638842** |
| Claimed initial tail above the trap depth | 0.133229498 | **Unknown**; no continuum/loading model |
| Actual discarded bound-state weight | Not distinguished | **0**; all six bound levels retained |
| Claimed quanta removed per photon | 0.04781796 | **Undefined** with appreciable loss |

The small population changes do not make the reporting fixes unimportant. A harmonic energy assignment gave the wrong temperature for this anharmonic spectrum. Also, 88.54% is post-selected on survival: the useful absolute target fraction is 71.93% of the initially trapped ensemble. The initial loading efficiency is unknown. None of these are 3D ground-state fractions.

The final distribution is not canonical: its total-variation distance from the energy-matched canonical distribution is 0.0692. **1.596 µK is a proxy, not a thermometry result.**

## Corrections implemented

### 1. Actual lattice energy and canonical temperature

Files: [`rb85rsc/analysis.py`](../rb85rsc/analysis.py), [`rb85rsc/motion.py`](../rb85rsc/motion.py), [`rb85rsc/plotting.py`](../rb85rsc/plotting.py).

Previously the code identified energy with `h * trap_frequency_hz * nbar` and converted `nbar` to temperature using the infinite harmonic ladder. These formulas do not apply to the six anharmonic bound levels used by this configuration.

Now the mean excitation energy is `sum_n p_n * (E_n - E_0)`, normalized to survivors. `fractional_energy_reduction` and `cooling_power_W` use that energy. New exports include `mean_excitation_energy_J` and `mean_excitation_energy_hz`. Here the mean excitation changes from `h × 104288.727 Hz` to `h × 11092.270 Hz`.

For lattice motion, `T_equiv_K` is obtained by solving for the canonical distribution on the actual bound spectrum with the same mean energy. A canonical state initialized at 10 µK now returns 10 µK. The zero-temperature limit is checked; an energy above the uniform-distribution mean is reported as having no positive-temperature canonical fit. The harmonic formula remains in harmonic mode.

`nbar` remains a useful **mean level index**, but a transition between neighboring lattice levels does not remove a fixed `hν_t`. The historical `*_quanta_*` fields should be read as level-index changes in lattice mode.

### 2. Initial bound-state conditioning and truncation

File: [`rb85rsc/model.py`](../rb85rsc/model.py).

For `thermal_temperature` in lattice mode, the initial state is canonical **conditional on initially bound atoms**. If `n_max` omits bound levels, the discarded probability is computed from the complete bound spectrum. With all six levels retained, it is zero.

The former 13.32% estimate was `r**N` from a harmonic geometric ladder. It was neither an anharmonic continuum population nor a prediction of how many atoms fail to load. That claim has been removed. `initial_unbound_fraction` is explicitly unspecified. A capped lattice overflow is also now labeled as a mixture of omitted bound states and physical escape, rather than purely numerical loss.

`thermal_nbar` still deliberately specifies a geometric distribution over level indices; it is explicitly identified as noncanonical for an anharmonic lattice.

### 3. Breit–Rabi ground energies and the failed magnetic check

Files: [`rb85rsc/atomic.py`](../rb85rsc/atomic.py), [`rb85rsc/optical.py`](../rb85rsc/optical.py), [`rb85rsc/model.py`](../rb85rsc/model.py).

The old check tried to instantiate ARC at runtime, caught `attempt to write a readonly database`, and returned a **zero** quadratic-Zeeman error with status OK. This happened during the baseline audit. Thus its claimed zero-error check was false.

Ground energies now use the analytic Breit–Rabi expression with hyperfine splitting and g factors inferred consistently from the cached ARC data. This does not require opening ARC's writable database. Both Raman and optical detunings use those ground energies. At 1 G the correction to the linear up/down splitting is approximately **+359 Hz**, and is included in the dynamics. The legacy diagnostic key `raman_quadratic_zeeman_error_hz` now describes the correction to the linear approximation, not an omitted error.

The formula and stretched-state treatment follow [Steck, Rubidium 85 D Line Data, Eqs. 26–28](https://steck.us/alkalidata/rubidium85numbers.pdf). Independent diagonalization of `A I·J + μ_B B(g_I I_z + g_J J_z)` checks all 12 ground energies at 0, 1 and 10 G.

Excited energies and dipoles still use the weak-field basis. This correction does not implement magnetic mixing of dipole amplitudes. The existing conservative excited-state estimate at 1 G is about 0.020 natural linewidth.

### 4. Raman momentum direction and four-tone phase consistency

File: [`rb85rsc/model.py`](../rb85rsc/model.py).

On `up → down`, the atom absorbs a low-tone photon and emits a high-tone photon. Its momentum kick is `+ħ(k_low - k_high)`. The former code put that displacement directly in the `|up><down|` block; that block describes the reverse process and needs the adjoint.

The lowering operator is now constructed first, with the raising block given by its adjoint. For the documented phase definitions, with `D(η)=exp(iη(a+a†))`,

```text
D_lower = sqrt(p1_low*p2_high) exp(+iθ) D(+η)
        + sqrt(p2_low*p1_high) exp[-i(χ+θ)] D(-η)
        + [sqrt(p1_low*p1_high) + sqrt(p2_low*p2_high) exp(-iχ)] I
D_raise = D_lower†
```

Here θ belongs to the lowering crossed pathway; χ is the beam-2 minus beam-1 **high-minus-low** phase. This also fixes the sign inconsistency between χ's definition and its previous use. An independent test factorizes the pathways into low-tone and conjugate high-tone field amplitudes. A momentum-expectation test verifies the kick direction.

For the experiment's balanced beams, θ = 90° and χ = 0°, the combined operator is real; this correction leaves its populations essentially unchanged. It matters for general phase settings, unequal pathways and phase-sensitive observables. **The red-sideband frequency sign was already correct**: `ν_beat - ν_ud = +(E_n-E_m)/h` for cooling from `up,n` to `down,m<n`.

### 5. Loss selection separated from cooling efficiency

File: [`rb85rsc/analysis.py`](../rb85rsc/analysis.py).

With loss, a falling survivor mean energy can be caused by hot atoms leaving. The old quanta-per-photon number also mixed a survivor-conditioned mean change with photons counted per initial atom.

That efficiency is now undefined when overflow exceeds the configured boundary tolerance. A warning identifies the conditioning. `selection_cooling_power_W` reports the exact instantaneous selection contribution for the modeled loss hazards:

```text
P_selection = sum_(s,n) [(E_n-E_0) - mean_energy] * loss_rate_(s,n) * P_abs_(s,n) / survival
```

The total reported `cooling_power_W` remains the decrease of mean excitation energy among survivors; it includes this selection term. Its smoothed numerical derivative should not be treated as an exact instantaneous power at protocol switches.

An optical-pumping-only control illustrates the issue: `nbar` falls from 1.7456 to 1.4296 even with Raman disabled. Preferential escape can produce an apparent improvement without Raman cooling.

### 6. More accurate diagnostics and documentation

Files: [`rb85rsc/model.py`](../rb85rsc/model.py), [`rb85rsc/runner.py`](../rb85rsc/runner.py), [`README.md`](../README.md).

- The secular-rate diagnostic now uses the smallest actual motional angular spacing. The experiment gives **0.410**, versus 0.224 when divided by the harmonic bottom frequency. This is not comfortably small.
- Raman isolation uses the actual anharmonic energy differences through `max_sideband_order`, and the largest scheduled Raman amplitude. The previously omitted third sideband gives a nearest unmodeled channel about **349.3 kHz** away, or **17.5** times the conservative summed carrier scale. Polarization strengths and shifts of the omitted channels remain unknown.
- Geometry mode now warns when its derived propagation direction differs from the inactive stored direction field.
- A separate warning states the additional recoil-coherence approximation; the secular-rate test alone does not establish its validity.
- Exported assumptions and the README no longer describe every run as an infinitely deep harmonic trap. They describe lattice escape, survivor conditioning, the updated magnetic model and the changed energy observables.

## Checks that did not require a physics change

- ARC dipole strengths, forbidden transitions, branching normalization and the specified decay branches pass the existing atomic tests. Weak-excitation rates and light-shift signs pass the isolated-transition limits. Cached atomic values agree with the existing Steck comparisons to better than 0.1%.
- `power_mw = 2e-5` means **20 nW**, not 20 µW. With a 200 µm intensity waist, each pumping beam has peak intensity **0.31831 W/m²**. This supersedes the inactive `peak_intensity_w_m2 = 0.1` field.
- The largest occupied-state excited fraction is about **0.00259**, so weak excitation is well supported for the stated powers.
- The spin-pump fractions are σ+ = 0.995113814, σ− = 0.004886186, π = 0. The active repump geometry produces **pure π** polarization relative to B along x, despite zero-valued inactive spherical-fraction fields.
- Exact motional overlaps are necessary: `η_R = 0.467976`, and the harmonic diagnostic `η_R²(2nbar_initial+1) = 0.984` is not deep in the Lamb–Dicke limit. The code already uses exact overlaps rather than a first-order expansion.
- Depth 15 µK and bottom frequency 70 kHz imply lattice recoil `E_r/h = 3919.382 Hz`, lattice wavelength **774.276 nm**, and six bound levels. Spacings are approximately **65.811, 61.204, 55.865, 49.140, 38.314 kHz**.
- `carrier_rabi_hz = 5000` is **per pathway** in the four-tone model. At θ = 90°, `|D_00| = 2` and `|D_10| = 0.855397`: realized carrier and first red-sideband Rabi frequencies are about **10.000 kHz** and **4.277 kHz**, respectively.
- `beat_frequency_hz` is inactive in `sideband_offset` mode. The corrected reference beat is **3.038133954 GHz**. Optical differential shifts make the actual residual red-sideband detuning about **−568.85 Hz** during cooling, even though the reference offset is zero.
- Absorption and spontaneous-emission recoil are combined before taking motional probabilities. Dipole emission patterns, recoil energy in the harmonic limit, explicit overflow conservation and the red/carrier/blue resonance signs pass the existing tests.

## Validation and numerical convergence

Original code: **52 tests passed** before editing. Corrected code: the full then-current suite passed **69/69** in 111.78 s. After adding the independent static-Hamiltonian check, the final changed-area suite passed **47/47** in 108.19 s (all dynamics, QuTiP, IO/GUI and new audit tests). Together with the unchanged atomic/lattice checks, all **70 tests at that historical stage** passed. The expanded 80-test suite is described above. `git diff --check` also passed; its only messages concerned Windows line-ending conversion.

After the final loss-vector caching cleanup, all **18 audit regression cases** passed again. The corrected headless export was rerun and the dashboard inspected; its diagnostics now wrap and expand the figure so the additional physics warnings are not clipped.

The new regression file is [`tests/test_physics_corrections.py`](../tests/test_physics_corrections.py). It checks canonical temperature inversion, true lattice energy/power, omitted bound-state weight, population inversion, loss selection, Breit–Rabi against an independent spin Hamiltonian, optical/Raman energy consistency, Raman kick direction, arbitrary four-tone phases, geometry warnings and higher-sideband isolation. Its additional coherent test compares lattice interaction-picture propagation against a **static rotating-frame Hamiltonian matrix exponential**, including arbitrary nonzero tone phases and detuning. The existing QuTiP comparison checks the implemented Lindblad equations, not the adequacy of their physical assumptions.

For the corrected experiment:

| Check | Change in final absolute target fraction | Change in final nbar |
| --- | ---: | ---: |
| rtol 1e-7 → 1e-9; atol 1e-10 → 1e-12 | +8.56e-9 | −1.59e-8 |
| Emission quadrature 24 → 48 | <1e-14 | <1e-14 |
| Raman sideband cutoff 3 → 5, all six bound levels | −7.77e-7 | +7.97e-7 |
| n_max 10 → 20 | 0 | 0 |

Increasing n_max makes no difference because the isolated site has only six bound levels; this is not a test of continuum or tunneling physics. Final probability-conservation error is **3.11e-15**, and the minimum sampled density eigenvalue/population is **+1.51e-8**.

An independent lattice-grid refinement from `dz = 0.05 x0` to `0.025 x0` changes levels by at most **41.8 Hz**, adjacent spacings by at most **12.6 Hz**, and Raman-overlap magnitudes by at most **6.19e-4**. Doubling the exterior domain changes levels by less than **7e-5 Hz**. The existing grid is adequate for these kHz-scale couplings; these checks do not validate an isolated well as a substitute for a periodic lattice near the barrier.

The `scripts` environment initially lacked pytest and threadpoolctl. I installed pytest 9.0.3 and threadpoolctl 3.5.0 there. Actual runtime versions include Python 3.13.10, NumPy 2.3.5, SciPy 1.18.1, ARC 3.10.2 and QuTiP 5.3.1. Other installed packages were not upgraded to the repository's pinned versions.

## Sensitivities and questions to ask later

### Experimental clarification received after the audit

The user clarified the following after the historical 1D audit. These inputs are now applied in the ensemble implementation described at the start of this report. The historical 1D numerical tables still describe the original configuration.

- Bias field along x; axial lattice along z; repump propagates along z with selectable transverse linear polarization; spin pump propagates along x.
- The 5 kHz Raman input is a **measured carrier-flop Rabi frequency between |2,2> and |3,3>**, not a one-pathway calibration. Its mapping to microscopic beam intensities needs the measurement conditions and polarization/phase conventions.
- Both Raman beams come from the same laser and carry both tones with equal powers. Their RCP/LCP handedness is defined along each beam's propagation direction. The 785 nm lattice uses a separate laser. Relative spatial registration of the Raman interference pattern and lattice minima is not measured; separate laser sources alone do not imply rapid fluctuations of their standing-wave patterns.
- The z lattice uses equal-power, same-linear-polarization beams at 785 nm; axial trap frequency 70 kHz and estimated depth 15 µK.
- Additional x and y lattices use 1188 nm optical wavelength, approximately 50 kHz and 15 µK each. Both paths come from the same laser, with a 160 MHz optical-carrier offset between x and y to suppress cross-interference. The user selects the quoted depths as authoritative; frequencies should follow from depth and wavelength in the sinusoidal model.
- Beam intensity waists are 200 µm. The loaded cloud is modeled as a spherical Gaussian with 150 µm 1/e² density radius. Pump/repump powers are adjustable; the agreed initial scan is 3 nW–3 µW.
- Raman scattering, differential shift and coherence decay have not been measured. The user requests ARC-based estimates with caching. Optical-scattering/shift estimates require the Raman wavelength, geometry and intensity calibration; technical laser/field noise cannot be inferred from ARC alone.
- The measured 10 µK temperature is before loading. The user explicitly authorizes assuming the loaded atoms are also 10 µK along all axes; the preloading-to-loaded survival fraction remains unknown. For this config, ground spin populations are uniform over F=3; custom populations should remain supported.
- The 5 kHz carrier calibration was measured across the approximately 10 µK ensemble with the 785 nm lattice, the same Raman operating configuration and pumping light off. Contrast is approximately 90%; the oscillation envelope falls to 1/e in 3–4 periods, or 0.6–0.8 ms. Spatial/thermal inhomogeneity and technical dephasing can contribute to that same measured envelope.
- Spin pump and repump are independent, frequency-referenced lasers, without established mutual phase coherence. Treating their phases as incoherent is the working assumption.
- Atoms need to remain anywhere in the lattice, not in their original site. The previous isolated-site escape counter is therefore **not automatically an experimental atom-loss probability**.
- Optimize the fraction of initially loaded atoms ending in |3,3,n_z=0>, and report the selected polarizations and why they were selected, all remaining spin populations, cooling and rough loss. Retain B = 1 G and the 4 ms cooling + 1 ms pumping schedule. Repump and lattice polarization choices are optimization variables within the stated beam geometry and linear-lattice constraints.
- Subsequent modeling decisions: the user approves spatial/relative-phase averaging and authorizes treating departure from a local well as irreversible loss. Implement this as population leaving the local bound-state manifold, with no recapture. This replaces the requirement to simulate retention anywhere in the lattice for the requested rough loss estimate; the difference from apparatus-level retention remains an explicitly stated approximation.

Transverse motion can decouple from a z-only coherent Raman operator in a separable trap, but that does not remove transverse recoil heating (especially absorption from the pump along x), lattice light shifts/scattering, or their effect on retention. Whether a reduced axial model suffices depends on the requested observable. A 160 MHz x/y carrier offset can suppress stationary cross-interference; it does not remove the individual trapping potentials or their optical effects.

Provisional consistency check, assuming optical wavelengths and counterpropagating sinusoidal lattices: at 15 µK, 785 nm gives a bottom frequency of 69.04 kHz and 1188 nm gives 45.62 kHz. The latter differs from the quoted 50 kHz; frequency and estimated depth should not both be treated as exact independent inputs.

The numbered questions below are the **historical first-round questions**; the clarification above supersedes their resolved parts.

### Meaning of the remaining explicit uncertainties

These are not additional unanswered configuration questions, and no statistical error bars have yet been calculated for the revised experiment.

1. **Spatial Raman/lattice registration:** the position of the Raman interference pattern relative to occupied lattice sites is not calibrated. Different wavelengths and a finite cloud produce a distribution of local phases; predictions should use the spatial ensemble and test sensitivity to the global offset. Separate lasers alone do not establish a phase-diffusion rate.
2. **Calibration versus dephasing:** the observed carrier frequency, contrast and decay do not uniquely determine microscopic intensity and homogeneous decoherence. Thermal motional overlaps, spatial intensity variation and genuine dephasing can give similar ensemble flops. Fit/calibrate with these effects included, and report residual ambiguity rather than setting T2 equal to the observed flop-envelope time.
3. **Estimated optical scattering and shifts:** ARC supplies atomic matrix elements, energies and lifetimes. The inferred Raman intensity, polarization purity and local lattice intensity determine the experimental rates. Cached atomic calculations do not eliminate those input uncertainties or predict technical laser/magnetic noise.
4. **Retention model (assumption now selected):** the user accepts no recapture and counts departure from a local well as loss. Crossing a local site barrier can physically mean transport rather than escape from the complete three-dimensional beam envelope, so this remains a well-loss estimate under an absorbing-boundary assumption, not a calibrated apparatus-level loss probability. A conservative static lattice does not remove the above-barrier energy merely because an atom enters a neighboring identical well; relocalization requires cooling, energy transfer or a change in the potential. Cooling-assisted recapture is possible, but its probability has not been calculated. For scale, Rb-85 kinetic energy k_B × 15 µK corresponds to 0.0542 m/s and a ballistic travel time of 3.69 ms over 200 µm, comparable to the sequence duration. This is only a dimensional estimate; the actual lattice trajectory is not free flight. Treating hopping and escape together as loss is also used in site-resolved RSC modeling, for example [Hilker's thesis, Chapter 4](https://edoc.ub.uni-muenchen.de/21633/1/Hilker_Timon.pdf). Ignoring recapture increases loss relative to the otherwise same model with recapture, but does not bound additional omitted loss mechanisms.
5. **Assumed ensemble and trap calibration:** loaded temperature 10 µK, Gaussian 150 µm cloud radius and quoted 15 µK depths are the agreed nominal model. Their experimental uncertainties and position-energy correlations are not supplied. Sensitivity cases can test robustness; they are not confidence intervals without an experimental uncertainty model.
6. **Remaining theoretical approximations:** the population-resolving recoil treatment drops coherence transfer, and a single reported temperature is a proxy when the motion is nonthermal. These are model limitations to improve or bound; the passing numerical tests do not quantify their physical error. Axial cooling also does not determine transverse temperatures.

These are one-at-a-time changes from the supplied config, **not inferred measurements or recommended settings**.

| Case | Target among survivors | Survival | Target / initial trapped atoms | Final nbar |
| --- | ---: | ---: | ---: | ---: |
| Supplied config, corrected code | 0.88537 | 0.81238 | 0.71925 | 0.18262 |
| Repump along lab z instead of active y | 0.84557 | 0.65098 | 0.55044 | 0.29033 |
| Kramers–Heisenberg optical paths | 0.88223 | 0.80689 | 0.71186 | 0.18759 |
| Lattice beat phase θ = 0° | 0.30791 | 0.77096 | 0.23738 | 1.47573 |
| θ = 45° | 0.82230 | 0.80509 | 0.66203 | 0.33625 |
| θ = 135° | 0.88434 | 0.81034 | 0.71662 | 0.20307 |
| Hypothetical Raman scattering 100/s | 0.88549 | 0.80894 | 0.71630 | 0.17800 |
| Hypothetical Raman scattering 1000/s | 0.88054 | 0.78295 | 0.68942 | 0.16419 |
| Raman disabled; optical pumping only | 0.32967 | 0.81397 | 0.26834 | 1.42963 |

The scattering examples show why survivor nbar alone is misleading: it can improve while the absolute target yield worsens. The hypothetical rates are sensitivity probes, not estimates from the 783 nm wavelength.

1. **Which way does the repump really propagate?** With B along lab x, the code's transverse frame is x_B = lab y and y_B = lab z. `beam_angle_deg = 90`, `beam_azimuth_deg = 0` therefore means lab y. The stored `direction = [0,0,1]` is inactive. I preserved the active angles. If lab z is intended, set the geometry azimuth to 90°; this materially changes absorption recoil.
2. **What does the measured 5 kHz Raman Rabi frequency calibrate?** One pathway, the realized carrier, or the measured red sideband? I retained the documented one-pathway interpretation. Polarization-dependent amplitudes can differ between co-propagating and counter-propagating pathways; a scalar equal-pathway model cannot establish them from tone powers alone.
3. **Are θ and χ stabilized, and how are they defined experimentally?** I retained θ = 90°, χ = 0°. If atoms sample different phases, average complete simulation outcomes over that distribution; averaging amplitudes would erase different physics. The large phase sensitivity is real within the assumed pathway model.
4. **What is the actual lattice wavelength/geometry and trap calibration?** The supplied depth/frequency imply 774.276 nm, whereas the Raman wavelength is 783 nm. These can describe separate beams, but should be reconciled if the same light forms the lattice. State-dependent trap depths and light shifts are also not specified.
5. **What are the measured Raman scattering, differential Stark shift and coherence decay?** All are currently zero. Wavelength and calibrated Rabi frequency alone do not determine them. I preserved these explicit idealizations and flagged them rather than inventing values.
6. **Is 10 µK the temperature of the already trapped ensemble? What is the loading fraction?** The simulation now explicitly conditions on bound atoms. A continuum/loading model or measured initial survival is required for yields relative to the original MOT ensemble.
7. **Are the spin-pump and repump phase coherent with each other?** The model assumes mutual incoherence. Phase-coherent optical fields can require CPT/dark-state physics outside these rate equations.
8. **How accurately must coherent recoil and Rayleigh decoherence be modeled?** This is the most important remaining theory limitation. Individual `|s',n'><s,n|` jumps act as which-level measurements. Even in the zero-recoil limit, a same-spin elastic event should preserve motional coherence, whereas these rank-one jumps erase it. Secular averaging groups equal Bohr frequencies; it does not justify splitting a degenerate carrier into independent n-resolving jumps. Correcting this requires keeping same-spin motional density matrices for the spectator states and using grouped amplitude-level jump operators. I retained the existing population-recoil approximation, added an explicit warning, and **did not certify it from the passing tests**. For general effective-jump construction see [Reiter and Sørensen, Effective operator formalism](https://arxiv.org/abs/1112.2806). Elastic spin decoherence depends on scattering-amplitude distinguishability; see [Ozeri et al., Hyperfine Coherence in the Presence of Spontaneous Photon Scattering](https://arxiv.org/abs/quant-ph/0502063).
9. **Is isolated-site escape an appropriate loss model?** Atoms excited above one site's barrier may tunnel, travel or be recaptured rather than disappear from the apparatus. Initially 18.3% of the modeled atoms occupy the highest two bound levels, where the periodic lattice and isolated-site approximations differ most. Continuum Raman excitation and intersite motion are absent.
10. **If background heating is used later, what is its noise mechanism/spectrum?** Its value is zero here. The existing `a,a†` heating model is harmonic and remains an approximation when applied to lattice level indices; lattice force-noise/parametric-noise matrix elements are not implemented. Do not interpret nonzero lattice heating as a calibrated physical noise spectrum.

Additional approximation retained: `optical.path_model = independent` neglects interference between intermediate excited hyperfine levels. The Kramers–Heisenberg option was tested above; its change in absolute target fraction is about 0.00739. The independent-path config was preserved so this assumption remains visible.

## Artifacts and reproduction

- Baseline: [`results/astra_audit/before/run_metadata.json`](../results/astra_audit/before/run_metadata.json).
- Corrected run: [`results/astra_audit/after/run_metadata.json`](../results/astra_audit/after/run_metadata.json), [`run_timeseries.csv`](../results/astra_audit/after/run_timeseries.csv), [`run_dashboard.png`](../results/astra_audit/after/run_dashboard.png), and the joint-population NPZ in the same directory.
- All sensitivity/convergence values and finer-grid checks: [`results/astra_audit/audit_results.json`](../results/astra_audit/audit_results.json).
- Reproduction script: [`results/astra_audit/reproduce.py`](../results/astra_audit/reproduce.py).

```powershell
conda run --no-capture-output -n scripts python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp/pytest
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config tests/fixtures/audit_experiment_1d.json --headless --output results/astra_audit/after
conda run --no-capture-output -n scripts python results/astra_audit/reproduce.py
```

`before/` is historical evidence and is not regenerated by the reproduction script. The tests and numerical convergence demonstrate correct implementation of the stated approximate model; they do not settle the experimental questions or omitted coherence/lattice physics above.
