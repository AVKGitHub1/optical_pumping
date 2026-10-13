# Rb-85 Raman sideband cooling simulator

Simulate optical pumping and Raman sideband cooling of Rb-85, including coherent Raman dynamics, photon recoil, finite lattice wells, and a spatial ensemble. The main outputs are spin populations, motional distributions, temperature proxies, and estimated well loss.

**[examples/experiment.json](examples/experiment.json) is the configuration to run.** It describes the clarified 3D lattice experiment with modeled carrier decay. Other configurations are retained in [examples/arxiv/](examples/arxiv/) for reference and legacy checks.

## Run the experiment

Run these commands from the repository root, always in the **`scripts` conda environment**:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --headless --output results/experiment
```

For the GUI:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --gui
```

In the GUI, open **3D spatial ensemble**, set **workers** to **8**, and click **Run**. Stop cancels the worker processes; closing a running window also stops and cleans them up. Use **Save config...** to retain the selected worker count.

The GUI also opens this config by default and returns to it on Reset. For headless runs, pass `--config` explicitly: omitting it retains the legacy built-in defaults. `python -m rb85rsc` accepts the same options as `python rb85_rsc_sim.py`.

The Trap tab provides separate X, Y, and Z lattice controls for wavelength,
waist, depth, and bottom frequency. Raman beams may counterpropagate along any
one of these lattice axes without rotating the trap. See
[Raman and lattice directions](docs/experiment-workflow.md#raman-and-lattice-directions)
for axis and polarization settings.

The **Raman (calibrated)** tab now has **State A (cooling target)** and **State B
(Raman partner)** selectors for the ground sublevels. The default remains
`|3,3> <-> |2,2>`. For `|3,3> <-> |2,1>` with linear Raman beams, select those
states, choose **Polarization input: jones**, and set both beam ellipticities to
**0 degrees**. The wavelength field accepts D1 or D2 detunings. In ensemble mode,
enter a carrier calibration for the selected transition and polarization.
See [Raman states and polarization](docs/experiment-workflow.md#raman-states-and-polarization).

If dependencies need installing into the existing environment:

```powershell
conda run --no-capture-output -n scripts python -m pip install -r requirements.txt
```

The pinned [requirements](requirements.txt) were tested with Python 3.14.6 on Windows 11. Install `ARC-Alkali-Rydberg-Calculator`, not the unrelated package named `arc`.

## Read the results

Runs export configuration and metadata JSON, time-series CSV, population arrays, and PNG/PDF figures. Ensemble runs also export per-sample 3D populations and carrier-calibration diagnostics. Inspect the validity diagnostics alongside the observables.

For a 3D ensemble, the GUI's **Thermal x/y/z** tab compares each lattice's temperature estimate, excitation energy, ground-state fraction, and cooling/heating rate. **p(n,t)** shows all three motional distributions, and **Overview** shows all three mean occupations. The thermal plots separate preferential-loss cooling from the remaining energy change; all of these curves describe surviving atoms. The same data and figures are included in exports.

`P_target_absolute` is the initially loaded fraction that remains bound in all three axes and reaches selected Raman state A with `n=0` along the Raman-driven lattice (`F=3,mF=3` and `z` in the active example). The full 3D ground-state fraction is reported separately. Temperatures are energy-matched proxies, and loss assumes no recapture after leaving a local well. See the [current physics report](docs/astra-physics-corrections.md) for saved predictions, assumptions, and remaining uncertainties.

Check the **model validity** banner before interpreting a result. Ensemble diagnostics include magnetic-field and ground-coherence checks as well as the motional-secular criterion. A numerically converged run can still lie outside these approximations. The energy-change plots separate preferential loss from the remaining dynamics; a decreasing survivor temperature does not by itself demonstrate cooling of retained atoms. "Well escape assuming no recapture" is not a prediction of apparatus-level atom loss.

For a **custom schedule**, its segment durations determine the run length; `total_duration_ms` is inactive. The GUI displays that length and disables unused duration and carrier-calibration inputs. If Raman light is off in every segment, carrier calibration is skipped and no carrier figure is exported. State A and the selected motional axis still define the reported target. Existing CSV observable names are retained for compatibility.

## Include magnetic mixing of optical states

Set **Optical model → zeeman_model → diagonalized** to include hyperfine + Zeeman mixing in the ground, D1, and D2 states. This updates optical pumping, branching probabilities, Raman calibration, and lattice light shifts consistently. State labels `F,mF` follow their zero-field branches adiabatically. The default `weak_field` retains the earlier approximation for reproducibility.

[examples/onlyOP_diagonalized.json](examples/onlyOP_diagonalized.json) is ready to run. To compare both atomic models with all other inputs fixed:

```powershell
conda run --no-capture-output -n scripts python scripts/check_finite_field.py --config examples/onlyOP.json --workers 8 --output results/onlyOP_finite_field
```

For the 3 G `onlyOP` example, the corrected 64-site run predicts **19.95% absolute target yield** and **48.02% well survival**, versus 19.31% and 46.59% with weak-field optical states. The run remains outside the motional-secular approximation; correcting magnetic mixing does not validate the entire model. See the [physics report](docs/astra-physics-corrections.md#finite-field-optical-states-2026-10-12).

## Check the motional population approximation

The separate [one-well reference solver](rb85rsc/recoil_reference.py) retains the motional density matrix in every ground spin state during optical pumping. It compares coherent recoil jumps with population-resolving jumps using the same optical rates, local spectrum, and absorbing escape boundary:

```powershell
conda run --no-capture-output -n scripts python scripts/check_coherent_recoil.py --config examples/onlyOP.json --axis z --power-scales 0.03 0.1 1 --output results/coherent_recoil_reference
```

The script writes `comparison.json`, exact configs, and time-series/density-matrix arrays in NPZ files. Power scales multiply both pumping beams; the schedule duration stays fixed. Use `--position-um X Y Z` for a different local site and `--axis x`, `y`, or `z` for its motional axis.

To check all three axes at the center plus reproducible cloud positions, and repeat each case with twice the angular quadrature:

```powershell
conda run --no-capture-output -n scripts python scripts/check_coherent_recoil.py --config examples/onlyOP.json --axes x y z --cloud-samples 4 --power-scales 0.03 0.1 1 --check-quadrature --output results/onlyOP_recoil_cloud
```

This runs 45 local comparisons and 45 quadrature repeats at the full configured duration. `survey.json` records coverage and the largest sampled differences in survival, spin/ground-state fractions, joint target yield, and survivor energy. `comparison.json` retains every case's physical-validity diagnostics, actual well parameters, numerical checks, and signed observable differences. The reference position is additional to the production Sobol cloud prefix, which uses `numerics.seed`. Failed or unsupported sites are recorded and cause a nonzero exit code. Numerical success does not remove shared physical approximations; local-axis yields are not combined into a 3D prediction. Maxima are over saved output times, not continuous-time error bounds.

The separate single-photon Cartesian-factorization check supports the same cloud sampling, per-axis lattice overrides, rotated apparatus, and configured initial motional state:

```powershell
conda run --no-capture-output -n scripts python scripts/check_recoil_factorization.py --config examples/onlyOP.json --cloud-samples 4 --output results/onlyOP_recoil_cloud/one_photon_recoil_check.json
```

The coherent benchmark requires Raman off, zero background heating, and 2–16 retained bound levels covering the full local bound spectrum. It isolates **motional coherence**: spin coherences, transverse recoil and escape, lattice scattering, and recapture are absent. Both solvers use the selected `optical.zeeman_model`. A small difference in this benchmark is not a validation of the full 3D experiment. The production ensemble solver is unchanged in its treatment of recoil. Effective-jump theory follows [Reiter and Sorensen](https://arxiv.org/abs/1112.2806).

## Documentation

| Document | Use it for |
| --- | --- |
| [Documentation index](docs/README.md) | Find current references and historical material |
| [Experiment workflow](docs/experiment-workflow.md) | Run, inspect, check convergence, and repeat optimization |
| [Performance](docs/performance.md) | Computational optimizations, benchmarks, and worker settings |
| [Current physics report](docs/astra-physics-corrections.md) | Experiment inputs, selected polarizations, results, validation, and limitations |
| [Simulator reference](docs/simulator-reference.md) | Detailed legacy 1D equations, configuration fields, outputs, and benchmarks |
| [Physics audit history](docs/astra-physics-history.md) | Original audit and intermediate calculations |
| [Archived documentation](docs/archive/README.md) | Earlier population-rate model and original specification |

## Repository layout

| Path | Contents |
| --- | --- |
| [rb85rsc/](rb85rsc/) | Simulator, CLI, GUI, analysis, plotting, and portable atomic/calibration caches |
| [rb85_rsc_sim.py](rb85_rsc_sim.py) | Compatible command-line entry point |
| [examples/experiment.json](examples/experiment.json) | Active experiment configuration |
| [examples/arxiv/](examples/arxiv/) | Archived alternative configurations and scan definitions |
| [scripts/](scripts/) | Experiment optimization and validation workflows |
| [scripts/archive/](scripts/archive/) | Legacy standalone population-rate calculation |
| [tests/](tests/) | Physics, numerical, configuration, CLI, GUI, and workflow checks |
| [docs/](docs/) | All supporting documentation |
| [results/](results/) | Saved scientific runs and audit artifacts; new runs can use a separate output directory |

## Development checks

```powershell
conda run --no-capture-output -n scripts python -m pytest -q
```

Headless code does not require importing Qt; GUI code is isolated in `rb85rsc/gui.py`. The [simulator reference](docs/simulator-reference.md#files) maps the physics modules. This repository also preserves the earlier [population-rate script](scripts/archive/rate-calcs.py); `rate-calcs.py` remains a compatibility entry point.
