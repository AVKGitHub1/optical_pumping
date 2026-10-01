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

If dependencies need installing into the existing environment:

```powershell
conda run --no-capture-output -n scripts python -m pip install -r requirements.txt
```

The pinned [requirements](requirements.txt) were tested with Python 3.14.6 on Windows 11. Install `ARC-Alkali-Rydberg-Calculator`, not the unrelated package named `arc`.

## Read the results

Runs export configuration and metadata JSON, time-series CSV, population arrays, and PNG/PDF figures. Ensemble runs also export per-sample 3D populations and carrier-calibration diagnostics. Inspect the validity diagnostics alongside the observables.

`P_target_absolute` is the initially loaded fraction that remains bound in all three axes and reaches `F=3,mF=3,nz=0`. It is an **axial** cooling target; the full 3D ground-state fraction is reported separately. Temperatures are energy-matched proxies, and loss assumes no recapture after leaving a local well. See the [current physics report](docs/astra-physics-corrections.md) for saved predictions, assumptions, and remaining uncertainties.

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
