# Documentation

Start with the [repository README](../README.md) to run **[examples/experiment.json](../examples/experiment.json)** in the `scripts` conda environment. Supporting documentation lives here; simulation configurations and saved scientific outputs remain in their own directories.

## Current experiment

| Document | Contents |
| --- | --- |
| [Simulation white paper (PDF, five pages)](../output/pdf/rb85_rsc_simulation_white_paper.pdf) | Illustrated overview of the current physics, calibration, numerical methods, reference results, and limitations |
| [Experiment workflow](experiment-workflow.md) | Running the active config, outputs, optimization, controls, and numerical checks |
| [Performance](performance.md) | Exact computational optimizations, benchmarks, validation, and worker settings |
| [RSC physics corrections and current results](astra-physics-corrections.md) | Agreed experimental inputs, implemented model, current results, validation, and unresolved physical uncertainties |
| [3D coherent optical pumping](coherent-pumping.md) | Pumping-only trajectory solver, motional coherence, Monte Carlo errors, and matched population checks |

The current config uses a spatial 3D ensemble with coherent axial Raman cooling and modeled carrier decay. Read the current physics report before interpreting temperature or loss estimates.

## Detailed reference and history

| Document | Contents |
| --- | --- |
| [Simulator reference: legacy 1D backend](simulator-reference.md) | Model equations, field conventions, configuration, exports, and preserved 1D benchmarks from the former README |
| [Historical physics audit](astra-physics-history.md) | Original audit, intermediate calibrations, convergence studies, and superseded predictions |
| [Archived documentation](archive/README.md) | Original specification and documentation/output for the separate population-rate model |

Older example configs and scan definitions are retained in [examples/arxiv/](../examples/arxiv/). Archived benchmarks describe their recorded inputs; they are not predictions for the active experiment. Scientific artifacts remain in [results/](../results/).
