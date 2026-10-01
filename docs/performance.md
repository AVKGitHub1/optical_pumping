# Simulator performance

The 3D ensemble implementation reduces repeated array handling and setup work while retaining the same physical model, floating-point precision, and numerical controls. The active [experiment config](../examples/experiment.json) remains unchanged.

## Implementation

- The [recoil contractions](../rb85rsc/ensemble.py) use fixed permutations for the `[spin, nx, ny, nz]` population layout. Identical x, x/y, or x/y/z intermediate results are reused within a gain evaluation. Keys include exact kernel values, shape, dtype, and strides; there is no additional rounding or pruning. Spin contractions, nonunit segment scales, and the sum of scattering terms retain their original order.
- Each segment prepares its scattering rate, stay probability, and contraction graph once. Poisson weights are cached by the exact timestep. The recurrence, termination threshold, reward integration, and missing-tail correction are unchanged.
- [Quadrature nodes and weights](../rb85rsc/motion.py) are cached as read-only arrays. [D-line responses](../rb85rsc/far_detuned.py) use bounded, per-instance caches keyed by their exact input values, including mutable polarization and atomic arrays. Public response calls return writable copies so callers cannot modify cached results.
- Parallel parameter scans use their outer worker pool to evaluate points; each point runs its spatial samples sequentially. This avoids nested pools. The execution-only override leaves the saved config and resume matching unchanged. Direct GUI and headless experiments use `ensemble.workers`, and results are accumulated in their original spatial-sample order.

These changes retain the bound-state basis, coherent Raman evolution, spin/motion correlations, recoil quadrature, spatial samples, timestep, and double precision. They do not change carrier calibration or any physical approximation described in the [physics report](astra-physics-corrections.md).

## Running and choosing workers

In the GUI, expand **3D spatial ensemble**, set **workers** to **8**, and click **Run**. **Stop** cancels queued work and signals active workers between integration steps. The GUI waits for all workers to exit before enabling another run, including when a worker fails. Closing the window requests the same cleanup. Save the config to persist the worker setting. Comparisons and GUI scans use this setting too; GUI scans process parameter points sequentially while parallelizing the spatial samples.

Run from the repository root in the `scripts` environment:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --headless --no-figures --output results/performance_check
```

The exported metadata's `final.wall_time_s` measures simulation time before export and plotting. Compare this value using the same config, cached carrier calibration, machine, and worker count. Avoid running other CPU-intensive work during comparisons.

To try a different process count without editing the config file:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --headless --no-figures --set ensemble.workers=8 --output results/performance_check_8_workers
```

The override is recorded in that run's exported config. More workers are not necessarily faster; process overhead, memory bandwidth, and CPU contention matter. BLAS remains limited to one thread per simulation process. For parameter scans, `--workers` controls the outer pool instead.

## Full experiment benchmark

The 2026-10-01 comparison used the canonical 64 spatial samples, 201 output times, 2 us maximum split timestep, 24 recoil quadrature points, and the same cached carrier calibration. The unchanged reference was repository revision `2779a10efab5b8463ce70577e8e230a52f5fbbf8`. Config SHA-256: `f13e18574eef1519033dd1903b822236d8a5b6c8d367eecc2ce21c0aadf77c09`.

| Implementation | Workers | Simulation wall time |
| --- | ---: | ---: |
| Original | 4 | 268.83 s |
| Optimized | 4 | 179.38 s |
| Optimized, execution override | 8 | 92.34 s |

At the same worker count, this is **33.3% less runtime (1.50x throughput)** in this measurement. These are individual full runs on the same machine, not a statistical timing study; absolute times vary with system load. Export and plotting are excluded.

Eight workers gave **2.91x throughput relative to the original four-worker run** on this 24-logical-CPU Windows machine. The four-worker default is retained; the command above enables eight for a particular run. The [benchmark record](../results/performance/experiment_speedup.json) includes the Python/package versions, unchanged numerical settings, timing scope, and comparison counts.

All **39,735 compared arrays were bit-for-bit identical**, and **8,869 non-timing scalar entries matched exactly**, both for original versus optimized and for four versus eight workers. The comparison includes every spatial sample's population histories, final joint 3D populations and Raman density matrix, axis distributions, photon counts, loss and energy fluxes, diagnostics, and the derived ensemble observables. Only 67 wall-time entries were excluded. Exact agreement here is a regression result for this config and environment, not a claim about physical-model accuracy or cross-platform floating-point reproducibility.

## Validation

The complete suite, including GUI multiprocessing, passed **130 tests in 122.59 s** in the `scripts` environment. A full single-node trajectory comparison also confirmed exact agreement after the mixed-dtype compatibility guard: all 614 saved arrays matched bit-for-bit over 2,120 integration steps.

[Prepared recoil tests](../tests/test_ensemble_performance.py) compare the optimized contractions and integrated loss/photon rewards against independently assembled dense operators and matrix exponentials. They cover unequal axes, noncontiguous and complex arrays, interference terms, nonunit scales, repeated timesteps, input preservation, and zero scattering. [Cache tests](../tests/test_setup_cache.py) check exact results and invalidation; [scheduling tests](../tests/test_parallel_scheduling.py) check worker overrides, deterministic ordering, and unchanged saved configs and resume behavior.

[GUI multiprocessing tests](../tests/test_gui_parallel.py) launch eight real worker processes from a Qt thread, compare their output exactly with sequential execution, stop a run with queued samples, restart successfully, and close a running window. They also check cancellation between integration steps with sparse output times, pre-cancelled runs, and worker-error cleanup. No physics or numerical settings change when enabling GUI workers.
