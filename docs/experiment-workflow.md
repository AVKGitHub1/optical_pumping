# Experiment workflow

The active configuration is **[examples/experiment.json](../examples/experiment.json)**. It defines the 785 nm axial lattice, 1188 nm transverse lattices, spatial cloud and beam envelopes, initial spin/motional populations, pumping sequence, and modeled carrier calibration. The [current physics report](astra-physics-corrections.md) records the inputs, chosen polarizations, saved predictions, and physical limitations.

All commands below run from the repository root in the **`scripts` conda environment**. They use explicit paths so that archived alternatives cannot be selected accidentally.

## Run and inspect

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --headless --output results/experiment
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --gui
```

Choose a separate output directory for runs you want to compare. The root entry point and `python -m rb85rsc` have the same command-line options. The GUI defaults to this experiment; headless runs without `--config` retain the legacy built-in defaults.

| Output | Contents |
| --- | --- |
| `run_config.json` | Exact inputs used for the run |
| `run_metadata.json` | Final metrics, all spin populations, calibration assumptions, and validity diagnostics |
| `run_timeseries.csv` | Time-dependent observables |
| `run_joint.npz` | Axial joint spin/motional population arrays |
| `run_ensemble.npz` | Per-node local spectra, axis distributions, and joint 3D populations |
| `run_dashboard.png` / `.pdf` | Population, cooling, scattering, and validity summary |
| `run_carrier_calibration.png` / `.pdf` | Modeled carrier trace and reported-envelope comparison |

The main objective is `P_target_absolute`: initially loaded atoms that remain in all three wells and end in `F=3,mF=3,n=0` along the Raman-driven lattice (`z` in the active example). Read absolute yields and survivor-conditioned fractions separately. `T_equiv_axes_K` contains finite-spectrum, energy-matched temperature proxies; a nonthermal distribution can have the same value. Leaving any local well counts as irreversible loss under this model.

## Raman and lattice directions

The **Trap** tab has separate **X lattice**, **Y lattice**, and **Z lattice**
sections. Each sets its own wavelength, beam waist, depth, and bottom frequency.
Choose **Set confinement using: Depth** or **Frequency**; wavelength and the
chosen value determine the other for a sinusoidal lattice. The calculated field
is read-only. These settings are saved under `trap.lattice_x`, `trap.lattice_y`,
and `trap.lattice_z`, with `wavelength_nm`, `waist_um`, `depth_uk`, `frequency_hz`,
and `confinement` (`depth` or `frequency`). Unspecified per-axis values in old
JSON files inherit the previous axial/transverse settings. The shared legacy
controls are hidden in the ensemble GUI.

Ensemble Raman beams can counterpropagate along any **one of the three lattice
axes**. Set `raman.beam_low_direction` and `raman.beam_high_direction` to opposite
vectors along the selected axis; either ordering and nonunit vectors work. The
solver selects that well's actual wavelength, local depth, motional spectrum,
beam envelope, and recoil projection. It does not rotate or overwrite the trap.
Oblique beams coupling multiple lattice axes coherently are still unsupported.

The ensemble's `x,y,z` labels describe a local lattice frame: `z` is `trap.axis`.
`ensemble.transverse_reference_direction` (default lab +x) is projected
perpendicular to it to define local `x`; local `y` is `z cross x`. If the reference
is parallel to the trap axis, the least aligned lab coordinate axis is used
instead (ties choose x, then y, then z). The existing +z configuration therefore
retains the original lab x,y,z frame.

`ensemble.lattice_linear_polarizations` still contains **lab vectors**, in local
x,y,z lattice order. Each must be perpendicular to its lattice direction. To
drive the x lattice of the existing lab-aligned trap, keep `trap.axis = [0,0,1]`
and use:

```text
raman.beam_low_direction                 = [1, 0, 0]
raman.beam_high_direction                = [-1, 0, 0]
```

Set the magnetic direction and pumping geometry to match the apparatus as well.
In particular, the active example has the magnetic field along x: Raman beams
parallel to that field cannot drive the modeled `|3,3> -> |2,2>` transition with
the available transverse polarizations. A positive measured carrier frequency
in that geometry is inconsistent, and calibration reports this explicitly.
For an example with nonzero x-axis coupling, use a magnetic direction such as
`[0,1,0]` and set the pumping directions/polarizations for that field.

`initial` still describes preparation in the z lattice; x and y start at
`ensemble.transverse_temperature_k`. Carrier calibration uses only the driven
lattice, its waist and spectrum, and that axis's configured temperature. The
supplied carrier measurement must correspond to this geometry and confinement.

Saved metadata includes `model_summary.lattice_axes_lab`, and the ensemble NPZ
includes `lattice_axes_lab` with rows in local x,y,z order. `raman_axis` identifies
the driven lattice. `n`, `P_n0`, target yields, and the primary cooling metrics
refer to this axis; three-axis temperatures and `P_final_spin_nx_ny_nz` retain
the original lattice order. The model summary also records the resolved
wavelength, frequency, depth, and waist for every lattice.

## Numerical controls and calibration

`ensemble.samples` controls deterministic Sobol spatial quadrature. `ensemble.time_step_us` controls the split integrator; legacy ODE tolerances `rtol` and `atol` do not control this backend. `ensemble.workers` enables parallel samples in both GUI and headless runs. In the GUI, expand **3D spatial ensemble**, set **workers** to **8**, and click **Run**. Stop cancels queued and active samples, and the GUI waits for worker cleanup before allowing another run. Closing a running window also cancels and joins the workers. **Save config...** retains your worker choice; the active example continues to default to four.

Parallel parameter scans and optimization stages use one outer process pool, with sequential spatial samples within each point. Their `--workers` setting controls the total simulation-worker budget without changing the saved point config. See [performance and validation](performance.md) for benchmarks and direct-run worker overrides.

GUI scans visit parameter points sequentially and use the chosen ensemble workers within each point. GUI protocol comparisons also honor this worker count.

For ensemble runs, each lattice's `confinement` selects depth or frequency as the input. In the legacy 1D backend, `trap.wavelength_nm` and depth determine the bottom frequency when wavelength is supplied. `n_max` must include all local bound levels for ensemble runs.

The current `raman.carrier_decay_mode = modeled` derives carrier dephasing from spatial and thermal averaging. The supplied carrier frequency uses the RMS local-frequency/initial-curvature convention, and no homogeneous damping is fitted to the reported envelope. Contrast is preparation/readout visibility. Configurations that omit this field retain the historical `measured_envelope` mode. Atomic data and carrier calibration are cached under [rb85rsc/cache/](../rb85rsc/cache/).

## Rerun the saved workflow

To run the current experiment through the workflow script and produce spatial-prefix convergence output:

```powershell
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage final --workers 4
```

This uses the current config's physics inputs, sets 64 spatial samples and 201 output times, applies the requested worker count, and rewrites `examples/experiment.json` with the normalized final-run settings. Use the direct CLI command above to run a config without rewriting it. The optimization scripts write modeled-decay results to `results/astra_modeled_decay/` and historical measured-envelope results to `results/astra_clarified/`. These paths already contain scientific artifacts; copy results you want to retain before repeating a workflow that writes to the same location.

For matched controls and numerical checks:

```powershell
conda run --no-capture-output -n scripts python scripts/check_experiment_controls.py
conda run --no-capture-output -n scripts python scripts/check_experiment_convergence.py
conda run --no-capture-output -n scripts python scripts/check_recoil_factorization.py
```

The controls compare Raman-off and zero-offset sequences with the same inputs. The convergence script checks timestep, emission quadrature, and sideband cutoff on a smaller matched ensemble. The recoil script checks one-photon Cartesian factorization; it is not a sequence-wide physical error bound.

## Repeat optimization

Optimization is optional. The active settings were selected before the change to modeled carrier decay; the calibration update alone does not establish a new optimum.

Run the stages in order:

```powershell
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage screen
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage refine
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage detuning
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage spatial_refine
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage polarizations
```

Scan resumes reuse only matching full configurations. Invalid weak-excitation or motional-secular points are excluded from selection. To explicitly promote the saved winner into `examples/experiment.json` and run it:

```powershell
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage final --use-selected
```

If comparing the chosen detuning against zero offset, first generate fresh matching controls, then run:

```powershell
conda run --no-capture-output -n scripts python scripts/check_experiment_controls.py
conda run --no-capture-output -n scripts python scripts/finalize_experiment.py
```

`finalize_experiment.py` selects between the matching final and zero-offset runs, updates the active config/results, and archives a displaced final. It rejects mismatched controls or occupied archive destinations. Control generation itself does not change the active config.

Other configurations and standalone scan definitions are preserved in [examples/arxiv/](../examples/arxiv/). Their purpose and the legacy 1D command options are covered in the [simulator reference](simulator-reference.md).
