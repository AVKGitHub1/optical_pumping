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
| `run_ensemble.npz` | Time grid, survival, aggregate x/y/z marginals, per-node spectra/distributions, and final joint 3D populations |
| `run_dashboard.png` / `.pdf` | Population, cooling, scattering, and validity summary |
| `run_carrier_calibration.png` / `.pdf` | Modeled carrier trace and reported-envelope comparison |
| `run_thermal_axes.png` / `.pdf` | x/y/z temperatures, excitation energies, ground fractions, and cooling/heating with loss selection |
| `run_motional_axes.png` / `.pdf` | x/y/z motional population maps through time |

The main objective is `P_target_absolute`: initially loaded atoms that remain in all three wells and end in Raman state A with `n=0` along the Raman-driven lattice (`F=3,mF=3` and `z` in the active example). Read absolute yields and survivor-conditioned fractions separately. `T_equiv_axes_K` contains finite-spectrum, energy-matched temperature proxies; a nonthermal distribution can have the same value. Leaving any local well counts as irreversible loss under this model.

## Thermal motion in all three lattices

After an ensemble run, open **Thermal x/y/z**. The upper panels compare temperature estimates, mean excitation energy above each local ground state, and the fraction in each axis's ground state. **Overview** and **Time traces** show all three mean occupations; **p(n,t)** shows the three distributions. Labels use the local lattice frame, and mark the Raman-driven axis. A ground-state fraction for one axis is not the joint 3D ground-state fraction.

Temperatures are obtained at every sample by matching that axis's mean energy to a mixture of its local bound-state spectra, using the spatial populations that survive at that time. They are not averages of local fitted temperatures and do not imply equilibration. Gaps mean no finite temperature estimate is available (for example, the energy exceeds the positive-temperature canonical range, no excited bound levels exist, or no atoms survive).

The lower panels show energy removal in microkelvin per millisecond: positive is cooling, negative is heating. **Net** is the smoothed derivative of survivor mean energy. **Loss selection** is the instantaneous contribution from preferential escape. **Net minus loss** retains Raman and recoil dynamics, so a decreasing temperature alone does not demonstrate cooling of retained atoms. The subtraction is approximate near sharp pulse edges because only the net derivative is smoothed.

`run_timeseries.csv` includes `T_equiv_x_K`, `nbar_x`, `P_n0_x`, `P_n0_x_absolute`, `mean_excitation_energy_x_hz`, `mean_excitation_energy_x_J`, `cooling_power_x_W`, `selection_cooling_power_x_W`, and `bound_cooling_power_x_W`, with matching y/z columns. `bound_cooling_power` denotes net minus loss selection. Energies use actual local level spacings, not mean occupation times a single trap frequency. In `run_ensemble.npz`, `pn_x` (and y/z) sums to one among survivors, whereas `pn_x_absolute` sums to `survival`; both have shape `(time, level)` and use `t_s`. Conditional observables are undefined once survival is zero. The solver still treats only the selected Raman axis coherently; adding these plots does not introduce Raman cooling on the other axes.

## Raman states and polarization

In **Raman (calibrated)**, choose **State A (cooling target)** and **State B
(Raman partner)** from the `F=2` and `F=3` ground sublevels. They are saved as
`raman.state_up: [F, mF]` and `raman.state_down: [F, mF]`. States must differ and
have `|delta mF| <= 2`; the actual polarization can impose further selection
rules. Old configurations default to `[3,3]` and `[2,2]`.

The selected pair determines the coherent block, Breit-Rabi splitting, pump and
Raman differential shifts, carrier calibration, and state labels. Population
arrays retain the same order: `F=2,mF=-2..2`, then `F=3,mF=-3..3`. `P_up`,
`P_down`, and `P_target` refer respectively to A, B, and A with motional `n=0`.
The exported model summary records both states and their indices; the joint NPZ
also records the spin order of the final Raman density-matrix block.

Changing this pair does not prepare atoms or retune the optical pumps. Under
**Initial state**, `spin_preset = raman_up` prepares A; `stretched` still means
`|3,3>`. Optical pumping still follows the configured pump frequencies and
polarizations.

For the `|3,3> <-> |2,1>` linear-polarization case, select those two states and
choose **Polarization input: jones**. Set both beam ellipticities to `0` degrees;
orientations specify their linear polarization directions in the transverse
beam frames. `+45` and `-45` degrees give circular polarization. The default
**helicity** mode preserves existing `beam_helicities` settings. Both tones in
each beam share that beam's polarization.

`raman.wavelength_nm` sets the absorbed tone's optical wavelength, including
detuning from D1 or D2 in the ensemble atomic calculation. With
`frequency_mode = sideband_offset`, the beat follows the selected pair and its
red sideband. In `absolute_beat` mode it is the signed frequency **emitted tone
minus absorbed tone** for A -> B; reversing the internal energy ordering can
make this negative. The JSON names `low` and `high` are retained for compatibility
and assume A is higher in energy than B.

The ensemble carrier measurement must belong to the selected pair, wavelength,
and polarization; the simulator does not reinterpret a measurement on another
transition. A zero atomic coupling produces an error naming the selected pair.
Legacy `calibration = pathway` uses the supplied Rabi frequency directly and
does not calculate its polarization dependence.

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

Ensemble Raman beams may point in **any lab direction**. Set
`raman.beam_low_direction` and `raman.beam_high_direction` to the two propagation
vectors; nonunit vectors are normalized. Counterpropagating, oblique, crossed,
and copropagating geometries are accepted, with either
`tone_layout = one_tone_per_beam` or `both_tones_both_beams`. The solver keeps
coherent Raman coupling across every lattice axis onto which the wavevector
difference projects, using each well's actual spectrum and displacement matrix.
Beam envelopes and recoil use the configured lab directions. The trap is not
rotated or overwritten.

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

For counterpropagating beams tilted between the local x and z axes of this trap,
use `[1,0,1]` and `[-1,0,-1]`; both wells then participate in coherent motion.

Set the magnetic direction and pumping geometry to match the apparatus as well.
In particular, the active example has the magnetic field along x: Raman beams
parallel to that field cannot drive the default `|3,3> -> |2,2>` transition with
the available transverse polarizations. A positive measured carrier frequency
in that geometry is inconsistent, and calibration reports this explicitly.
For an example with nonzero x-axis coupling, use a magnetic direction such as
`[0,1,0]` and set the pumping directions/polarizations for that field.

`initial` still describes preparation in the z lattice; x and y start at
`ensemble.transverse_temperature_k`. Carrier calibration includes the thermal
carrier overlaps of every coupled lattice and the envelopes of both beams.
The supplied carrier measurement must correspond to this geometry and
confinement. Coupling several lattice axes increases the coherent basis size
and can substantially increase runtime and memory use. In this product basis,
`raman.max_sideband_order` limits the total change
`|nx-mx| + |ny-my| + |nz-mz|`; increase it to check sideband convergence.

Saved metadata includes `model_summary.lattice_axes_lab`, and the ensemble NPZ
includes `lattice_axes_lab` with rows in local x,y,z order. `raman_axis` identifies
the lattice with the largest absolute wavevector-difference projection; this
is the reference for `sideband_offset` tuning and for the existing `n`, `P_n0`,
target yields, and primary cooling metrics. Ties select x, then y, then z;
copropagating beams select the largest projection of their propagation direction
as the reporting reference. `raman_coupled_axes` records
all coherently coupled axes. Three-axis temperatures and
`P_final_spin_nx_ny_nz` retain the original lattice order. The model summary also
records the beam directions and each lattice's resolved wavelength, frequency,
depth, and waist.

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
