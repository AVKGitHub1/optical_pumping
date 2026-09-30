# Build an Rb-85 optical-pumping and Raman sideband cooling simulator

Act as an AMO physicist and scientific Python developer. Implement a working, tunable simulator, rather than only describing a model. Use ARC, NumPy, SciPy, Matplotlib, and PyQt6. Use QuTiP where it simplifies a physically justified master-equation calculation. Deliver runnable source, example configurations, documentation, and meaningful physics checks.

The central question is: **How efficiently can an initially unpolarized Rb-85 sample be pumped into the stretched state and cooled when Raman driving, repumping, and spin-pumping operate simultaneously, and how does this compare with staged or pulsed operation?**

## 1. Experiment and scope

- Isotope: Rb-85, nuclear spin I = 5/2.
- Ground electronic state: 5S1/2; include all 12 ground sublevels: F = 2 with mF = -2,...,2 and F = 3 with mF = -3,...,3.
- Target spin: `up = |F=3,mF=3>`.
- Raman partner: `down = |F=2,mF=2>`.
- Harmonic lattice vibration frequency: `trap_frequency_hz = 70e3`. This is frequency in cycles/s, not angular frequency and not the lattice optical frequency.
- Raman beams: near 783 nm, driving `|up,n> -> |down,n-1>`.
- Repump: `5S1/2,F=2 -> 5P3/2,F'=3`.
- Spin pump: sigma+ light on `5S1/2,F=3 -> 5P3/2,F'=3`.
- Assume these two optical-pumping beams use D2 near 780 nm. This is an explicit assumption; their wavelengths are distinct from the 783 nm Raman wavelength.
- Start with an incoherent spin mixture and a thermal vibrational distribution representative of MOT + polarization-gradient cooling. Its actual populations and temperature are unknown and must be configurable.

Implement a single-atom, one-dimensional harmonic-motion model first. The reported ground-state fraction is for the modeled axis. Do not present it as a three-dimensional ground-state fraction. Use the same harmonic potential for all ground spin states initially, with this approximation documented. Collisions, photon reabsorption, tunneling, finite-depth loss, and trap anharmonicity are outside the baseline model. High-energy predictions must carry the appropriate validity diagnostic.

## 2. Atomic data and selection rules

Use `from arc import Rubidium85`; the distribution is `ARC-Alkali-Rydberg-Calculator`, not an unrelated package named `arc`. Inspect the installed ARC API and units instead of guessing method signatures. Useful APIs include `getDipoleMatrixElementHFS`, `getBranchingRatio`, `getHFSCoefficients`, and `getHFSEnergyShift`. Convert atomic-unit dipole moments to SI before calculating optical Rabi frequencies. Cross-check representative results against Steck's Rb-85 data.

Generate dipole couplings and normalized spontaneous-emission branching probabilities from atomic matrix elements. Do not assign equal branching fractions or manually force decay into the target state. Cache the atomic data, with package version and source metadata.

Include all D2 excited hyperfine levels F' = 1,2,3,4 and their magnetic sublevels in the optical transition calculation. They may be adiabatically eliminated rather than retained in the propagated state. Provide an explicitly labeled idealized F'=3-only switch for comparisons. Include ground and excited Zeeman shifts and configurable optical transition shifts. A documented weak-field Zeeman approximation is sufficient; flag fields outside its domain instead of extending it silently.

The following must emerge from the matrix elements:

- Pure sigma+ light on F=3 -> F'=3 cannot excite |3,3>, because F'=3 has no mF'=4.
- A sigma+ repump drives |2,2> -> |F'=3,mF'=3>. That excited state can decay into |3,3>, |3,2>, or |2,2>.
- The F=3 pump is needed to recover population in unwanted F=3 sublevels during repeated cooling cycles.
- On D2, |3,3> can scatter off-resonantly through |F'=4,mF'=4>. Its subsequent return to the same spin state still deposits recoil energy.

Default the repump to sigma+ as a configurable design choice. Both beams need independently adjustable sigma+, pi, and sigma- fractions relative to a stated bias-field axis. Require nonnegative fractions summing to one. Distinguish polarization impurity from beam propagation angle. If arbitrary angles are supported, make the polarization geometry physically consistent; do not label a transverse beam as pure sigma+ relative to the field without checking transversality.

## 3. Dynamics: a controlled model, not an imposed cooling curve

### Optical pumping

For the baseline, treat the two near-resonant pumping fields as mutually incoherent, with weak-excitation optical rates and eliminated excited populations. Explain that coherent population trapping between the two optical frequencies is excluded by this assumption. If the experimental fields are phase coherent, the corresponding optical coherences require an extension.

Derive excitation rates from beam intensity, dipole matrix elements, detuning, and spontaneous decay rate. With the convention `H_ge/hbar = Omega_ge/2`, the isolated weak-excitation rate should reduce to:

`R_ge = Gamma * |Omega_ge|^2 / (Gamma^2 + 4*Delta_ge^2)`.

Here Gamma is a decay rate in s^-1; Omega and Delta use rad/s. Branch each absorption event through the actual decay probabilities. Include relevant cross-excitation by each beam, or quantitatively justify terms neglected because of detuning. Document the independent excited-path approximation and its limits.

Calculate a saturation/adiabatic-elimination validity indicator for the occupied states. Flag invalid settings or use a justified saturated multilevel treatment; do not hide invalid weak-excitation calculations behind a two-level saturation factor. Compute state-dependent optical light shifts consistently within the chosen model, and permit measured shift overrides without double counting.

### Raman coupling and frequency convention

Support a primary **calibrated coupling mode** with independently specified carrier two-photon Rabi frequency, Raman coherence decay, differential light shift, geometry, and residual Raman photon-scattering rate. Wavelength alone does not determine Raman coupling or scattering. A zero default for an unknown scattering rate must be visibly labeled an idealized assumption.

Define `Omega_carrier_rad_s = 2*pi*raman_carrier_rabi_hz`. The carrier Rabi parameter is the coupling coefficient before the motional overlap is applied. In the Lamb-Dicke limit, the first red-sideband Rabi frequency is approximately `eta_R * Omega_carrier_rad_s * sqrt(n)`.

Make the Raman frequency sign unambiguous. Let:

- `nu_ud = (E_up - E_down)/h`, including the configured Zeeman and differential light shifts;
- `nu_beat = |nu_high - nu_low|`;
- `delta_beat_hz = nu_beat - nu_ud`.

Because the initial F=3 state has higher internal energy, cooling from |up,n> to |down,n-1> requires **delta_beat_hz = +trap_frequency_hz** for equal trap frequencies. A UI control called `red_sideband_offset_hz` can be defined as `delta_beat_hz - trap_frequency_hz`, so zero means the cooling resonance. Derive the rotating-frame Hamiltonian and verify this sign with energy conservation; do not blindly set the positive beat magnitude to the hyperfine frequency minus 70 kHz.

Retain coherent Raman dynamics in a reference solver, using a ground-manifold Lindblad model, QuTiP, or an equivalent justified approach. This must capture bidirectional stimulated transitions, finite linewidth, off-resonant carrier and blue-sideband excitation, and the dependence of Raman coherence on optical scattering. An always-successful, one-way `n -> n-1` jump is inadequate for comparing simultaneous and pulsed operation.

A fast population-rate solver for scans is encouraged, but optional. If implemented, derive its Raman rates by eliminating coherences in a stated regime and compare it against the coherent solver. It must not diverge when decoherence vanishes or assume an arbitrary constant linewidth independent of pumping. Reject or switch away from the rate approximation when its assumptions fail.

The baseline Raman coupling may address only the selected up/down pair. Clearly identify this approximation and require the chosen bias field/polarizations to isolate it; otherwise warn that additional Raman channels are missing. Do not require a full microscopic Raman calculation to make the calibrated mode usable.

An optional later extension may calculate Raman coupling, light shifts, and scattering from both beam powers/waists/polarizations at 783 nm. Such a mode must sum amplitudes through the relevant D1 and D2 intermediates with consistent phases and detuning conventions. It must not sum interfering Raman paths as probabilities or claim an arbitrary fitted rate was calculated from ARC.

### Motion and recoil

Use vibrational states n = 0,...,n_max and calculate:

`omega_t = 2*pi*trap_frequency_hz`

`x0 = sqrt(hbar/(2*m_Rb85*omega_t))`

`eta_R = abs(dot(k_abs - k_emit, trap_axis))*x0`.

The Raman beam geometry must determine momentum transfer; do not replace it automatically by a single optical wavevector. Prefer exact harmonic-oscillator displacement matrix elements for Raman coupling. If a truncated Lamb-Dicke expansion is offered for speed, display its validity criterion over the occupied n distribution.

Include both absorption and spontaneous-emission recoil for every optical scattering event, including events returning to the same spin state. Construct a normalized motional redistribution channel from the oscillator recoil operator and the emission angular distribution appropriate to the dipole channel. Use angular quadrature or a converged trajectory method. An isotropic emission approximation is allowed only as an explicit optional approximation, compared against dipole emission for at least one case.

Do not simply preserve n on all optical-pumping events or add a fitted temperature rise. If separate absorption/emission motional kernels are composed, justify any intermediate dephasing; otherwise combine the amplitudes consistently before forming probabilities. In a quantum implementation, document the secular assumptions behind the collapse operators. Scattering that leaves populations unchanged can still affect coherence; do not double count its decoherence with an additional phenomenological term.

Check probability that reaches the upper vibrational boundary and convergence with n_max. A truncated displacement matrix must not silently reflect escaped population and create artificial cooling. Use padding, adaptive expansion, or an explicit numerical overflow diagnostic. Numerical overflow is not physical atom loss.

For orientation, a 780 nm photon in a 70 kHz Rb-85 trap gives a single-photon Lamb-Dicke parameter near 0.235. Calculate it from constants. This is not necessarily the Raman eta, and small eta alone does not establish the Lamb-Dicke limit for a hot initial distribution.

## 4. Tunable configuration and initial state

Use a typed configuration object shared by GUI, command line, and saved JSON. All adjustable controls need units, validation, and a short explanation. Include:

| Group | Parameters |
| --- | --- |
| Trap | Frequency, n_max, trap-axis direction; optional background motional heating rate in quanta/s |
| Initial state | F=2/F=3 fractions; uniform, stretched, or user-specified mF populations; thermal nbar or temperature; optional custom p(n) |
| Magnetic field | Magnitude and quantization-axis direction, with weak-field validity limits |
| Raman | Carrier Rabi frequency, red-sideband offset, relative beam geometry, additional coherence decay rate, differential light shift, residual scattering rate and stated scattering model |
| Each pump | Enable, peak intensity or power plus 1/e^2 waist, detuning, independent pi and wrong-handed sigma impurity controls, beam direction/misalignment, optional measured light shifts |
| Timing | Total duration, preparation duration, Raman/optical pulse durations, repetitions, piecewise amplitudes, output sampling |
| Numerics | Solver, tolerances, recoil quadrature accuracy, seed where applicable, boundary tolerance |

Use `I_peak = 2*P/(pi*w^2)` for a Gaussian beam with the stated waist convention. Define optical detunings relative to named zero-field hyperfine transitions; add sublevel and trap shifts when computing individual transitions.

Provide these initial-state presets: uniform across the seven F=3 sublevels; uniform across all 12 ground sublevels; fully stretched |3,3>; custom populations. Do not confuse equal weight per hyperfine manifold with equal weight per sublevel.

For a thermal distribution use `p_n = nbar^n/(1+nbar)^(n+1)` and `nbar = 1/expm1(h*nu_t/(k_B*T))`. Report the discarded initial tail before normalization. Reject negative populations and zero-total distributions. Initialize spin and motion independently unless the user supplies a correlated distribution.

Choose clearly marked illustrative defaults for parameters not supplied by the experiment. A reasonable starting template is B = 0.5 G, nbar = 3, n_max = 40, carrier Rabi frequency = 5 kHz, duration = 10 ms, pure sigma+ pumping, and weak peak intensities around 0.1 W/m^2 per pump. Verify the numerical and physical validity of the actual default; adjust demonstration settings if needed and explain why. These are not measured values or guaranteed optimal settings. Wavelength, trap frequency, and state labels above are the supplied experimental facts.

### Required polarization-impurity knobs

Make polarization impurity a prominent, independently adjustable control for **each** of the spin-pump and repump beams. It must change the transition strengths used by the solver, not just a label or fitted depolarization rate.

In an effective spherical-component input mode, expose:

- `pi_fraction`: fraction of total beam intensity in q = 0.
- `sigma_minus_fraction`: fraction in the wrong-handed q = -1 component when the intended polarization is sigma+.
- `sigma_plus_fraction = 1 - pi_fraction - sigma_minus_fraction`, calculated and displayed.
- `total_impurity = pi_fraction + sigma_minus_fraction`, displayed as a percentage.

These are **intensity fractions**, not field amplitudes. Use square roots when constructing field amplitudes, and do not multiply dipole amplitudes directly by intensity fractions. Keep total beam intensity fixed as impurity changes. Reject fractions outside [0,1] or a sum above one instead of silently normalizing erroneous inputs.

Offer a convenient linked control `total_impurity` plus `pi_share_of_impurity`, with `pi_fraction = total_impurity*pi_share_of_impurity` and `sigma_minus_fraction = total_impurity*(1-pi_share_of_impurity)`. Editing either representation should update the other without feedback loops. Provide fine numerical entry and log-spaced nonzero settings from approximately 1e-6 to 1e-1, plus an explicit zero-impurity option and access to the full physical range. Include pure sigma+, pi-contaminated, and wrong-handed-contaminated presets.

Keep a distinct **geometry/misalignment mode** that accepts a physically transverse Jones polarization in the beam frame and rotates it into the bias-field spherical basis. Expose beam-to-field angle, ellipticity/handedness, and polarization-ellipse orientation, and display the resulting three intensity fractions. In this mode the fractions are derived rather than independently overridden. At exact propagation along the field, changing ellipticity creates opposite-handed contamination but cannot create a pi field component; tilting the axis can. Label freely specified spherical fractions as an effective illumination model when no single transverse plane wave realizes them. Do not count the same impurity once through fractions and again through geometry.

In calibrated Raman mode, show that Raman polarization imperfections are not inferred from a single calibrated Rabi frequency. Their modeled effects must enter explicitly through calibrated unwanted couplings, dephasing, or scattering. If microscopic Raman mode is implemented, give each Raman beam its own physically consistent polarization controls and recalculate allowed/undesired Raman paths and shifts.

Provide a dedicated impurity scan: sweep spin-pump impurity and repump impurity separately, and offer a two-dimensional scan of both. Compare pi contamination with opposite-handed contamination at the same total impurity and fixed beam power. Plot final P_up, P_n0, P_target, nbar, and target-state scattering rate. Save the complete polarization specification with every result. Include sample impurity values 0, 0.001, 0.01, and 0.05, clearly expressed as fractions or percentages in each context.

## 5. Required operating protocols

Implement, using the same physical parameters and initial conditions:

1. **Optical pumping only:** both optical beams, Raman off.
2. **Continuous combined cooling:** both optical beams and Raman on together.
3. **Preparation followed by continuous cooling:** initial spin preparation, then simultaneous weak pumping and Raman driving.
4. **Pulsed cooling:** alternate Raman pulses and optical-reset pulses, with independent durations and repetition count.
5. **Controls:** Raman only; repump plus Raman with the F=3 pump disabled; all fields off.

Keep optical branching and recoil active during reset pulses. Never force a perfect spin reset between pulses. Propagate state continuously through schedule changes, integrate exactly to switch times, and preserve accumulated photon counts. Include preparation time when comparing protocols at equal total wall-clock duration.

## 6. Outputs and definitions of efficiency

From the joint probabilities `P(F,mF,n,t)`, calculate and plot:

- Every spin-sublevel population, plus total F=2 and F=3 populations.
- Spin-pumping efficiency: `P_up(t) = sum_n P(3,3,n,t)`.
- Motional distribution p(n,t), mean nbar(t), and motional ground-state probability `P_n0(t) = sum_(F,mF) P(F,mF,0,t)`.
- Joint success probability: `P_target(t) = P(3,3,0,t)`. Compute it directly; it is generally not `P_up * P_n0` because spin and motion become correlated.
- Conditional ground-state probability `P(n=0 | up)`, undefined when P_up is zero.
- Fractional motional-energy reduction `1 - nbar(t)/nbar(0)` for nonzero initial nbar, labeled as such; allow negative values when the sample heats.
- Net cooling rate in quanta/s and energy/s, with the derivative/smoothing method documented.
- Integrated scattered-photon counts separately for spin pump, repump, and modeled Raman scattering; optionally net quanta removed per photon, with zero-denominator handling.
- Time to a user-defined spin-polarization threshold and to a joint-success threshold; report "not reached" if appropriate.
- Thermal-equivalent temperature from nbar, explicitly labeled as a proxy if p(n) is not thermal.

Include initial/final spin bars, time traces, a p(n,t) heatmap, and a final joint spin-motion heatmap. Show conservation error, motional truncation diagnostics, and model-validity status alongside the results. If physical loss is added later, distinguish unconditional success from conditional-on-survival probabilities.

## 7. Desktop interface, scans, and reproducibility

Build a PyQt6 window with grouped parameter controls, protocol selector, Run/Stop/Reset controls, progress/status, and embedded Matplotlib plots. Use the Qt6-compatible Matplotlib canvas (`backend_qtagg`). Keep simulation work off the GUI thread and update plots through signals. Stop must work during long integrations/scans, not only after the entire job finishes. Separate physics from widget code and support headless operation without importing Qt.

Support JSON save/load and export of time-series CSV, joint distributions in NPZ, plots in PNG/PDF, and metadata containing the exact configuration, constants, assumptions, solver/package versions, diagnostics, and seed.

Provide 1D and 2D parameter scans. Include at least pump/repump intensity versus Raman coupling, the required polarization-impurity scans above, and a Raman-detuning scan. Plot final P_up, P_n0, P_target, nbar, and photon count. Use equal total durations and consistent starting distributions for protocol comparisons. Mark unconverged or physically invalid points rather than ranking them as optima. Cache atomic and recoil calculations when their inputs have not changed.

## 8. Physics checks and acceptance criteria

Write focused numerical checks for the actual physical risks:

1. Atomic branching probabilities are nonnegative and sum to one; forbidden dipole elements vanish. Validate the three decay destinations of |F'=3,mF'=3> and the closed F'=4,mF'=4 cycling transition.
2. In an idealized F'=3-only, pure-sigma+ model with Raman off, |3,3> is dark and the initial mixture approaches that state when both pumps address the necessary transitions. Including F'=4 or polarization impurity creates the appropriate scattering channels; it need not always reduce spin polarization, since cycling scattering can heat without changing spin.
3. With every drive and background heating off, populations and nbar remain constant. Raman-only evolution is coherent and reversible, not irreversible relaxation to a fabricated cooling equilibrium.
4. A selected |up,n> red-sideband resonance transfers to |down,n-1>; |up,0> has no resonant n=-1 partner. Verify the positive-beat frequency convention, carrier position, and blue sideband. Off-resonant excitation must remain possible in the full model.
5. An ideal isolated red-sideband pi pulse removes one vibrational quantum from a prepared n>0 state. Optical reset adds the recoil predicted by its channel; it is not automatically recoil-free.
6. Trace/probability conservation, positivity within numerical tolerance, and branch/count consistency hold. Do not clip large negative populations or repeatedly renormalize to conceal numerical failures.
7. Compare n_max, time-step/tolerance, and recoil-quadrature refinements on representative runs. Document convergence tolerances and report the unresolved initial tail and boundary population.
8. If a rate backend exists, compare against the coherent reference in its valid regime and demonstrate its failure flag outside that regime. Increased optical scattering must enter Raman decoherence consistently; test the strong-dephasing suppression limit in a simplified model without violating optical weak-excitation assumptions.
9. Demonstrate a net-cooling regime with justified parameters and at least one failure/heating regime. Report actual computed results; do not tune plotted data or hard-code an expected final efficiency.
10. Verify headless execution, configuration round-trip, export, and a GUI smoke test where a display is available. State explicitly if interactive GUI behavior could not be tested.
11. Check polarization bookkeeping: fractions sum to one, total optical power is conserved as impurity changes, and the zero-impurity limit reproduces the pure-sigma+ calculation. Verify that pi and sigma- contamination open the correct distinct transitions from |3,3>. In geometry mode verify the aligned circular, aligned linear, and tilted-axis limits. A change to repump impurity must not silently change the spin-pump settings.

## 9. Deliverables and working approach

Provide a main entry point such as `rb85_rsc_sim.py`. Small supporting modules for atomic data, dynamics, GUI, and tests are welcome if they improve clarity; avoid unnecessary application infrastructure. Include dependency instructions with tested versions, a default JSON configuration, example protocol configurations, and a README explaining equations, units, assumptions, and how to interpret the plots.

Expected usage should be comparable to:

```text
python rb85_rsc_sim.py --gui
python rb85_rsc_sim.py --config examples/continuous.json --headless --output results/continuous
python rb85_rsc_sim.py --compare-protocols --config examples/default.json --output results/comparison
python rb85_rsc_sim.py --scan-config examples/intensity_scan.json --headless --output results/scan
```

First implement and validate the atomic transition graph and a small coherent cooling model, then add recoil and all spin states, protocols, exports, and GUI. Use sparse operators and sensible basis sizes. Avoid propagating optical-frequency oscillations or a huge explicit excited-state density matrix when adiabatic elimination is valid. Show estimated resource cost before expensive runs.

Proceed using labeled assumptions for unknown experimental inputs instead of blocking on every missing parameter. Ask only if a missing choice prevents a meaningful implementation. Complete the baseline and report what was actually run, representative results, accuracy checks, and limitations. Optional microscopic Raman calculations, D1 pumping, high-field state mixing, and full 3D motion must not delay delivery of the working 1D calibrated simulator.

## 10. References to verify during implementation

Use current official APIs and inspect installed versions. These references support atomic data, library interfaces, and the continuous-cooling concept; the proposed simulator still requires its own derivation and validation.

- [Steck: Rubidium 85 D Line Data](https://steck.us/alkalidata/rubidium85numbers.pdf)
- [ARC: hyperfine dipole matrix elements](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFS.html)
- [ARC: hyperfine branching ratios](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getBranchingRatio.html)
- [ARC: atomic functions and hyperfine data](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/alkali_atom_functions.html)
- [QuTiP: Lindblad master-equation solver](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html)
- [QuTiP: time-dependent Hamiltonians and collapse operators](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-time.html)
- [Matplotlib: embedding in Qt](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html)
- [Continuous Raman sideband cooling, Phys. Rev. Research 5, 023022 (2023)](https://doi.org/10.1103/PhysRevResearch.5.023022) (a trapped-ion demonstration, not a direct validation of this Rb-85 scheme).
