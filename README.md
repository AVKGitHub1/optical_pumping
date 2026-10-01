# Rb-85 optical pumping + Raman sideband cooling simulator

A tunable, single-atom simulator with a legacy **one-dimensional** backend and an experimental **3D recoil / spatial ensemble** backend: how efficiently can an initially unpolarized
Rb-85 sample be pumped into `|up> = |F=3, mF=3>` and cooled to the vibrational ground state when Raman driving,
repumping, and spin pumping run simultaneously? It also compares that with staged and pulsed operation.

* Physics is in [rb85rsc/](rb85rsc/), the entry point is [rb85_rsc_sim.py](rb85_rsc_sim.py), example configurations are in [examples/](examples/), and the physics checks are in [tests/](tests/).
* The earlier population-rate script is still available as [rate-calcs.py](rate-calcs.py), with its own documentation in [README_rate_calcs.md](README_rate_calcs.md) and [physics.md](physics.md).

> **Scope.** `P_n0` and `P_target` refer to the *modeled trap axis only*. They are not three-dimensional ground-state
> fractions. All ground spin states share one configured potential. Lattice mode includes anharmonic bound levels
> and recoil escape from an isolated site; collisions, reabsorption and tunneling are outside the model. For harmonic motion without a supplied depth, high-energy results carry a
> "harmonic approximation not assessed" diagnostic.

### Clarified experiment

`examples/experiment.json` uses the experimental ensemble backend. Run commands in the `scripts` conda environment:

```powershell
conda run --no-capture-output -n scripts python rb85_rsc_sim.py --config examples/experiment.json --headless --output results/experiment
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage screen
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage refine
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage detuning
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage spatial_refine
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage polarizations
conda run --no-capture-output -n scripts python scripts/optimize_experiment.py --stage final --use-selected
conda run --no-capture-output -n scripts python scripts/check_experiment_controls.py
conda run --no-capture-output -n scripts python scripts/finalize_experiment.py
```

Use `--stage final` without `--use-selected` to rerun the current example config. The explicit flag promotes the saved optimization winner. Scan resumes reuse only matching configs; detuning selection requires controls with matching physics and numerical settings. Current results and limitations are summarized in [astra-physics-corrections.md](astra-physics-corrections.md); intermediate calculations are preserved in the [audit history](docs/astra-physics-history.md).

This mode includes 785 nm z and 1188 nm x/y lattices, a Gaussian cloud and beam envelopes, static Raman registration averaging, and joint `P(spin,nx,ny,nz)` populations. It retains coherent axial Raman dynamics at each transverse occupation. Exiting any local bound manifold counts as **well loss with no recapture**. All yields refer to initially loaded atoms; no loading-efficiency prediction is made.

When `trap.wavelength_nm` is supplied, wavelength and depth determine the bottom frequency. The retained `frequency_hz` field is inactive. At 15 uK the central bottom frequencies are 69.04 kHz (z) and 45.62 kHz (x/y). `n_max` must retain every local bound level for ensemble runs.

`raman.calibration = measured_carrier` uses the supplied 5 kHz carrier reference. The example sets `raman.carrier_decay_mode = modeled`: spatial/thermal averaging predicts the decay, and **no homogeneous damping is fitted to the reported envelope**. With no raw trace, 5 kHz is interpreted as the RMS local carrier frequency, fixing the initial quadratic rise of the ensemble population. Cached ARC D1+D2 amplitudes, thermal carrier overlaps, beam envelopes and spatial phases then imply 1.363 mW per tone per Raman beam. The modeled nonexponential envelope first crosses 1/e at about 89 us. The reported 3.5-period decay is retained only for comparison; it does not constrain intensity or dynamics. Contrast is preparation/readout visibility and never removes atoms. Explicit `extra_coherence_decay_rate_s` remains available for independently specified technical noise (zero in this config).

For backward compatibility, configs without `carrier_decay_mode` retain `measured_envelope`, which fits a synthetic trace reconstructed from the reported frequency/contrast/decay and flags poor fits. Modeling the decay does not establish experimental agreement: the new calibration convention is an explicit assumption. The optimization/check scripts separate modeled-decay outputs under `results/astra_modeled_decay/` from the earlier measured-envelope results under `results/astra_clarified/`. The current power/polarization settings are retained from the earlier search; the revised calibration alone does not establish a new optimum.

`ensemble.samples` controls deterministic Sobol spatial quadrature. `ensemble.time_step_us` controls the positivity-preserving split integrator; halve it to check convergence. Legacy ODE `rtol`/`atol` do not control this backend. `ensemble.workers` permits parallel headless samples; interactive cancellable runs remain sequential. ARC's temporary database is kept in memory; portable atomic and calibration results are saved under `rb85rsc/cache/`.

The optimized observable is `P_target_absolute = P(remaining in all three wells AND spin=3,3 AND nz=0)` per initially loaded atom. Spin populations are exported both absolutely and conditional on survival. `T_equiv_axes_K` gives energy-matched finite-bound-spectrum mixture proxies for x/y/z, not thermometry or proof of a thermal state. `run_ensemble.npz` stores each node's local spectra, axis distributions and final joint 3D populations.

Remaining approximations include separable scalar well shapes, population-resolving recoil, factorized Cartesian emission marginals, omitted tunnelling and coherent Raman escape into the continuum, and D-line-only trap optical estimates. Excited-state lattice Stark shifts of the near-resonant pumping transitions remain the configurable `optical.excited_state_shift_hz` approximation. Power scans exclude points outside weak-excitation or motional-secular validity; the finite search does not establish an optimum in the saturated regime. See [the physics audit](astra-physics-corrections.md) for numerical checks, selected settings and limitations.

The remaining sections describe the legacy 1D backend unless explicitly stated otherwise.

---

## 1. Install and run

```text
conda activate scripts
pip install -r requirements.txt
python rb85_rsc_sim.py --gui                     # opens examples/experiment.json (Reset returns to it); --config overrides
python rb85_rsc_sim.py --config examples/continuous.json --headless --output results/continuous
python rb85_rsc_sim.py --compare-protocols --config examples/default.json --output results/comparison
python rb85_rsc_sim.py --scan-config examples/intensity_scan.json --headless --output results/scan
python rb85_rsc_sim.py --config examples/default.json --estimate-only          # resource estimate only
python rb85_rsc_sim.py --config examples/default.json --set raman.carrier_rabi_hz=1e4 --set timing.protocol=pulsed
python -m pytest                                                                 # physics checks (~2.5 min)
```

**Tested versions** (Windows 11, Python 3.14.6): ARC-Alkali-Rydberg-Calculator 3.10.2, numpy 2.4.6, scipy 1.18.0,
matplotlib 3.11.0, PyQt6 6.11.0, qutip 5.3.1, threadpoolctl 3.5.0, pytest 9.0.3.

The ARC package is `ARC-Alkali-Rydberg-Calculator` (`from arc import Rubidium85`), not the unrelated PyPI package named
`arc`. Atomic data is generated once and cached in `rb85rsc/cache/rb85_d2_atomic_data.json` along with the ARC version and the functions used. It is regenerated automatically if a different ARC version is installed.

Before each run, the CLI prints a resource estimate: state dimension, operator memory, RHS evaluations × a
micro-benchmarked cost per evaluation. The GUI shows the same estimate in its status bar. A default 10 ms coherent run takes about 18 s.

### Files

| File | Contents |
| --- | --- |
| `rb85rsc/atomic.py` | ARC wrapper and cache, SI constants, Steck cross-check, analytic Breit-Rabi ground energies and magnetic validity |
| `rb85rsc/polarization.py` | spherical fractions, Jones-vector geometry mode, realizability test |
| `rb85rsc/motion.py` | exact displacement matrix elements, recoil kernels, thermal states |
| `rb85rsc/ensemble.py` | spatial ensemble, joint 3D recoil/loss, coherent axial Raman dynamics and temperature mixtures |
| `rb85rsc/far_detuned.py` | ARC D1+D2 estimates, modeled/measured-envelope carrier calibration and atomic cache publication |
| `rb85rsc/optical.py` | weak-excitation rates, branching, light shifts, Kramers-Heisenberg option |
| `rb85rsc/model.py` | assembles operators for one configuration, static validity checks |
| `rb85rsc/dynamics.py` | coherent Lindblad reference solver, checked rate solver, cost estimate |
| `rb85rsc/protocols.py` | piecewise-constant schedules for all protocols |
| `rb85rsc/analysis.py` | all output metrics and dynamic diagnostics |
| `rb85rsc/plotting.py`, `runner.py`, `scan.py` | figures, export and metadata, scans (no Qt imports) |
| `rb85rsc/gui.py` | PyQt6 window (the only Qt module) |

---

## 2. Model

### 2.1 State space

All 12 ground sublevels (`F=2, mF=-2..2`; `F=3, mF=-3..3`) are included, together with vibrational levels `n = 0..n_max`.

* The **Raman pair** `{up=|3,3>, down=|2,2>} ⊗ motion` carries a full density matrix of dimension 2N (N = n_max+1),
  because the calibrated Raman coupling acts only there.
* The other **10 sublevels** carry motional populations only. Nothing couples them coherently, and
  the population recoil approximation below drops coherence transfer into these states. This goes beyond secular averaging.

For the default n_max = 40 this is 3N² + 10N ≈ 5.5k complex numbers. The excited states are adiabatically eliminated, so
optical-frequency oscillations are never propagated.

### 2.2 Units and constants

Frequencies called `*_hz` are cycles/s. Rabi frequencies, detunings and rates inside the equations are rad/s or
s⁻¹. `trap_frequency_hz = 70e3` is ν_t, with `omega_t = 2π ν_t`, `x0 = sqrt(ħ / (2 m ω_t)) = 29.2 nm`. The
single 780 nm photon Lamb-Dicke parameter along the axis is computed from constants: `k x0 = 0.2348`. The Raman
`eta_R = |(k_abs − k_emit)·trap_axis| x0` is set by the configured Raman beam directions. With the default
beams (90° apart, Δk ∥ x) it is 0.331.

With `raman.tone_layout = "both_tones_both_beams"`, each beam carries both tones. The four pathways (low tone from beam i,
high tone from beam j) are resonant at the same beat, so they add coherently in the coupling matrix:

`D_raise = √(p₁ˡp₂ʰ) e^{-iθ} D(−η_R) + √(p₂ˡp₁ʰ) e^{i(χ+θ)} D(η_R) + [√(p₁ˡp₁ʰ) + √(p₂ˡp₂ʰ) e^{iχ}] 𝟙`

Here `D(η) = exp(iη(a+a†))`, θ is the phase of the **up→down** crossed pathway, and χ is the beam-2 minus beam-1 **high-minus-low** tone phase. The lowering operator is `D_raise†`: absorption from the low tone and emission into the high tone impart momentum `ħ(k_low−k_high)`.

The two co-propagating pathways have Δk = 0, so they drive only the carrier. The tone powers are
`raman.tone_powers_low/high` (each is [beam 1, beam 2]), and `carrier_rabi_hz` is the Rabi frequency of one pathway at unit
relative power. θ = `raman.lattice_phase_deg` is the atom's position in the static beat interference pattern, and
χ = `raman.beam_beat_phase_deg`. For balanced beams with χ = 0, the red sideband scales as 2 sin θ times the one-pathway
value. The motion-free carrier term is 2 + 2 cos θ·D_nn. At θ = 0 the sideband vanishes. The diagnostics report the realized
|D₁₀| and |D₀₀|. If θ is not stabilized in the experiment, scan `raman.lattice_phase_deg`.

### 2.3 Atomic data (ARC)

The model uses ARC's `getDipoleMatrixElementHFS` (absorption amplitude with q = m' − m, units e·a0), `getBranchingRatio`,
`getHFSCoefficients`/`getHFSEnergyShift`, `getStateLifetime`, `getTransitionFrequency`, and `getLandegfExact`. Dipoles are
converted to SI (× e a0) before any Rabi frequency is formed. The table below compares ARC-derived values with Steck's Rb-85 D-line data (see the test `test_steck_cross_check`):

| quantity | ARC-derived | Steck | rel. dev. |
| --- | --- | --- | --- |
| ground hyperfine splitting | 3.035732 GHz | 3.0357324 GHz | 0 |
| Γ/2π | 6.0659 MHz | 6.0666 MHz | −1.2e-4 |
| F'=4↔3, 3↔2, 2↔1 | 120.640, 63.4005, 29.372 MHz | 120.640, 63.401, 29.372 | ≤ 8e-6 |
| ⟨J=1/2‖er‖J'=3/2⟩ (Steck convention) | 3.5840e-29 C m | 3.5842e-29 | −5e-5 |
| I_sat (σ+ cycling) | 16.691 W/m² | 16.693 W/m² | −1.1e-4 |

Branching comes from ARC and is checked to equal |d|²/Σ|d|². The prescribed facts follow from the matrix elements; none of them is imposed:
`|F'=3,m'=3>` decays to `|3,3>` 5/12, `|3,2>` 5/36, `|2,2>` 4/9, and `|F'=4,m'=4> → |3,3>` with probability 1.

### 2.4 Optical pumping (spin pump F=3→F'=3, repump F=2→F'=3, both on D2 near 780 nm — an explicit assumption)

With `H_ge/ħ = Ω_ge/2`, `Δ_ge = ω_L − ω_ge`, and Γ = 1/τ, each excited amplitude driven from `|g>` is eliminated:

```
c_ge  = (Ω_ge/2) / (Δ_ge + iΓ/2)
R_ge  = Γ |c_ge|² = Γ Ω_ge² / (Γ² + 4 Δ_ge²)          (isolated weak-excitation rate; tested)
δ_g   = Σ_e Ω_ge² Δ_ge / (Γ² + 4 Δ_ge²)                 (ground light shift; blue detuning shifts up; tested)
Ω_ge  = E0 sqrt(f_q) d_ge / ħ,  E0 = sqrt(2 I / (c ε0)),  q = m_e − m_g,  I = 2P/(π w²) in power/waist mode
```

* **All** D2 levels `F'=1,2,3,4` and every sublevel are included for **both** beams. This covers cross-excitation, such as the
  spin pump acting on F=2 3 GHz away, and off-resonant F'=4 scattering of `|3,3>`. Ground energies use analytic Breit-Rabi;
  excited shifts and all dipoles retain the weak-field basis. `optical.excited_state_shift_hz` shifts every optical transition, for example to represent a differential
  trap shift. Detunings are defined relative to the named *zero-field* hyperfine lines, and sublevel shifts are added per
  transition.
* Each absorption is routed through the actual decay probabilities. The default `path_model = "independent"`
  sums rates over excited sublevels. The optional `kramers_heisenberg` sums amplitudes through different F' with the same m'
  before squaring. Total rates are identical by the dipole sum rule, and only the final-state distribution differs.
  With the defaults, the two change the final n̄ by 7e-5 and P_target by 3e-5. The independent-path picture fails
  when several F' are excited with comparable amplitude, i.e. at detunings comparable to the hyperfine splittings.
* The two beams are **mutually incoherent**. This excludes two-frequency coherent population trapping between the
  spin-pump and repump frequencies. Phase-coherent fields would need the optical coherences restored. Coherences
  between ground Zeeman sublevels created by a single beam are also dropped. That is valid when the Zeeman splitting ≫ pumping rate, and
  it is checked (default ratio 34). At B → 0 the check fails, because dark superpositions would matter.
* **Weak-excitation validity.** The excited fraction Σ_e|c_ge|² is computed for every ground state that is occupied
  (population > 1e-3) in any segment. Above 0.02 it is a warning, and above 0.1 the result is marked invalid. There is no
  two-level saturation factor hiding an invalid calculation. The default gives 1.1e-3.
* Light shifts of up and down from the pump beams enter the Raman detuning. A measured shift
  (`light_shift_override_up_hz`/`_down_hz`) **replaces** the computed value for that beam, so it is never added twice.

### 2.5 Polarization and impurity

Polarization is specified relative to the bias-field axis in one of two exclusive modes:

* **Spherical (effective) mode.** `pi_fraction` and `sigma_minus_fraction` are **intensity** fractions, and
  `sigma_plus = 1 − pi − sigma_minus`. Field amplitudes use `sqrt(f_q)`, and the total intensity is fixed. Negative
  fractions or a sum above 1 are rejected, never renormalized. There is also a linked control: `total_impurity` together with
  `pi_share_of_impurity` (the GUI adds a log slider from 1e-6 to 1e-1, a zero-impurity checkbox, and presets). The code tests whether
  a transverse plane wave **along the configured beam direction** can realize the fractions. For three complex
  components with free phases, the condition is a triangle inequality. If it fails, the beam is labeled
  *EFFECTIVE ILLUMINATION MODEL*. For example, any π admixture on a beam along B is labeled this way.
* **Geometry mode.** The inputs are beam angle and azimuth relative to B, ellipticity χ (+45° is positive helicity), and ellipse orientation ψ.
  A transverse Jones vector is rotated into the field-frame spherical basis, and the fractions are *derived*, never overridden. The beam
  direction used for recoil is the derived one. Tested limits: an aligned circular beam gives pure σ+. An aligned linear beam gives 50/50 σ±
  with no π, and aligned ellipticity never creates π. A 10° tilt gives π = sin²θ/2.

### 2.6 Raman coupling (calibrated mode) and the frequency sign

```
Ω_c = 2π · raman.carrier_rabi_hz        (before motional overlap)
H_R/ħ = (Ω_c/2) [ |up><down| ⊗ D + h.c. ],  D = exp(-i η_R (a + a†))   (one pathway, raising block)
ν_ud = (E_up − E_down)/h = hyperfine + Breit-Rabi Zeeman + (calibrated Raman shift) + (pump light shifts)
ν_beat = |ν_high − ν_low|,   δ_beat = ν_beat − ν_ud
```

**Sign.** The atom absorbs from the low-frequency beam and emits into the high-frequency beam, so
`E_up + ħω_low = E_down + ħω_high + ħω_t` for `|up,n> → |down,n−1>`. Energy conservation therefore requires
**δ_beat = +ν_t**. In the rotating frame, `H' = −Δ_R|up><up| + ω_t a†a + H_R`, with Δ_R = 2π δ_beat. The interaction picture is then taken with
respect to `H0 = ω_t(a†a − |up><up|)`:

```
H_I(t)/ħ = −δ |up><up| + Σ_nm (Ω_c/2) D_nm e^{i(n−m−1) ω_t t} |up,n><down,m| + h.c.,   δ = 2π(δ_beat − ν_t)
```

The first red sideband is static, while the carrier rotates at −ω_t and the blue sideband at −2ω_t. All of them are retained,
together with the higher orders up to `max_sideband_order` (default 3, with convergence tested at 6). `red_sideband_offset_hz = δ_beat − ν_t`,
so 0 is the cooling resonance. The sign was verified numerically (`test_sideband_frequency_convention`). Starting from `|up,2>`, offset 0 → `|down,1>`,
−70 kHz (carrier) → `|down,2>`, and −140 kHz (blue) → `|down,3>`, each with transfer above 0.99. `|up,0>` reaches at most 2e-4
(off-resonant only). The `absolute_beat` mode takes ν_beat directly, so Zeeman and light shifts then move the resonance.

**Calibrated inputs.** Wavelength alone determines neither the coupling nor the scattering, so these are supplied directly: carrier Rabi, an *additional* coherence decay γ_x (laser or field noise), the differential light shift, and the
residual Raman scattering rate with its model. `spin_preserving` gives recoil plus full which-state decoherence, a conservative choice that ignores
coherence-preserving Rayleigh scattering. `depolarizing` is a crude worst case. **The default scattering rate of 0 is an IDEALIZED assumption**, and it is
shown as a warning on every result. The Raman amplitude multiplier in a schedule scales Ω_c, the Raman shift, and the Raman scattering
linearly, which corresponds to both beam intensities scaling together. Raman polarization imperfections are *not* inferred from Ω_c. They enter only
through these calibrated inputs, and a validity line states so.

**Only the up/down pair** is coupled. The isolation diagnostic checks every other `|3,m>↔|2,m'>` channel (|Δm| ≤ 2), using actual motional energy differences through `max_sideband_order`, and
flags the result when the nearest one lies within 20× (warning) or 3× (invalid) of the summed carrier Rabi scale at the largest scheduled amplitude. This is a conservative bound: unmodeled polarization strengths and light shifts are unknown. The optional microscopic Raman mode, with amplitude
sums over D1 and D2, is **not implemented**.

### 2.7 Recoil and motion

Every optical scattering event, including one that returns to the same spin state, applies
`exp(i x0 (k_abs − k_emit)·u (a+a†))`. Absorption and emission are combined into **one displacement**, because the 26 ns
excited lifetime is much shorter than the 14 µs trap period. The amplitudes therefore combine before squaring, and no intermediate dephasing is assumed.

```
K_cls(n'|n) = ∫ dc  w_cls(c; β) |<n'| D(η(c)) |n>|²,   η(c) = k x0 (k̂_abs·u − c),  c = ŝ·u
w_σ = 3/8 (1 + <cos²θ_B>),  w_π = 3/4 (1 − <cos²θ_B>),  <cos²θ_B> = c² cos²β + ½(1−c²) sin²β
```

Here β is the angle between the trap axis and B. The azimuth is integrated analytically, and c uses Gauss-Legendre quadrature (24 points by default; 48 points agrees within the 2e-5 convergence-test tolerance).
Isotropic emission is an explicit option. For the default geometry it changes n̄ by 1.3e-4 in continuous cooling
(0.02706 → 0.02693) and by 9e-4 in optical-pumping-only. The mean recoil per photon, η²[(k̂·u)² + ⟨c²⟩], is reproduced
to 1e-6 (`test_red_pi_pulse_removes_one_quantum_then_reset_adds_predicted_recoil`). Optical reset is not recoil-free.

**Secular approximations**, documented and checked:

* The jump operators are `sqrt(W K(n'|n)) |s',n'><s,n|`. They resolve individual motional levels, dropping coherence transfer even between degenerate transitions. Scattering rates much smaller than the minimum motional angular spacing are necessary but **not sufficient** to justify this extra approximation. A separate recoil-coherence warning is always shown when scattering is active.
* Their anticommutator damps every Raman coherence at (Γ_out(up,n) + Γ_out(down,m))/2. This is the
  *only* place optical scattering enters Raman decoherence. `extra_coherence_decay_rate_s` is purely additional, so nothing is counted twice
  (`test_optical_scattering_damps_raman_coherence_consistently`).
* Background heating uses L = sqrt(Γ_h) a and sqrt(Γ_h) a†, acting identically on both spins, so it preserves the Raman coherence structure.

**Truncation.** Kernel probability that would land above n_max, and heating out of n_max, goes into an explicit
**overflow** bin. It is never reflected back into the basis. In harmonic mode it is numerical; in a complete lattice bound basis it is physical escape; a capped lattice basis mixes both. The diagnostics report the
overflow, the population in the top two levels, `|trace + overflow − 1|`, and the thermal tail discarded before normalization.

**Lattice trap** (`trap.potential = "lattice"`). The motion is one site of a 1D lattice `V0 sin²(k_L z)` along `trap.axis`,
with `V0 = trap.depth_uk` and `frequency_hz` read as the harmonic frequency at the bottom of a site, `h ν_t = 2 sqrt(V0 E_r)`.
That fixes `E_r = (h ν_t)² / (4 V0)` and k_L; the implied lattice wavelength is printed in the diagnostics so it can be checked against
the real one. The site keeps the exact lattice shape inside `|z| ≤ λ_L/4` and is held at V0 outside (isolated-site approximation).
The Schrödinger equation is solved by finite differences ([motion.py](rb85rsc/motion.py), `LatticeSite`), and the basis is every
bound level (E < V0), capped by `n_max`. Then:

* Raman matrix elements `D_nm = <n|exp(i Δk z)|m>` and the recoil kernels are overlaps of the site states.
* Each Raman element `|up,n><down,m|` rotates at its own frequency `E_n − E_m − ω_10`, so carrier and sidebands are anharmonic.
  `ω_10 = (E_1 − E_0)/ħ` replaces ω_t as the reference: `red_sideband_offset_hz = 0` is resonant with n = 1 → 0.
* Probability that a recoil kick promotes above V0 goes to the overflow bin, which is now **physical loss** (the atom leaves its site),
  reported as `lattice loss` and `final.lattice_loss`.
* All reported populations are **per trapped atom** (post-selected on survival): P_up, P_n0, P_target, P(F), the spin bars,
  p(n,t), the final joint P(F,mF,n), and `nbar` with everything derived from it (T_equiv, energy reduction, cooling rate).
  Losing hot atoms therefore lowers n̄ without cooling anyone. `P_trapped` is the surviving fraction, and `P_up_absolute`,
  `P_n0_absolute`, `P_target_absolute` are fractions of all atoms (= per-trapped value × P_trapped). The dashboard draws
  P_trapped and prints the absolute values when there is loss; scans gain a survival panel. The `_joint.npz` export keeps the raw
  absolute P[t, s, n]. In a harmonic trap P_trapped is 1 − numerical overflow, so the normalization changes values by < 1e-4 in a valid run.
* `thermal_temperature` initial states are Boltzmann over the exact bound energies, conditional on initially bound atoms. `initial_tail_discarded` measures omitted **bound-state** weight if the basis is capped. The initially unbound fraction is unspecified; a harmonic geometric tail cannot determine it.
* Lattice `nbar` is a mean level index. Energy and cooling power use the actual `E_n-E_0`; `T_equiv_K` matches that mean energy to a canonical distribution on the bound spectrum. It is only a proxy for nonthermal distributions, and is undefined above the positive-temperature energy range. `selection_cooling_power_W` separates the instantaneous contribution from preferential loss of hot atoms.

Example: 15 µK and 70 kHz give E_r = 3.92 kHz (λ_L ≈ 774 nm) and 6 bound levels with spacings 65.8, 61.2, 55.9, 49.1, 38.3 kHz.
Levels n = 0–3 agree with the exact Mathieu band centres to within 0.1 kHz. Not modeled: tunnelling between sites (negligible for the
lowest bands, 0.004 Hz here, but the top band is about 16 kHz wide), coherent Raman coupling into unbound states, and anharmonic
corrections to the background-heating matrix elements.

### 2.8 Solvers

* **`coherent` (reference, default).** The master equation above is integrated with scipy `DOP853` (rtol 1e-7, atol 1e-10), exactly
  up to every schedule switch time. The step is capped at half the fastest coherent period and at 1/(fastest decay rate). The decay cap is necessary because scipy's RMS error
  norm across thousands of near-zero coherences would otherwise let a stiff population drift negative. BLAS is
  pinned to one thread, which is about 2.5× faster for 41×41 matrices. **It agrees with QuTiP `mesolve` to < 1e-6** in populations and Raman coherences
  on a reduced basis, using identical operators and an explicit sink level (`test_qutip_crosscheck.py`).
* **`rate` (fast, optional).** Each (up,n)-(down,m) coherence is eliminated, giving
  `W_nm = (Ω_nm²/2) γ_nm/(γ_nm² + δ_nm²)`, where γ_nm is built from the *actual* scattering rates in that segment plus γ_x and heating.
  Every segment start is checked for `Ω_nm ≤ ε sqrt(γ² + δ²)` (ε = 0.2) on occupied pairs. If the check fails, `RateInvalid` is raised, so the solver can
  never diverge as γ → 0. Integration uses exact matrix exponentials. The rate model omits the carrier-induced AC-Stark
  shift of the sideband. In the valid regime (Ω_c = 1 kHz, n̄₀ = 1) it matches the coherent solver to < 3e-3 while running 100× faster. It
  **is rejected at the defaults** (ratio 0.8, near critical damping), for Raman-only, and for Raman pulses with the pumps off.
* **`auto`** uses the rate backend when every segment passes and otherwise falls back to coherent. The switch is recorded in the output.

### 2.9 Protocols (equal wall-clock `timing.total_duration_ms`, preparation included)

| protocol | schedule |
| --- | --- |
| `optical_pumping_only` | both pumps, Raman off |
| `continuous` | both pumps + Raman throughout |
| `prep_then_continuous` | `prep_duration_ms` of pumping, then pumping (scaled by `cooling_*_scale`) + Raman |
| `pulsed` | [reset (pumps) → Raman pulse] × reps; reps = 0 fills the duration, the remainder is idle |
| `raman_only`, `repump_raman_no_spin_pump`, `all_off` | controls |
| `custom` | list of `{duration_ms, raman, pump, repump}` amplitude multipliers |

Branching and recoil stay active during reset pulses. There is never a forced spin reset, the state and photon counters carry
across every switch, and the laser phase is continuous because the interaction-picture phases use absolute time.

---

## 3. Configuration

One typed dataclass tree, [rb85rsc/config.py](rb85rsc/config.py), is shared by the GUI, CLI (`--set path=value`) and JSON. Each field
carries a unit, a help string, bounds, and choices. Unknown keys, out-of-range values, bad polarization fractions, negative or zero-sum
populations, and inconsistent schedules are all rejected. Groups: `trap`, `initial`, `magnetic`, `raman`, `spin_pump`, `repump`
(with a `polarization` block each), `optical`, `timing`, `numerics`, `analysis`.

Initial-state presets: `uniform_all` (1/12 per sublevel), `uniform_F3` (1/7 per F=3 sublevel), `stretched`,
`manifold_split` (a given F=3 total, uniform per sublevel within each F, so it is not "equal per manifold" by accident), and `custom`.
Motion can be thermal, set by n̄ or T (`p_n = n̄ⁿ/(1+n̄)ⁿ⁺¹`, `n̄ = 1/expm1(hν_t/k_B T)`), Fock, or a custom p(n).

**Illustrative defaults.** These are not measured values and not optimal settings: B = 0.5 G, n̄₀ = 3, n_max = 40, carrier Rabi 5 kHz, pure σ+
pumping at 0.1 W/m² for each pump, both pumps along B, trap axis ⊥ B, Raman beams at 90° (η_R = 0.331), and 10 ms duration. Verification of these defaults:
weak excitation 1.1e-3, Zeeman secular ratio 34, motional secular ratio 0.098, and Raman isolation 33× Ω_c. The initial tail is 7.5e-6,
the final overflow 8e-6, and the conservation error 3e-15. **One change was made:** the pulsed defaults were set to 300 µs Raman pulses, close
to the n=1 red-sideband π time of about 0.32 ms, with 100 µs resets. The earlier 200 µs pulses were not matched to that π time.

Example files: `default`, `continuous`, `prep_then_continuous` (1 ms preparation, then pumps at 0.5), `pulsed`,
`optical_pumping_only`, `controls_*`, `heating_blue_sideband` (a failure regime), `spin_pump_pi_contaminated`,
`spin_pump_wrong_handed`, `geometry_tilted_pump`, `realistic_raman_scattering` (non-zero calibrated scattering, extra
dephasing, and heating, all illustrative), `custom_schedule`, and scans `intensity_scan`, `impurity_scan_spin_pump`,
`impurity_scan_repump`, `impurity_scan_2d`, `detuning_scan`.

---

## 4. Outputs and how to read them

These are computed directly from `P(F, mF, n, t)`:

| output | definition |
| --- | --- |
| spin populations | all 12 sublevels plus the F=2 and F=3 totals |
| `P_up` | Σ_n P(3,3,n) — spin-pumping efficiency |
| `p(n,t)`, `nbar`, `P_n0` | motional marginal over all spins |
| `P_target` | P(3,3,0) **computed directly**. It generally differs from P_up·P_n0 (both are reported) because spin and motion correlate |
| `P_n0_given_up` | P_target/P_up (undefined when P_up = 0) |
| `fractional_energy_reduction` | 1 − ⟨E−E_0⟩(t)/⟨E−E_0⟩(0), conditional on survival; includes loss selection |
| cooling rate | −dn̄/dt and separately −d⟨E−E_0⟩/dt in W, from Savitzky-Golay derivatives; only a harmonic ladder permits multiplication by hν_t |
| photons | integrated scattered photons per initially trapped atom; quanta removed per photon is None for zero photons or appreciable overflow/loss |
| thresholds | time to `spin_threshold`, and to `joint_threshold`, or "not reached" |
| `T_equiv_K` | harmonic formula hν_t/(k_B ln(1+1/n̄)); lattice: canonical fit to actual mean energy; labeled *PROXY* for nonthermal p(n) (TVD > 0.05) |

Figures (dashboard PNG/PDF and GUI tabs): initial and final spin bars, time traces, n̄(t) on its own axis, p(n,t) and
final P(F,mF,n) heatmaps on a log color scale, photon counts, cooling rate, and a text block with every validity item.
Protocol comparisons and scans mark invalid points with a dashed line, "INVALID", or ×, and never rank them as optima.

Exports: `*_timeseries.csv`; `*_joint.npz` (P[t,s,n], overflow, counts, final Raman-block ρ); `*_metadata.json` with the exact
configuration, derived polarization fractions and their source, schedule, SI constants, assumptions, ARC metadata, package versions,
validity list, solver notes, and seed. The solvers are deterministic, so the seed is recorded only for reproducibility.

---

## 5. Representative results (actual computed output, default parameters unless stated)

**Protocol comparison, 10 ms each** (`--compare-protocols`, 1.8 min total):

| protocol | P_up | P_n0 | P_target | n̄ | photons/atom |
| --- | --- | --- | --- | --- | --- |
| optical pumping only | 1.0000 | 0.244 | 0.244 | 3.095 (heated) | 5.3 |
| **continuous** | 0.9957 | 0.9940 | **0.9902** | 0.027 | 14.2 |
| prep (1 ms) + continuous | 0.9956 | 0.9935 | 0.9897 | 0.030 | 14.1 |
| pulsed (300 µs Raman / 100 µs reset, 25 reps) | 0.943 | 0.873 | 0.860 | 0.76 | 10.6 |
| Raman only | 0.092 | 0.247 | 0.021 | 3.01 | 0 |
| repump + Raman, no F=3 pump | 0.097 | 0.331 | 0.097 | 2.69 | 1.5 |
| all off | 0.0833 | 0.250 | 0.021 | 3.00 | 0 |

With these illustrative parameters, simultaneous operation reaches P_target > 0.5 after 1.15 ms and 0.99 after
10 ms. It removes about 0.21 quanta per scattered photon. The fixed-duration pulsed sequence is slower because a single pulse
length cannot be a π pulse for every n, and it was not optimized. Without the F=3 spin pump, population that falls into
|3,m<3> is stranded and P_up stays below 0.1. That is why the F=3 pump is required. In the optical-pumping-only case, |3,3> keeps scattering
via F'=4 at 71 s⁻¹ and heats without changing spin.

**Failure and heating regimes.** A drive on the blue sideband (offset −140 kHz) heats to n̄ ≈ 18. The distribution becomes
non-thermal and stalls near a node of the blue-sideband matrix element L_n¹(η²), below the basis edge. Driving the carrier
(−70 kHz) depolarizes the atom and heats it. The detuning scan in `results/detuning_scan` shows the cooling line at 0 as a single peak
with P_target = 0.967 in 5 ms. It also flags the −150 and −160 kHz points as **invalid**, because the drive comes within 13 and 3 kHz of the
unmodeled |3,2>↔|2,2> channel.

**Polarization impurity** (continuous, 5 ms, fixed power; `impurity_scan_*`). Impurities are intensity fractions; the percentages in parentheses are the same values.

| spin-pump impurity | π: P_target | π: \|3,3> scatter (1/s) | σ⁻: P_target | σ⁻: \|3,3> scatter (1/s) |
| --- | --- | --- | --- | --- |
| 0 | 0.9670 | 71 | 0.9670 | 71 |
| 0.001 (0.1%) | 0.9621 | 119 | 0.9642 | 87 |
| 0.01 (1%) | 0.9190 | 546 | 0.9387 | 229 |
| 0.05 (5%) | 0.7464 | 2444 | 0.8288 | 861 |

π contamination is worse than σ⁻ at the same total impurity, because it drives |3,3>→|3′,3> resonantly. Repump impurity up to 5% changes P_target by
< 2e-3 in either direction, and 5% σ⁻ is marginally *better* (0.9686 vs 0.9670). The repump is 3 GHz from F=3, and the extra |2,m> channels
slightly speed the return to F=3. In the 2D scan, spin-pump impurity dominates.

**Intensity vs coupling** (`intensity_scan`, 5 ms). The best valid point is 0.128 W/m² per pump with a 10 kHz carrier Rabi
(P_target 0.974). Too little pumping (0.01 W/m²) limits the cycle rate. Strong pumping combined with weak Raman coupling (0.3 W/m², 1 kHz)
over-damps the sideband coherence (Ω²/γ suppression) and ends with n̄ ≈ 2.8.

---

## 6. Physics checks (`python -m pytest`; 52 tests, about 3 min)

| check | test(s) | status |
| --- | --- | --- |
| 1 branching sums to 1, forbidden elements vanish, \|3′,3> destinations, closed \|4′,4> cycle, Steck agreement | `test_atomic_optical.py` | pass |
| 2 F'=3-only pure σ+: \|3,3> dark and reached (> 0.999); F'=4 and impurity open target scattering | `test_idealized_pumping…`, `test_F4_and_impurity…` | pass |
| 3 all off is static; Raman-only 2π pulse returns (> 0.995) | `test_all_off_is_static`, `test_raman_only_is_coherent…` | pass |
| 4 red, carrier and blue sidebands at +ν_t, 0, −ν_t beat offsets; \|up,0> has no red partner; off-resonant excitation retained | `test_sideband_*`, `test_up_n0_*` | pass |
| 5 red π pulse removes one quantum; reset adds exactly the predicted recoil | `test_red_pi_pulse_*` | pass |
| 6 conservation < 1e-9, positivity, monotone photon counters | `test_net_cooling_conservation_positivity` | pass |
| 7 n_max 40→55, quadrature 24→48, rtol 1e-7→1e-9, sideband order 3→6 (agreement ≤ 2e-5 in P_up and P_target; n̄ ≤ 1e-3, set by the discarded tail) | `test_convergence_*`, `test_max_sideband_order_converged` | pass |
| 7b overflow tallied and the run flagged invalid, never reflected (n_max = 8) | `test_overflow_is_flagged_not_reflected` | pass |
| 8 rate vs coherent in the valid regime; rejection outside it; coherence damping equals (Γ_u+Γ_d)/2; strong-dephasing limit (1−e^{−Ω²t/γ})/2 within 3% | `test_rate_*`, `test_optical_scattering_damps…`, `test_strong_dephasing_suppression` | pass |
| 9 net cooling at the defaults; blue-sideband heating | `test_net_cooling…`, `test_blue_sideband_heats` | pass |
| 10 config round trip, unknown keys, all examples load, headless CLI + every export, headless imports no Qt, GUI smoke (offscreen run + Stop mid-integration) | `test_io_gui.py` | pass |
| 11 fractions sum to 1, power fixed, zero-impurity limit is identical, π and σ⁻ open distinct transitions with the expected Rabi rates, repump edit leaves the spin pump unchanged, geometry limits | `test_atomic_optical.py` | pass |
| independent master-equation cross-check with QuTiP | `test_qutip_crosscheck.py` | pass (< 1e-6) |
| 12 lattice: levels match Mathieu band centres, deep limit is harmonic, recoil kernels conserve probability, zero offset hits the exact 1→0 spacing, loss is reported and conserved, rate matches coherent | `test_lattice.py` | pass |

The GUI was exercised programmatically: an offscreen Qt run, linked-impurity edits, a completed run, Stop during a 20 ms integration, and a
rendered screenshot. **Interactive mouse and keyboard use and the Matplotlib toolbar were not hand-tested.**

On this Windows install, importing ARC triggers a handled `gmpy2` DLL load exception inside mpmath. `pytest.ini` disables
faulthandler so that it doesn't print a spurious stack dump. It does not affect results.

---

## 7. Limitations and not implemented

* 1D motion only; the same potential for all spins. Harmonic traps have no depth or anharmonicity (a diagnostic is shown when
  `trap.depth_uk` is given); `trap.potential = "lattice"` adds both for one isolated lattice site, without tunnelling.
* Calibrated Raman mode only. The microscopic Raman mode (D1+D2 amplitude sums from beam powers and polarizations), D1 pumping,
  and high-field dipole mixing are not implemented. Ground-state Breit-Rabi energies are included: at 0.5 G, the correction to linear ν_ud is about 90 Hz. The neglected excited quadratic shift is about 0.005 Γ.
* Pump beams are incoherent with each other; pump-induced ground Zeeman coherences are neglected (checked).
* Raman scattering has no coherence-preserving Rayleigh component, which is conservative. The rate backend omits the AC-Stark shift of the sideband
  and approximates heating-induced coherence decay.
* Directional interference between different emitted polarizations is ignored in the recoil pattern. It integrates to zero in total rates.
* The motional secular ratio is close to its warning level at the default pump intensity (0.098). Stronger pumping is flagged.
