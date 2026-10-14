# Learnings from the PGC atom model for the OP/RSC simulator

Review date: 2026-10-13. Reviewed simulator revision: `d7fd425`.

Implementation update (2026-10-14): the counter-rotating polarization correction is now implemented in `DLineResponse`. The response records `light_shift_model_version=2`, which also makes carrier-calibration caches refresh their stored shifts. All 89 affected checks passed, including 19 new regression cases covering independent scalar/vector limits, helicity reversal, the static limit, unchanged linear-polarization shifts, and cache invalidation. The excited-state and internal-coherence extensions below remain future work.

## Recommendation

The PGC notes contain a useful atomic-operator design, particularly the treatment of **trap-induced excited-state shifts and mixing**. The angular-momentum construction is sound. However, the specified dynamics contain inconsistencies in the ground hyperfine frame, absorption-beam sampling, and saturation treatment. I would reuse selected ideas after validation, rather than adopt this model as a replacement for our atom model or trajectory solver.

The most useful next steps are:

1. **Fix the counter-rotating polarization term in our existing ground light-shift calculation.** Implemented on 2026-10-14; the original finding and numerical comparison are retained below.
2. **Add local ground-to-excited differential trap shifts to optical pumping**, first as a spectroscopy diagnostic, then as validated transition energies and dipoles used by the simulator.
3. **Develop a consistent internal-coherence benchmark** before attempting general spin-coherent optical pumping or PGC in the existing motional trajectory solver.

Keep our Rb85 data, finite-field treatment, quantized motion, correlated dipole recoil, and weak-excitation validity checks. The other model does not supersede these.

Scope: this is a review of [the supplied notes and reference algorithms](atom-model-learnings.md), compared with the code in this repository. The upstream PGC repository, its original implementation, and converged cooling results were not available. No production code was changed during this review.

## What differs between the models

| Feature | PGC reference in the notes | Current OP/RSC simulator | Assessment |
|---|---|---|---|
| Species and basis | Rb87: 8 ground and 16 D2 excited states | Rb85: 12 ground and 24 D2 excited states | Port operator construction, not arrays, constants, or state indices. |
| Atomic data | Explicit constants and an approximate polarizability pole set | ARC-derived cached energies, signed dipoles, branching data, and reference comparisons | Retain our data infrastructure; expand its coverage where needed. |
| Magnetic mixing | Electronic Zeeman operators; ground-frame problem discussed below | Hyperfine plus Zeeman diagonalization, including nuclear contributions | Our `diagonalized` option already addresses magnetic mixing. |
| Trap effects on excited states | Local scalar, vector, and tensor Hamiltonian, diagonalized at each position | Pumping has a configured common excited-state shift; lattice shifts are calculated separately for ground states | This is the most useful missing physics to borrow. |
| Internal coherence | Ground-state amplitudes, with important approximations and inconsistencies | Population treatment of pumping; selected Raman coherence; motional trajectory mode uses a resolved spin state | Full optical spin coherence would be a new capability. |
| Motion | Classical positions and velocities | Quantized local well levels; optional coherent motional trajectories | Classical PGC motion is not a replacement for RSC near the ground state. |
| Photon recoil | Isotropic emission and heuristic absorption-beam selection | Dipole emission patterns and correlated recoil across axes | Keep the current recoil treatment. |
| Excited-state elimination | Effective operators with heuristic saturation broadening | Weak-excitation pumping with explicit validity diagnostics | A saturation denominator is not a validated extension to arbitrary intensity. |

Relevant current implementation: [atomic.py](../rb85rsc/atomic.py), [finite_field.py](../rb85rsc/finite_field.py), [optical.py](../rb85rsc/optical.py), [far_detuned.py](../rb85rsc/far_detuned.py), [ensemble.py](../rb85rsc/ensemble.py), and [coherent_pumping.py](../rb85rsc/coherent_pumping.py).

## What is good in the PGC atom model

### Angular momentum and dipole normalization

The uncoupled electronic/nuclear basis is a good way to assemble hyperfine, Zeeman, and Stark operators. The hyperfine Hamiltonian in the notes has the standard magnetic-dipole and electric-quadrupole form. These conventions can be checked against the isotope-specific [Rb87](https://steck.us/alkalidata/rubidium87numbers.pdf) and [Rb85](https://steck.us/alkalidata/rubidium85numbers.pdf) reference sheets.

Independent numerical assembly using the notes' rounded Rb87 constants gives these D2 excited hyperfine energies relative to the fine-structure centroid:

| Excited F | Energy / h (MHz) |
|---|---:|
| 0 | -302.07375 |
| 1 | -229.85175 |
| 2 | -72.91125 |
| 3 | 193.74075 |

For dimensionless emission matrices `D_q` mapping the 16 excited states to the 8 ground states, with the stretched transition normalized to one, the independently assembled electronic coefficients satisfy:

```text
sum_q D_q^dagger D_q = identity on the excited manifold
sum_(q,e) |D_q[g,e]|^2 = 2 for each ground state
```

The maximum numerical errors were zero and `2.22e-16`, respectively. These are useful completeness tests to preserve in any generalized atom model. They validate the stated angular coefficients, not the entire PGC implementation.

The quoted `D_D1` and `D_D2` parameters also use a particular line-strength convention. They must not be substituted directly for our signed hyperfine matrix elements. Check lifetimes, oscillator strengths, and cycling-transition Rabi frequencies after converting conventions.

### Local excited-state Hamiltonians

The strongest feature is constructing and diagonalizing

```text
H_e(r) = H_hyperfine + H_Zeeman + H_scalar(r) + H_vector(r) + H_tensor(r).
```

This allows trap light to change both transition frequencies and optical selection strengths. A single common shift cannot represent those effects. Scalar/vector/tensor operators and diagonalization of the combined interaction have a sound theoretical basis; see [Le Kien, Schneeweiss, and Rauschenbeutel, general theory](https://arxiv.org/html/1211.2673v2).

Our current `zeeman_model="diagonalized"` means magnetic hyperfine mixing. It does **not** include general trap-induced excited-state mixing. In `LocalExperiment`, the optical `Model` is built before the lattice light shifts are calculated, and those shifts do not become state-dependent pumping detunings.

### Coherent amplitudes and field phases

Keeping amplitudes until physically distinguishable channels are selected is the right design principle. The notes retain interference between excited intermediate states and represent coherent beam groups with complex polarization vectors and phases.

We already retain intermediate-state interference in the Kramers-Heisenberg pumping option and D1/D2 interference in far-detuned scattering. The additional opportunity is coherence between ground sublevels and between phase-related pump beams. Pump `BeamSpec` currently carries spherical polarization fractions rather than the full complex field information needed for that extension.

Classical motion through a polarization texture can produce cooling through delayed internal-state response without inserting a fitted friction coefficient. That is a useful mechanism to benchmark, consistent with the original [Dalibard–Cohen-Tannoudji PGC analysis](https://www.phys.ens.psl.eu/~dalibard/publi2/josaB_89.pdf). It does not by itself validate this particular force and jump implementation.

## Problems that should not be copied

### 1. Removing ground hyperfine energies while retaining static inter-F couplings

Sections 5 and 7 of the notes omit the large ground hyperfine splitting from propagated `Hg`, but retain the full electronic `JG` matrices, including their matrix elements between F=1 and F=2. The laser operators, in contrast, are constructed separately in each F block.

This is not a consistent removal of fast hyperfine evolution. In the corresponding interaction picture, inter-F matrix elements acquire fast phases; they cannot simply remain static after the splitting is removed.

A simple counterexample uses only a 1 G static field, no trap, and no optical field. The notes' electronic Zeeman term couples `|F=2,m=0>` and `|F=1,m=0>` with approximately `V/h = 1.401 MHz`. With the hyperfine splitting removed, these two states undergo complete transfer after about **0.178 microseconds**. Restoring the approximately 6.835 GHz splitting reduces the maximum transfer probability to approximately **1.68e-7** in the same two-state calculation.

This is an algebraic counterexample, not a simulated PGC cooling result. It demonstrates why a superficially small omitted term can change the physics qualitatively.

Use the complete ground Hamiltonian/eigenbasis, or derive a consistent rotating-frame and secular approximation. Preserve deliberately resonant Raman couplings with their correct phases. Our existing finite-field atomic treatment is a better starting point.

### 2. Absorption-beam probabilities depend on arbitrary eigenvector phases

Section 8.3 uses

```text
B_b = sum_k L_k * a_kb
P_beam(b) proportional to |B_b|^2 * Doppler_factor(b).
```

The excited eigenvectors can independently change phase without changing the atom. Their projected amplitudes `a_kb` change phase correspondingly. Reconstructing a physical excited vector as `sum_k V[:,k] * L_k * a_kb` cancels those phases; the scalar sum `B_b` does not.

For a two-channel, two-beam algebra example with equal denominators, take the channel amplitudes of beam 1 to be `(1,1)` and beam 2 to be `(1,-1)`. The prescribed beam probabilities are `(1,0)`. Flipping the sign of the second excited eigenvector changes them to `(0,1)`, although nothing physical changed.

This invalidates the prescription as a general absorption-recoil rule. Replacing the scalar sum by a vector norm would remove that phase defect, but would not by itself establish a correct momentum-transfer model for interfering coherent beams. Derive recoil and force from the same optical operators.

### 3. Saturation changes the hazard but not the reconstructed jump operator

Sections 7–8 use a broadened denominator

```text
1 / (Delta^2 + Gamma^2 * (1+s_local)/4)
```

for `R` and channel selection, but reconstruct the excited amplitude using `1/(Delta+i*Gamma/2)`. Therefore, the displayed jump maps do not generally satisfy `R = sum_mu L_mu^dagger L_mu` at finite saturation.

For illustration, at `s_local=4`, the ratio between the unsaturated jump norm and the broadened hazard is 5 on resonance, but only about 1.235 at `Delta=2*Gamma`. There is no single channel-independent normalization that fixes all detunings. Normalizing each post-jump state does not resolve this mismatch for a general coherent multichannel state.

A consistent eliminated model must derive its Hamiltonian, jumps, and no-jump loss together, within its approximation regime. [Reiter and Sørensen's effective-operator formalism](https://arxiv.org/html/1112.2806v2) provides a suitable weak-drive framework. At appreciable excitation, compare against an optical Bloch/master-equation calculation instead of assuming a shared saturation factor is sufficient.

### 4. Additional physical limitations

| Issue in the notes | Consequence / action |
|---|---|
| Doppler shifts only reweight beam selection after a scattering event | They do not change the total scattering hazard or the optical operators. This cannot supply a complete velocity-dependent optical response. Spatial polarization dynamics can still produce PGC, but a general Doppler-cooling claim needs further validation. |
| Ground Zeeman/vector energies are absent from transition denominators | Ground shifts appear in evolution but not consistently in spectroscopy. Use differences between local excited and ground energies. |
| Emission is isotropic and independent of sampled dipole component | Directional momentum diffusion is inaccurate for a polarized atom. Retain our dipole-pattern and correlated-axis treatment. |
| Post-emission vectors retain both ground F manifolds while their hyperfine evolution is removed | Cross-F coherence lacks the required phase/frequency treatment. Use consistent emitted-frequency channels and secularization, or retain the necessary dynamics explicitly. |
| Beam phase roll is not also included as a frequency offset in optical denominators | Treat this as a restricted approximation, not an exact multifrequency response. |
| Cooler/repumper inter-F Raman interference is excluded | Do not interpret the model as a complete coherent dark-state or gray-molasses calculation. |

The excited polarizability numbers are also not a precision data set. Section 4 uses a limited set of upward transitions, approximate wavelengths/lifetimes/branching fractions, an effective pole, and no comprehensive core/tail contribution. That is useful for exploration but requires independent validation, especially near resonances, cancellations, and tune-out wavelengths. The supplied table does not cover our 1188 nm transverse lattices.

There are numerical caveats too: fixed-step approximate exponentials, cached optical operators, forces refreshed on a different schedule, and eigenstates identified by sorted energy rather than overlap tracking. The specified loss/record order and delayed cache updates deserve separate tests before interpreting survival or photon counts. None of these should be imported as an assumed improvement over our solver.

## A concrete issue found in our current simulator

At the reviewed revision, `DLineResponse._shift_hz_uncached` in [far_detuned.py](../rb85rsc/far_detuned.py) used the same `absorption(eps)` matrix for the rotating and counter-rotating light-shift terms. For a complex positive-frequency polarization vector, the counter-rotating interaction uses the **complex-conjugated polarization**. The implementation now makes this distinction.

Neglecting linewidths far from resonance, the diagonal ground shift has the form

```text
delta_E_g / hbar = (1/4) sum_e [
    |Omega_eg(eps)|^2  / (omega - omega_eg)
  - |Omega_eg(eps*)|^2 / (omega + omega_eg)
].
```

The supplied notes encode this correctly through opposite spherical helicities in the two terms. It also follows from the positive- and negative-frequency interactions in [Le Kien et al., equations (4)–(7)](https://arxiv.org/html/1211.2673v2).

I compared the current function with an independently written version that changes only that numerator. Conditions: Rb85, zero magnetic field, quantization axis z, `eps=(x+i*y)/sqrt(2)`, and the existing D-line data and `diagonalized` basis. The observable below is the **F=3 stretched-state differential shift**, `shift(m=+3)-shift(m=-3)`, per intensity:

| Wavelength (nm) | Before fix (Hz per W/m²) | Conjugated counter term (Hz per W/m²) | Relative difference, previous/corrected − 1 |
|---|---:|---:|---:|
| 783 | -0.16772246 | -0.16772805 | -0.00333% |
| 785 | -0.11730288 | -0.11730848 | -0.00477% |
| 850 | 0.0017202729 | 0.0017142619 | +0.351% |
| 1064 | 0.00014193277 | 0.00013470046 | +5.37% |
| 1188 | 0.000086812803 | 0.000078954172 | +9.95% |

These percentages concern this vector-sensitive diagnostic, not the total trap depth, cooling rate, or target yield. Linear polarization produced no change beyond roundoff, at most `2.1e-17 Hz per W/m²` in these checks. The example lattices are linearly polarized, so this is not evidence for a 10% error in their trapping potentials. Circular/elliptical light-shift calculations do need the correction.

The independently evaluated numerator change was:

```python
nu = response.ee[:, None] - response.eg
rotating = abs(response.absorption(eps) * response.eunit / at.HBAR)**2
counter = abs(response.absorption(eps.conj()) * response.eunit / at.HBAR)**2
corrected = np.sum(
    rotating / (2*np.pi*(frequency_hz - nu))
    - counter / (2*np.pi*(frequency_hz + nu)), axis=0
) / (8*np.pi)
```

The correction was subsequently implemented and covered by [test_far_detuned_shifts.py](../tests/test_far_detuned_shifts.py). Eleven new regression cases failed before the change and passed afterward; eight additional cases also passed, including the linear-polarization limits. The comparison retains the existing D-line truncation, so it does not certify an absolute full-spectrum polarizability.

## Recommended implementation sequence

### First: correct and test counter-rotating polarization

Completed on 2026-10-14: the counter-rotating numerator uses the conjugated polarization. Regression checks cover real polarization, helicity reversal, an independent scalar/vector reference at 785 and 1188 nm, and the static limit in both atomic basis models. Carrier-cache provenance was updated to prevent reuse of old light shifts. Existing response-cache, finite-field, ensemble-physics, and ensemble-geometry checks passed. Three GUI checks initially failed because the unactivated environment loaded incompatible Qt DLLs; they passed after placing the `scripts` environment's DLL directory first on the process path.

### Second: add local optical transition spectroscopy

Start with a diagnostic for the actual Rb85 examples: 785 nm axial and 1188 nm transverse lattices, configured magnetic field and polarizations, and sampled cloud positions. Report differential transition shifts, mixing strengths, and their spread compared with the natural linewidth and neighboring level spacings. This establishes whether each refinement materially affects pumping before making a large solver change.

Use sourced Rb85 matrix elements/transition energies, including the excited-state transitions needed for the polarizability. ARC's [dynamic-polarizability API](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.calculations_atom_single.DynamicPolarizability.getPolarizability.html) is one possible independent comparison: it returns scalar, vector, tensor, core, and ponderomotive contributions. Its documented SI output uses `Hz V^-2 m^2`; explicitly reconcile that with energy polarizability, electric-field amplitude, and intensity before comparing. Do not copy the Rb87 ratio table into Rb85 configuration.

Then integrate a local atomic-response object supplying ground/excited energies and polarization-resolved dipoles to `beam_rates`:

```text
local fields -> H_g, H_e -> eigenvalues/eigenvectors
             -> transition frequencies and transformed D_q
             -> optical amplitudes, rates, shifts, and validity diagnostics
```

Important constraints for this integration:

- **General mixing requires explicit dipole tensors.** Current magnetic diagonalization conserves m and can use real transforms. A general transverse vector/tensor Stark Hamiltonian can mix m and have complex eigenvectors. Keep each `D_q` and transform it with the appropriate conjugate-transpose basis matrices; do not infer every allowed polarization from one pair of m labels.
- **Construct the optical response after the local fields are known.** Reorder the relevant setup in `LocalExperiment` so pumping sees these fields.
- **Use the complete transition difference.** Optical resonance depends on `(E_e-E_g)/h`, including the scalar trap shift. The common ground scalar potential is already present in the motional Hamiltonian: include it in spectroscopy without adding it to motion a second time.
- **Sample motion within a well as well as cloud position.** A site-center correction is only a first approximation. Check variation over the occupied motional wavefunctions. Position-dependent denominators can produce nonseparable jump operators across axes, which the present product-state trajectory representation cannot automatically handle.
- **Track states by overlaps/projectors.** Sorted eigenvalue indices alone are unreliable through crossings and degeneracies. Physical predictions must be invariant under arbitrary eigenvector phases and rotations within an exactly degenerate subspace.

### Third: benchmark internal coherence before extending trajectories

Build a small internal-state calculation at fixed position, initially in the weak-drive regime. Use full Rb85 ground/excited structure, consistent field phases, and frequency-resolved decay channels. Compare effective ground-state operators against a full internal master equation in controlled limits.

Require Hermitian Hamiltonians, trace preservation, and `R=sum L^dagger L`, as well as agreement with existing rate-equation results when internal coherences are secularly suppressed. Add polarization-gradient/dark-state examples only after these limits pass. An optically incoherent beam description should remain available when relative laser phases are physically randomized.

This is not a drop-in addition to `coherent_pumping.py`. That solver retains motional coherences while representing a trajectory by one resolved spin state and separate motional vectors for the axes. General spin-coherent optical jumps or Raman driving can entangle spin and motion. Supporting them may require a joint state or a separately justified approximation.

### Lower-priority transfers

The focused-field longitudinal polarization correction is useful for tightly focused traps. Our example waists are 200 micrometers, whereas the PGC example uses 1 micrometer; the paraxial longitudinal correction scales roughly as `1/(k*w0)`. It is a lower priority here than excited-state spectroscopy. A balanced, linearly polarized standing wave in the specified geometry also has zero local vector helicity despite its longitudinal component.

Do not replace the current quantum observables with classical kinetic temperature. The example harmonic spacings correspond to roughly `h*nu/kB = 2.19 microkelvin` transversely and `3.31 microkelvin` axially. Classical trajectories cannot resolve ground-state occupation or sideband structure at those scales. Where classical kinetic temperature is reported, separate thermal velocity variance from center-of-mass drift; keep it distinct from our motional-energy temperature proxies.

Atomic-model improvements also do not remove a motional-secular validity warning. Excited-state elimination, secularization of internal states, and secularization of motion are separate approximations, with separate tests.

## Validation performed and limits

The following existing tests passed during this review:

```text
python -m pytest tests/test_atomic_optical.py tests/test_finite_field.py tests/test_ensemble_physics.py tests/test_correlated_recoil.py -q --basetemp .test_tmp/pgc_review_tests

71 passed in 7.75s
```

The current atomic reference comparison had a maximum relative difference of approximately `1.16e-4` for the quantities it checks. These tests establish the existing tested limits, not the validity of every model assumption, and they do not cover the counter-rotating polarization defect identified above.

Separately written algebra probes checked the hyperfine spectrum, dipole completeness, rotating-frame counterexample, eigenvector-phase counterexample, saturation mismatch, and current-repository light-shift comparison. The numerical values are recorded above. The PGC reference code was reviewed statically; its full trajectory example was not executed. No new cooling curve, absolute excited polarizability, or quantitative PGC temperature has been validated here.

Before accepting a future implementation, require zero-trap recovery of existing pumping results; correct weak-field and finite-field atomic spectra; phase-invariant optical predictions; consistent total jump hazards; agreement with a full internal master equation in a weak-drive limit; appropriate recoil moments; and convergence across cloud samples, occupied motional states, and numerical resolution.
