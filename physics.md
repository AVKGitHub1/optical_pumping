# Physics Behind The Model

This document explains the physics implemented in [rate-calcs.py](c:\Abhishek\Stanford\Simon Lab\Code\optical_pumping\rate-calcs.py).

## 1. Physical System

The model describes `85Rb` optical pumping on the D2 line.

### Ground states

- `|g, F=3, mF>`
- `|g, F=2, mF>`

with all allowed Zeeman sublevels in each hyperfine manifold.

### Excited states

The actual simulation includes all D2 excited hyperfine manifolds:

- `|e, F'=1, mF'>`
- `|e, F'=2, mF'>`
- `|e, F'=3, mF'>`
- `|e, F'=4, mF'>`

The nominal addressed transition for both lasers is into `F'=3`, but off-resonant coupling to `F'=1,2,4` is also included.

### Lasers

Two classical lasers are used:

- optical pump: nominally `F=3 -> F'=3`
- repump: nominally `F=2 -> F'=3`

Each beam is assigned spherical polarization amplitudes:

- `epsilon_plus`
- `epsilon_pi`
- `epsilon_minus`

The model uses polarization power fractions `|epsilon_q|^2` after normalization.

## 2. Why A Rate-Equation Model?

The script uses population rate equations only.

That means the dynamical variables are populations:

- `p_i(t)` for every Zeeman sublevel

There are no coherences such as:

- `rho_ij` with `i != j`

So the model can represent:

- optical pumping into stretched states
- leakage into other hyperfine manifolds
- repumping
- spontaneous redistribution
- B-field-dependent detuning shifts

But it cannot represent:

- coherent dark states
- CPT
- EIT
- transient Rabi oscillations
- any optical Bloch or density-matrix effects

## 3. Angular-Momentum Structure

The strength of each dipole-allowed hyperfine/Zeeman transition is computed from angular-momentum algebra rather than hardcoded lookup tables.

For a transition

- `|F, mF> -> |F', mF'>`

with spherical component `q = mF' - mF`, the relative dipole strength is proportional to

- a Wigner `6j` factor for hyperfine recoupling
- a Wigner `3j` factor for the Zeeman sublevel coupling

In the script this is implemented by `dipole_strength(...)` using `sympy.physics.wigner`.

The same relative dipole strengths are used for:

- laser excitation strengths
- spontaneous-emission branching ratios

That keeps the angular factors self-consistent.

## 4. Selection Rules

Allowed couplings satisfy:

- `q in {-1, 0, +1}`
- `mF' = mF + q`

These correspond to:

- `q = +1`: `sigma+`
- `q = 0`: `pi`
- `q = -1`: `sigma-`

If a chosen polarization component is zero, that part of the coupling is absent.

## 5. Laser Excitation Model

For each laser-driven channel `g -> e`, the script uses a Lorentzian scattering-rate style expression:

`W_ge = (Gamma/2) * s_ge / (1 + s_total(g) + (2 Delta_ge / Gamma)^2)`

where:

- `Gamma` is the natural linewidth
- `s_ge` is a channel-dependent saturation parameter
- `s_total(g)` is an effective total saturation load out of the ground state
- `Delta_ge` is the channel detuning

This is not a full multilevel derivation, but it is a standard, physically transparent rate-equation approximation.

### Channel saturation parameter

For one channel,

`s_ge ~ (I / I_sat) * polarization_fraction * relative_line_strength`

where the relative line strength comes from the Wigner-structure calculation.

### Effective detuning

The detuning used for each channel is:

`Delta_ge = Delta_laser - Delta_hfs + Delta_Zeeman,g - Delta_Zeeman,e`

The hyperfine part allows the same laser to couple off resonance to transitions other than its nominal target.

## 6. Off-Resonant Effects

This is one of the most important pieces of the model.

Even though the optical pump is intended for `F=3 -> F'=3` and the repump is intended for `F=2 -> F'=3`, each beam can also weakly excite:

- the wrong excited hyperfine manifold
- the opposite ground hyperfine state

because the real atom contains nearby D2 hyperfine structure.

This matters because:

1. A laser can off-resonantly excite population from the “wrong” ground hyperfine state.
2. Once the atom is excited, spontaneous decay can return it to either ground hyperfine manifold.
3. This creates effective population transfer between `F=2` and `F=3` even when the laser is not resonant with that manifold.

So in this script, the ground-manifold transfer is not hardcoded by hand; it emerges naturally from:

- off-resonant excitation
- spontaneous branching

## 7. Spontaneous Emission

Each excited state `|e, F', mF'>` can decay to all allowed ground states `|g, F, mF>`.

The branching ratio is:

`B_(e->g) = strength_(e->g) / sum_all_allowed_g strength_(e->g)`

and the decay rate is:

`Gamma_(e->g) = Gamma * B_(e->g)`

The script explicitly checks that the branching ratios out of each excited state sum to `1`.

## 8. Magnetic Field

A uniform bias field `B` along the quantization axis is included.

The script uses the linear Zeeman shift:

`Delta_Z = (mu_B / hbar) g_F m_F B`

with hyperfine Landé factors `g_F` computed from:

- electronic `g_J`
- nuclear contribution
- angular-momentum coupling formula

This shifts each Zeeman sublevel and therefore changes the detuning of each optical channel independently.

## 9. ODE System

All populations are collected into a vector:

`p(t)`

The evolution is linear:

`dp/dt = M p`

where the rate matrix `M` contains:

- positive inflow from absorption into excited states
- negative outflow from the driven ground state
- positive inflow from spontaneous emission into ground states
- negative outflow from each excited state due to decay

Because the model is linear in populations, it is efficient and stable to solve numerically with `solve_ivp`.

## 10. Population Conservation

The model should conserve total population:

`sum_i p_i(t) = 1`

up to numerical integration error.

The script reports:

- initial total population
- final total population
- maximum drift over the simulation

This is a useful sanity check that the rate matrix and solver are behaving properly.

## 11. Dark States In This Model

The script reports “dark” or unaddressed states by checking whether the total excitation rate out of a ground state is exactly zero for the chosen beam configuration.

In this rate-equation sense, a state is dark if there is no allowed excitation channel out of it with nonzero beam power.

This is different from a coherent dark state.

For example:

- a stretched state can be dark in a pure `sigma+` pumping geometry because there is no allowed `Delta mF = +1` transition out of it

But the script does not describe:

- destructive interference dark states
- coherent superpositions

## 12. What Is Shown vs What Is Simulated

The simulation includes:

- all excited manifolds `F'=1,2,3,4`
- off-resonant couplings from both lasers

The current visualization intentionally shows less:

- the level diagram only displays `F'=3`
- the transition listing only displays the nominal couplings

This is a presentation choice for readability.

The hidden off-resonant channels still affect the dynamics.

## 13. Interpretation Notes

This model is most useful for:

- estimating optical-pumping timescales
- understanding leakage and repumping
- exploring polarization impurity
- scanning detuning or power
- seeing how off-resonant scattering redistributes population

It is less appropriate when:

- coherence matters
- the laser linewidth or phase noise must be modeled explicitly
- the system enters a regime where optical Bloch equations are required

## 14. Summary

The script implements a multilevel, mF-resolved, hyperfine-aware optical-pumping model in which:

- transition strengths come from angular-momentum algebra
- detunings include hyperfine and Zeeman structure
- spontaneous decay is treated with proper branching ratios
- off-resonant excitation causes real transfer between ground hyperfine manifolds
- dynamics are solved with population rate equations only

That makes it a strong middle ground between a toy optical-pumping picture and a full density-matrix treatment.
