# Detailed Output

```text
=== Configuration Summary ===
Gamma / (2 pi)                  =  6.065000 MHz
Optical-pump detuning / (2 pi)  =  10.000000 MHz
Repump detuning / (2 pi)        =  0.000000 MHz
B field                         =  2.350000 G
Saturation intensity scale      = 1 (arbitrary units)
Include sat broadening          = True
gF(F=3, ground)                 =  0.33396755
gF(F=2, ground)                 = -0.33337788
gF(F'=3, excited)               =  0.38993384

Laser: optical_pump
  addressed manifold           = F=3 -> F'=3
  intensity                    = 1 (same units as I_sat)
  sigma+ power fraction        = 1.000000
  pi power fraction            = 0.000000
  sigma- power fraction        = 0.000000

Laser: repump
  addressed manifold           = F=2 -> F'=3
  intensity                    = 0.5 (same units as I_sat)
  sigma+ power fraction        = 0.000000
  pi power fraction            = 1.000000
  sigma- power fraction        = 0.000000

=== Branching-Ratio Sanity Check ===
|e, F=3, mF=-3>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=-2>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=-1>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+0>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+1>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+2>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+3>: sum(branching ratios) = 1.000000000000
Maximum branching-ratio normalization error = 0.000e+00

=== Dark / Unaddressed Ground States ===
|g, F=3, mF=+3>: zero excitation rate in this rate model

=== Laser-Driven Transition Network ===
optical_pump: |g, F=3, mF=-3> -> |e, F=3, mF=-2>  sigma+  rate =  5.587516e+05 s^-1  Delta/(2pi) =  9.269701 MHz
optical_pump: |g, F=3, mF=-2> -> |e, F=3, mF=-1>  sigma+  rate =  9.453659e+05 s^-1  Delta/(2pi) =  9.085621 MHz
optical_pump: |g, F=3, mF=-1> -> |e, F=3, mF=+0>  sigma+  rate =  1.162784e+06 s^-1  Delta/(2pi) =  8.901541 MHz
optical_pump: |g, F=3, mF=+0> -> |e, F=3, mF=+1>  sigma+  rate =  1.204254e+06 s^-1  Delta/(2pi) =  8.717462 MHz
optical_pump: |g, F=3, mF=+1> -> |e, F=3, mF=+2>  sigma+  rate =  1.051332e+06 s^-1  Delta/(2pi) =  8.533382 MHz
optical_pump: |g, F=3, mF=+2> -> |e, F=3, mF=+3>  sigma+  rate =  6.695479e+05 s^-1  Delta/(2pi) =  8.349302 MHz
repump: |g, F=2, mF=-2> -> |e, F=3, mF=-2>  pi  rate =  8.751768e+05 s^-1  Delta/(2pi) =  4.758115 MHz
repump: |g, F=2, mF=-1> -> |e, F=3, mF=-1>  pi  rate =  2.699590e+06 s^-1  Delta/(2pi) =  2.379057 MHz
repump: |g, F=2, mF=+0> -> |e, F=3, mF=+0>  pi  rate =  4.397021e+06 s^-1  Delta/(2pi) =  0.000000 MHz
repump: |g, F=2, mF=+1> -> |e, F=3, mF=+1>  pi  rate =  2.699590e+06 s^-1  Delta/(2pi) = -2.379057 MHz
repump: |g, F=2, mF=+2> -> |e, F=3, mF=+2>  pi  rate =  8.751768e+05 s^-1  Delta/(2pi) = -4.758115 MHz

=== Population-Conservation Check ===
Initial total population       = 1.000000000000
Final total population         = 1.000000000000
Maximum |sum(p)-1| over time   = 6.661e-16

=== Final Populations ===
|g, F=3, mF=-3>: 0.0000000000
|g, F=3, mF=-2>: 0.0000000006
|g, F=3, mF=-1>: 0.0000000026
|g, F=3, mF=+0>: 0.0000000051
|g, F=3, mF=+1>: 0.0000000087
|g, F=3, mF=+2>: 0.0000000178
|g, F=3, mF=+3>: 0.9999999454
|g, F=2, mF=-2>: 0.0000000001
|g, F=2, mF=-1>: 0.0000000005
|g, F=2, mF=+0>: 0.0000000014
|g, F=2, mF=+1>: 0.0000000042
|g, F=2, mF=+2>: 0.0000000119
|e, F=3, mF=-3>: 0.0000000000
|e, F=3, mF=-2>: 0.0000000000
|e, F=3, mF=-1>: 0.0000000001
|e, F=3, mF=+0>: 0.0000000002
|e, F=3, mF=+1>: 0.0000000005
|e, F=3, mF=+2>: 0.0000000005
|e, F=3, mF=+3>: 0.0000000003

=== Final Manifold Totals ===
Total F=3 ground   = 0.9999999802
Total F=2 ground   = 0.0000000182
Total excited      = 0.0000000016
Grand total        = 1.0000000000

```
