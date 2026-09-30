# Detailed Output

```text
=== Configuration Summary ===
Gamma / (2 pi)                  =  6.065000 MHz
Optical-pump detuning / (2 pi)  =  10.000000 MHz
Repump detuning / (2 pi)        =  10.000000 MHz
B field                         =  0.000000 G
Saturation intensity scale      = 1 (arbitrary units)
Include sat broadening          = True
Included excited manifolds      = (1, 2, 3, 4)
Ground F=2 offset from F=3      = -3035.732439 MHz
gF(F=3, ground)                 =  0.33396755
gF(F=2, ground)                 = -0.33337788
gF(F'=1, excited)              = -1.00163404
Excited F'=1 offset            = -92.772400 MHz
gF(F'=2, excited)              =  0.11162026
Excited F'=2 offset            = -63.401700 MHz
gF(F'=3, excited)              =  0.38993384
Excited F'=3 offset            =  0.000000 MHz
gF(F'=4, excited)              =  0.50125927
Excited F'=4 offset            =  120.640700 MHz

Laser: optical_pump
  nominal target              = F=3 -> F'=3
  intensity                    = 10 (same units as I_sat)
  sigma+ power fraction        = 0.987805
  pi power fraction            = 0.000000
  sigma- power fraction        = 0.012195

Laser: repump
  nominal target              = F=2 -> F'=3
  intensity                    = 100 (same units as I_sat)
  sigma+ power fraction        = 0.000000
  pi power fraction            = 1.000000
  sigma- power fraction        = 0.000000

=== Branching-Ratio Sanity Check ===
|e, F=1, mF=-1>: sum(branching ratios) = 1.000000000000
|e, F=1, mF=+0>: sum(branching ratios) = 1.000000000000
|e, F=1, mF=+1>: sum(branching ratios) = 1.000000000000
|e, F=2, mF=-2>: sum(branching ratios) = 1.000000000000
|e, F=2, mF=-1>: sum(branching ratios) = 1.000000000000
|e, F=2, mF=+0>: sum(branching ratios) = 1.000000000000
|e, F=2, mF=+1>: sum(branching ratios) = 1.000000000000
|e, F=2, mF=+2>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=-3>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=-2>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=-1>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+0>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+1>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+2>: sum(branching ratios) = 1.000000000000
|e, F=3, mF=+3>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=-4>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=-3>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=-2>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=-1>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=+0>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=+1>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=+2>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=+3>: sum(branching ratios) = 1.000000000000
|e, F=4, mF=+4>: sum(branching ratios) = 1.000000000000
Maximum branching-ratio normalization error = 1.110e-16

=== Dark / Unaddressed Ground States ===
No exactly dark ground states found from the chosen laser couplings.

=== Laser-Driven Transition Network ===
optical_pump: |g, F=3, mF=-3> -> |e, F=3, mF=-2>  sigma+  rate =  2.179711e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=-2> -> |e, F=3, mF=-1>  sigma+  rate =  3.609356e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=-2> -> |e, F=3, mF=-3>  sigma-  rate =  2.673597e+04 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=-1> -> |e, F=3, mF=+0>  sigma+  rate =  4.317090e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=-1> -> |e, F=3, mF=-2>  sigma-  rate =  4.441451e+04 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+0> -> |e, F=3, mF=+1>  sigma+  rate =  4.316738e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+0> -> |e, F=3, mF=-1>  sigma-  rate =  5.329306e+04 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+1> -> |e, F=3, mF=+2>  sigma+  rate =  3.608470e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+1> -> |e, F=3, mF=+0>  sigma-  rate =  5.345881e+04 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+2> -> |e, F=3, mF=+3>  sigma+  rate =  2.178813e+06 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+2> -> |e, F=3, mF=+1>  sigma-  rate =  4.483155e+04 s^-1  Delta/(2pi) =  10.000000 MHz
optical_pump: |g, F=3, mF=+3> -> |e, F=3, mF=+2>  sigma-  rate =  2.715841e+04 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=-2> -> |e, F=3, mF=-2>  pi  rate =  2.136814e+07 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=-1> -> |e, F=3, mF=-1>  pi  rate =  3.244890e+07 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=+0> -> |e, F=3, mF=+0>  pi  rate =  3.589601e+07 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=+1> -> |e, F=3, mF=+1>  pi  rate =  3.244890e+07 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=+2> -> |e, F=3, mF=+2>  pi  rate =  2.136814e+07 s^-1  Delta/(2pi) =  10.000000 MHz

=== Population-Conservation Check ===
Initial total population       = 1.000000000000
Final total population         = 1.000000000000
Maximum |sum(p)-1| over time   = 3.553e-15

=== Final Populations ===
|g, F=3, mF=-3>: 0.0000296640
|g, F=3, mF=-2>: 0.0003705846
|g, F=3, mF=-1>: 0.0018395777
|g, F=3, mF=+0>: 0.0042385184
|g, F=3, mF=+1>: 0.0083178142
|g, F=3, mF=+2>: 0.0160222754
|g, F=3, mF=+3>: 0.9563242137
|g, F=2, mF=-2>: 0.0000128036
|g, F=2, mF=-1>: 0.0001218233
|g, F=2, mF=+0>: 0.0005425620
|g, F=2, mF=+1>: 0.0013122412
|g, F=2, mF=+2>: 0.0014120063
|e, F=1, mF=-1>: 0.0000015868
|e, F=1, mF=+0>: 0.0000094211
|e, F=1, mF=+1>: 0.0000170928
|e, F=2, mF=-2>: 0.0000006069
|e, F=2, mF=-1>: 0.0000016921
|e, F=2, mF=+0>: 0.0000010569
|e, F=2, mF=+1>: 0.0000157846
|e, F=2, mF=+2>: 0.0000787883
|e, F=3, mF=-3>: 0.0000002606
|e, F=3, mF=-2>: 0.0000110236
|e, F=3, mF=-1>: 0.0001447653
|e, F=3, mF=+0>: 0.0007311443
|e, F=3, mF=+1>: 0.0016163846
|e, F=3, mF=+2>: 0.0022610881
|e, F=3, mF=+3>: 0.0009358340
|e, F=4, mF=-4>: 0.0000000014
|e, F=4, mF=-3>: 0.0000000131
|e, F=4, mF=-2>: 0.0000000576
|e, F=4, mF=-1>: 0.0000002694
|e, F=4, mF=+0>: 0.0000016730
|e, F=4, mF=+1>: 0.0000059294
|e, F=4, mF=+2>: 0.0000184506
|e, F=4, mF=+3>: 0.0000573964
|e, F=4, mF=+4>: 0.0035455947

=== Final Manifold Totals ===
Total F=3 ground   = 0.9871426481
Total F=2 ground   = 0.0034014364
Total excited      = 0.0094559155
Grand total        = 1.0000000000

```
