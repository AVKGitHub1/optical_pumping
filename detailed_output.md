# Detailed Output

```text
=== Configuration Summary ===
Gamma / (2 pi)                  =  6.065000 MHz
Optical-pump detuning / (2 pi)  =  10.000000 MHz
Repump detuning / (2 pi)        =  10.000000 MHz
B field                         =  2.350000 G
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
  sigma+ power fraction        = 0.999898
  pi power fraction            = 0.000000
  sigma- power fraction        = 0.000102

Laser: repump
  nominal target              = F=2 -> F'=3
  intensity                    = 100 (same units as I_sat)
  sigma+ power fraction        = 0.333333
  pi power fraction            = 0.333333
  sigma- power fraction        = 0.333333

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
optical_pump: |g, F=3, mF=-3> -> |e, F=3, mF=-2>  sigma+  rate =  2.524562e+06 s^-1  Delta/(2pi) =  9.269701 MHz
optical_pump: |g, F=3, mF=-2> -> |e, F=3, mF=-1>  sigma+  rate =  4.318920e+06 s^-1  Delta/(2pi) =  9.085621 MHz
optical_pump: |g, F=3, mF=-2> -> |e, F=3, mF=-3>  sigma-  rate =  1.687884e+02 s^-1  Delta/(2pi) =  11.650698 MHz
optical_pump: |g, F=3, mF=-1> -> |e, F=3, mF=+0>  sigma+  rate =  5.341315e+06 s^-1  Delta/(2pi) =  8.901541 MHz
optical_pump: |g, F=3, mF=-1> -> |e, F=3, mF=-2>  sigma-  rate =  2.886388e+02 s^-1  Delta/(2pi) =  11.466618 MHz
optical_pump: |g, F=3, mF=+0> -> |e, F=3, mF=+1>  sigma+  rate =  5.531828e+06 s^-1  Delta/(2pi) =  8.717462 MHz
optical_pump: |g, F=3, mF=+0> -> |e, F=3, mF=-1>  sigma-  rate =  3.565188e+02 s^-1  Delta/(2pi) =  11.282538 MHz
optical_pump: |g, F=3, mF=+1> -> |e, F=3, mF=+2>  sigma+  rate =  4.803003e+06 s^-1  Delta/(2pi) =  8.533382 MHz
optical_pump: |g, F=3, mF=+1> -> |e, F=3, mF=+0>  sigma-  rate =  3.683737e+02 s^-1  Delta/(2pi) =  11.098459 MHz
optical_pump: |g, F=3, mF=+2> -> |e, F=3, mF=+3>  sigma+  rate =  3.024878e+06 s^-1  Delta/(2pi) =  8.349302 MHz
optical_pump: |g, F=3, mF=+2> -> |e, F=3, mF=+1>  sigma-  rate =  3.186520e+02 s^-1  Delta/(2pi) =  10.914379 MHz
optical_pump: |g, F=3, mF=+3> -> |e, F=3, mF=+2>  sigma-  rate =  1.995780e+02 s^-1  Delta/(2pi) =  10.730299 MHz
repump: |g, F=2, mF=-2> -> |e, F=3, mF=-1>  sigma+  rate =  8.726714e+05 s^-1  Delta/(2pi) =  13.475576 MHz
repump: |g, F=2, mF=-2> -> |e, F=3, mF=-2>  pi  rate =  3.689629e+06 s^-1  Delta/(2pi) =  14.758115 MHz
repump: |g, F=2, mF=-2> -> |e, F=3, mF=-3>  sigma-  rate =  9.473270e+06 s^-1  Delta/(2pi) =  16.040653 MHz
repump: |g, F=2, mF=-1> -> |e, F=3, mF=+0>  sigma+  rate =  3.626648e+06 s^-1  Delta/(2pi) =  11.096519 MHz
repump: |g, F=2, mF=-1> -> |e, F=3, mF=-1>  pi  rate =  7.990490e+06 s^-1  Delta/(2pi) =  12.379057 MHz
repump: |g, F=2, mF=-1> -> |e, F=3, mF=-2>  sigma-  rate =  8.373945e+06 s^-1  Delta/(2pi) =  13.661596 MHz
repump: |g, F=2, mF=+0> -> |e, F=3, mF=+1>  sigma+  rate =  1.016037e+07 s^-1  Delta/(2pi) =  8.717462 MHz
repump: |g, F=2, mF=+0> -> |e, F=3, mF=+0>  pi  rate =  1.234150e+07 s^-1  Delta/(2pi) =  10.000000 MHz
repump: |g, F=2, mF=+0> -> |e, F=3, mF=-1>  sigma-  rate =  6.764572e+06 s^-1  Delta/(2pi) =  11.282538 MHz
repump: |g, F=2, mF=+1> -> |e, F=3, mF=+2>  sigma+  rate =  2.173147e+07 s^-1  Delta/(2pi) =  6.338404 MHz
repump: |g, F=2, mF=+1> -> |e, F=3, mF=+1>  pi  rate =  1.419405e+07 s^-1  Delta/(2pi) =  7.620943 MHz
repump: |g, F=2, mF=+1> -> |e, F=3, mF=+0>  sigma-  rate =  4.372663e+06 s^-1  Delta/(2pi) =  8.903481 MHz
repump: |g, F=2, mF=+2> -> |e, F=3, mF=+3>  sigma+  rate =  2.921925e+07 s^-1  Delta/(2pi) =  3.959347 MHz
repump: |g, F=2, mF=+2> -> |e, F=3, mF=+2>  pi  rate =  8.597691e+06 s^-1  Delta/(2pi) =  5.241885 MHz
repump: |g, F=2, mF=+2> -> |e, F=3, mF=+1>  sigma-  rate =  1.495321e+06 s^-1  Delta/(2pi) =  6.524424 MHz

=== Population-Conservation Check ===
Initial total population       = 1.000000000000
Final total population         = 1.000000000000
Maximum |sum(p)-1| over time   = 2.220e-15

=== Final Populations ===
|g, F=3, mF=-3>: 0.0000021363
|g, F=3, mF=-2>: 0.0000042322
|g, F=3, mF=-1>: 0.0000090049
|g, F=3, mF=+0>: 0.0000190550
|g, F=3, mF=+1>: 0.0000624905
|g, F=3, mF=+2>: 0.0002028394
|g, F=3, mF=+3>: 0.9960285887
|g, F=2, mF=-2>: 0.0000005858
|g, F=2, mF=-1>: 0.0000017359
|g, F=2, mF=+0>: 0.0000036997
|g, F=2, mF=+1>: 0.0000101722
|g, F=2, mF=+2>: 0.0000223241
|e, F=1, mF=-1>: 0.0000000188
|e, F=1, mF=+0>: 0.0000000739
|e, F=1, mF=+1>: 0.0000002363
|e, F=2, mF=-2>: 0.0000000232
|e, F=2, mF=-1>: 0.0000000545
|e, F=2, mF=+0>: 0.0000001388
|e, F=2, mF=+1>: 0.0000002598
|e, F=2, mF=+2>: 0.0000030670
|e, F=3, mF=-3>: 0.0000001457
|e, F=3, mF=-2>: 0.0000005798
|e, F=3, mF=-1>: 0.0000015141
|e, F=3, mF=+0>: 0.0000037937
|e, F=3, mF=+1>: 0.0000084200
|e, F=3, mF=+2>: 0.0000262156
|e, F=3, mF=+3>: 0.0000400779
|e, F=4, mF=-4>: 0.0000000000
|e, F=4, mF=-3>: 0.0000000001
|e, F=4, mF=-2>: 0.0000000004
|e, F=4, mF=-1>: 0.0000000019
|e, F=4, mF=+0>: 0.0000000076
|e, F=4, mF=+1>: 0.0000000259
|e, F=4, mF=+2>: 0.0000007738
|e, F=4, mF=+3>: 0.0000050118
|e, F=4, mF=+4>: 0.0035426947

=== Final Manifold Totals ===
Total F=3 ground   = 0.9963283470
Total F=2 ground   = 0.0000385177
Total excited      = 0.0036331353
Grand total        = 1.0000000000

```
