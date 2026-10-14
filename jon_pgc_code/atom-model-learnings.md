# A standalone specification of the 87Rb atom-cooling model

This document specifies semiclassical polarization-gradient cooling on the
87Rb D2 line. It contains the constants, basis conventions, fields,
Hamiltonians, scattering rules, integration algorithm, and observables
needed to implement the model independently. It covers a Monte Carlo
wavefunction (MCWF) engine and a dressed-state rate-equation alternative.

**The Python blocks form a self-contained MCWF reference implementation.**
Copy all blocks marked `python`, in order, into one file and run it with
NumPy and SymPy installed. No repository modules, downloaded atomic data,
or external tables are required. Equation and pseudocode blocks marked
`text` are explanatory. Use double-precision real and complex arrays.

## 1. Physical state and scope

Each of `N` noninteracting trajectories carries classical position `X[n,3]`,
velocity `V[n,3]`, an internal ground state, an alive flag, and photon
counters. MCWF stores a normalized complex vector `psi[n,8]`; the alternative
rate engine stores an index of a local dressed ground eigenstate.

The ground manifold is `5S_1/2` with nuclear spin `I=3/2`, spanning `F=2,1`.
The excited manifold is `5P_3/2`, with 16 states spanning `F'=0,1,2,3`.
Excited states are diagonalized locally and adiabatically eliminated.
There is no explicitly propagated excited population or excited-state
residence-time trajectory.

Cooling emerges from position-dependent optical forces, internal evolution,
optical pumping, and recoil. No target temperature or fitted friction is
imposed. Motion is classical; motional quantization, collisions, and
collective optical effects are absent.

## 2. Units and constants

Use SI units: metres, seconds, kilograms, joules. Temperatures and input
trap depths are in kelvin. Hamiltonians are energies; linewidths, detunings,
and hyperfine coefficients are angular frequencies in rad/s. Magnetic-field
inputs are in gauss. Distinguish cooling wavenumber `k_c` from trap
wavenumber `k_t`.

| Quantity | Convention |
|---|---|
| `U0 = KB * depth_K` | Positive scalar trap depth; the central ground potential is `-U0`. |
| `Gamma` | D2 population decay rate `2*pi*6.0666e6 /s`; optical coherence half-width `Gamma/2`. |
| `delta_c` | Detuning from free-space `F=2 -> F'=3`; red is negative. |
| `delta_r` | Detuning from free-space `F=1 -> F'=2`. |
| `s_cool`, `s_rep` | Saturation parameters per beam, referenced to the stretched cycling transition. |
| `Omega_b` | Stretched-transition Rabi frequency `Gamma*sqrt(s_b/2)`. |
| `v_rec` | Recoil speed `HBAR*k_c/mass`. |

For example, minus three linewidths means `delta_c=-3*Gamma`.

```python
import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from sympy import Rational
from sympy.physics.quantum.cg import CG

HBAR = 1.054_571_8e-34
KB = 1.380_649e-23
AMU = 1.660_539_07e-27
C_LIGHT = 2.997_924_58e8
EPS0 = 8.854_187_82e-12
E_CH = 1.602_176_634e-19
A_BOHR = 5.291_772_11e-11
ALPHA_AU = 1.648_777_27e-41
MASS = 86.909_180_5 * AMU
LAMBDA_D2 = 780.241_209e-9
LAMBDA_D1 = 794.978_851e-9
GAMMA = 2*np.pi*6.0666e6
OMEGA_D2 = 2*np.pi*C_LIGHT/LAMBDA_D2
OMEGA_D1 = 2*np.pi*C_LIGHT/LAMBDA_D1
K_COOL = 2*np.pi/LAMBDA_D2
D_D2 = 4.227*E_CH*A_BOHR
D_D1 = 2.992*E_CH*A_BOHR
A_G = 2*np.pi*3417.341e6
A_E = 2*np.pi*84.7185e6
B_E = 2*np.pi*12.4965e6
MU_B_OVER_HBAR_PER_G = 2*np.pi*1.3996e6

# Hyperfine centroid offsets in rad/s, not joules.
E_G_F2 = 0.75*A_G
E_G_F1 = -1.25*A_G
E_E_F3 = 2*np.pi*193.7407e6
E_E_F2 = -2*np.pi*72.9113e6
BLOCKS = ((slice(0, 5), E_G_F2), (slice(5, 8), E_G_F1))
```

Derived values are `mass=1.4431608976193214e-25 kg`,
`v_rec=5.88454e-3 m/s`, recoil energy `E_rec/KB=0.180978 microkelvin`,
and Doppler scale `HBAR*Gamma/(2*KB)=145.575 microkelvin`. The last two
are diagnostic scales, not imposed temperature floors.

## 3. Angular momentum, basis order, and dipoles

### 3.1 Angular-momentum and hyperfine matrices

Angular-momentum matrices are dimensionless, with eigenvalues `m` rather
than `HBAR*m`. In descending order `m=j,j-1,...,-j`, use
`Jz[m,m]=m`, `J+[m+1,m]=sqrt(j*(j+1)-m*(m+1))`,
`Jx=(J+ + J-)/2`, and `Jy=(J+ - J-)/(2i)`.

Electronic indices are outer and nuclear indices inner in product bases.
Set `K=Jx tensor Ix + Jy tensor Iy + Jz tensor Iz`. Then

```text
H_hfs = hbar * [A K + B (3K^2 + 1.5K - i(i+1)j(j+1) identity)
                           / (2i(2i-1)j(2j-1))].
```

Omit the quadrupole term when either spin is at most one half. Ground
hyperfine energies enter laser frequency references rather than the
propagated ground Hamiltonian.

```python
def jmat(j):
    dim = int(round(2*j)) + 1
    ms = j-np.arange(dim)
    jp = np.zeros((dim, dim), complex)
    for a in range(1, dim):
        jp[a-1, a] = np.sqrt(j*(j+1)-ms[a]*(ms[a]+1))
    return (jp+jp.conj().T)/2, -0.5j*(jp-jp.conj().T), np.diag(ms+0j)


def hfs(j, i, A, B=0.0):
    K = sum(np.kron(a, b) for a, b in zip(jmat(j), jmat(i)))
    H = A*K
    if B != 0 and j > 0.5 and i > 0.5:
        H += B*(3*K@K+1.5*K-i*(i+1)*j*(j+1)*np.eye(len(K))) / (
            2*i*(2*i-1)*j*(2*j-1))
    return HBAR*H
```

### 3.2 Explicit basis transformation

The ground product and coupled bases are ordered as follows:

```text
p = 0..7:
(mJ,mI) = (+1/2,+3/2), (+1/2,+1/2), (+1/2,-1/2), (+1/2,-3/2),
          (-1/2,+3/2), (-1/2,+1/2), (-1/2,-1/2), (-1/2,-3/2).

g = 0..7:
(F,mF) = (2,2), (2,1), (2,0), (2,-1), (2,-2), (1,1), (1,0), (1,-1).
```

The excited product basis uses `mJ'=3/2,1/2,-1/2,-3/2`, each followed by
`mI=3/2,1/2,-1/2,-3/2`. The real orthogonal matrix `U[g,p]` below fixes
all ground phases. Transform a ground operator as `U @ operator @ U.T`.

For absorption polarization `q=+1,0,-1`, the dipole tensor conserves `mI`
and has `mJ'=mJ+q`. Its nonzero electronic amplitudes are:

| `mJ` | `q=+1` | `q=0` | `q=-1` |
|---|---:|---:|---:|
| `+1/2` | `1` | `sqrt(2/3)` | `sqrt(1/3)` |
| `-1/2` | `sqrt(1/3)` | `sqrt(2/3)` | `1` |

Do not renormalize these amplitudes. The stretched transition has amplitude
one, and the squared amplitudes summed over `q,e` equal two for each ground
state. `D[q,g,e]` has shape `(3,8,16)` and acts on an excited vector to
produce a ground vector for emission polarization `q`.

```python
r, h, u = np.sqrt(3)/2, 0.5, np.sqrt(0.5)
U = np.array([
    [1,0,0,0,0,0,0,0], [0,r,0,0,h,0,0,0],
    [0,0,u,0,0,u,0,0], [0,0,0,h,0,0,r,0],
    [0,0,0,0,0,0,0,1], [0,h,0,0,-r,0,0,0],
    [0,0,u,0,0,-u,0,0], [0,0,0,r,0,0,-h,0],
], dtype=float)
D_product = np.zeros((3,8,16))
cg_electronic = np.sqrt([[1,2/3,1/3], [1/3,2/3,1]])
for a, mj in enumerate([0.5,-0.5]):
    for b, q in enumerate([1,0,-1]):
        ep = [1.5,0.5,-0.5,-1.5].index(mj+q)
        for n in range(4):
            D_product[b,4*a+n,4*ep+n] = cg_electronic[a,b]
D = np.einsum('gp,qpe->qge', U, D_product)
JG = [U @ np.kron(a,np.eye(4)) @ U.T for a in jmat(0.5)]
JE = [np.kron(a,np.eye(4)) for a in jmat(1.5)]
H_HFS_E = hfs(1.5,1.5,A_E,B_E)
```

## 4. Polarizability at an arbitrary trap wavelength

The following finite sum of poles defines ground scalar/vector and excited
scalar/vector/tensor polarizabilities. Additional core and high-lying-state
contributions are omitted. Wavelengths at poles, or where the ground scalar
polarizability vanishes, are outside this parameterization's usable domain.

For initial spin `j`, intermediate spin `jp`, signed level spacing `dE`,
trap photon energy `hw`, and squared dipole strength `d2`, define

```text
C_q(m) = |CG(j,m;1,q | jp,m+q)|^2, or zero if the state is forbidden
b_q(m) = C_q(m)/(dE-hw) + C_-q(m)/(dE+hw)
S0 = mean_m b_0(m)
T0 = [b_0(j)-S0] j(2j-1)/[3j^2-j(j+1)] for j>1/2, else zero
V0 = [b_+1(j)-b_+1(-j)] (2j)/(j-(-j))
S_exact = (d2/3) * 2dE/(dE^2-hw^2)
(alpha_s,alpha_v,alpha_t) = (S_exact/S0)*(S0,V0,T0).
```

Use standard Condon-Shortley Clebsch-Gordan coefficients. A downward line
has negative `dE`; retain both rotating and counter-rotating terms. The
strength convention is fixed by `S_exact`; insert no extra `2j+1` factor.

Ground polarizability sums the D2 and D1 upward lines with strengths
`D_D2^2` and `D_D1^2`. Excited polarizability contains the downward D2 line
with strength `D_D2^2/2`, plus these upward poles:

| Pole | `jp` | Wavelength (nm) | Lifetime (ns) | Branch fraction to `5P_3/2` |
|---|---:|---:|---:|---:|
| 5D_5/2 | 2.5 | 776.0 | 238.5 | 0.74 |
| 5D_3/2 | 1.5 | 776.2 | 246.3 | 0.12 |
| 7S_1/2 | 0.5 | 741.0 | 88.0 | 0.63 |
| 6S_1/2 | 0.5 | 1366.9 | 45.6 | 0.67 |
| Effective 4D | 2.5 | 1529.3 | 90.0 | 0.85 |

The last row is one effective `jp=5/2` pole. For each upward line set
`omega0=2*pi*c/lambda`, `G_partial=branch/lifetime`, and

```text
d2 = 3*pi*eps0*hbar*c^3 * (2jp+1)/(2*1.5+1) * G_partial/omega0^3.
```

Multiply only upward excited contributions by `alpha_e_scale` (default
one). Divide all polarizabilities by the ground scalar value to obtain
ratios `ag_s=1`, `ag_v`, `ae_s`, `ae_v`, `ae_t` multiplying `U0`.

```python
def line_alpha(j, jp, dE, hw, d2):
    def rat(x):
        return Rational(int(round(2*x)),2)
    def strength(m,q):
        mp = m+q
        if abs(mp) > jp+1e-9:
            return 0.0
        return float(CG(rat(j),rat(m),1,q,rat(jp),rat(mp)).doit())**2
    def b(m,q):
        return strength(m,q)/(dE-hw)+strength(m,-q)/(dE+hw)
    ms = j-np.arange(int(round(2*j))+1)
    scalar = np.mean([b(m,0) for m in ms])
    tensor = ((b(j,0)-scalar)*j*(2*j-1)/(3*j*j-j*(j+1))
              if j > 0.5 else 0.0)
    vector = (b(j,1)-b(-j,1))/(2*j)*(2*j)
    exact = (d2/3)*2*dE/(dE*dE-hw*hw)
    return (exact/scalar)*np.array([scalar,vector,tensor])


def make_atom(lambda_trap=785e-9, alpha_e_scale=1.0):
    hw = HBAR*2*np.pi*C_LIGHT/lambda_trap
    ag = (line_alpha(0.5,1.5,HBAR*OMEGA_D2,hw,D_D2**2)
          + line_alpha(0.5,0.5,HBAR*OMEGA_D1,hw,D_D1**2))
    ae = line_alpha(1.5,0.5,-HBAR*OMEGA_D2,hw,D_D2**2/2)
    poles = [(2.5,776.0e-9,238.5e-9,0.74), (1.5,776.2e-9,246.3e-9,0.12),
             (0.5,741.0e-9,88e-9,0.63), (0.5,1366.9e-9,45.6e-9,0.67),
             (2.5,1529.3e-9,90e-9,0.85)]
    for jp, wavelength, lifetime, branch in poles:
        w = 2*np.pi*C_LIGHT/wavelength
        d2 = (3*np.pi*EPS0*HBAR*C_LIGHT**3*(2*jp+1)/4
              * (branch/lifetime)/w**3)
        ae += alpha_e_scale*line_alpha(1.5,jp,HBAR*w,hw,d2)
    return SimpleNamespace(
        lambda_trap=lambda_trap, mass=MASS, gamma=GAMMA, k=K_COOL,
        vrec=HBAR*K_COOL/MASS, alpha_g_SI=ag[0], ag_v=ag[1]/ag[0],
        ae_s=ae[0]/ag[0], ae_v=ae[1]/ag[0], ae_t=ae[2]/ag[0])
```

At default upward scaling, useful numerical values are:

| Trap wavelength | `ag_v` | `ae_s` | `ae_v` | `ae_t` |
|---|---:|---:|---:|---:|
| 785 nm | +1.9536 | -0.6299 | -1.9292 | +0.6588 |
| 850 nm | -0.1717 | -0.4238 | -0.9422 | +0.3638 |
| 852 nm | -0.1664 | -0.4304 | -0.9465 | +0.3672 |
| 1064 nm | -0.0423 | -1.4572 | -1.1020 | +0.8748 |

Use the pole calculation rather than rounded table entries for simulation.

## 5. Trap field and Hamiltonians

### 5.1 Complex Gaussian field

The trap propagates along laboratory z with waist `w0`. Define

```text
k_t = 2*pi/lambda_trap; zR = pi*w0^2/lambda_trap
w2 = w0^2*(1+(z/zR)^2); rho2 = x^2+y^2
a = w0/sqrt(w2)*exp(-rho2/w2); invR = z/(z^2+zR^2)
phi = k_t*z-atan(z/zR)+k_t*rho2*invR/2
gx+ = x*(-2/w2+i*k_t*invR); gx- = x*(-2/w2-i*k_t*invR)
gy+ = y*(-2/w2+i*k_t*invR); gy- = y*(-2/w2-i*k_t*invR).
```

Let field ellipticity be `ell`, so input polarization is
`(x_hat+i*ell*y_hat)/sqrt(1+ell^2)`. For a standing wave set
`u+=a*exp(i*phi)/2`, `u-=a*exp(-i*phi)/2`, and

```text
e_x = (u+ + u-)/sqrt(1+ell^2)
e_y = i*ell*(u+ + u-)/sqrt(1+ell^2)
e_z = (i/k_t)*[u+*gx+ - u-*gx- + i*ell*(u+*gy+ - u-*gy-)]/sqrt(1+ell^2).
```

For a traveling wave set `u=a*exp(i*phi)` and

```text
e_x = u/sqrt(1+ell^2)
e_y = i*ell*u/sqrt(1+ell^2)
e_z = (i/k_t)*u*(gx+ + i*ell*gy+)/sqrt(1+ell^2).
```

The dimensionless field has `e*.e=1` at the origin. Its longitudinal
component is a first-order focusing correction. A balanced standing wave
with `ell=0` is real and has zero vector Stark field. A traveling wave can
have local ellipticity even with linear input polarization.

### 5.2 Scalar, vector, tensor, and magnetic energies

Define `I=e*.e` and `Cvec=real[-i*(e* cross e)]`. For electronic spin `j`,

```text
H_Stark(j) = -U0*[alpha_s*I*identity + alpha_v*(Cvec.J)/(2j)
              + alpha_t*(3*Sym[(e*.J)(e.J)]-I*J^2)/(j*(2j-1))]
Sym[A B] = (A B + B A)/2.
```

Omit the tensor term for `j=1/2`. The scalar ground potential is `Ug=-U0*I`.
The remaining ground and excited Hamiltonians are

```text
Hg = -U0*ag_v*sum_a Cvec_a*JG_a
     + hbar*2.002*muB_over_hbar*(Bx*JG_x+Bg*JG_z)
     + hbar*2*pi*quad_MHz*1e6*diag(4,1,0,1,4,0,0,0)
He = H_hfs_excited + H_Stark(3/2) tensor identity_4
     + hbar*(4/3)*muB_over_hbar*(Bx*JE_x+Bz*JE_z).
```

`muB_over_hbar=2*pi*1.3996e6 rad/s/G`; nuclear Zeeman terms are omitted.
`Bg` equals the explicit ground z-field override if supplied, otherwise
zero when `excited_only=True`, otherwise `Bz`. Suppressing ground Zeeman
energy does not suppress the trap vector shift. `quad_MHz` is an optional
phenomenological quadratic shift in `F=2`.

Retain all entries of `Hg`, including off-block `F=1/F=2` elements. Laser
terms below are block diagonal; this ground term is not. This prescription
has a rotating-frame/secular-approximation limitation because the large
ground splitting has been removed. Projecting away the off-block entries
would define a different model.

```python
def stark(e,U0,j,scalar,vector,tensor):
    J = np.array(jmat(j))
    J2 = sum(a@a for a in J)
    intensity = np.real(np.einsum('na,na->n',e.conj(),e))
    cv = np.real(-1j*np.cross(e.conj(),e))
    edj = np.einsum('na,aij->nij',e,J)
    ecj = np.einsum('na,aij->nij',e.conj(),J)
    sym = 0.5*(np.einsum('nab,nbc->nac',ecj,edj)
               + np.einsum('nab,nbc->nac',edj,ecj))
    H = scalar*intensity[:,None,None]*np.eye(len(J2))[None]
    H = H+np.einsum('na,aij->nij',cv,J)/(2*j)*vector
    if j > 0.5:
        H += tensor*(3*sym-intensity[:,None,None]*J2)/(j*(2*j-1))
    return -U0*H


@dataclass
class Trap:
    atom: object
    depth_K: float = 1e-3
    waist: float = 1e-6
    standing: bool = True
    ellipticity: float = 0.0
    Bz: float = 0.0
    Bx: float = 0.0
    ground_Bz: object = None
    excited_only: bool = False
    quad_MHz: float = 0.0

    def __post_init__(self):
        self.U0 = KB*self.depth_K
        self.kt = 2*np.pi/self.atom.lambda_trap
        self.zR = np.pi*self.waist**2/self.atom.lambda_trap

    def field(self,X):
        x,y,z = X.T
        w2 = self.waist**2*(1+(z/self.zR)**2)
        rho2 = x*x+y*y
        amp = self.waist/np.sqrt(w2)*np.exp(-rho2/w2)
        invR = z/(z*z+self.zR**2)
        phi = self.kt*z-np.arctan(z/self.zR)+0.5*self.kt*rho2*invR
        ell = self.ellipticity
        norm = np.sqrt(1+ell*ell)
        gp = -2/w2+1j*self.kt*invR
        gm = -2/w2-1j*self.kt*invR
        gxp,gxm,gyp,gym = x*gp,x*gm,y*gp,y*gm
        e = np.zeros((len(X),3),complex)
        if self.standing:
            up,um = 0.5*amp*np.exp(1j*phi),0.5*amp*np.exp(-1j*phi)
            e[:,0] = (up+um)/norm
            e[:,1] = 1j*ell*(up+um)/norm
            e[:,2] = (1j/self.kt)*(up*gxp-um*gxm
                                   + 1j*ell*(up*gyp-um*gym))/norm
        else:
            wave = amp*np.exp(1j*phi)
            e[:,0] = wave/norm
            e[:,1] = 1j*ell*wave/norm
            e[:,2] = (1j/self.kt)*wave*(gxp+1j*ell*gyp)/norm
        return e

    def scalar(self,X):
        e = self.field(X)
        return -self.U0*np.real(np.einsum('na,na->n',e.conj(),e))

    def ground(self,X):
        e = self.field(X)
        cv = np.real(-1j*np.cross(e.conj(),e))
        H = -self.U0*self.atom.ag_v*np.einsum('na,aij->nij',cv,np.array(JG))
        bg = (self.ground_Bz if self.ground_Bz is not None
              else (0.0 if self.excited_only else self.Bz))
        H += HBAR*2.002*MU_B_OVER_HBAR_PER_G*(self.Bx*JG[0]+bg*JG[2])
        H += HBAR*2*np.pi*self.quad_MHz*1e6*np.diag([4,1,0,1,4,0,0,0])
        return H

    def excited(self,X):
        a = self.atom
        H4 = stark(self.field(X),self.U0,1.5,a.ae_s,a.ae_v,a.ae_t)
        H16 = np.einsum('nab,cd->nacbd',H4,np.eye(4)).reshape(-1,16,16)
        if self.Bz:
            H16 = H16+HBAR*(4/3)*MU_B_OVER_HBAR_PER_G*self.Bz*JE[2]
        if self.Bx:
            H16 = H16+HBAR*(4/3)*MU_B_OVER_HBAR_PER_G*self.Bx*JE[0]
        return H16+H_HFS_E

    def frequencies(self):
        wr = np.sqrt(4*self.U0/(MASS*self.waist**2))
        wz = (self.kt*np.sqrt(2*self.U0/MASS) if self.standing
              else np.sqrt(2*self.U0/(MASS*self.zR**2)))
        return np.array([wr,wr,wz])
```

Nominal harmonic frequencies initialize positions; actual forces use the
full field. The frequency formulas do not include the full longitudinal
and Gouy corrections that a numerical Hessian would contain.

## 6. Beam geometry, polarization, and phase groups

A beam has propagation unit vector `d_b`, complex unit Jones vector `eps_b`,
saturation `s_b`, phase `phi_b`, phase-roll rate `roll_b`, and a frequency
label. Plane-wave cooling beams have no spatial intensity envelope.

The spherical basis about laboratory z is

```text
e_+ = (-x_hat-i*y_hat)/sqrt(2)
e_0 = z_hat
e_- = ( x_hat-i*y_hat)/sqrt(2)
p[q,b] = conj(e_q).eps_b.
```

Its explicit conjugation is important. Use the following beam order and
Jones vectors for circular pairs; defining vectors avoids helicity naming
ambiguities for reverse propagation:

| Beam index | Direction | Jones vector |
|---|---|---|
| 0 | `+x` | `(y+i*z)/sqrt(2)` |
| 1 | `-x` | `(y-i*z)/sqrt(2)` |
| 2 | `+y` | `(z+i*x)/sqrt(2)` |
| 3 | `-y` | `(z-i*x)/sqrt(2)` |
| 4 | `+z` | `(x+i*y)/sqrt(2)` |
| 5 | `-z` | `(x-i*y)/sqrt(2)` |

For linear-perpendicular-linear pairs, keep those directions and use Jones
vectors `[y,z,z,x,x,y]` in that order. A transverse 2D setup takes the first
four beams; 1D setups take a selected pair. A single-beam setup takes `+x`.

Two additional six-beam layouts are defined here without experimental
assumptions:

- `split_plane`: pair directions `nA=(0,cos(theta),sin(theta))` and
  `nB=(0,cos(theta),-sin(theta))`, followed by an x pair. For each of the
  first two pairs, the forward Jones vector is `(x+i*(n cross x))/sqrt(2)`
  and the reverse uses the minus sign. The x cooling pair has twice the
  intensity of each other cooling beam; repump intensities remain equal.
- `oblique`: the x and y pairs above, followed by a pair along the normalized
  direction `(sin(10 degrees),cos(10 degrees)/sqrt(2),cos(10 degrees)/sqrt(2))`.
  Let `t1=normalize(n cross z)` and `t2=n cross t1`. Its Jones vectors are
  `(t1 +/- i*t2)/sqrt(2)`.

The `tilt` two-beam layout uses directions `(cos(theta),0,sin(theta))` and
`(-cos(theta),0,sin(theta))` and Jones vector `(y+i*(d cross y))/sqrt(2)`
for each. These directions are not generally counterpropagating. A separate
rigid tilt rotates all directions and polarizations by
`Ry(beta)=[[cos(beta),0,sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]]`.

Repump beams can copy every cooling direction/polarization, form an x pair,
be a single +x beam, or be absent. Append repump beams after cooling beams.
All repump beams share the `rep` label. Ordinarily all cooling beams share
`cool`; `2d_inc` instead labels the y pair `cool2`, eliminating cooler x/y
interference. Even in that case all repump beams remain in one group.

Beam amplitudes and frequency offsets are

```text
A[q,b](X,t) = Gamma*sqrt(s_b/2)*p[q,b]
              * exp(i*(k_c*d_b.X + phi_b + roll_b*t))
off_cool = delta_c + E_E_F3 - E_G_F2
off_rep  = delta_r + E_E_F2 - E_G_F1.
```

`cool2` has the same offset as `cool`. Coherently sum amplitudes within a
label, and add different groups incoherently in the effective operators.
Independent uniform phases on `[0,2*pi)` are drawn once per beam set.
Rolling adds `2*pi*roll_hz` to the second cooling pair's phase rates only;
it changes neither detuning offsets nor repump phases. Frequency separation
is therefore approximated by group separation or a phase ramp, rather than
by a complete multicolor optical treatment.

```python
E_SPH = np.array([[-1,-1j,0],[0,0,np.sqrt(2)],[1,-1j,0]])/np.sqrt(2)


def make_beams(atom,s_cool,delta_c,geometry='3d',s_rep=0.05,delta_r=0.0,
               repump='all',seed=0,phases=None,roll_hz=0.0,
               theta_deg=45.0,tilt_deg=0.0):
    ex,ey,ez = np.eye(3)
    dirs = [ex,-ex,ey,-ey,ez,-ez]
    pols = [(ey+1j*ez),(ey-1j*ez),(ez+1j*ex),
            (ez-1j*ex),(ex+1j*ey),(ex-1j*ey)]
    circular = [(d,p/np.sqrt(2)) for d,p in zip(dirs,pols)]
    linear = list(zip(dirs,[ey,ez,ez,ex,ex,ey]))
    if geometry == '3d':
        geo = circular
    elif geometry in ('2d','2d_inc'):
        geo = circular[:4]
    elif geometry == '3d_lpl':
        geo = linear
    elif geometry == '2d_lpl':
        geo = linear[:4]
    elif geometry == '1d_x':
        geo = circular[:2]
    elif geometry == '1d_y':
        geo = circular[2:4]
    elif geometry == 'single_x':
        geo = circular[:1]
    elif geometry == 'split_plane':
        th = np.deg2rad(theta_deg)
        geo = []
        for n in (np.array([0,np.cos(th),np.sin(th)]),
                  np.array([0,np.cos(th),-np.sin(th)])):
            tangent = np.cross(n,ex)
            geo += [(n,(ex+1j*tangent)/np.sqrt(2)),
                    (-n,(ex-1j*tangent)/np.sqrt(2))]
        geo += circular[:2]
    elif geometry == 'oblique':
        a = np.deg2rad(10)
        n = np.array([np.sin(a),np.cos(a)/np.sqrt(2),np.cos(a)/np.sqrt(2)])
        n /= np.linalg.norm(n)
        t1 = np.cross(n,ez)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n,t1)
        geo = circular[:4]+[(n,(t1+1j*t2)/np.sqrt(2)),
                            (-n,(t1-1j*t2)/np.sqrt(2))]
    elif geometry == 'tilt':
        th = np.deg2rad(theta_deg)
        geo = []
        for sign in (1,-1):
            n = np.array([sign*np.cos(th),0,np.sin(th)])
            geo.append((n,(ey+1j*np.cross(n,ey))/np.sqrt(2)))
    else:
        raise ValueError(geometry)
    nc = len(geo)
    labels = [('cool2' if geometry == '2d_inc' and b >= 2 else 'cool')
              for b in range(nc)]
    rep_geo = {'all':geo,'pair_x':circular[:2],
               'single_x':circular[:1],'none':[]}[repump]
    all_geo = geo+rep_geo
    labels += ['rep']*len(rep_geo)
    directions = np.array([d for d,p in all_geo],float)
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    jones = np.array([p for d,p in all_geo],complex)
    angle = np.deg2rad(tilt_deg)
    c,s = np.cos(angle),np.sin(angle)
    rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    directions,jones = directions@rotation.T,jones@rotation.T
    labels = np.array(labels)
    nb = len(labels)
    rng = np.random.default_rng(seed)
    phase = (rng.uniform(0,2*np.pi,nb) if phases is None
             else np.asarray(phases,float))
    roll = np.zeros(nb)
    if nc >= 4:
        roll[2:4] = 2*np.pi*roll_hz
    Omega = atom.gamma*np.sqrt(np.where(labels=='rep',s_rep,s_cool)/2)
    if geometry == 'split_plane':
        Omega[4:6] *= np.sqrt(2)
    off = np.where(labels=='rep',delta_r+E_E_F2-E_G_F1,
                   delta_c+E_E_F3-E_G_F2)
    return SimpleNamespace(dirs=directions,kvec=atom.k*directions,
        pol=np.einsum('qi,bi->qb',E_SPH.conj(),jones),Omega=Omega,
        labels=labels,phases=phase,roll=roll,off=off)
```

## 7. Elimination into ground-state operators

At each evaluated position and time, diagonalize the Hermitian excited
Hamiltonian: `He V = V diag(E)`. Eigenvectors are columns `V[e,k]`, sorted
by ascending eigenvalue. `E[k]` is in joules. The following index-level
formula fixes the complex-conjugation convention:

```text
w[k,g,b] = sum_{q,e} D[q,g,e] * conj(V[e,k]) * A[q,b]
w_L[k,g] = sum_{b in group L} w[k,g,b]
s_local = (2/Gamma^2) * sum_L sum_q |sum_{b in L} A[q,b]|^2
Delta[k,L,F] = off[L] + E_G_F + Ug/hbar - E[k]/hbar
ell[k,L,F] = 1 / (Delta[k,L,F]^2 + Gamma^2*(1+s_local)/4)
Q[k,g,h] = conj(w_L[k,g])*w_L[k,h].
```

For each group and each ground block separately, add

```text
H_light[g,h] += (hbar/4) * sum_k Delta[k,L,F]*ell[k,L,F]*Q[k,g,h]
R[g,h]       += (Gamma/4) * sum_k ell[k,L,F]*Q[k,g,h]
H_eff = H_light + Hg.
```

Both cooler and repumper are evaluated against both ground blocks. Each
frequency group must contain beams with the same offset. These laser terms
have no cross-block or cooler-repumper Raman interference. `Hg` is then
added as specified in section 5, without extra block projection.

`R` has units of inverse seconds and is positive semidefinite up to
floating-point error. `H_eff` is Hermitian. Ground vector/Zeeman energies
enter `H_eff`, but are not separately inserted into the line denominators.
Local saturation broadens every channel through one shared `s_local`;
this is an approximate multilevel saturation prescription.

There is **no velocity in this operator calculation**. Explicit Doppler
weighting appears only in absorption-beam selection at a jump. Motion also
changes the sampled spatial polarization texture.

```python
def operators(X,t,trap,beams):
    G = trap.atom.gamma
    E,V = np.linalg.eigh(trap.excited(X))
    phase = np.exp(1j*(X@beams.kvec.T+(beams.phases+beams.roll*t)[None]))
    A = beams.Omega[None,None,:]*beams.pol[None,:,:]*phase[:,None,:]
    DV = np.einsum('qge,nek->nqgk',D,V.conj())
    wb = np.einsum('nqgk,nqb->nkgb',DV,A)
    groups = list(dict.fromkeys(beams.labels))
    s_local = np.zeros(len(X))
    for label in groups:
        field = A[:,:,beams.labels==label].sum(axis=2)
        s_local += 2*np.real(np.einsum('nq,nq->n',field.conj(),field))/G**2
    width2 = (G**2/4)*(1+s_local)[:,None]
    base = -E/HBAR+(trap.scalar(X)/HBAR)[:,None]
    H = np.zeros((len(X),8,8),complex)
    R = np.zeros_like(H)
    channels = []
    for label in groups:
        mask = beams.labels==label
        wl = wb[:,:,:,mask].sum(axis=3)
        off = beams.off[mask][0]
        for block,ground_energy in BLOCKS:
            delta = off+base+ground_energy
            lor = 1/(delta*delta+width2)
            w = wl[:,:,block]
            Q = np.einsum('nkg,nkh->nkgh',w.conj(),w)
            H[:,block,block] += (HBAR/4)*np.einsum('nk,nkgh->ngh',delta*lor,Q)
            R[:,block,block] += (G/4)*np.einsum('nk,nkgh->ngh',lor,Q)
            channels.append((label,mask,block,lor,delta))
    return SimpleNamespace(H=H+trap.ground(X),R=R,wb=wb,
                           V=V,channels=channels)
```

A return value is also a scattering cache: retain `wb`, excited
eigenvectors, and every channel's detunings and Lorentzians together.
They must refer to the same positions and beam phases.

## 8. MCWF propagation and photon scattering

### 8.1 No-jump propagation

For a time step `dt`, form `M=(-i*H_eff/HBAR-R/2)*dt`. Approximate its
exponential by a fourth-order Taylor polynomial at half the argument,
followed by one squaring:

```text
Z=M/2
P=identity+Z+Z^2/2+Z^3/6+Z^4/24
Prop=P P.
```

Apply `psi_tilde=Prop*psi`, calculate `p_survive=||psi_tilde||^2`, and draw
`u` uniformly on `[0,1)`. Jump if `u>p_survive`; otherwise normalize
`psi_tilde`. There is at most one ordinary scattering jump per time step.
This requires small jump probability and a sufficiently accurate matrix
exponential approximation. There is no adaptive step-size control.

### 8.2 Jump channel and coherent excited state

For a jump, first normalize the propagated state. For each group/block,
compute

```text
a_k = sum_{g in F} w_L[k,g]*psi_g
P(L,F) = sum_k |a_k|^2 * ell[k,L,F].
```

Normalize these nonnegative weights and sample a group/block. For that
choice reconstruct

```text
L_k = 1/(Delta_k+i*Gamma/2)
exc_e = sum_k V[e,k]*a_k*L_k.
```

Keep this coherent sum over excited eigenchannels. Sampling just one
excited eigenstate defines a different optical-pumping model. Notice that
jump reconstruction uses the **unsaturated** half-width `Gamma/2`, even
though channel-selection weights use the broadened denominator. Preserve
this distinction to reproduce this approximation.

### 8.3 Absorption direction, emission, and recoil

For each candidate absorption beam in the selected group, calculate

```text
a_kb = sum_{g in F} w[k,g,b]*psi_g
B_b = sum_k L_k*a_kb
Delta_eff = sum_k |a_k|^2*Delta_k / max(sum_k |a_k|^2,1e-300)
f_Doppler(b) = (Delta_eff^2+Gamma^2/4)
               / ((Delta_eff-kvec_b.velocity)^2+Gamma^2/4)
P_beam(b) proportional to |B_b|^2*f_Doppler(b).
```

If all beam weights vanish numerically, choose uniformly within the group.
This beam-selection prescription is an approximation, not a full
velocity-dependent optical Bloch calculation. In particular, `B_b` is a
scalar sum of eigenchannel amplitudes; do not replace it with a sum of
squared channel amplitudes when matching the specified algorithm.

For emission compute `c[q,g]=sum_e D[q,g,e]*exc_e`. Sample `q` with
probability proportional to `sum_g |c[q,g]|^2`, and set
`psi_new=c[q,:]/||c[q,:]||`. Do not collapse further onto a bare ground
state or a single hyperfine block.

Independently draw a 3-vector of unit-variance normal samples and normalize
it to get isotropic direction `n_emit`. Update

```text
velocity += v_rec*(d_absorption+n_emit).
```

The plus sign on the random vector is equivalent in distribution to a minus
sign because its distribution is inversion symmetric. Emission direction
is independent of emission polarization; the dipole radiation pattern is
not resolved. Count one ordinary scatter and, if label `rep` was selected,
one repump scatter.

```python
def propagator(H,R,dt):
    Z = 0.5*((-1j*H/HBAR-0.5*R)*dt)
    Z2 = Z@Z
    Z3 = Z2@Z
    Z4 = Z3@Z
    P = np.eye(8)[None]+Z+Z2/2+Z3/6+Z4/24
    return P@P


def scatter(rng,psi,velocity,j,cache,beams,atom):
    psi = psi/max(np.linalg.norm(psi),1e-300)
    weights,amplitudes = [],[]
    for label,mask,block,lor,delta in cache.channels:
        wl = cache.wb[j][:,:,mask].sum(axis=2)
        ak = wl[:,block]@psi[block]
        weights.append(np.sum(np.abs(ak)**2*lor[j]))
        amplitudes.append(ak)
    weights = np.asarray(weights)
    if weights.sum() <= 0:
        return psi,velocity,None
    selected = rng.choice(len(weights),p=weights/weights.sum())
    label,mask,block,lor,delta = cache.channels[selected]
    ak = amplitudes[selected]
    G = atom.gamma
    L = 1/(delta[j]+1j*G/2)
    exc = np.einsum('ek,k->e',cache.V[j],ak*L)
    wkb = cache.wb[j][:,:,mask]
    akb = np.einsum('kgb,g->kb',wkb[:,block,:],psi[block])
    B = np.einsum('k,kb->b',L,akb)
    w2 = np.abs(ak)**2
    deff = np.sum(w2*delta[j])/max(w2.sum(),1e-300)
    dop = beams.kvec[mask]@velocity
    prob = np.abs(B)**2*(deff**2+G**2/4)/((deff-dop)**2+G**2/4)
    candidates = np.flatnonzero(mask)
    b = (rng.choice(candidates,p=prob/prob.sum()) if prob.sum()>0
         else candidates[rng.integers(len(candidates))])
    c = np.einsum('qge,e->qg',D,exc)
    pq = np.sum(np.abs(c)**2,axis=1)
    q = rng.choice(3,p=pq/pq.sum())
    newpsi = c[q]/np.linalg.norm(c[q])
    nem = rng.normal(size=3)
    nem /= np.linalg.norm(nem)
    return newpsi,velocity+atom.vrec*(beams.dirs[b]+nem),label
```

## 9. Forces, initialization, and exact MCWF step order

### 9.1 Mechanical forces

For a fixed internal vector `chi`, define
`U_internal(X,t;chi)=real[chi^dagger H_eff(X,t) chi]`. The force is

```text
F = -grad Ug(X) - grad U_internal(X,t;chi).
```

Evaluate both gradients by central differences with displacement
`h_fd=5e-9 m`. Hold `chi` and time fixed for all six displaced operator
calculations. Do not differentiate eigenvectors or the normalized MCWF
state with position during this calculation.

Scalar trap force is recomputed as the atom moves. The internal force is
stored as a vector and held constant between refreshes; it is not merely
an internal state cached for fresh force evaluations. Default `dt=10 ns`
and `n_cache=8` give an 80 ns optical-cache interval.

```python
H_FD = 5e-9


def force_trap(X,trap):
    F = np.empty_like(X)
    for axis in range(3):
        xp,xm = X.copy(),X.copy()
        xp[:,axis] += H_FD
        xm[:,axis] -= H_FD
        F[:,axis] = -(trap.scalar(xp)-trap.scalar(xm))/(2*H_FD)
    return F


def force_internal(X,psi,t,trap,beams):
    F = np.empty_like(X)
    for axis in range(3):
        xp,xm = X.copy(),X.copy()
        xp[:,axis] += H_FD
        xm[:,axis] -= H_FD
        hp = operators(xp,t,trap,beams).H
        hm = operators(xm,t,trap,beams).H
        ep = np.real(np.einsum('ng,ngh,nh->n',psi.conj(),hp,psi))
        em = np.real(np.einsum('ng,ngh,nh->n',psi.conj(),hm,psi))
        F[:,axis] = -(ep-em)/(2*H_FD)
    return F
```

### 9.2 Initial ensemble

For input temperatures `T_x,T_y,T_z` use independent normal samples with
`sigma_v,i=sqrt(KB*T_i/MASS)` and `sigma_x,i=sigma_v,i/omega_i`.
The nominal trap frequencies are

```text
omega_r = sqrt(4U0/(MASS*w0^2))
omega_z,standing = k_t*sqrt(2U0/MASS)
omega_z,traveling = sqrt(2U0/(MASS*zR^2)).
```

This samples a harmonic thermal distribution, not the full finite-depth
Gaussian-trap Boltzmann distribution. It does not reject initially unbound
atoms. For MCWF, choose one of the five bare `F=2` states uniformly per
atom. All ordinary counters initially vanish and all atoms start alive.

For free space, set `depth_K=0`, explicitly supply finite initial `X` and
`V`, and disable geometric loss. One useful choice is a position normal
width of 50 micrometres and Maxwellian velocities at the desired temperature.
The reference rejects missing free-space initial positions rather than
using undefined harmonic widths.

An initial-state dictionary accepts `X`, `V`, `alive`, `nscat`, and optional
`t`. It supplies classical state, not an exact MCWF checkpoint: a new
`psi` is initialized, repump counts reset, and a new RNG is used. Exact
checkpointing would additionally require `psi`, counters, RNG state, and
integration/cache state. Both distinctions matter when chaining bursts.

### 9.3 One step, in order

At a regular cache refresh, compact the currently alive atoms, evaluate
operators at their current `(X,t)`, construct the propagator, and compute
internal forces from their current states. Keep the global-to-compact index
mapping with that cache.

For every step:

1. If needed, set acceleration to `(F_trap(X)+F_internal_cached)/MASS`.
2. For alive atoms, advance `X += V*dt + acceleration*dt^2/2`.
3. Recompute scalar trap force at the new positions, add the same cached
   internal force, and update `V += (acceleration+new_acceleration)*dt/2`.
4. Propagate the cached atoms' internal states. Draw jump decisions from
   norm loss and execute ordinary jumps in ascending compact-atom order.
   Doppler weighting uses the velocity after the mechanical update.
5. Apply optional compensation scattering, described below.
6. Store the new internal states. For atoms with ordinary jumps, refresh
   their internal force at the new position and state, using the current
   pre-increment time `t`. Their propagators and scattering-channel data
   remain unchanged until the next regular cache refresh. Invalidate the
   acceleration when a force changes.
7. Increment `t` by `dt`. At a recording threshold, record observables and
   then apply geometric loss.

Because internal propagation follows the mechanical step, the optical
cache represents an earlier position throughout a window. The phase time
used in an immediate post-jump force refresh is also the pre-increment
step time. These are part of this discretization, not exact simultaneous
Hamiltonian evolution.

### 9.4 Optional compensation scattering

A phenomenological extra beam can be specified by a scattering rate
`cancel_R` in inverse seconds. Each time step and cached atom draws a
Poisson count with mean `cancel_R*dt`. Each event adds recoil
`v_rec*(z_hat+isotropic_direction)`. With probability one half, apply the
`F=2` raising operator and normalize if its norm exceeds `1e-12`:

```text
F_plus |2,m> = sqrt(6-m(m+1)) |2,m+1>;
F_plus vanishes on F=1 and |2,2>.
```

These events do not increment ordinary scattering counters. A
compensation-only event does not trigger the immediate force refresh in
this discretization; its force catches up at the next regular refresh.
The recoil wavelength is still D2. This option is a simplified additional
heating/pumping channel, not a separately eliminated multilevel laser.

### 9.5 Reference trajectory engine

The implementation below preserves the step order and cache behavior above.
It makes a preliminary random surface-index draw before the MCWF initial
state draw so that the shared initialization convention is explicit.

```python
F_PLUS = np.zeros((8,8))
for a,m in enumerate([1,0,-1,-2],start=1):
    F_PLUS[a-1,a] = np.sqrt(6-m*(m+1))


class MCWF:
    def __init__(self,trap,beams,N,T0_K,seed,dt=10e-9,n_cache=8,
                 record_dt=5e-6,loss_off=False,loss_rho=2e-6,loss_z=8e-6,
                 T0_axes_K=None,init_state=None,cancel_R=0.0):
        self.trap,self.beams,self.N = trap,beams,N
        self.atom = trap.atom
        self.dt,self.n_cache,self.record_dt = dt,n_cache,record_dt
        self.loss_off,self.loss_rho,self.loss_z = loss_off,loss_rho,loss_z
        self.cancel_R = cancel_R
        self.rng = np.random.default_rng(seed)
        T = np.array([T0_K]*3 if T0_axes_K is None else T0_axes_K)
        sv = np.sqrt(KB*T/MASS)
        omega = trap.frequencies()
        if np.any(omega==0) and init_state is None:
            raise ValueError('Supply finite X and V for an unconfined initial state')
        sx = np.divide(sv,omega,out=np.zeros(3),where=omega!=0)
        self.X = self.rng.normal(size=(N,3))*sx
        self.V = self.rng.normal(size=(N,3))*sv
        self.surface_index = self.rng.integers(0,5,N)
        self.alive = np.ones(N,bool)
        self.nscat = np.zeros(N,int)
        self.nscat_rep = np.zeros(N,int)
        self.t = 0.0
        if init_state is not None:
            for key in ('X','V','alive','nscat'):
                setattr(self,key,np.array(init_state[key],copy=True))
            self.t = float(init_state.get('t',0.0))
        self.psi = np.zeros((N,8),complex)
        self.psi[np.arange(N),self.rng.integers(0,5,N)] = 1
        self.records = {k:[] for k in ('t','KEax','Ug','alive','nscat','pF1','Rinst')}

    def record(self,R=None,idx=None):
        self.records['t'].append(self.t)
        self.records['KEax'].append(0.5*MASS*self.V**2)
        self.records['Ug'].append(self.trap.scalar(self.X).copy())
        self.records['alive'].append(self.alive.copy())
        self.records['nscat'].append(self.nscat.copy())
        self.records['pF1'].append(np.sum(np.abs(self.psi[:,5:])**2,axis=1))
        rate = float('nan')
        if R is not None and idx is not None and len(idx):
            ps = self.psi[idx]
            rate = float(np.real(np.einsum('ng,ngh,nh->n',ps.conj(),R,ps)).mean())
        self.records['Rinst'].append(rate)

    def run(self,t_end):
        dt = self.dt
        steps = int(round((t_end-self.t)/dt))
        next_record = 0.0
        acceleration = None
        for step in range(steps):
            if step % self.n_cache == 0:
                idx = np.flatnonzero(self.alive)
                if len(idx)==0:
                    break
                cache = operators(self.X[idx],self.t,self.trap,self.beams)
                prop = propagator(cache.H,cache.R,dt)
                Fc = np.zeros((self.N,3))
                Fc[idx] = force_internal(self.X[idx],self.psi[idx],self.t,
                                         self.trap,self.beams)
                acceleration = None
            alive = self.alive
            if acceleration is None:
                acceleration = np.zeros((self.N,3))
                acceleration[alive] = (force_trap(self.X[alive],self.trap)
                                       + Fc[alive])/MASS
            self.X[alive] += self.V[alive]*dt+0.5*acceleration[alive]*dt**2
            anew = np.zeros((self.N,3))
            anew[alive] = (force_trap(self.X[alive],self.trap)+Fc[alive])/MASS
            self.V[alive] += 0.5*(acceleration[alive]+anew[alive])*dt
            acceleration = anew
            ps = np.einsum('ngh,nh->ng',prop,self.psi[idx])
            norm2 = np.real(np.einsum('ng,ng->n',ps.conj(),ps))
            jumping = self.rng.uniform(size=len(idx)) > norm2
            ps[~jumping] /= np.sqrt(np.maximum(norm2[~jumping],1e-300))[:,None]
            jlocs = np.flatnonzero(jumping)
            for j in jlocs:
                n = idx[j]
                ps[j],self.V[n],label = scatter(
                    self.rng,ps[j],self.V[n],j,cache,self.beams,self.atom)
                if label is not None:
                    self.nscat[n] += 1
                    self.nscat_rep[n] += int(label=='rep')
            if self.cancel_R:
                counts = self.rng.poisson(self.cancel_R*dt,size=len(idx))
                for j in np.flatnonzero(counts):
                    for event in range(counts[j]):
                        nem = self.rng.normal(size=3)
                        nem /= np.linalg.norm(nem)
                        self.V[idx[j]] += self.atom.vrec*(np.array([0,0,1])+nem)
                        if self.rng.uniform() < 0.5:
                            raised = F_PLUS@ps[j]
                            norm = np.linalg.norm(raised)
                            if norm > 1e-12:
                                ps[j] = raised/norm
            self.psi[idx] = ps
            if len(jlocs):
                changed = idx[jlocs]
                Fc[changed] = force_internal(self.X[changed],self.psi[changed],
                                            self.t,self.trap,self.beams)
                acceleration = None
            self.t += dt
            if self.t >= next_record-1e-15:
                self.record(cache.R,idx)
                next_record += self.record_dt
                if not self.loss_off:
                    lost = ((self.X[:,0]**2+self.X[:,1]**2 > self.loss_rho**2)
                            | (np.abs(self.X[:,2]) > self.loss_z))
                    self.alive &= ~lost
        self.record()
        return self.records
```

## 10. Dressed-state rate-equation alternative

The rate-equation engine uses the same atomic data, trap, beams, eliminated
operators, force stencil, and recoil rules. Replace the MCWF state by a
single local eigenstate of `H_eff`. This removes coherence between dressed
surfaces and does not explicitly model Landau-Zener transitions.

At each regular cache refresh:

1. Compute `H_eff,R` and the scattering cache, then diagonalize
   `H_eff W = W diag(epsilon)` with ascending eigenvalues.
2. On the first refresh, compute `p2[a]=sum_{g=0..4}|W[g,a]|^2` for each
   atom. Draw eight independent uniforms in `[0.5,1)` and choose the
   surface maximizing `p2[a]*uniform[a]`. This initializes an F=2-like
   surface; it is not projection of one prescribed bare state.
3. On later refreshes retain the integer surface index. There is no
   eigenvector-overlap matching if eigenvalues reorder.
4. Set `chi=W[:,surface_index]`, `rate=real[chi^dagger R chi]`, and compute
   the cached internal force with `chi` fixed during the stencil.

For the rate engine's operator stencil, form `dH=(H(X+h)-H(X-h))/(2*h)`
first and evaluate `-real[chi^dagger dH chi]`. This is mathematically the
same frozen-state derivative as section 9, but fixes the floating-point
operation order for that alternative engine.

Perform the same velocity-Verlet motion step. Draw a Poisson count with
mean `rate*dt` per cached atom. For each count, select a group/block and
construct `exc` exactly as in section 8 using `chi`. The absorption-beam
probability and recoil direction use the same formulas. Emission now
selects a **dressed surface**, with

```text
c[q,g] = sum_e D[q,g,e]*exc_e
b[q,a] = sum_g c[q,g]*conj(W[g,a])
P_surface(a) proportional to sum_q |b[q,a]|^2.
```

Sample `a`, set the surface index, apply absorption plus isotropic emission
recoil, and increment counters. The rate engine does not sample a final
emission-polarization quantum number separately.

After all ordinary jumps in a step, rediagonalize at each jumped atom's
current position and pre-increment time. Select its new eigenvector by the
sampled integer index, refresh its rate and internal force, and invalidate
mechanical acceleration. The original window's per-beam amplitudes,
channel detunings, excited eigenvectors, and full `W` remain cached until
the next regular refresh. For multiple Poisson jumps in the same step,
the original selected vector is used until that post-step refresh; recoil
uses the successively updated velocity. These details require `rate*dt`
to be small for a reliable surface-jump approximation.

The instantaneous F=1 observable is `sum_{g=5..7}|chi_g|^2`. A final record
without an active cache uses zero as its F=1 placeholder and `NaN` for the
instantaneous rate. Unlike MCWF, the rate engine needs no matrix
exponential or no-jump normalization.

## 11. Loss, recording, and observables

Loss is geometric: at each recording event, remove an atom if
`X_x^2+X_y^2 > loss_rho^2` or `abs(X_z)>loss_z`. Defaults are 2 micrometres
radially and 8 micrometres axially. This is not an energy-based escape test.
Setting `loss_off=True` disables it.

The reference records before applying loss. Its compact optical cache is
updated only at a regular cache refresh. Therefore a newly lost atom stops
mechanical motion immediately, but can still receive internal propagation
and scattering updates through the stale compact cache until the next
refresh. With ordinary parameters this interval is at most one cache
window; implement this timing if matching the algorithm exactly. Filtering
lost atoms from the cache immediately would be a different discretization.

| Record key | Shape after stacking | Meaning |
|---|---|---|
| `t` | `(nt,)` | Time in seconds. |
| `KEax` | `(nt,N,3)` | `0.5*MASS*V_i^2`, joules per atom and axis. |
| `Ug` | `(nt,N)` | Scalar trap energy only, excluding internal/light energy. |
| `alive` | `(nt,N)` | Survival mask. |
| `nscat` | `(nt,N)` | Cumulative ordinary scattering counts. |
| `pF1` | `(nt,N)` | MCWF probability `sum_{g=5..7} abs(psi_g)^2`, or dressed-surface weight. |
| `Rinst` | `(nt,)` | Mean `real[psi^dagger R psi]` over cached atom indices, in inverse seconds. |

Ordinary and repump-only counters both exist in state, but only the ordinary
counter is recorded by default. Add a copied `nscat_rep` array to records
if time-resolved repump statistics are needed.

For survivor-conditioned kinetic temperatures, use

```text
T_i(t) = 2 mean_alive KE_i(t)/KB = MASS*mean_alive(V_i^2)/KB
T_radial = (T_x+T_y)/2
T_three_axis = (T_x+T_y+T_z)/3.
```

These are velocity second moments. A drifting cloud has a different
thermal estimator `T_variance,i=MASS*(mean(V_i^2)-mean(V_i)^2)/KB`.
The reference records the former, so directed acceleration contributes to
its temperature. Decreasing survivor temperature can also reflect selective
loss; always retain survival. If no atoms survive, temperature is `NaN`.

Recording starts after the first step, at `dt`, rather than at exactly zero.
An unconditional final record may duplicate the last ordinary timestamp;
its `Rinst` is `NaN` and its survival mask includes the latest loss update.
Floating-point accumulation can put nominal sample times a few ulps away
from their requested values. If removing duplicates, deliberately choose
whether to retain the final post-loss entry.

`run(t_end)` takes an absolute end time, rounded to an integer number of
steps from the current time. Repeated calls retain state and append records,
but reset the recording threshold to zero; several consecutive initial
steps can then be recorded while that threshold catches up. For a clean
sampling schedule, run one continuous interval, or explicitly manage the
threshold and distinguish that output change from a dynamics change.

```python
def temperatures(records):
    ke = np.asarray(records['KEax'])
    alive = np.asarray(records['alive'])
    axes = np.array([2*e[a].mean(axis=0)/KB if a.any() else np.full(3,np.nan)
                     for e,a in zip(ke,alive)])
    return SimpleNamespace(t=np.asarray(records['t']),axes_K=axes,
        radial_K=axes[:,:2].mean(axis=1),mean_K=axes.mean(axis=1),
        survival=alive.mean(axis=1))
```

## 12. Fully specified example and numerical checks

The following example is a four-beam transverse PGC configuration with
phase rolling in a standing-wave trap:

| Parameter | Value |
|---|---|
| Species / cooling line | 87Rb D2, using all constants in section 2. |
| Trap wavelength / scalar depth / waist | `785e-9 m`, `1e-3 K`, `1e-6 m`. |
| Trap polarization / magnetic fields | `ellipticity=0`, all real fields and quadratic shift zero. |
| Trap type | Balanced standing wave along z. |
| Cooling directions | `+x,-x,+y,-y`, circular vectors in section 6. |
| Cooling detuning / saturation | `-3*Gamma`, `s_cool=2` per beam. |
| Repump | Copies all four beams; `delta_r=0`, `s_rep=0.1` per beam. |
| Pair phase roll | `roll_hz=30e3`, on the second cooling pair only. |
| Initial temperature | `50e-6 K` on all three axes. |
| Step / cache / finite difference | `10e-9 s`, 8 steps, `5e-9 m`. |
| Short execution | 4 atoms, `10e-6 s`, records every `1e-6 s`, loss disabled. |
| Seeds | Beam phases 7700; trajectory RNG 88000. |

Nominal frequencies are approximately `omega_r/(2*pi)=98.5 kHz` and
`omega_z/(2*pi)=557.2 kHz`. A short run checks execution and finite states;
it is too short and too small to estimate an equilibrium cooling floor.

For a cooling curve, change to 16 or more atoms per texture, `t_end=2.5e-3`,
`record_dt=20e-6`, and geometric loss enabled. Use several independent beam
phase textures, such as six with beam seeds `7700+texture` and trajectory
seeds `88000+37*texture`. Each texture is a separate simulation. More
trajectories at one phase realization do not replace multiple realizations.
Record raw per-texture curves before averaging, including their survival.

Install the two dependencies with `python -m pip install numpy sympy`.
The driver below completes the reference program. It prints final
short-run observables and performs algebraic and state-normalization checks;
it does not launch a long cooling sweep or write output files.

```python
def check_algebra(atom,trap,beams):
    assert np.allclose(U@U.T,np.eye(8),atol=1e-13)
    assert np.allclose(np.sum(np.abs(D)**2,axis=(0,2)),2.0,atol=1e-13)
    assert np.isclose(D[0,0,0],1.0)
    levels = np.linalg.eigvalsh(H_HFS_E)/HBAR/(2*np.pi*1e6)
    target = np.array([-302.0738]+[-229.8518]*3+[-72.9113]*5+[193.7407]*7)
    assert np.allclose(levels,target,atol=0.05)
    assert np.isclose(-trap.scalar(np.zeros((1,3)))[0]/trap.U0,1.0)
    points = np.array([[0,0,0],[1e-7,-2e-7,5e-8]])
    op = operators(points,0.0,trap,beams)
    hscale = max(np.max(np.abs(op.H)),1e-300)
    rscale = max(np.max(np.abs(op.R)),1e-300)
    assert np.max(np.abs(op.H-op.H.conj().transpose(0,2,1)))/hscale < 1e-11
    assert np.max(np.abs(op.R-op.R.conj().transpose(0,2,1)))/rscale < 1e-11
    assert np.linalg.eigvalsh(op.R).min() > -1e-11*rscale


def short_example():
    atom = make_atom(lambda_trap=785e-9)
    trap = Trap(atom,depth_K=1e-3,waist=1e-6,standing=True)
    beams = make_beams(atom,s_cool=2.0,delta_c=-3*GAMMA,
        geometry='2d',repump='all',s_rep=0.1,roll_hz=30e3,seed=7700)
    check_algebra(atom,trap,beams)
    sim = MCWF(trap,beams,N=4,T0_K=50e-6,seed=88000,
               record_dt=1e-6,loss_off=True)
    result = sim.run(10e-6)
    assert np.isfinite(sim.X).all() and np.isfinite(sim.V).all()
    assert np.allclose(np.sum(np.abs(sim.psi)**2,axis=1),1.0,atol=1e-10)
    summary = temperatures(result)
    print('Final kinetic temperatures [microkelvin]:',summary.axes_K[-1]*1e6)
    print('Survival:',summary.survival[-1])
    print('Scattering counts:',sim.nscat)
    return sim


if __name__ == '__main__':
    short_example()
```

Further numerical acceptance checks can be performed without experimental
data:

- For linear polarization along z and excited `j=3/2`, the Stark
  eigenvalues are `-U0*(a_s+a_t)` twice and `-U0*(a_s-a_t)` twice.
  For pure circular polarization with only the vector term, they are
  `-U0*a_v*m/3`, `m=3/2,1/2,-1/2,-3/2`.
- With both laser intensities zero, no compensation scattering, zero fields,
  and a linearly polarized balanced standing trap, `R=0` and `Hg=0`.
  Scalar mechanical energy `0.5*MASS*sum(V^2)+Ug` should be approximately
  conserved. For depth `800e-6 K`, waist `1e-6 m`, a 100-microsecond run
  at `dt=10 ns` should keep relative drift small compared with `2e-3`
  for well-bound trajectories. Avoid dividing by near-zero initial energy.
- At 785 nm, ground scalar polarizability should be approximately
  `2.1113087575221877e-37` in SI, or `12805.30` atomic units.
- Check normalized state norms, finite trajectories, nonnegative
  scattering rates, and a jump probability well below one at every step.
  Large violations should prompt a smaller step, not silent probability
  clipping.

## 13. Alternative operating modes and repeated recooling

### 13.1 Free-space and grey-molasses settings

For free space, use zero trap depth, explicit initial positions/velocities,
and loss disabled. All trap-induced shifts and forces then vanish. Keep
the chosen laser geometry and the full D2 hyperfine manifold.

To put the cooler blue of `F=2 -> F'=2` by `Delta22`, set

```text
delta_c = Delta22 - (E_E_F3-E_E_F2)
delta_r = Delta22.
```

The two excited reference offsets differ by `2*pi*266.6520e6 rad/s` with
the specified constants. Positive `Delta22` is blue of the open F'=2 line
while it can still be red of F'=3. Incoherent cooler/repumper groups remain
as defined above; coherent two-photon Lambda coupling is not added.

### 13.2 A periodic heating-and-cooling cycle

A driven steady state can be modeled by alternating a heating stage with
a short MCWF cooling burst. This is separate from equilibrium under
continuous molasses. A concrete cycle is:

| Parameter | Value |
|---|---|
| Trap | Traveling Gaussian, `lambda=850e-9 m`, `depth=2.32e-3 K`, `waist=2.05e-6 m`. |
| Initial ensemble | 24 atoms with harmonic thermal positions/velocities at `30e-6 K`. |
| Heating attempts per cycle | 40. |
| Harmonic evolution per attempt | `5e-6 s`. |
| Pump photons per attempt | Choose `N_pump`, for example 6; one additional excitation photon. |
| Trap-off displacement interval | `150e-9 s`. |
| Cooling burst | Six-beam circular geometry; `delta_c=-3*Gamma`; `350e-6 s`. |
| Saturations | Choose `s_cool`, for example 2; `s_rep=max(0.05,0.05*s_cool)`. |
| Number of cycles | 12, or enough to assess approach to a stationary cycle. |

For each attempt and axis, evolve the harmonic motion analytically for
`tau_try` using

```text
c=cos(omega_i*tau_try); s=sin(omega_i*tau_try)
X_i_new = c*X_i + (s/omega_i)*V_i
V_i_new = -omega_i*s*X_i + c*V_i.
```

Use copies of the old position and velocity on both right-hand sides.
Then draw `2*(N_pump+1)` independent isotropic unit-vector kicks per atom,
adding `v_rec*n` to velocity for each. Both absorption and emission are
isotropically approximated in this heating stage. Finally apply
`X += V*tau_off`; velocities are unchanged during this displacement.
The specified harmonic evolution and free-flight interval are sequential.

After 40 attempts, construct a fresh MCWF burst carrying `X,V,alive,nscat`
through its initial-state dictionary. Reset internal states to random
bare F=2 states, reset the burst clock to zero, and optionally draw new beam
phases for each cycle. One fully specified random-seed schedule for texture
index `j` and cycle index `c`, both starting at zero, is heating RNG
`9100+j`, beam seed `700+17*j+c`, and MCWF seed `5000+91*j+7*c`.
The heating RNG is persistent across attempts and cycles.

For the supplied cycle prescription, heating updates all stored atoms even
if some are already marked lost. The hot second moment can therefore be
recorded over all stored atoms, while the cold temperature after a burst
is survivor-conditioned. Label these populations separately; they cease
to be directly comparable when loss is appreciable. For a physically
survivor-only heating protocol, explicitly restrict both the motion and
hot estimator to alive atoms, recognizing the change in prescription.

This cycle has two distinct temperature observables: immediately before
and immediately after recooling. Their arithmetic midpoint is an endpoint
summary, not the exact time average over the heating and cooling waveform.
To obtain a time average, record and integrate that waveform.

## 14. Approximation boundaries and reproducibility

The following choices are part of the specified model:

- Classical motion with an eight-state ground wavefunction; no quantized
  motional levels or explicit excited-state residence-time motion.
- Adiabatic elimination with one local saturation broadening shared across
  channels; unsaturated resolvents for jump reconstruction and recoil.
- Explicit Doppler weighting in absorption direction only, not in the
  total scattering operator or the light-shift denominators.
- Incoherent laser-frequency groups, without cooler-repumper Raman terms.
  MCWF collapse may nevertheless populate both hyperfine blocks, and full
  ground vector/Zeeman operators retain off-block entries after the large
  ground splitting has been removed.
- Isotropic, polarization-independent emission recoil and the scalar
  eigenchannel sum used for absorption-beam selection.
- Cached internal force, propagator, and jump data, with the precise
  post-jump refresh rules in section 9.
- A harmonic initial distribution and geometric, periodically sampled loss.
- Approximate excited-state pole strengths and no additional core terms.

The scalar beam amplitude `B_b=sum_k L_k*a_kb` deserves special care:
unlike reconstructed `exc`, it does not multiply each channel by its
excited eigenvector. Consequently that recoil-selection rule can depend on
eigenvector phase conventions or degenerate-subspace choices. Keeping a
consistent Hermitian eigensolver and recording numerical-library versions
helps reproduce this implementation. A reformulation that makes this
selection invariant would be a model change, not merely a code optimization.

Random seeds alone do not guarantee identical trajectories across different
implementations. Beam ordering, basis phases, eigensolver choices, RNG
algorithm, draw ordering, and floating-point roundoff all matter. The
reference uses `numpy.random.default_rng` and ordered group/block loops.
Compare operators and ensemble statistics as well as single seeded runs.

For quantitative work, converge time step, cache duration, and force
stencil; then increase trajectories and independent phase textures.
A smaller cache duration is particularly relevant when atoms move rapidly
through polarization gradients or when phase rolling is fast. Check
survival, drift, hyperfine populations, and scattering counts alongside
kinetic temperatures. Finite-difference calculations near strongly varying
resonances deserve an independent stencil-convergence check.

Finally compare predicted kinetic scales with `HBAR*omega_i/KB` and the
recoil scale. For the 1 mK standing-wave example these harmonic spacings are
approximately 4.73 microkelvin radially and 26.74 microkelvin axially.
A classical result well below such a spacing is not a prediction of a
resolved quantum motional-state population. Likewise, agreement of a
cooling-off scalar energy test does not establish energy consistency for
finite-saturation, light-on stochastic dynamics.



