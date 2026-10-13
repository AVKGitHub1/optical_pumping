"""Hyperfine + Zeeman eigenstates within an isolated fine-structure manifold.

The coupled basis is |(J I) F m>, matching ARC's dipole phase convention.
H/h = diag(E_F/h) + (mu_B B/h)[g_I F_z + (g_J-g_I) J_z].
Only states with equal m mix. Fine-structure mixing and diamagnetism are absent.
See Steck, Rb-85 D Line Data, and ARC's breitRabi documentation.
"""
from functools import lru_cache

import numpy as np

from .atomic import H, MU_B


def electronic_g(nuclear_spin, j, f, g_f, g_i):
    """Recover g_J from a cached weak-field g_F and the common nuclear g_I."""
    projection = (f*(f+1) + j*(j+1) - nuclear_spin*(nuclear_spin+1))/(2*f*(f+1))
    return g_i + (g_f-g_i)/projection


@lru_cache(maxsize=128)
def eigensystem(nuclear_spin, j, states, hyperfine_hz, g_j, g_i, b_tesla):
    """Energies relative to the centroid and columns in the B=0 coupled basis.

    Labels follow each fixed-m branch adiabatically from B=0. Eigenvalue order
    within a fixed-m block is preserved through avoided crossings; sorting the
    entire manifold or assigning labels by the largest overlap would fail.
    Results are cached and read-only.
    """
    if not np.isfinite(b_tesla):
        raise ValueError('magnetic field must be finite')
    energies = np.array(hyperfine_hz, float)
    vectors = np.eye(len(states))
    if b_tesla != 0:
        f = np.array([s[0] for s in states])
        m = np.array([s[1] for s in states])
        # Diagonal and adjacent-F elements of J_z in the |(J I) F m> basis.
        jz = np.diag(m*(f*(f+1)+j*(j+1)-nuclear_spin*(nuclear_spin+1))/(2*f*(f+1)))
        for row, (ff, mm) in enumerate(states):
            if (ff-1, mm) in states:
                col = states.index((ff-1, mm))
                value = np.sqrt(((nuclear_spin+j+1)**2-ff**2)
                                *(ff**2-(nuclear_spin-j)**2)*(ff**2-mm**2)
                                /(4*ff**2*(4*ff**2-1)))
                jz[row, col] = jz[col, row] = value
        hamiltonian = np.diag(hyperfine_hz) + MU_B*b_tesla/H*(g_i*np.diag(m)+(g_j-g_i)*jz)
        for mm in np.unique(m):
            indices = np.flatnonzero(m == mm)
            indices = indices[np.argsort(np.asarray(hyperfine_hz)[indices])]
            values, block = np.linalg.eigh(hamiltonian[np.ix_(indices, indices)])
            # Fix the phase by the original F component. Finite-field rates
            # are phase invariant, but Raman amplitudes need consistent gauges.
            block *= np.where(np.diag(block) < 0, -1., 1.)
            energies[indices] = values
            vectors[np.ix_(indices, indices)] = block
    energies.setflags(write=False)
    vectors.setflags(write=False)
    return energies, vectors


def ground_eigensystem(atom, b_tesla):
    from .atomic import GROUND_STATES
    g_i = (atom.g_ground[3]+atom.g_ground[2])/2
    g_j = g_i+(atom.nuclear_spin+.5)*(atom.g_ground[3]-atom.g_ground[2])
    return eigensystem(atom.nuclear_spin, .5, GROUND_STATES,
                       tuple(atom.ground_hfs_hz[f] for f, _ in GROUND_STATES), g_j, g_i, b_tesla)


def d2_eigensystem(atom, b_tesla):
    from .atomic import EXCITED_STATES
    g_i = (atom.g_ground[3]+atom.g_ground[2])/2
    return eigensystem(atom.nuclear_spin, 1.5, EXCITED_STATES,
                       tuple(atom.excited_hfs_hz[f] for f, _ in EXCITED_STATES),
                       atom.gJ_excited, g_i, b_tesla)


def d2_dipoles(atom, b_tesla):
    """Mixed-state dipoles in atomic units, with q still fixed by delta m."""
    _, ground = ground_eigensystem(atom, b_tesla)
    _, excited = d2_eigensystem(atom, b_tesla)
    return excited.T @ atom.dipole_au @ ground
