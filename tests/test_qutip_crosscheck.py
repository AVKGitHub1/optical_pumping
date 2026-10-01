"""Cross-check the custom coherent solver against QuTiP mesolve on a small basis.

The same model operators (Raman couplings with their time-dependent phases,
secular optical jump operators, overflow) are handed to QuTiP as a full
(12N+1)-dimensional Lindblad problem with an explicit overflow sink level.
"""
import numpy as np
import pytest

qutip = pytest.importorskip("qutip")

from rb85rsc.atomic import DOWN, UP
from rb85rsc.dynamics import solve
from rb85rsc.model import Model

from helpers import cfg


def test_coherent_solver_matches_qutip():
    c = cfg(trap__n_max=4, initial__nbar=0.8, timing__total_duration_ms=0.4, timing__n_samples=5,
            numerics__rtol=1e-9, numerics__atol=1e-12)
    m = Model(c)
    r = solve(m)
    N = m.N
    dim = 12 * N + 1
    sink = dim - 1
    o = m.ops(m.schedule[0])

    def ket(i):
        return qutip.basis(dim, i)

    def op(i, j):
        return ket(i) * ket(j).dag()

    H0 = -o.delta * sum(op(UP * N + n, UP * N + n) for n in range(N))
    H = [H0]
    for w, V in o.vk:
        A = sum(V[i, j] * op(UP * N + i, DOWN * N + j) for i, j in zip(*np.nonzero(V)))
        H.append([A, lambda t, w=w: np.exp(1j * w * t)])
        H.append([A.dag(), lambda t, w=w: np.exp(-1j * w * t)])
    cops = []
    G = o.G
    for i, j in zip(*np.nonzero(G > 1e-12 * G.max())):
        cops.append(np.sqrt(G[i, j]) * op(i, j))
    for j in np.nonzero(o.ovr > 0)[0]:
        cops.append(np.sqrt(o.ovr[j]) * op(sink, j))
    p0 = np.concatenate([m.P0.ravel(), [0.0]])
    rho0 = qutip.Qobj(np.diag(p0))
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**7, "max_step": 2e-7}
    out = qutip.mesolve(H, rho0, r.t, c_ops=cops, options=opts)
    pq = np.array([np.real(np.diag(s.full()))[:-1].reshape(12, N) for s in out.states])
    assert np.max(np.abs(pq - r.P)) < 1e-6
    rq = out.states[-1].full()
    coh_q = rq[UP * N : (UP + 1) * N, DOWN * N : (DOWN + 1) * N]
    np.testing.assert_allclose(coh_q, r.rho_final[:N, N:], atol=1e-6)
