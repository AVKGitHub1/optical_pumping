"""Time evolution: coherent reference solver and checked population-rate solver.

Coherent solver (reference)
---------------------------
Lindblad master equation for the {up, down} x motion density-matrix block plus
motional populations of the 10 spectator sublevels.  Dissipators:

* optical/Raman scattering: population-resolving jump operators sqrt(r) |s',n'><s,n| with
  r = W(s->s', emitted class) * K(n'|n); the anticommutator term damps every
  up-down coherence at (Gamma_out(up,n) + Gamma_out(down,m))/2, including events
  that return to the same spin state.  No separate phenomenological term is added
  for this scattering (``extra_coherence_decay_rate_s`` is only for *additional*
  mechanisms). Resolving each n separately drops coherence transfer even for
  degenerate transitions; this is an extra approximation beyond secular averaging;
* background heating: L = sqrt(G_h) a, sqrt(G_h) a^+ acting on motion for every spin;
* extra up-down dephasing gamma_x (calibrated input).

Population pushed above n_max (recoil kernel tail, heating from n_max) goes into
an explicit numerical-overflow bin; it is *not* reflected into the basis.  For a
harmonic trap it is *not* physical atom loss; for trap.potential = lattice with
all bound levels kept it is the probability of promotion above the lattice depth.

Rate solver
-----------
Adiabatic elimination of each up-down coherence (n, m): with Rabi Omega_nm,
pair detuning delta_nm and coherence decay gamma_nm,

   W_nm = (Omega_nm^2 / 2) gamma_nm / (gamma_nm^2 + delta_nm^2).

Valid only if Omega_nm << sqrt(gamma^2 + delta^2) for every occupied pair; the
solver checks this at every segment start and raises ``RateInvalid`` otherwise,
so it never diverges as gamma -> 0.  Off-resonant AC-Stark shifts of the
sideband by the carrier are not included in the rate model.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .atomic import DOWN, N_GROUND, UP
from .model import COUNTERS, SPECTATOR_IDX, Model, SegmentOps, Validity
from .motion import annihilation


class SimulationStopped(Exception):
    pass


class RateInvalid(Exception):
    pass


@dataclass
class RunResult:
    t: np.ndarray  # (T,)
    P: np.ndarray  # (T, 12, N) joint populations P(s, n, t)
    overflow: np.ndarray  # (T,) numerical overflow above n_max
    counts: np.ndarray  # (T, 3) cumulative scattered photons per atom
    min_eig: np.ndarray  # (T,) most negative eigenvalue of the Raman block (coherent) or min population
    coherence: np.ndarray  # (T,) sum |rho_up,down| (0 for rate solver)
    solver: str
    segments: list
    wall_time_s: float
    n_rhs: int = 0
    notes: list = field(default_factory=list)
    rate_validity: list = field(default_factory=list)
    rho_final: np.ndarray | None = None


def _sample_times(model: Model) -> np.ndarray:
    T = model.schedule[-1].t1 if model.schedule else 0.0
    return np.linspace(0.0, T, model.cfg.timing.n_samples)


class _Progress:
    def __init__(self, cb, T):
        self.cb, self.T, self.last = cb, max(T, 1e-30), 0.0

    def __call__(self, t, msg=""):
        if self.cb is None:
            return
        now = time.perf_counter()
        if now - self.last > 0.2:
            self.last = now
            self.cb(min(t / self.T, 1.0), msg)


# ---------------------------------------------------------------------------
# Coherent solver
# ---------------------------------------------------------------------------
class CoherentSolver:
    def __init__(self, model: Model, stop: threading.Event | None = None, progress=None):
        self.m = model
        self.stop = stop or threading.Event()
        self.progress = progress
        N = model.N
        self.N = N
        self.N2 = N * N
        self.a = annihilation(N)
        self.ad = self.a.T.copy()
        self.two_n1 = 2.0 * np.arange(N) + 1.0
        self.nrhs = 0
        self.n_spec = len(SPECTATOR_IDX)
        self.size = 3 * self.N2 + self.n_spec * N + 2 + len(COUNTERS)
        self.up_sl = slice(UP * N, (UP + 1) * N)
        self.dn_sl = slice(DOWN * N, (DOWN + 1) * N)
        self.spec_idx = (SPECTATOR_IDX[:, None] * N + np.arange(N)[None, :]).ravel()

    # -- packing ---------------------------------------------------------
    def pack(self, P0: np.ndarray) -> np.ndarray:
        N = self.N
        y = np.zeros(self.size, dtype=complex)
        y[: self.N2] = np.diag(P0[UP]).ravel()
        y[2 * self.N2 : 3 * self.N2] = np.diag(P0[DOWN]).ravel()
        y[3 * self.N2 : 3 * self.N2 + self.n_spec * N] = P0[SPECTATOR_IDX].ravel()
        return y

    def unpack(self, y):
        N, N2 = self.N, self.N2
        A = y[:N2].reshape(N, N)
        Bm = y[N2 : 2 * N2].reshape(N, N)
        Dm = y[2 * N2 : 3 * N2].reshape(N, N)
        Ps = y[3 * N2 : 3 * N2 + self.n_spec * N].reshape(self.n_spec, N)
        tail = y[3 * N2 + self.n_spec * N :]
        return A, Bm, Dm, Ps, tail

    def populations(self, y) -> np.ndarray:
        A, _, Dm, Ps, _ = self.unpack(y)
        P = np.zeros((N_GROUND, self.N))
        P[UP] = np.diag(A).real
        P[DOWN] = np.diag(Dm).real
        P[SPECTATOR_IDX] = Ps.real
        return P

    def rho_block(self, y) -> np.ndarray:
        A, Bm, Dm, _, _ = self.unpack(y)
        return np.block([[A, Bm], [Bm.conj().T, Dm]])

    # -- right-hand side -----------------------------------------------------
    def make_rhs(self, o: SegmentOps, prog: _Progress):
        N, N2, ns = self.N, self.N2, self.n_spec
        G, gout, ovr, cnt = o.G, o.gout, o.ovr, o.counts
        gu, gd = gout[self.up_sl], gout[self.dn_sl]
        gs = gout[self.spec_idx].reshape(ns, N)
        dampA = 0.5 * (gu[:, None] + gu[None, :])
        dampB = 0.5 * (gu[:, None] + gd[None, :]) + o.gamma_extra
        dampD = 0.5 * (gd[:, None] + gd[None, :])
        hu = -o.delta
        vk = o.vk
        freqs = np.array([w for w, _ in vk])
        Vstack = np.array([V for _, V in vk]) if vk else None
        # few distinct frequencies (harmonic: k*omega_t): sum over groups; many (lattice): one phase per element
        elementwise = len(vk) > 12
        if elementwise:
            Vamp = Vstack.sum(axis=0)
            Fmat = np.zeros((N, N))
            for w, V in vk:
                Fmat[V != 0] = w
        gh = o.heating
        a, ad, t2 = self.a, self.ad, self.two_n1
        nv = np.arange(N, dtype=float)
        stop = self.stop
        di = np.diag_indices(N)
        spec_idx = self.spec_idx
        p = np.zeros(N_GROUND * N)
        up_sl, dn_sl = self.up_sl, self.dn_sl

        def rhs(t, y):
            self.nrhs += 1
            if stop.is_set():
                raise SimulationStopped()
            if (self.nrhs & 255) == 0:
                prog(t)
            A = y[:N2].reshape(N, N)
            Bm = y[N2 : 2 * N2].reshape(N, N)
            Dm = y[2 * N2 : 3 * N2].reshape(N, N)
            Ps = y[3 * N2 : 3 * N2 + ns * N].reshape(ns, N).real
            p[up_sl] = A[di].real
            p[dn_sl] = Dm[di].real
            p[spec_idx] = Ps.ravel()
            gain = G @ p
            if Vstack is not None:
                V = Vamp * np.exp(1j * Fmat * t) if elementwise else np.tensordot(np.exp(1j * freqs * t), Vstack, axes=1)
                Vh = V.conj().T
                C = Bm.conj().T
                dA = -1j * (V @ C - Bm @ Vh)
                dB = -1j * (hu * Bm + V @ Dm - A @ V)
                dD = -1j * (Vh @ Bm - C @ V)
            else:
                dA = np.zeros((N, N), complex)
                dB = -1j * hu * Bm
                dD = np.zeros((N, N), complex)
            dA -= dampA * A
            dB -= dampB * Bm
            dD -= dampD * Dm
            dA[di] += gain[up_sl]
            dD[di] += gain[dn_sl]
            dPs = gain[spec_idx].reshape(ns, N) - gs * Ps
            if gh > 0:
                for X, dX in ((A, dA), (Bm, dB), (Dm, dD)):
                    dX += gh * (a @ X @ ad + ad @ X @ a - 0.5 * (t2[:, None] * X + X * t2[None, :]))
                up = np.zeros_like(Ps)
                up[:, 1:] = nv[1:] * Ps[:, :-1]
                down = np.zeros_like(Ps)
                down[:, :-1] = (nv[:-1] + 1) * Ps[:, 1:]
                dPs += gh * (up + down - t2 * Ps)
            dy = np.empty_like(y)
            dy[:N2] = dA.ravel()
            dy[N2 : 2 * N2] = dB.ravel()
            dy[2 * N2 : 3 * N2] = dD.ravel()
            dy[3 * N2 : 3 * N2 + ns * N] = dPs.ravel()
            k = 3 * N2 + ns * N
            dy[k] = ovr @ p
            top = p[N - 1 :: N].sum()
            dy[k + 1] = gh * N * top
            dy[k + 2 :] = cnt @ p
            return dy

        return rhs

    def max_frequency(self, o: SegmentOps) -> float:
        f = abs(o.delta)
        if o.vk:
            f = max(f, max(abs(w) for w, _ in o.vk) + abs(o.delta))
        return f

    def run(self) -> RunResult:
        m = self.m
        num = m.cfg.numerics
        ts = _sample_times(m)
        T = ts[-1]
        prog = _Progress(self.progress, T)
        y = self.pack(m.P0)
        out_y = {0: y.copy()}
        t_start = time.perf_counter()
        ops_cache = {}
        for seg in m.schedule:
            if seg.key not in ops_cache:
                ops_cache[seg.key] = m.ops(seg)
            o = ops_cache[seg.key]
            idx = np.nonzero((ts > seg.t0) & (ts <= seg.t1))[0]
            teval = np.unique(np.concatenate([ts[idx], [seg.t1]]))
            # Step cap: half the fastest coherent period, and 1/(fastest decay rate).
            # The scipy error norm is an RMS over all components (mostly tiny
            # coherences), so without the decay cap a single stiff population can
            # be integrated near the explicit stability limit unnoticed.
            fmax = self.max_frequency(o)
            rmax = float(o.gout.max()) + o.gamma_extra + 2 * o.heating * self.N
            max_step = min(np.inf if fmax == 0 else np.pi / fmax, np.inf if rmax == 0 else 1.0 / rmax)
            sol = solve_ivp(
                self.make_rhs(o, prog), (seg.t0, seg.t1), y, method=num.method, t_eval=teval,
                rtol=num.rtol, atol=num.atol, max_step=max_step,
            )
            if not sol.success:
                raise RuntimeError(f"integration failed in segment '{seg.label}': {sol.message}")
            for i in idx:
                j = np.searchsorted(sol.t, ts[i])
                out_y[i] = sol.y[:, j]
            y = sol.y[:, -1]
            prog(seg.t1, seg.label)
        wall = time.perf_counter() - t_start
        return self._collect(ts, out_y, wall, y)

    def _collect(self, ts, out_y, wall, y_final) -> RunResult:
        nT = len(ts)
        P = np.zeros((nT, N_GROUND, self.N))
        ov = np.zeros(nT)
        cn = np.zeros((nT, len(COUNTERS)))
        me = np.zeros(nT)
        coh = np.zeros(nT)
        for i in range(nT):
            y = out_y[i]
            P[i] = self.populations(y)
            tail = y[3 * self.N2 + self.n_spec * self.N :].real
            ov[i] = tail[0] + tail[1]
            cn[i] = tail[2:]
            rho = self.rho_block(y)
            rho = 0.5 * (rho + rho.conj().T)
            me[i] = min(np.linalg.eigvalsh(rho).min(), P[i][SPECTATOR_IDX].min())
            coh[i] = np.abs(self.unpack(y)[1]).sum()
        return RunResult(ts, P, ov, cn, me, coh, "coherent", self.m.schedule, wall, self.nrhs,
                         rho_final=self.rho_block(y_final))


# ---------------------------------------------------------------------------
# Rate solver
# ---------------------------------------------------------------------------
class RateSolver:
    def __init__(self, model: Model, stop: threading.Event | None = None, progress=None):
        self.m = model
        self.stop = stop or threading.Event()
        self.progress = progress
        self.N = model.N
        self.dim = N_GROUND * self.N
        self.validity_log = []

    def generator(self, o: SegmentOps, p_start: np.ndarray | None = None) -> np.ndarray:
        """Augmented generator for [p (12N), overflow, counts(3)]."""
        m, N, dim = self.m, self.N, self.dim
        eps = m.cfg.numerics.rate_validity_epsilon
        L = np.zeros((dim + 1 + len(COUNTERS), dim + 1 + len(COUNTERS)))
        M = o.G - np.diag(o.gout)
        gh = o.heating
        n = np.arange(N)
        if gh > 0:
            for s in range(N_GROUND):
                b = s * N
                for k in range(N):
                    M[b + k, b + k] -= gh * (2 * k + 1)
                    if k + 1 < N:
                        M[b + k + 1, b + k] += gh * (k + 1)
                        M[b + k, b + k + 1] += gh * (k + 1)
        worst = 0.0
        if o.vk:
            gu = o.gout[UP * N : (UP + 1) * N]
            gd = o.gout[DOWN * N : (DOWN + 1) * N]
            V = sum(Vk for _, Vk in o.vk)
            occ = np.ones(N, bool)
            if p_start is not None:
                pn = p_start.reshape(N_GROUND, N).sum(axis=0)
                occ = pn > 1e-5
                occ = occ | np.roll(occ, 1) | np.roll(occ, -1)
            for i in range(N):
                for j in range(N):
                    om = 2 * abs(V[i, j])
                    if om == 0:
                        continue
                    dl = m.F_raman[i, j] - o.delta
                    g = 0.5 * (gu[i] + gd[j]) + o.gamma_extra + 0.5 * gh * (2 * i + 1 + 2 * j + 1)
                    den = np.hypot(g, dl)
                    if occ[i] or occ[j]:
                        r = om / den if den > 0 else np.inf
                        worst = max(worst, r)
                    if den == 0:
                        continue
                    w = 0.5 * om**2 * g / (g**2 + dl**2)
                    a, b = UP * N + i, DOWN * N + j
                    M[a, a] -= w
                    M[b, a] += w
                    M[b, b] -= w
                    M[a, b] += w
        self.validity_log.append((o.seg.label, worst, worst <= eps))
        if worst > eps:
            raise RateInvalid(
                f"segment '{o.seg.label}': max Omega_nm/sqrt(gamma^2+delta^2) = {worst:.3g} > {eps} "
                "(coherent Raman dynamics; rate elimination invalid)"
            )
        L[:dim, :dim] = M
        L[dim, :dim] = o.ovr
        L[dim, :dim] += gh * N * np.isin(np.arange(dim) % N, [N - 1])
        L[dim + 1 :, :dim] = o.counts
        return L

    def run(self) -> RunResult:
        m = self.m
        ts = _sample_times(m)
        T = ts[-1]
        prog = _Progress(self.progress, T)
        t_start = time.perf_counter()
        x = np.concatenate([m.P0.ravel(), np.zeros(1 + len(COUNTERS))])
        samples = {0: x.copy()}
        cache = {}
        for seg in m.schedule:
            if self.stop.is_set():
                raise SimulationStopped()
            o = m.ops(seg)
            L = self.generator(o, x[: self.dim])
            idx = np.nonzero((ts > seg.t0) & (ts <= seg.t1))[0]
            tcur = seg.t0
            for t_target in list(ts[idx]) + [seg.t1]:
                dt = t_target - tcur
                if dt > 0:
                    key = (seg.key, round(dt, 15))
                    if key not in cache:
                        cache[key] = expm(L * dt)
                    x = cache[key] @ x
                    tcur = t_target
                if self.stop.is_set():
                    raise SimulationStopped()
                for i in idx:
                    if ts[i] == t_target:
                        samples[i] = x.copy()
            prog(seg.t1, seg.label)
        nT = len(ts)
        dim = self.dim
        P = np.array([samples[i][:dim].reshape(N_GROUND, self.N) for i in range(nT)])
        ov = np.array([samples[i][dim] for i in range(nT)])
        cn = np.array([samples[i][dim + 1 :] for i in range(nT)])
        me = P.reshape(nT, -1).min(axis=1)
        res = RunResult(ts, P, ov, cn, me, np.zeros(nT), "rate", m.schedule, time.perf_counter() - t_start)
        res.rate_validity = list(self.validity_log)
        return res


def _single_thread():
    """Small (N x N) BLAS calls are much faster single-threaded."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(1)
    except Exception:  # pragma: no cover
        import contextlib

        return contextlib.nullcontext()


def solve(model: Model, stop=None, progress=None) -> RunResult:
    with _single_thread():
        return _solve(model, stop, progress)


def _solve(model: Model, stop=None, progress=None) -> RunResult:
    solver = model.cfg.numerics.solver
    if solver == "coherent":
        return CoherentSolver(model, stop, progress).run()
    if solver == "rate":
        return RateSolver(model, stop, progress).run()
    try:
        r = RateSolver(model, stop, progress)
        res = r.run()
        res.notes.append("auto: rate approximation valid in every segment")
        return res
    except RateInvalid as exc:
        res = CoherentSolver(model, stop, progress).run()
        res.notes.append(f"auto: switched to coherent solver ({exc})")
        res.rate_validity = list(r.validity_log)
        return res


def estimate_cost(model: Model, calibrate: bool = True) -> dict:
    """Rough runtime estimate from a micro-benchmark of the RHS."""
    cs = CoherentSolver(model)
    N = model.N
    dim_state = cs.size
    evals = 0.0
    for seg in model.schedule:
        o = model.ops(seg)
        f = cs.max_frequency(o) / (2 * np.pi)
        rmax = float(o.gout.max()) + o.gamma_extra
        # ~4.5 DOP853 steps per fastest period (measured), or the decay step cap
        steps = max(20.0, (seg.t1 - seg.t0) * max(f * 4.5, rmax, 1.0))
        evals += 12 * steps
    per_eval = np.nan
    if calibrate:
        # benchmark the most expensive segment type (one with Raman on, if any)
        seg = max(model.schedule, key=lambda s: s.raman)
        rhs = cs.make_rhs(model.ops(seg), _Progress(None, 1.0))
        y = cs.pack(model.P0)
        with _single_thread():
            rhs(0.0, y)
            t0 = time.perf_counter()
            for i in range(30):
                rhs(1e-6 * i, y)
            per_eval = (time.perf_counter() - t0) / 30
    return {
        "state_dimension_complex": dim_state,
        "raman_block_dim": 2 * N,
        "population_dim": N_GROUND * N,
        "memory_state_MB": dim_state * 16 / 1e6,
        "memory_transfer_matrices_MB": 3 * (N_GROUND * N) ** 2 * 8 / 1e6,
        "segments": len(model.schedule),
        "est_rhs_evals": evals,
        "rhs_eval_s": per_eval,
        "est_coherent_runtime_s": evals * per_eval if calibrate else np.nan,
    }
