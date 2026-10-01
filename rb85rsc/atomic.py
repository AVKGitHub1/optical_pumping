"""Rb-85 5S1/2 -> 5P3/2 (D2) atomic data generated from ARC and cached on disk.

Everything the dynamics needs from atomic structure is collected here:

* hyperfine energies of 5S1/2 (F=2,3) and 5P3/2 (F'=1..4),
* Lande g_F factors (ARC ``getLandegfExact``, includes g_I),
* hyperfine-resolved dipole amplitudes <F',m'| e r_q |F,m> with q = m' - m,
  in units of e*a0 (ARC ``getDipoleMatrixElementHFS``),
* spontaneous branching probabilities (ARC ``getBranchingRatio``),
* the 5P3/2 decay rate (ARC ``getStateLifetime``).

The package is ``ARC-Alkali-Rydberg-Calculator`` (``from arc import Rubidium85``),
not the unrelated PyPI package called ``arc``.  ARC is only imported when the
cache is missing or was produced by a different ARC version.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SI constants (CODATA 2018 exact / recommended values)
# ---------------------------------------------------------------------------
H = 6.62607015e-34
HBAR = H / (2.0 * np.pi)
C_LIGHT = 299792458.0
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
A0 = 5.29177210903e-11
MU_B = 9.2740100783e-24
K_B = 1.380649e-23
EA0 = E_CHARGE * A0  # C*m per atomic unit of dipole moment

CACHE_FORMAT_VERSION = 2
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache" / "rb85_d2_atomic_data.json"

# Ground sublevels in the fixed order used everywhere in the package.
GROUND_STATES: tuple[tuple[int, int], ...] = tuple(
    [(2, m) for m in range(-2, 3)] + [(3, m) for m in range(-3, 4)]
)
EXCITED_STATES: tuple[tuple[int, int], ...] = tuple(
    (fp, m) for fp in (1, 2, 3, 4) for m in range(-fp, fp + 1)
)
N_GROUND = len(GROUND_STATES)  # 12
N_EXCITED = len(EXCITED_STATES)  # 24
UP = GROUND_STATES.index((3, 3))
DOWN = GROUND_STATES.index((2, 2))
SPECTATORS = tuple(i for i in range(N_GROUND) if i not in (UP, DOWN))


def ground_label(i: int) -> str:
    f, m = GROUND_STATES[i]
    return f"|{f},{m:+d}>"


# Reference values transcribed from D. A. Steck, "Rubidium 85 D Line Data"
# (rev. 2.3.x), used only for cross-checks.  They are *not* used by the model.
STECK_RB85 = {
    "ground_hfs_splitting_hz": 3.0357324390e9,
    "gamma_2pi_hz": 6.0666e6,
    "lifetime_s": 26.2348e-9,
    "excited_splitting_43_hz": 120.640e6,
    "excited_splitting_32_hz": 63.401e6,
    "excited_splitting_21_hz": 29.372e6,
    "reduced_dipole_d2_Cm": 3.58424e-29,  # Steck's <J=1/2||er||J'=3/2> convention
    "isat_cycling_w_m2": 16.6933,  # sigma+-polarized |3,3> -> |4',4'>
    "mass_kg": 1.409993199e-25,
    "gJ_ground": 2.00233113,
    "gI": -0.000293640,
}


@dataclass(frozen=True)
class AtomicData:
    ground_hfs_hz: dict  # F -> energy relative to 5S1/2 centroid [Hz]
    excited_hfs_hz: dict  # F' -> energy relative to 5P3/2 centroid [Hz]
    d2_centroid_hz: float
    gamma: float  # 5P3/2 decay rate [s^-1] (not angular, 1/lifetime)
    dipole_au: np.ndarray  # (24, 12) <e|e r_q|g>, q = m_e - m_g, units e*a0
    branching: np.ndarray  # (24, 12) probability e -> g
    g_ground: dict  # F -> g_F
    g_excited: dict  # F' -> g_F'
    gJ_excited: float
    mass_kg: float
    nuclear_spin: float
    metadata: dict = field(default_factory=dict)

    # -- derived helpers ---------------------------------------------------
    @property
    def dipole_si(self) -> np.ndarray:
        return self.dipole_au * EA0

    @property
    def d2_wavelength_m(self) -> float:
        return C_LIGHT / self.d2_centroid_hz

    @property
    def hfs_ground_splitting_hz(self) -> float:
        return self.ground_hfs_hz[3] - self.ground_hfs_hz[2]

    def transition_hz(self, f: int, fp: int) -> float:
        """Zero-field optical frequency of 5S1/2 F -> 5P3/2 F'."""
        return self.d2_centroid_hz + self.excited_hfs_hz[fp] - self.ground_hfs_hz[f]

    def ground_zeeman_hz(self, b_tesla: float) -> np.ndarray:
        """Linear (weak-field) Zeeman shift of each ground sublevel [Hz]."""
        return np.array([self.g_ground[f] * m for f, m in GROUND_STATES]) * MU_B * b_tesla / H

    def excited_zeeman_hz(self, b_tesla: float) -> np.ndarray:
        return np.array([self.g_excited[f] * m for f, m in EXCITED_STATES]) * MU_B * b_tesla / H

    def ground_energy_hz(self, b_tesla: float) -> np.ndarray:
        """Ground energies from Breit-Rabi, with labels adiabatic from B=0.

        Infer g_I and g_J from the cached weak-field g_F values, so no ARC
        database access is needed at runtime. Dipoles still use the weak-field
        basis; the magnetic validity check bounds that approximation.
        """
        e0 = np.array([self.ground_hfs_hz[f] for f, _ in GROUND_STATES])
        gi = 0.5 * (self.g_ground[3] + self.g_ground[2])
        gj_minus_gi = (self.nuclear_spin + 0.5) * (self.g_ground[3] - self.g_ground[2])
        ze = MU_B * b_tesla / H
        split = self.hfs_ground_splitting_hz
        x = gj_minus_gi * ze / split
        shifts = []
        for f, m in GROUND_STATES:
            if abs(m) == self.nuclear_spin + 0.5:
                shifts.append(self.g_ground[f] * m * ze)
                continue
            y = 4 * m * x / (2 * self.nuclear_spin + 1) + x * x
            # Rationalized sqrt(1+y)-1 is accurate even at very small B.
            sign = 1 if f == 3 else -1
            shifts.append(gi * m * ze + sign * split / 2 * y / (np.sqrt(1 + y) + 1))
        return e0 + np.asarray(shifts)

    def ground_energy_linear_hz(self, b_tesla: float) -> np.ndarray:
        """Linear approximation, retained for the magnetic validity diagnostic."""
        e0 = np.array([self.ground_hfs_hz[f] for f, _ in GROUND_STATES])
        return e0 + self.ground_zeeman_hz(b_tesla)

    def emission_norm(self) -> float:
        """D^2 = sum_{g,q} |<e|er_q|g>|^2, identical for every excited sublevel."""
        return float(np.mean(np.sum(self.dipole_au**2, axis=1)))


def _arc_version() -> str:
    try:
        from importlib.metadata import version

        return version("ARC-Alkali-Rydberg-Calculator")
    except Exception:  # pragma: no cover - ARC missing
        return "unavailable"


def _generate_from_arc() -> AtomicData:
    from arc import Rubidium85  # ARC-Alkali-Rydberg-Calculator

    atom = Rubidium85()
    n = 5
    a_g, b_g = atom.getHFSCoefficients(n, 0, 0.5)
    a_e, b_e = atom.getHFSCoefficients(n, 1, 1.5)
    ground_hfs = {f: float(atom.getHFSEnergyShift(0.5, f, a_g, b_g)) for f in (2, 3)}
    excited_hfs = {f: float(atom.getHFSEnergyShift(1.5, f, a_e, b_e)) for f in (1, 2, 3, 4)}
    d = np.zeros((N_EXCITED, N_GROUND))
    br = np.zeros((N_EXCITED, N_GROUND))
    for ie, (fe, me) in enumerate(EXCITED_STATES):
        for ig, (fg, mg) in enumerate(GROUND_STATES):
            q = me - mg
            if abs(q) <= 1:
                d[ie, ig] = atom.getDipoleMatrixElementHFS(n, 0, 0.5, fg, mg, n, 1, 1.5, fe, me, q)
            br[ie, ig] = atom.getBranchingRatio(0.5, fg, mg, 1.5, fe, me)
    tau = float(atom.getStateLifetime(n, 1, 1.5))
    meta = {
        "source": "ARC-Alkali-Rydberg-Calculator, arc.Rubidium85()",
        "arc_version": _arc_version(),
        "functions": [
            "getHFSCoefficients",
            "getHFSEnergyShift",
            "getDipoleMatrixElementHFS",
            "getBranchingRatio",
            "getStateLifetime",
            "getTransitionFrequency",
            "getLandegfExact",
        ],
        "dipole_units": "e*a0, <F',m'| e r_q |F,m> with q=m'-m (absorption)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cache_format_version": CACHE_FORMAT_VERSION,
        "arc_reduced_dipole_symmetric_au": float(atom.getReducedMatrixElementJ(n, 0, 0.5, n, 1, 1.5)),
    }
    return AtomicData(
        ground_hfs_hz=ground_hfs,
        excited_hfs_hz=excited_hfs,
        d2_centroid_hz=float(atom.getTransitionFrequency(n, 0, 0.5, n, 1, 1.5)),
        gamma=1.0 / tau,
        dipole_au=d,
        branching=br,
        g_ground={f: float(atom.getLandegfExact(0, 0.5, f)) for f in (2, 3)},
        g_excited={f: float(atom.getLandegfExact(1, 1.5, f)) for f in (1, 2, 3, 4)},
        # g_J(5P3/2) from ARC's g_L and g_S (L=1, S=1/2, J=3/2)
        gJ_excited=float(atom.gL * 2.0 / 3.0 + atom.gS / 3.0),
        mass_kg=float(atom.mass),
        nuclear_spin=float(atom.I),
        metadata=meta,
    )


def _to_json(ad: AtomicData) -> dict:
    return {
        "ground_hfs_hz": {str(k): v for k, v in ad.ground_hfs_hz.items()},
        "excited_hfs_hz": {str(k): v for k, v in ad.excited_hfs_hz.items()},
        "d2_centroid_hz": ad.d2_centroid_hz,
        "gamma": ad.gamma,
        "dipole_au": ad.dipole_au.tolist(),
        "branching": ad.branching.tolist(),
        "g_ground": {str(k): v for k, v in ad.g_ground.items()},
        "g_excited": {str(k): v for k, v in ad.g_excited.items()},
        "gJ_excited": ad.gJ_excited,
        "mass_kg": ad.mass_kg,
        "nuclear_spin": ad.nuclear_spin,
        "metadata": ad.metadata,
    }


def _from_json(d: dict) -> AtomicData:
    return AtomicData(
        ground_hfs_hz={int(k): v for k, v in d["ground_hfs_hz"].items()},
        excited_hfs_hz={int(k): v for k, v in d["excited_hfs_hz"].items()},
        d2_centroid_hz=d["d2_centroid_hz"],
        gamma=d["gamma"],
        dipole_au=np.array(d["dipole_au"]),
        branching=np.array(d["branching"]),
        g_ground={int(k): v for k, v in d["g_ground"].items()},
        g_excited={int(k): v for k, v in d["g_excited"].items()},
        gJ_excited=d["gJ_excited"],
        mass_kg=d["mass_kg"],
        nuclear_spin=d["nuclear_spin"],
        metadata=d["metadata"],
    )


@lru_cache(maxsize=4)
def load_atomic_data(cache_path: str | None = None, regenerate: bool = False) -> AtomicData:
    """Return cached ARC data, regenerating when missing or from another ARC version."""
    path = Path(cache_path) if cache_path else DEFAULT_CACHE
    installed = _arc_version()
    if path.exists() and not regenerate:
        data = json.loads(path.read_text())
        meta = data.get("metadata", {})
        stale = meta.get("cache_format_version") != CACHE_FORMAT_VERSION or (
            installed != "unavailable" and meta.get("arc_version") != installed
        )
        if not stale:
            return _from_json(data)
    if installed == "unavailable":
        raise RuntimeError(
            "ARC (pip install ARC-Alkali-Rydberg-Calculator) is required to generate "
            f"the atomic-data cache at {path}."
        )
    ad = _generate_from_arc()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json(ad), indent=1))
    return ad


# ---------------------------------------------------------------------------
# Magnetic-field validity
# ---------------------------------------------------------------------------
def zeeman_validity(ad: AtomicData, b_tesla: float, trap_hz: float) -> dict:
    """Report the included Breit-Rabi correction and weak-field dipole validity.

    The legacy quadratic-error key is the correction relative to linear ground
    energies, now included in the dynamics. Excited energies remain linear.
    """
    out = {"B_gauss": b_tesla * 1e4}
    lin = ad.ground_energy_linear_hz(b_tesla)
    exact = ad.ground_energy_hz(b_tesla)
    err_ud = float((exact[UP] - exact[DOWN]) - (lin[UP] - lin[DOWN]))
    out["breit_rabi_source"] = "analytic Breit-Rabi using cached ARC hyperfine splitting and g factors"
    out["ground_energy_model"] = "Breit-Rabi"
    out["raman_quadratic_zeeman_error_hz"] = err_ud
    spacing = min(
        abs(ad.excited_hfs_hz[f + 1] - ad.excited_hfs_hz[f]) for f in (1, 2, 3)
    )
    ez = ad.gJ_excited * MU_B * b_tesla / H
    out["excited_zeeman_over_min_hfs"] = ez / spacing
    # Leading neglected (second-order) excited-state shift ~ E_Z^2 / Delta_hfs,
    # compared with the natural linewidth that sets optical detuning scales.
    quad_exc = ez**2 / spacing
    out["excited_quadratic_zeeman_hz"] = quad_exc
    out["excited_quadratic_over_gamma"] = quad_exc / (ad.gamma / (2 * np.pi))
    status = "ok"
    notes = []
    if out["excited_quadratic_over_gamma"] > 0.3 or out["excited_zeeman_over_min_hfs"] > 0.3:
        status = "invalid"
        notes.append("5P3/2 Zeeman energy not small against hyperfine intervals (Paschen-Back mixing)")
    elif out["excited_quadratic_over_gamma"] > 0.05:
        status = "warning"
        notes.append("neglected quadratic 5P3/2 Zeeman shift >5% of Gamma")
    # err_ud is the correction now INCLUDED, not an error in the dynamics.
    out["status"] = status
    out["notes"] = notes
    return out


@lru_cache(maxsize=64)
def _breit_rabi_ground(b_tesla: float):
    import warnings

    from arc import Rubidium85

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Rubidium85().breitRabi(5, 0, 0.5, np.array([b_tesla]))


def isat_cycling(ad: AtomicData) -> float:
    """Saturation intensity of the sigma+ |3,3> -> |4',4'> cycling transition [W/m^2]."""
    ie = EXCITED_STATES.index((4, 4))
    d = abs(ad.dipole_si[ie, UP])
    return C_LIGHT * EPS0 * ad.gamma**2 * HBAR**2 / (4.0 * d**2)


def steck_comparison(ad: AtomicData) -> dict:
    """Relative deviation of ARC-derived quantities from Steck's tabulation."""
    s = STECK_RB85
    red_steck_au = ad.metadata.get("arc_reduced_dipole_symmetric_au", np.nan) / np.sqrt(2.0)
    rows = {
        "ground_hfs_splitting_hz": (ad.hfs_ground_splitting_hz, s["ground_hfs_splitting_hz"]),
        "gamma_2pi_hz": (ad.gamma / (2 * np.pi), s["gamma_2pi_hz"]),
        "excited_splitting_43_hz": (ad.excited_hfs_hz[4] - ad.excited_hfs_hz[3], s["excited_splitting_43_hz"]),
        "excited_splitting_32_hz": (ad.excited_hfs_hz[3] - ad.excited_hfs_hz[2], s["excited_splitting_32_hz"]),
        "excited_splitting_21_hz": (ad.excited_hfs_hz[2] - ad.excited_hfs_hz[1], s["excited_splitting_21_hz"]),
        "reduced_dipole_d2_Cm": (red_steck_au * EA0, s["reduced_dipole_d2_Cm"]),
        "isat_cycling_w_m2": (isat_cycling(ad), s["isat_cycling_w_m2"]),
        "mass_kg": (ad.mass_kg, s["mass_kg"]),
    }
    return {k: {"arc": a, "steck": b, "rel_dev": (a - b) / b} for k, (a, b) in rows.items()}
