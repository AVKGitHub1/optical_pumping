import numpy as np

from rb85rsc.analysis import analyze
from rb85rsc.config import SimConfig, set_param
from rb85rsc.dynamics import solve
from rb85rsc.model import Model


def cfg(**kw) -> SimConfig:
    c = SimConfig()
    c.timing.n_samples = 21
    for k, v in kw.items():
        path = k.replace("__", ".")
        if isinstance(v, (list, dict)):
            obj = c
            for p in path.split(".")[:-1]:
                obj = getattr(obj, p)
            setattr(obj, path.split(".")[-1], v)
        else:
            set_param(c, path, v)
    c.validate()
    return c


def run(c: SimConfig):
    m = Model(c)
    r = solve(m)
    return m, r, analyze(m, r)


def nbar(P):
    return float(np.sum(P.sum(axis=-2) * np.arange(P.shape[-1]), axis=-1)) if P.ndim == 2 else None
