# Archived documentation

These documents preserve earlier work. The active experiment is described in the [current physics report](../astra-physics-corrections.md) and [experiment workflow](../experiment-workflow.md).

| Document | Historical purpose |
| --- | --- |
| [Population-rate model guide](rate-equations.md) | Running and configuring the original standalone optical-pumping model |
| [Population-rate physics](rate-equations-physics.md) | State space, transition rates, branching, and approximations for that model |
| [Population-rate output](rate-equations-output.md) | Preserved text output; the matching figure is in [results/archive/rate-calcs/](../../results/archive/rate-calcs/) |
| [Original development specification](original-specification.md) | Original requested simulator scope, before later experimental clarifications |

The earlier script now lives at [scripts/archive/rate-calcs.py](../../scripts/archive/rate-calcs.py); the root `rate-calcs.py` entry point remains compatible. Earlier RSC configurations are retained in [examples/arxiv/](../../examples/arxiv/). The separate [physics audit history](../astra-physics-history.md) preserves subsequent corrections and intermediate RSC calculations.
