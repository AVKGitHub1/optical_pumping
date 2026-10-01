"""Compatibility entry point for the archived population-rate model.

Edit the legacy model inputs in ``scripts/archive/rate-calcs.py``.
"""
from pathlib import Path as _Path


# Keep the original module namespace for callers that import the script and
# override its input globals before calling main(). The archived file retains
# its own __main__ guard and is the single source of the legacy implementation.
_source = _Path(__file__).resolve().parent / "scripts/archive/rate-calcs.py"
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
