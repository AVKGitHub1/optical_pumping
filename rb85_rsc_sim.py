#!/usr/bin/env python
"""Compatibility entry point for the Rb-85 RSC simulator.

The command-line implementation lives in :mod:`rb85rsc.cli`.
"""
from rb85rsc.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
