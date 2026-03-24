#!/usr/bin/env python3
"""Loads the canonical piper_core.py from the project root, avoids circular import."""
import importlib.util
import os
import sys

_root_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "piper_core.py"
)
_spec = importlib.util.spec_from_file_location("piper_core", _root_file)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["piper_core"] = _mod
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
