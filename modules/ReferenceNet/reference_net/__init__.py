"""REFERENCE NET — the first-generation recursive multi-scale
network family: net, trainer, curriculum, growth policy, bodies,
instruments, and its own SPU bindings. Runs on modules/Engine."""
import sys as _sys
from pathlib import Path as _Path

_ENGINE = _Path(__file__).resolve().parents[2] / "Engine"
if _ENGINE.is_dir() and str(_ENGINE) not in _sys.path:
    _sys.path.insert(0, str(_ENGINE))
