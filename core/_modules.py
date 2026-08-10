"""The ONE import shim: exposes the module packages.

Every system file that needs module code imports it from here.

    from core._modules import generator, reference_net

When the project is pip-installed, `generator` and `reference_net` are
importable directly. In the source layout they live under `modules/`,
so we add those paths as a fallback before importing.
"""
import sys
from pathlib import Path

try:                                  # installed layout
    import generator                 # noqa: F401
    import reference_net                  # noqa: F401
except ModuleNotFoundError:           # source layout
    _MODULES = Path(__file__).resolve().parent.parent / "modules"
    for _name in ("Generator", "Engine", "ReferenceNet",
                  "RLTrainer"):
        _p = str(_MODULES / _name)
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import generator                 # noqa: E402,F401
    import reference_net                  # noqa: E402,F401

# rl_trainer (Track B, plan 84 D-P4): the editable install maps
# only generator/reference_net, so the RLTrainer path needs its
# own fallback in BOTH layouts (fence v2.19 registration point).
try:
    import rl_trainer                # noqa: F401
except ModuleNotFoundError:
    _RLP = str(Path(__file__).resolve().parent.parent
               / "modules" / "RLTrainer")
    if _RLP not in sys.path:
        sys.path.insert(0, _RLP)
    import rl_trainer                # noqa: E402,F401
