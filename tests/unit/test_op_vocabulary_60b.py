"""60b V-2/V-3: operator-vocabulary constants + the rho
rename. Value box asserts VIA the constants; residue sweep
is permanent."""
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_v2_decide_returns_constants():
    from reference_net.growthpolicy import (OP_OMEGA, OP_RHO,
                                            GROWTH_OPERATORS)
    from core.plasticity import policy as siting
    from reference_net.net import Network
    assert GROWTH_OPERATORS == {"widen", "refine", "deepen"}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3))
    y = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    net = Network(3, 8, lr=1e-2, seed=7)
    for _ in range(40):
        net.train_step(X, y)
    d = siting.decide(net)
    assert d["action"] in (OP_OMEGA, OP_RHO)
    # localized fixture (per decide()'s own criterion,
    # policy.py:56-57): ONE unit unstable (u=1), rest calm
    # (u=0) -> max >= factor*mean -> non-uniform -> refine
    net._ema_dw = np.ones((8, 3)); net._ema_dw[2] = 0.0
    net._ema_adw = np.ones((8, 3))
    d = siting.decide(net)
    assert d["action"] == OP_RHO          # THE rename point
    # widespread fixture: every unit equally unstable (u=1)
    # -> uniform + saturation 1 >= widen_sat -> widen
    net._ema_dw = np.zeros((8, 3))
    net._ema_adw = np.ones((8, 3))
    d2 = siting.decide(net)
    assert d2["action"] == OP_OMEGA


def test_v3_residue_sweep_permanent():
    """Converted files: zero BARE 'widen'/'deepen'/'refine'
    string literals outside the constants block; the three
    constants exist with exactly the wire values."""
    from reference_net import growthpolicy as gp
    assert gp.OP_OMEGA == "widen"
    assert gp.OP_RHO == "refine"
    assert gp.OP_DELTA == "deepen"
    files = [
        REPO / "core/plasticity/policy.py",
        REPO / "core/plasticity/run_self.py",
        REPO / "modules/ReferenceNet/reference_net/"
               "growthpolicy/combiner_threshold.py",
    ]
    lit = re.compile(r'''["'](widen|deepen|refine)["']''')
    for f in files:
        hits = [(i + 1, ln.strip()) for i, ln in
                enumerate(f.read_text().splitlines())
                if lit.search(ln) and "OP_" not in ln]
        # exemption (60b V-3): a wire word cited BESIDE its
        # constant (line contains OP_) is documentation, not
        # a bare literal.
        assert hits == [], f"{f.name}: bare literals {hits}"
    # __init__: only the constants block may carry the words
    init = (REPO / "modules/ReferenceNet/reference_net/"
                   "growthpolicy/__init__.py").read_text()
    body = init.split("GROWTH_OPERATORS", 1)[1]
    hits = [(i + 1, ln.strip()) for i, ln in
            enumerate(body.splitlines()) if lit.search(ln)]
    assert hits == [], f"__init__ post-block literals {hits}"
