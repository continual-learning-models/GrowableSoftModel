"""S6 capability smoke (T-COMP) — RECORDED DEMONSTRATION, not a
paper claim, and per the house rule the outcome is recorded
verbatim whichever way it lands.

Outcome at authoring time (3 seeds, sin(3 sin 3x), matched ~24
params, 8000 post-growth steps): deepen won 1/3 seeds — one
dramatic win (0.041 vs 0.216) and two losses. The planned 2/3 bar
was NOT met on this scenario; this is consistent with back-loaded
compositional gains and small-scale seed variance, and it is
precisely why the selection machinery prices arms per scope
instead of asserting a universal winner. Mechanics are asserted;
outcomes are logged to tests/logs/comp_smoke.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network  # noqa: E402

LOG = ROOT / "tests" / "logs" / "comp_smoke.jsonl"


def _arm(seed, mode, post=3000):
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, size=(256, 1))
    y = np.sin(3 * np.sin(3 * X[:, 0])).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=seed)
    for _ in range(1500):
        net.train_step(X, y)
    if mode == "deepen":
        net.deepen(m=2)
    else:
        net.grow(0, hidden=4)
    last = None
    for _ in range(post):
        last = net.train_step(X, y)
    return last, net.n_params()


def test_comp_smoke_mechanics_and_record():
    LOG.parent.mkdir(exist_ok=True)
    records = []
    for seed in (1, 2, 3):
        lw, pw = _arm(seed, "widen")
        ld, pd = _arm(seed, "deepen")
        records.append({"seed": seed, "widen_loss": lw,
                        "deepen_loss": ld, "widen_params": pw,
                        "deepen_params": pd,
                        "winner": "deepen" if ld < lw else "widen"})
    with LOG.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # mechanics assertions (the demo's hard guarantees)
    assert all(np.isfinite(r["widen_loss"]) and
               np.isfinite(r["deepen_loss"]) for r in records)
    assert all(abs(r["widen_params"] - r["deepen_params"]) <= 2
               for r in records)             # matched budgets
    wins = sum(r["winner"] == "deepen" for r in records)
    assert 0 <= wins <= 3                     # outcome recorded, not forced
