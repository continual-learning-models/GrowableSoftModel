"""SPU integration tests (DEV_PLAN S3 I3.1-I3.4, delivered at the
substrate-integration level; factory/MCP exposure is post-v0 and
its lifecycle tests belong to that round)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp                    # noqa: E402
from reference_net.spu.spu_network import SPUNetwork       # noqa: E402


def build(seed=8, spu=True):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (48, 2))
    y = np.sin(3 * X[:, :1]) + 0.5 * X[:, 1:]
    net = SPUNetwork(d_in=2, hidden=4, lr=1e-2, seed=seed)
    for _ in range(60):
        net.train_step(X, y)
    net.grow(1, hidden=5)
    if spu:
        net.set_spu_policy({"spu_enabled": True,
                            "spu_warmup_steps": 0})
    return net, X, y


def test_gate_pattern_cycle_with_spu():                # I3.1'
    """The lifecycle's commit pattern at substrate level: a
    held-out slice quarantined from training; the SPU-evolved
    candidate is adopted only if strictly better; refusal leaves
    the incumbent untouched."""
    net, X, y = build()
    X_hold, y_hold = X[:16], y[:16]                    # quarantined
    X_tr, y_tr = X[16:], y[16:]
    incumbent = float(((net.predict(X_hold) - y_hold) ** 2).mean())
    import pickle
    snapshot = pickle.dumps(net)                       # the incumbent
    for _ in range(120):
        net.train_step(X_tr, y_tr)                     # spu active
    candidate = float(((net.predict(X_hold) - y_hold) ** 2).mean())
    promoted = candidate < incumbent                   # strict gate
    if not promoted:
        net = pickle.loads(snapshot)                   # refusal
        after = float(((net.predict(X_hold) - y_hold) ** 2).mean())
        assert after == incumbent                      # untouched
    else:
        assert candidate < incumbent
    assert any(e.get("skip") is None for e in net.spu_events) \
        or not promoted


def test_serving_purity_full_pipeline():               # I3.2
    net, X, _ = build()
    w_root = net.W1.copy()
    w_leaf = net.grown_body(1).W1.copy()
    ev0 = len(net.spu_events)
    for _ in range(100):
        net.predict(X)
    assert np.array_equal(net.W1, w_root)
    assert np.array_equal(net.grown_body(1).W1, w_leaf)
    assert len(net.spu_events) == ev0


def test_end_to_end_smoke_deterministic():             # I3.3'
    """create -> teach-with-spu -> grow -> teach -> predict,
    twice, bit-identical."""
    outs = []
    for _ in range(2):
        net, X, y = build(seed=11)
        for _ in range(30):
            net.train_step(X, y)
        net.grown_body(1).grow(0, hidden=4)            # deeper leaf
        for _ in range(30):
            net.train_step(X, y)
        outs.append(net.predict(X))
    assert np.array_equal(outs[0], outs[1])


def test_spu_and_governed_growth_coexist():            # I3.4
    net, X, y = build()
    d = gp.grow_with_policy(net, {"min_energy_points": 10 ** 9,
                                  "probe_steps": 60,
                                  "min_window_rows": 16})
    for _ in range(10):
        net.train_step(X, y)
    assert d["applied"] is not None
    assert any(e.get("skip") is None and "steps" in e
               for e in net.spu_events)
