"""A5 zero-attach pricer tests (7)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network  # noqa: E402
from reference_net.growthpolicy.pricer_zero_attach import (  # noqa: E402
    ZeroAttachPricer, _fingerprint)

P = ZeroAttachPricer()
POL = {"min_window_rows": 64, "probe_steps": 120, "probe_lr": 0.05}


def _scope(seed=2, hidden=4, steps=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(64, 2))
    y = (np.sin(2 * X[:, 0]) + 0.4 * X[:, 1]).reshape(-1, 1)
    net = Network(d_in=2, hidden=hidden, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net


def test_living_scope_fingerprint_unchanged():
    net = _scope()
    before = _fingerprint(net)
    P.price(net, POL)
    assert _fingerprint(net) == before
    assert net.blocks == [] and net.gain_ledger == []


def test_both_curves_same_units_and_improving():
    net = _scope()
    r = P.price(net, POL)
    w, d = r["widen_curve"], r["deepen_curve"]
    assert len(w) == len(d) == POL["probe_steps"]
    assert w[-1] <= w[0] and d[-1] <= d[0]


def test_probe_seeding_determinism():
    net = _scope()
    assert P.price(net, POL) == P.price(net, POL)


def test_H1_scope_runs():
    net = _scope(hidden=1)
    r = P.price(net, POL)
    assert "widen_curve" in r


def test_short_window_refusal():
    net = Network(d_in=2, hidden=3, lr=1e-2, seed=1)
    X = np.random.default_rng(0).normal(size=(8, 2))
    net.train_step(X, X[:, :1])
    assert "refusal" in P.price(net, POL)


def test_untrained_scope_refusal():
    net = Network(d_in=2, hidden=3, lr=1e-2, seed=1)
    assert "refusal" in P.price(net, POL)


def test_curves_are_plain_floats():
    net = _scope()
    r = P.price(net, POL)
    assert all(isinstance(v, float) for v in r["widen_curve"][:5])
