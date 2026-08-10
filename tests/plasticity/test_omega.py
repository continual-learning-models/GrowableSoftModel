"""Stage-3 unit tests: omega preservation (exact), EMA warm-start,
optimizer extension, composition with rho, siting policy."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import reference_net                    # noqa: E402
from reference_net.net import Network                      # noqa: E402
from core.plasticity.net_ops import widen_net, widen_at  # noqa: E402
from core.plasticity import policy                    # noqa: E402


def _trained_net(seed=7, hidden=4, steps=60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(128, 3))
    y = (X[:, :1] * X[:, 1:2] + X[:, 2:3])
    net = Network(d_in=3, hidden=hidden, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_widen_preserves_function_exactly():
    net, X, _ = _trained_net()
    before = net.predict(X)
    rep = widen_net(net, k=3)
    after = net.predict(X)
    assert np.max(np.abs(after - before)) < 1e-12
    assert net.H == 7 and rep["new_units"] == [4, 5, 6]


def test_widen_extends_optimizer_and_ema_consistently():
    net, X, y = _trained_net()
    ema_mean = net._ema_adw.mean(axis=0)
    widen_net(net, k=2)
    assert net.opt.m[0].shape == net.W1.shape
    assert net.opt.v[2].shape == net.W2.shape
    assert net._ema_dw.shape == net.W1.shape
    # warm-start: new rows equal the pre-widen row-mean
    assert np.allclose(net._ema_adw[-1], ema_mean)
    assert len(net.instability()) == net.H
    net.train_step(X, y)                      # still trains fine


def test_new_units_are_trainable_from_step_one():
    net, X, y = _trained_net()
    widen_net(net, k=2)
    w1_new0 = net.W1[-2:].copy()
    w2_new0 = net.W2[:, -2:].copy()
    for _ in range(5):
        net.train_step(X, y)
    assert np.abs(net.W1[-2:] - w1_new0).mean() > 0
    assert np.abs(net.W2[:, -2:] - w2_new0).mean() > 0   # left zero-init,
    # moved by training — zero was an initial value, never a mask


def test_widen_at_every_scale_and_compose_with_rho():
    net, X, y = _trained_net()
    net.grow(1, hidden=3)                     # rho: inner net at node 1
    inner_before = net.grown_body(1).H
    out = widen_at(net, "1", k=2)             # omega INSIDE scale 1
    assert net.grown_body(1).H == inner_before + 2
    assert out["path"] == "1"
    before = net.predict(X)
    widen_net(net, k=1)                       # omega at root
    net.grow(net.H - 1, hidden=2)             # rho ON the new unit
    assert np.max(np.abs(net.predict(X) - before)) < 1e-12
    assert net.grown_body(net.H - 1) is not None  # operators compose


def test_widen_rejects_bad_k():
    net, _, _ = _trained_net()
    with pytest.raises(ValueError):
        widen_net(net, k=0)


def test_policy_widen_on_uniform_saturation_fixture():
    net, _, _ = _trained_net()
    net._ema_dw = np.zeros_like(net._ema_dw)          # oscillation ->
    net._ema_adw = np.ones_like(net._ema_adw)         # u = 1 everywhere
    d = policy.decide(net)
    assert d["action"] == "widen" and d["container"] == "root"


def test_policy_deepen_on_localized_conflict_fixture():
    net, _, _ = _trained_net()
    net._ema_dw = np.ones_like(net._ema_dw) * 0.99    # steady drift:
    net._ema_adw = np.ones_like(net._ema_adw)         # u ~ 0.01 ...
    net._ema_dw[2] = 0.0                              # ... except unit 2
    d = policy.decide(net)
    from reference_net.growthpolicy import OP_RHO
    assert d["action"] == OP_RHO and d["site"].startswith("root[2]")


def test_policy_escalates_after_disposed_deepens():
    net, _, _ = _trained_net()
    d = policy.decide(net, recent_disposed=2)
    assert d["action"] == "widen"
