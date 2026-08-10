"""Stage-4 unit tests: sigma (input-schema growth) — exact
preservation for ANY new-column values, recursion into inner nets,
trainability of the new column, no-freeze after sigma."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import reference_net                    # noqa: E402
from reference_net.net import Network                      # noqa: E402
from core.plasticity.net_ops import add_feature_net, widen_net  # noqa: E402


def _trained(seed=7, steps=80):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(128, 2))
    y = X[:, :1] * X[:, 1:2] + X[:, :1]
    net = Network(d_in=2, hidden=6, seed=seed)
    net.grow(1, hidden=3)                     # include an inner scale
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_sigma_preserves_exactly_for_any_new_column():
    net, X, _ = _trained()
    before = net.predict(X)
    add_feature_net(net, default=0.0)
    rng = np.random.default_rng(1)
    for col in (np.zeros((len(X), 1)),                 # backfill
                rng.normal(size=(len(X), 1)) * 50):    # wild values
        X3 = np.hstack([X, col])
        assert np.max(np.abs(net.predict(X3) - before)) < 1e-12


def test_sigma_recurses_into_inner_networks():
    net, _, _ = _trained()
    add_feature_net(net)
    assert net.d_in == 3 and net.grown_body(1).d_in == 3
    assert net.W1.shape == (6, 3)
    assert net.grown_body(1).W1.shape == (3, 3)
    assert net.opt.m[0].shape == net.W1.shape


def test_new_feature_is_learnable_and_earns_participation():
    net, X, _ = _trained()
    add_feature_net(net, default=0.0)
    rng = np.random.default_rng(2)
    c = rng.normal(size=(len(X), 1))
    X3 = np.hstack([X, c])
    y3 = X[:, :1] * X[:, 1:2] + X[:, :1] + 2.0 * c    # c now informative
    mse0 = float(np.mean((net.predict(X3) - y3) ** 2))
    w_col0 = net.W1[:, -1].copy()
    for _ in range(400):
        net.train_step(X3, y3)
    mseT = float(np.mean((net.predict(X3) - y3) ** 2))
    assert np.abs(net.W1[:, -1] - w_col0).mean() > 0   # column trains
    assert mseT < 0.25 * mse0                          # c exploited


def test_sigma_composes_with_omega_and_rho():
    net, X, _ = _trained()
    add_feature_net(net)
    widen_net(net, k=2)
    net.grow(net.H - 1, hidden=2)
    X3 = np.hstack([X, np.zeros((len(X), 1))])
    y3 = X[:, :1] * 2.0
    for _ in range(5):
        net.train_step(X3, y3)                         # trains fine
    assert net.d_in == 3 and net.H == 8
    assert net.grown_body(net.H - 1).d_in == 3         # newborn inner
    # nets are born at the CURRENT schema
