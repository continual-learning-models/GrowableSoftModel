"""The no-freeze audit — the axiom, mechanized (design §7).

For every host and after EVERY operator: (1) every parameter array is
in the training path (moves under a generic batch); (2) nothing is
excluded from the optimizer; (3) no masks/stop-gradients anywhere.
Movement is averaged over several steps so genuine zero-gradient
coincidences don't false-alarm."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import reference_net                      # noqa: E402
from reference_net.net import Network                        # noqa: E402
from core.plasticity.net_ops import widen_net           # noqa: E402
from core.substrates.transformer_plus import TransformerPlus  # noqa: E402


def _net_arrays(net, prefix=""):
    out = {f"{prefix}W1": net.W1, f"{prefix}b1": net.b1,
           f"{prefix}W2": net.W2, f"{prefix}c": net.c}
    for j, inner in net.inner.items():
        out.update(_net_arrays(inner, prefix=f"{prefix}{j}/"))
    return out


def _audit_network(net, X, y, steps=5):
    before = {k: v.copy() for k, v in _net_arrays(net).items()}
    for _ in range(steps):
        net.train_step(X, y)
    after = _net_arrays(net)
    dead = [k for k in before
            if np.abs(after[k] - before[k]).mean() == 0.0]
    return dead


def _audit_transformer(tf, X, y, steps=5):
    before = {k: v.copy() for k, v in tf.P.items()}
    inner_before = {f"{lj}:{k}": v.copy()
                    for lj, net in tf.inner.items()
                    for k, v in _net_arrays(net).items()}
    for _ in range(steps):
        tf.train_step(X, y)
    dead = [k for k in before
            if np.abs(tf.P[k] - before[k]).mean() == 0.0]
    for lj, net in tf.inner.items():
        arrs = _net_arrays(net)
        dead += [f"{lj}:{k}" for k, v in arrs.items()
                 if np.abs(v - inner_before[f"{lj}:{k}"]).mean() == 0.0]
    return dead


def _data(seed=7, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, d))
    y = X[:, :1] * X[:, 1:2] + 0.5 * X[:, 2:3]
    return X, y


def test_no_freeze_mlp_fresh_grown_widened():
    X, y = _data()
    net = Network(d_in=3, hidden=4, seed=7)
    assert _audit_network(net, X, y) == []                 # fresh
    net.grow(0, hidden=3)                                  # rho
    for _ in range(3):
        net.train_step(X, y)      # inner scalers fit before audit
    assert _audit_network(net, X, y) == []                 # after rho
    widen_net(net, k=2)                                    # omega
    assert _audit_network(net, X, y) == []                 # after omega


def test_no_freeze_transformer_base_widened_grown():
    X, y = _data()
    tf = TransformerPlus(d_in=3, hidden=6, seed=7)
    for _ in range(3):
        tf.train_step(X, y)
    assert _audit_transformer(tf, X, y) == []              # base
    tf.widen_ffn(k=2)                                      # omega
    assert _audit_transformer(tf, X, y) == []              # after omega
    site = tf.growth_sites()[0][0]                         # rho on top
    tf.grow_site(site)
    for _ in range(3):
        tf.train_step(X, y)
    assert _audit_transformer(tf, X, y) == []              # after rho


def test_optimizer_covers_every_array():
    net = Network(d_in=3, hidden=4, seed=7)
    widen_net(net, k=2)
    shapes = [net.W1.shape, net.b1.shape, net.W2.shape, net.c.shape]
    assert [m.shape for m in net.opt.m] == shapes
    assert [v.shape for v in net.opt.v] == shapes
    tf = TransformerPlus(d_in=3, hidden=6, seed=7)
    tf.widen_ffn(k=1)
    for k, v in tf.P.items():
        assert tf._adam[k][0].shape == v.shape, k
        assert tf._adam[k][1].shape == v.shape, k
