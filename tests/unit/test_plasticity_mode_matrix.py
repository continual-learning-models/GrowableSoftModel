"""The plasticity-verb x head-mode MATRIX (owner order 2026-07-10).

The hole class this closes: verbs were each tested only on the mode
they were developed with (numeric); the cross of dimensions had no
owning test. Nine cells, EFFECTIVENESS assertions (function
preserved, training moves, serving flows) — never existence-only.

Contract encoded here:
- omega (widen) works on EVERY head mode: exact function
  preservation (zero-extended head columns) + trains after.
- sigma (add_feature) works on every head mode.
- Phi (deepen) works on numeric; on categorical/numeric_dist it
  must REFUSE LOUDLY (the head paths read Hact directly, so blocks
  would neither train nor serve — a silent no-op is the bug).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from core.substrates.mlp import MLPSubstrate           # noqa: E402
from core.plasticity.net_ops import (widen_net,        # noqa: E402
                                     add_feature_net)

MODES = ["numeric", "categorical", "numeric_dist"]


def _make(mode, trained=True):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    if mode == "categorical":
        y = ["a" if v > 0 else "b" for v in X[:, 0]]
        org = MLPSubstrate(3, 8, mode=mode, vocab=["a", "b"])
    else:
        y = (X[:, 0] * 2 + 0.3 * X[:, 1]).reshape(-1, 1) \
            if mode == "numeric" else X[:, 0] * 2
        org = MLPSubstrate(3, 8, mode=mode)
    if trained:
        for _ in range(30):
            org.train_step(X, y)
    return org, X, y


def _serve(org, X):
    if org.mode == "categorical":
        return org.predict_proba(X)
    if org.mode == "numeric_dist":
        v, s = org.predict_dist(X)
        return np.stack([v, s], axis=1)
    return np.asarray(org.predict(X))


# ---------------- omega (widen) x 3 modes ----------------
@pytest.mark.parametrize("mode", MODES)
def test_omega_widen_every_mode(mode):
    """Widen must preserve the function EXACTLY (zero head columns)
    and keep training on, for every head mode."""
    org, X, y = _make(mode)
    before = _serve(org, X[:8])
    out = widen_net(org, k=2)
    assert out["params"] > 0 if "params" in out else True
    after = _serve(org, X[:8])
    # the existing omega standard (test_omega.py): exact up to BLAS
    # summation reassociation, 1e-12
    assert np.max(np.abs(np.asarray(after)
                         - np.asarray(before))) < 1e-12, \
        "widen must be exactly function-preserving"
    losses = [org.train_step(X, y) for _ in range(10)]
    assert np.isfinite(losses).all()


# ---------------- sigma (add_feature) x 3 modes ----------------
@pytest.mark.parametrize("mode", MODES)
def test_sigma_add_feature_every_mode(mode):
    org, X, y = _make(mode)
    before = _serve(org, X[:8])
    add_feature_net(org)
    X4 = np.hstack([X, np.zeros((len(X), 1))])
    after = _serve(org, X4[:8])
    assert np.allclose(np.asarray(before), np.asarray(after),
                       atol=1e-12), \
        "zero-weight new feature must preserve the function"
    losses = [org.train_step(X4, y) for _ in range(10)]
    assert np.isfinite(losses).all()


# ---------------- Phi (deepen) x 3 modes ----------------
def test_phi_deepen_numeric_blocks_actually_train():
    """EFFECTIVENESS, not existence: after deepen, block params
    must MOVE under training (the silent-no-op probe)."""
    org, X, y = _make("numeric")
    org.deepen()
    assert org.blocks
    before = [np.array(b["Bin"], copy=True) for b in org.blocks]
    for _ in range(20):
        org.train_step(X, y)
    moved = any(not np.array_equal(b0, np.asarray(b["Bin"]))
                for b0, b in zip(before, org.blocks))
    assert moved, "deepen blocks must actually train on numeric"


@pytest.mark.parametrize("mode", ["categorical", "numeric_dist"])
def test_phi_deepen_refuses_on_head_modes(mode):
    """The head paths read Hact directly: blocks would neither train
    nor serve. Silent acceptance is the bug; the contract is a LOUD
    refusal naming the boundary."""
    org, X, y = _make(mode)
    with pytest.raises(ValueError, match="numeric"):
        org.deepen()
