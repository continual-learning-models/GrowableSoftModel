"""GSM-I2 S2: the causal + categorical transformer paths on
the kernel contract — the tests the plan promised for S2
(FD spot checks on the causal path; growth exactness on torch
for the newly enabled modes)."""
import pickle   # safe: round-trips objects THIS test creates
                # in-process (artifact contract is pickle-based;
                # no untrusted data is ever loaded)
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.backends import (resolve_backend,        # noqa: E402
                               set_compute_policy)
from core.substrates.sequence import SequenceSubstrate  # noqa: E402
from core.substrates.transformer import (                # noqa: E402
    TransformerSubstrate)

DEVICES = [("torch", "cpu", "float32", 2e-3)]
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICES.append(("torch", "mps", "float32", 2e-3))
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _judge_after():
    yield
    set_compute_policy("numpy", "cpu", None)


def _seq_data(n=8, T=6, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, d))
    y = X[:, -1, :1] * 0.5
    return X, y


def test_causal_fd_spot_check_on_judge():
    """FD spot check of the causal-path gradients (judge).

    The host computes grads inside train_step; SGD mode is
    exactly P -= lr*G, so G is recovered from parameter deltas
    on a pickled clone and checked entry-by-entry against
    central differences of the before-step loss."""
    set_compute_policy("numpy", "cpu", None)
    X, y = _seq_data()
    sq = SequenceSubstrate(2, 8, d_model=8, n_layers=1,
                           n_heads=2)
    for _ in range(3):
        sq.train_step(X, y)          # scalers fitted, params moved

    def loss_at(s):
        return pickle.loads(pickle.dumps(s)).train_step(
            X, y, sgd_lr=0.0)

    lr = 1e-4
    clone = pickle.loads(pickle.dumps(sq))
    P0 = {k: np.array(clone.P[k], copy=True) for k in clone.P}
    clone.train_step(X, y, sgd_lr=lr)
    G = {k: (P0[k] - clone.P[k]) / lr for k in P0}

    eps = 1e-6
    for k in ["Wv", "Wh", "Wq_0", "Wk_0", "Wo_0", "W1_0",
              "W2_0", "g1_0"]:
        flat = sq.P[k].reshape(-1)
        for ix in range(min(3, flat.size)):
            keep = flat[ix]
            flat[ix] = keep + eps
            lp = loss_at(sq)
            flat[ix] = keep - eps
            lm = loss_at(sq)
            flat[ix] = keep
            fd = (lp - lm) / (2 * eps)
            g = G[k].reshape(-1)[ix]
            assert abs(fd - g) < 3e-5 * max(1.0, abs(fd)), \
                (k, ix, fd, g)


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_categorical_transformer_add_class_on_device(bk, dev,
                                                     dt, tol):
    """Vocab growth (2a surgery) on device: old-class ordering
    preserved, new class born ~silent, trains after growth."""
    set_compute_policy(bk, dev, dt,
                       acknowledge_f32_precision=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4))
    y = ["a" if v > 0 else "b" for v in X[:, 0]]
    tc = TransformerSubstrate(4, 8, mode="categorical",
                              vocab=["a", "b"], d_model=8,
                              n_layers=1, n_heads=2)
    for _ in range(12):
        tc.train_step(X, y)
    p_before = tc.predict_proba(X[:6])
    tc.add_class("c")
    p_after = tc.predict_proba(X[:6])
    assert np.array_equal(np.argmax(p_before, 1),
                          np.argmax(p_after[:, :2], 1))
    assert p_after[:, 2].max() < 0.05
    ce = tc.train_step(X[:10], ["c"] * 10)   # trains after growth
    assert np.isfinite(ce)


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_causal_grow_on_device_parity(bk, dev, dt, tol):
    """grow_site on the causal host per device: same grow point,
    trajectory parity vs the judge within tolerance."""
    X, y = _seq_data(n=16)

    def run(policy):
        set_compute_policy(*policy,
                           acknowledge_f32_precision=True)
        sq = SequenceSubstrate(2, 8, d_model=8, n_layers=1,
                               n_heads=2)
        for _ in range(10):
            sq.train_step(X, y)
        sq.grow_site("layer0/ffn[1]", hidden=4)
        mses = [sq.train_step(X, y) for _ in range(10)]
        p = sq.predict(X[:5])
        return mses[-1], np.asarray(sq._bk.to_numpy(p), float)

    m_j, p_j = run(("numpy", "cpu", None))
    m_t, p_t = run((bk, dev, dt))
    assert abs(m_j - m_t) < tol
    scale = max(1.0, float(np.abs(p_j).max()))
    assert float(np.abs(p_j - p_t).max()) / scale < tol
