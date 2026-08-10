"""WP1 acceptance tests for the recursive network core (PLAN A-WP1):
function preservation at growth, recursion with zero code difference,
and learnability of a stage-1 linear law by the atomic net."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference_net.net import Network


def _data(rng, n, fn):
    X = rng.uniform(0, 4, (n, 3))
    y = fn(X).reshape(-1, 1)
    return X, y


def test_growth_preserves_function_exactly():
    rng = np.random.default_rng(0)
    net = Network(3, 16, seed=1)
    X = rng.uniform(0, 4, (32, 3))
    before = net.predict(X).copy()
    net.grow(3)                      # depth 1 -> 2
    assert np.allclose(net.predict(X), before), "growth must not change f"
    net.inner[3].grow(0)             # depth 2 -> 3: SAME operator, zero new code
    assert np.allclose(net.predict(X), before)
    assert net.depth() == 3


def test_atomic_learns_stage1_linear():
    rng = np.random.default_rng(1)
    fn = lambda X: 2 * X[:, 0] + X[:, 1] - X[:, 2]
    Xtr, ytr = _data(rng, 200, fn)
    Xho, yho = _data(rng, 50, fn)
    net = Network(3, 16, seed=2)
    for _ in range(1500):
        net.train_step(Xtr, ytr)
    mse = float(np.mean((net.predict(Xho) - yho) ** 2))
    assert mse < 0.01, mse


def test_recursive_target_handoff_trains_inner():
    """A composite node's inner net must receive targets and learn:
    the grown net must beat its frozen-atomic twin on a harder law."""
    rng = np.random.default_rng(2)
    fn = lambda X: 2 * X[:, 0] + X[:, 1] - X[:, 2] + X[:, 0] * X[:, 1]
    Xtr, ytr = _data(rng, 300, fn)
    Xho, yho = _data(rng, 60, fn)
    grown = Network(3, 8, seed=3)
    for j in range(4):
        grown.grow(j)
    flatc = Network(3, 8, seed=3)     # same seed, no growth
    for _ in range(3000):
        grown.train_step(Xtr, ytr)
        flatc.train_step(Xtr, ytr)
    mse_g = float(np.mean((grown.predict(Xho) - yho) ** 2))
    mse_f = float(np.mean((flatc.predict(Xho) - yho) ** 2))
    assert mse_g < mse_f, (mse_g, mse_f)
    # inner nets actually moved (received and used targets)
    assert any(np.abs(n.W2).sum() > 0 for n in grown.inner.values())


def test_instability_signal_reacts_to_underfit():
    """Nodes under a too-hard law should show higher instability than the
    same net converged on an easy law."""
    rng = np.random.default_rng(3)
    easy = lambda X: X[:, 0]
    hard = lambda X: np.sin(3 * X[:, 0]) * X[:, 1] - (X[:, 2] > 2) * X[:, 0]
    Xe, ye = _data(rng, 200, easy)
    Xh, yh = _data(rng, 200, hard)
    net_e = Network(3, 8, seed=4)
    net_h = Network(3, 8, seed=4)
    for _ in range(2000):
        net_e.train_step(Xe, ye)
        net_h.train_step(Xh, yh)
    assert net_h.instability().mean() > net_e.instability().mean()


if __name__ == "__main__":
    test_growth_preserves_function_exactly()
    test_atomic_learns_stage1_linear()
    test_recursive_target_handoff_trains_inner()
    test_instability_signal_reacts_to_underfit()
    print("net core tests passed")
