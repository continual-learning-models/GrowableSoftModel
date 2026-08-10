"""IWP1 acceptance (UT1.1-UT1.7): MSOrgan substrate — dual heads on the
FROZEN recursive Network via subclassing only."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.substrate import MSOrgan, B_NEG


def _law(X):
    return X[:, 0] * X[:, 1] + 2 * X[:, 2]


def test_ut1_1_numeric_parity_behavior():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, (200, 3))
    y = _law(X).reshape(-1, 1)
    m = MSOrgan(3, 16, mode="numeric", seed=1)
    for _ in range(800):
        m.train_step(X, y)
    mse = float(np.mean((m.predict(X) - y) ** 2))
    assert mse < 0.01, mse                    # parent behavior reached


def test_ut1_2_categorical_learns():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2, (300, 3))
    labels = np.where(_law(X) > 3.0, "HIGH", "LOW")
    m = MSOrgan(3, 16, mode="categorical", vocab=["LOW", "HIGH"], seed=2)
    for _ in range(600):
        m.train_step(X, labels)
    pred, conf = m.predict_label(X)
    acc = float(np.mean(np.array(pred) == labels))
    assert acc >= 0.95, acc
    assert conf.min() >= 0 and conf.max() <= 1


def test_ut1_3_growth_preserves_function_both_heads():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 2, (64, 3))
    # numeric
    m = MSOrgan(3, 8, mode="numeric", seed=3)
    m.train_step(X, _law(X).reshape(-1, 1))
    before = m.predict(X).copy()
    m.grow(2)
    m.grown_body(2).grow(1)                   # depth 3, same operator
    assert np.allclose(m.predict(X), before)
    assert m.depth() == 3
    # categorical
    c = MSOrgan(3, 8, mode="categorical", vocab=["A", "B"], seed=4)
    c.train_step(X, np.where(X[:, 0] > 1, "A", "B"))
    pb = c.predict_proba(X).copy()
    c.grow(1)
    c.grown_body(1).grow(0)
    assert np.allclose(c.predict_proba(X), pb)
    assert c.depth() == 3


def test_ut1_4_vocab_growth_epsilon():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 2, (128, 3))
    m = MSOrgan(3, 12, mode="categorical", vocab=["A", "B"], seed=5)
    for _ in range(300):
        m.train_step(X, np.where(X[:, 0] > 1, "A", "B"))
    before = m.predict_proba(X).copy()
    m.add_class("C")
    after = m.predict_proba(X)
    # old-class mass preserved within documented epsilon
    assert np.abs(after[:, :2] - before).max() < 1e-3
    assert after[:, 2].max() < 1e-3            # new class ~ e^B_NEG
    assert len(m.vocab) == 3 and m.c[-1] == B_NEG


def test_ut1_5_sgd_mode_inherited():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 2, (64, 3))
    m = MSOrgan(3, 8, mode="categorical", vocab=["A", "B"], seed=6)
    for _ in range(300):
        m.train_step(X, np.where(X[:, 0] > 1, "A", "B"))
    before = m.predict_proba(X).copy()
    for _ in range(10):                        # self-anchored consolidation
        m.train_step(X, np.array(m.predict_label(X)[0]), sgd_lr=1e-3)
    assert np.abs(m.predict_proba(X) - before).max() < 0.05


def test_ut1_6_instability_across_depths():
    m = MSOrgan(3, 8, mode="categorical", vocab=["A", "B"], seed=7)
    m.grow(2)
    from reference_net.net import Network
    assert isinstance(m.grown_body(2), Network)  # body stays parent class
    assert len(m.instability()) == 8


def test_ut1_7_artifact_roundtrip():
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 2, (32, 3))
    m = MSOrgan(3, 8, mode="categorical", vocab=["A", "B"], seed=8)
    m.train_step(X, np.where(X[:, 0] > 1, "A", "B"))
    m.grow(0)
    tmp = tempfile.mkdtemp()
    try:
        m.save(tmp)
        m2 = MSOrgan.load(tmp)
        assert np.allclose(m2.predict_proba(X), m.predict_proba(X))
        assert m2.shape_record() == m.shape_record()
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut1_1_numeric_parity_behavior()
    test_ut1_2_categorical_learns()
    test_ut1_3_growth_preserves_function_both_heads()
    test_ut1_4_vocab_growth_epsilon()
    test_ut1_5_sgd_mode_inherited()
    test_ut1_6_instability_across_depths()
    test_ut1_7_artifact_roundtrip()
    print("iwp1 tests passed")
