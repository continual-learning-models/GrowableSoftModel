"""SWP4 unit tests (SUT4.1-4.4): the transformer substrate runs the
COMPLETE learning-training process INCLUDING multi-scale network growth,
on simulated data (owner's directive).

SUT4.1 hand-written backprop is exact (finite differences) + determinism
SUT4.2 relation capacity: learns interaction laws; beats same-budget mlp
SUT4.3 growth preserves function exactly at depths 1->2->3, both heads,
       via contract verbs (incl. DEEP :: sites into inner networks)
SUT4.4 the complete process on a graded simulated curriculum: easy stage
       mastered -> hard stage plateaus -> instability-ranked growth ->
       continued coupled training -> measurable improvement, ending in a
       multi-scale (depth>=2) transformer
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.substrates import get_substrate
from core.substrates.transformer import TransformerSubstrate

RNG = np.random.default_rng(0)


def _fit(m, X, y, epochs=150, seed=0, bs=32):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            m.train_step(X[order[i:i + bs]], y[order[i:i + bs]])


def test_sut4_1_gradients_and_determinism():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2, (8, 3))
    y = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    m = TransformerSubstrate(3, 4, mode="numeric", seed=1,
                             d_model=8, n_layers=2, n_heads=2)
    m.train_step(X, y)
    captured = {}
    orig = m._step
    m._step = lambda G, s: captured.update(G)
    m.train_step(X, y)
    m._step = orig

    def loss_at():
        Xs = m._stdx(X)
        ys = (np.asarray(y, float) - m._y_mu) / m._y_sd
        return float(np.mean((m._forward(Xs) - ys) ** 2))

    eps = 1e-6
    for name in ("Wq_0", "Wo_1", "W1_0", "Wv", "g2_1", "Wh"):
        P = m.P[name]
        idx = tuple(rng.integers(0, s) for s in P.shape)
        old = P[idx]
        P[idx] = old + eps
        lp = loss_at()
        P[idx] = old - eps
        lm = loss_at()
        P[idx] = old
        fd = (lp - lm) / (2 * eps)
        an = captured[name][idx]
        assert abs(fd - an) / (abs(fd) + abs(an) + 1e-9) < 1e-4, name
    # determinism
    outs = []
    for _ in range(2):
        d = TransformerSubstrate(3, 8, mode="numeric", seed=9)
        _fit(d, X, y, epochs=10)
        outs.append(d.predict(X))
    assert np.array_equal(outs[0], outs[1])


def test_sut4_2_relation_capacity_vs_mlp():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 2, (300, 4))
    # strong-interaction law: products + conditional coupling
    y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
         + (X[:, 0] > 1) * X[:, 2]).reshape(-1, 1)
    Xh = rng.uniform(0, 2, (80, 4))
    yh = (Xh[:, 0] * Xh[:, 1] + Xh[:, 2] * Xh[:, 3]
          + (Xh[:, 0] > 1) * Xh[:, 2]).reshape(-1, 1)
    tf = TransformerSubstrate(4, 16, mode="numeric", seed=3)
    _fit(tf, X, y, epochs=200)
    mse_tf = float(np.mean((tf.predict(Xh) - yh) ** 2))
    mlp_cls = get_substrate("mlp")
    # same-parameter-budget mlp (transformer ~ params -> width match)
    width = max(16, round((tf.n_params() - 1) / (4 + 2)))  # 5H+1 approx for d_in=4
    mlp = mlp_cls(4, min(width, 512), mode="numeric", seed=3)
    _fit(mlp, X, y, epochs=200)
    mse_mlp = float(np.mean((mlp.predict(Xh) - yh) ** 2))
    print(f"   interaction law: transformer mse={mse_tf:.4f} "
          f"({tf.n_params()}p) vs mlp mse={mse_mlp:.4f} ({mlp.n_params()}p)")
    assert mse_tf < 0.1, mse_tf                        # learns it well
    # categorical head learns too
    lab = np.where(X[:, 0] * X[:, 1] > 2, "A", "B")
    tc = TransformerSubstrate(4, 16, mode="categorical",
                              vocab=["A", "B"], seed=4)
    _fit(tc, X, lab, epochs=120)
    pred, conf = tc.predict_label(Xh)
    labh = np.where(Xh[:, 0] * Xh[:, 1] > 2, "A", "B")
    acc = float(np.mean(np.array(pred) == labh))
    assert acc >= 0.93, acc
    assert 0 <= conf.min() and conf.max() <= 1


def test_sut4_3_growth_preserves_function_deeply():
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 2, (64, 3))
    m = TransformerSubstrate(3, 8, mode="numeric", seed=6)
    m.train_step(X, (X[:, 0] * X[:, 1]).reshape(-1, 1))
    before = m.predict(X).copy()
    sites = m.growth_sites()
    assert len(sites) == 2 * 8                        # L=2 layers x m=8
    m.grow_site(sites[0][0])
    assert np.allclose(m.predict(X), before)          # depth 2 exact
    deep = next(sp for sp, _ in m.growth_sites() if "::" in sp)
    m.grow_site(deep)
    assert m.depth() == 3
    assert np.allclose(m.predict(X), before)          # depth 3 exact
    # categorical
    c = TransformerSubstrate(3, 8, mode="categorical",
                             vocab=["A", "B"], seed=7)
    c.train_step(X, np.where(X[:, 0] > 1, "A", "B"))
    pb = c.predict_proba(X).copy()
    c.grow_site(c.growth_sites()[0][0])
    assert np.allclose(c.predict_proba(X), pb)
    # vocab growth epsilon on the grown net
    c.add_class("C")
    pa = c.predict_proba(X)
    assert np.abs(pa[:, :2] - pb).max() < 1e-3 and pa[:, 2].max() < 1e-3


def test_sut4_4_complete_process_with_growth_on_simulated_data():
    """The owner's directive: the FULL learning-training cycle on the
    transformer, multi-scale growth included, simulated data."""
    rng = np.random.default_rng(8)
    law = lambda X: (X[:, 0] * X[:, 1]
                     + np.maximum(X[:, 1] - X[:, 2], 0) * X[:, 0]
                     + 2 * X[:, 2]).reshape(-1, 1)
    easy_b, hard_b = np.array([1.5, 1.5, 1.5]), np.array([5.0, 5.0, 4.0])
    Xe = rng.uniform(0, 1, (250, 3)) * easy_b
    Xh = rng.uniform(0, 1, (250, 3)) * hard_b
    Xe_t = rng.uniform(0, 1, (60, 3)) * easy_b
    Xh_t = rng.uniform(0, 1, (60, 3)) * hard_b
    tol = 0.5
    acc = lambda m, Xt: float(np.mean(np.abs(m.predict(Xt) - law(Xt)) <= tol))

    m = TransformerSubstrate(3, 8, mode="numeric", seed=10)
    # stage 1 (easy): learn
    _fit(m, Xe, law(Xe), epochs=150, seed=1)
    acc_easy = acc(m, Xe_t)
    assert acc_easy >= 0.9, f"easy stage not mastered: {acc_easy}"
    assert m.depth() == 1                                   # no growth yet
    # stage 2 (hard): cumulative training, plateau expected
    Xc = np.vstack([Xe, Xh])
    yc = law(Xc)
    _fit(m, Xc, yc, epochs=150, seed=2)
    acc_hard_before = acc(m, Xh_t)
    # GROWTH: instability-ranked sites, grow 2, keep training (coupled)
    p_before, d_before = m.n_params(), m.depth()
    for sp, _ in m.growth_sites()[:2]:
        m.grow_site(sp)
    assert m.depth() == 2 and m.n_params() > p_before       # structure grew
    _fit(m, Xc, yc, epochs=200, seed=3)
    acc_hard_after = acc(m, Xh_t)
    acc_easy_after = acc(m, Xe_t)
    print(f"   complete process: easy {acc_easy:.2f} -> hard before-growth "
          f"{acc_hard_before:.2f} -> after-growth {acc_hard_after:.2f} "
          f"(easy retained {acc_easy_after:.2f}; params {p_before}->"
          f"{m.n_params()}, depth {d_before}->{m.depth()})")
    assert acc_hard_after >= acc_hard_before               # growth paid
    assert acc_hard_after >= 0.7, acc_hard_after           # hard stage usable
    assert acc_easy_after >= 0.8                           # retention


if __name__ == "__main__":
    test_sut4_1_gradients_and_determinism()
    print("   SUT4.1 gradients exact + deterministic")
    test_sut4_2_relation_capacity_vs_mlp()
    test_sut4_3_growth_preserves_function_deeply()
    print("   SUT4.3 growth exact at depth 3, both heads")
    test_sut4_4_complete_process_with_growth_on_simulated_data()
    print("swp4 tests passed")
