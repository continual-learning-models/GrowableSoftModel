"""S9.2 categorical mode for growable_attention (docs/system/6
D-G4; plan doc 7 T20 a-f). The port follows the transformer
host's proven path; the attention theory (heads, J_att, growth,
gate principle) is untouched — (e)/(f) pin that mechanically.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

from core.substrates.growable_attention import (      # noqa: E402
    GrowableAttentionSubstrate)
from core.substrates.heads import B_NEG               # noqa: E402


def _mk_cat(seed=8, heads_spec=None, vocab=("a", "b")):
    m = GrowableAttentionSubstrate(
        2, 4, mode="categorical", vocab=list(vocab),
        d_model=6, n_layers=2, seed=seed,
        heads_spec=heads_spec or [[1, 3], [2, 1]])
    m._x_mu, m._x_sd = np.zeros(2), np.ones(2)
    return m


def _ce_loss(m, Xs, onehot):
    logits = m._forward(Xs)
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    return float(-np.mean(
        np.log((probs * onehot).sum(axis=1) + 1e-12)))


def test_t20a_ce_backward_full_fd_ragged():
    """T20a: CE backward, EVERY parameter entry-by-entry vs
    central differences, ragged heads (the T5 standard applied
    to the categorical branch)."""
    m = _mk_cat()
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 2))
    y = np.array(["a", "b", "a", "a", "b"])
    onehot = np.zeros((5, 2))
    onehot[np.arange(5), [0, 1, 0, 0, 1]] = 1.0
    m0 = copy.deepcopy(m)
    lr = 1.0
    m.train_step(X, y, sgd_lr=lr)
    eps = 1e-6
    for k in m0.P:
        g_an = (m0.P[k] - m.P[k]) / lr
        W = m0.P[k]
        g_fd = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = W[i]
            W[i] = old + eps
            fp = _ce_loss(m0, X, onehot)
            W[i] = old - eps
            fm = _ce_loss(m0, X, onehot)
            W[i] = old
            g_fd[i] = (fp - fm) / (2 * eps)
            it.iternext()
        tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
        assert (np.abs(g_an - g_fd) <= tol).all(), \
            (k, np.abs(g_an - g_fd).max())
    for l, layer in enumerate(m0.heads):
        for h, HS in enumerate(layer):
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                g_an = (getattr(HS, nm)
                        - getattr(m.heads[l][h], nm)) / lr
                W = getattr(HS, nm)
                g_fd = np.zeros_like(W)
                it = np.nditer(W, flags=["multi_index"])
                while not it.finished:
                    i = it.multi_index
                    old = W[i]
                    W[i] = old + eps
                    fp = _ce_loss(m0, X, onehot)
                    W[i] = old - eps
                    fm = _ce_loss(m0, X, onehot)
                    W[i] = old
                    g_fd[i] = (fp - fm) / (2 * eps)
                    it.iternext()
                tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
                assert (np.abs(g_an - g_fd) <= tol).all(), \
                    (l, h, nm, np.abs(g_an - g_fd).max())


def test_t20b_learning_smoke_separable():
    """T20b: labels learned on a separable fixture — accuracy
    > 0.9 and the head matrices MOVE (attention participates)."""
    m = GrowableAttentionSubstrate(
        2, 4, mode="categorical", vocab=["lo", "hi"],
        d_model=8, n_layers=1, seed=5, heads_spec=[[2]])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, "hi", "lo")
    wq0 = m.heads[0][0].Wq.copy()
    for _ in range(300):
        m.train_step(X, y)
    labels, conf = m.predict_label(X)
    acc = float(np.mean(np.array(labels) == y))
    assert acc > 0.9, acc
    assert np.abs(m.heads[0][0].Wq - wq0).max() > 1e-6


def test_t20c_proba_rows_and_uniform_prior():
    """T20c: predict_proba rows sum to 1; uniform BEFORE any
    scalers exist (transformer host contract)."""
    m = GrowableAttentionSubstrate(
        2, 4, mode="categorical", vocab=["a", "b", "c"],
        d_model=6, n_layers=1, seed=3, heads_spec=[[1]])
    p0 = m.predict_proba(np.zeros((4, 2)))
    assert p0.shape == (4, 3)
    assert np.allclose(p0, 1.0 / 3.0)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 2))
    y = np.array((["a", "b", "c"] * 6)[:16])
    for _ in range(5):
        m.train_step(X, y)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(1), 1.0, atol=1e-12)
    assert (p >= 0).all()


def test_t20d_add_class_function_preserving():
    """T20d: add_class appends a zero logit column + B_NEG bias —
    old-class probabilities move only within epsilon, argmax
    unchanged, adam slots rebuilt to the new shapes."""
    m = _mk_cat(vocab=("a", "b"))
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 2))
    y = np.array((["a", "b"] * 6))
    for _ in range(30):
        m.train_step(X, y)
    p_before = m.predict_proba(X)
    m.add_class("c")
    assert m.vocab == ["a", "b", "c"]
    assert m.P["Wh"].shape[1] == 3
    assert float(m.P["bh"][-1]) == B_NEG
    p_after = m.predict_proba(X)
    assert np.abs(p_after[:, :2] - p_before).max() < 1e-3
    assert (p_after[:, :2].argmax(1) == p_before.argmax(1)).all()
    assert m._adam["Wh"][0].shape == m.P["Wh"].shape
    assert m._adam["bh"][0].shape == m.P["bh"].shape


def test_t20e_numeric_path_bit_identical_pin():
    """T20e: the NUMERIC path is bit-identical to pre-S9.2 —
    regression pin computed on the frozen pre-change tree
    (GrowableSoftModel-20260711, same numpy 2.4.6) with this
    exact fixture."""
    m = GrowableAttentionSubstrate(
        2, 4, d_model=6, n_layers=1, seed=8, heads_spec=[[1, 3]])
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 2))
    y = np.sin(X.sum(1))
    for _ in range(25):
        m.train_step(X, y)
    p = m.predict(X)
    assert float(p[0, 0]) == -0.32042419503816477
    assert float(p[7, 0]) == -0.8837641378845049
    assert float(np.abs(p).sum()) == 6.306182331565485


def test_t20f_growth_and_discipline_mode_agnostic():
    """T20f: growth + discipline are UPSTREAM of the output head:
    (1) head_add runs on a categorical model (its internal
    bitwise self-check guards exactness) and adds one head;
    (2) J_att on layer 0 is IDENTICAL for same-seed numeric and
    categorical models (heads/Wv/Bf draw before the head matrix,
    layer-0 attention never sees the output head); (3) row
    entropies stay finite/computable."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 2))
    mc = _mk_cat(seed=8, heads_spec=[[1, 3], [2, 1]])
    mn = GrowableAttentionSubstrate(
        2, 4, d_model=6, n_layers=2, seed=8,
        heads_spec=[[1, 3], [2, 1]])
    mn._x_mu, mn._x_sd = np.zeros(2), np.ones(2)
    for h in range(2):
        jc = mc.j_att_value(0, h, X)
        jn = mn.j_att_value(0, h, X)
        assert jc == jn, (h, jc, jn)          # bitwise equality
    n_before = len(mc.heads[0])
    mc.head_add(0)                            # self-checks inside
    assert len(mc.heads[0]) == n_before + 1
    ents = mc.row_entropies(0, 0, X)
    assert np.isfinite(ents).all()


def test_t20g_product_path_categorical_gate(tmp_path, monkeypatch):
    """T20g (product path): create/study/grow_attention on a
    CATEGORICAL model through the facade — the gate runs with
    the error-rate metric and returns a governed verdict; labels
    survive the holdout round-trip."""
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    from core.facade import System
    ws = System()
    rows = [{"input": {"x": float(i) / 12.0,
                       "y": float(11 - i) / 12.0},
             "target": ("hi" if i % 2 else "lo")}
            for i in range(24)]
    r = ws.create_model("t20g", holdout=rows[:8],
                        substrate="growable_attention",
                        policy={"substrate_params": {
                            "d_model": 8, "n_layers": 1,
                            "heads_spec": [[1]], "seed": 3}})
    assert "refusal" not in r, r
    ws.study("t20g", rows[8:], steps=40)
    row = ws.grow_attention("t20g", layer=0, tol=1e9)
    assert row.get("verdict") in ("accepted", "no_trigger"), row
    labels, conf = ws.lc._load_working("t20g")[0].predict_label(
        np.array([[0.2, 0.7]]))
    assert labels[0] in ("hi", "lo")
