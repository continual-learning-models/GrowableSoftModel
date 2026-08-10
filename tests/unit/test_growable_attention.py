"""growable_attention build tests (DEV_PLAN v1.5 D3; S1 delivers
T1 + T14). T-numbers are the plan's — AUTHORITATIVE table there.

T1  parity vs the packed transformer host on the same RAW weights
    (vector AND causal) — the independent forward reference: FD
    cannot catch a self-consistent wrong forward.
T14 artifact round-trip through the REGISTRY entry, ragged widths,
    per-head d_h census asserted (round-trip equality alone is
    vacuous for a missing field).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

from core.substrates import REGISTRY, GUIDANCE, get_substrate  # noqa: E402
from core.substrates.growable_attention import (               # noqa: E402
    POLICY, GrowableAttentionSubstrate, attention_grow_event)
from core.substrates.sequence import SequenceSubstrate         # noqa: E402
from core.substrates.transformer import TransformerSubstrate   # noqa: E402


def _transfer_packed_weights(ga, tr):
    """Copy the packed host's RAW weights into the per-head layout:
    slice each d x d matrix into h columns blocks of width d_h;
    W~q absorbs 1/sqrt(d_h) (the harness transfers RAW weights and
    lets the head absorb — plan T1)."""
    d, hh = tr.d, tr.h
    dh = d // hh
    for k in ("Wv", "Bf", "Wh", "bh"):
        ga.P[k] = np.array(tr._bk.to_numpy(tr.P[k]), copy=True)
    for l in range(tr.L):
        for k in ("g1", "b1n", "g2", "b2n", "W1", "b1", "W2", "b2"):
            ga.P[f"{k}_{l}"] = np.array(
                tr._bk.to_numpy(tr.P[f"{k}_{l}"]), copy=True)
        Wq = tr._bk.to_numpy(tr.P[f"Wq_{l}"])
        Wk = tr._bk.to_numpy(tr.P[f"Wk_{l}"])
        Wv = tr._bk.to_numpy(tr.P[f"Wk2_{l}"])    # packed V is Wk2
        Wo = tr._bk.to_numpy(tr.P[f"Wo_{l}"])
        for h, HS in enumerate(ga.heads[l]):
            sl = slice(h * dh, (h + 1) * dh)
            HS.Wq = Wq[:, sl] / np.sqrt(dh)       # absorb at birth
            HS.Wk = np.array(Wk[:, sl], copy=True)
            HS.Wv = np.array(Wv[:, sl], copy=True)
            HS.Wo = np.array(Wo[sl, :], copy=True)


def test_t1_parity_vector():
    tr = TransformerSubstrate(3, 8, d_model=8, n_layers=2, seed=11)
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])              # uniform = packed
    _transfer_packed_weights(ga, tr)
    X = np.random.default_rng(0).normal(size=(6, 3))
    out_tr = tr._bk.to_numpy(tr._forward(tr._bk.ingest(X)))
    out_ga = ga._forward(X)
    assert np.abs(out_tr - out_ga).max() <= 1e-12, \
        np.abs(out_tr - out_ga).max()


def test_t1_parity_causal():
    tr = SequenceSubstrate(2, 8, d_model=8, n_layers=2, seed=13)
    ga = GrowableAttentionSubstrate(
        2, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]], causal=True)
    _transfer_packed_weights(ga, tr)
    X = np.random.default_rng(1).normal(size=(5, 7, 2))  # (n,T,f)
    out_tr = tr._bk.to_numpy(tr._forward(tr._bk.ingest(X)))
    out_ga = ga._forward(X)
    assert np.abs(out_tr - out_ga).max() <= 1e-12, \
        np.abs(out_tr - out_ga).max()


def test_t14_artifact_roundtrip_ragged(tmp_path):
    cls = get_substrate("growable_attention")     # via the REGISTRY
    assert cls is GrowableAttentionSubstrate
    assert "growable_attention" in GUIDANCE       # service exposure
    m = cls(3, 8, d_model=8, n_layers=2, seed=5,
            heads_spec=[[1, 3], [2, 1]])          # ragged widths
    # give it serving state so predict is nontrivial
    m._fit_x_scalers(np.random.default_rng(2).normal(size=(20, 3)))
    m._y_mu, m._y_sd = 1.5, 2.0
    X = np.random.default_rng(3).normal(size=(6, 3))
    before = m.predict(X)
    rec = m.shape_record()
    assert rec["heads"] == [[1, 3], [2, 1]]       # per-head census
    m.save(tmp_path)
    m2 = REGISTRY["growable_attention"].load(tmp_path)
    after = m2.predict(X)
    assert np.array_equal(before, after)          # bit-identical
    assert m2.shape_record() == rec
    assert m2.heads[0][1].d_h == 3 and m2.heads[1][0].d_h == 2


def test_t14_mode_boundary_refusals():
    """Mode-scope guard — CONSCIOUSLY UPDATED at S9.2 (docs/system/6
    D-G4 lifts the v1 numeric-only boundary BY DESIGN; the census-pin
    precedent: boundary tests move WITH the designed boundary, on the
    record). New boundary: numeric + categorical serve; numeric_dist
    and junk modes refuse loudly."""
    import pytest
    m = GrowableAttentionSubstrate(3, 8, mode="categorical",
                                   vocab=["a", "b"])   # now serves
    assert m.mode == "categorical" and m.P["Wh"].shape[1] == 2
    # 60A boundary change (owner-approved): numeric_dist now
    # SERVES (zero-born heteroscedastic head); junk modes refuse
    # via the L3 whitelist assertion (ValueError)
    md = GrowableAttentionSubstrate(3, 8, mode="numeric_dist")
    assert md.P["Wh"].shape[1] == 2
    assert np.all(np.asarray(md.P["Wh"]) == 0.0)
    with pytest.raises(ValueError):
        GrowableAttentionSubstrate(3, 8, mode="nonsense")


# ---------------------------------------------------------- S2 tests

def _fd_loss(m, X, ys):
    """Scalar training loss exactly as train_step computes it,
    parameters read fresh each call (for finite differences)."""
    logits = m._forward(X)
    return float(((logits - ys) ** 2).mean())


def test_t5_full_backward_fd_ragged():
    """T5: every parameter entry-by-entry vs central differences,
    ragged widths, BOTH vector and causal paths (plan D3)."""
    for causal in (False, True):
        m = GrowableAttentionSubstrate(
            2, 4, d_model=6, n_layers=2, seed=8,
            heads_spec=[[1, 3], [2, 1]], causal=causal)
        rng = np.random.default_rng(7)
        X = (rng.normal(size=(4, 5, 2)) if causal
             else rng.normal(size=(5, 2)))
        n = len(X)
        ys = rng.normal(size=(n, 1))
        # capture analytic grads via one SGD step of known lr
        import copy
        m0 = copy.deepcopy(m)
        m._x_mu, m._x_sd = (np.zeros(2), np.ones(2))
        m._y_mu, m._y_sd = 0.0, 1.0
        m0._x_mu, m0._x_sd = m._x_mu, m._x_sd
        m0._y_mu, m0._y_sd = 0.0, 1.0
        lr = 1.0
        m.train_step(X, ys.ravel(), sgd_lr=lr)
        eps = 1e-6
        worst = 0.0
        # host params
        for k in m0.P:
            g_an = (m0.P[k] - m.P[k]) / lr
            W = m0.P[k]
            g_fd = np.zeros_like(W)
            it = np.nditer(W, flags=["multi_index"])
            while not it.finished:
                i = it.multi_index
                old = W[i]
                W[i] = old + eps
                fp = _fd_loss(m0, np.asarray(X, float), ys)
                W[i] = old - eps
                fm = _fd_loss(m0, np.asarray(X, float), ys)
                W[i] = old
                g_fd[i] = (fp - fm) / (2 * eps)
                it.iternext()
            tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
            assert (np.abs(g_an - g_fd) <= tol).all(), \
                (causal, k, np.abs(g_an - g_fd).max())
            worst = max(worst, float(np.abs(g_an - g_fd).max()))
        # head params
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
                        fp = _fd_loss(m0, np.asarray(X, float), ys)
                        W[i] = old - eps
                        fm = _fd_loss(m0, np.asarray(X, float), ys)
                        W[i] = old
                        g_fd[i] = (fp - fm) / (2 * eps)
                        it.iternext()
                    tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
                    assert (np.abs(g_an - g_fd) <= tol).all(), \
                        (causal, l, h, nm,
                         np.abs(g_an - g_fd).max())


def test_t8a_dtheta_assembly_order():
    """T8a: the u_h EMA update ingests the flattened concatenation
    in the [Wq, Wk, Wv, Wo] ORDER (the NEW logic; plan D3):
    scripted step with known distinct per-matrix deltas."""
    m = GrowableAttentionSubstrate(2, 4, d_model=4, n_layers=1,
                                   seed=3, heads_spec=[[2]])
    HS = m.heads[0][0]
    # scripted: overwrite matrices by known deltas, replay the EMA
    deltas = {"Wq": 1.0, "Wk": 2.0, "Wv": 3.0, "Wo": 4.0}
    old = {nm: getattr(HS, nm).copy() for nm in deltas}
    for nm, v in deltas.items():
        setattr(HS, nm, old[nm] + v)
    dtheta = np.concatenate([
        (getattr(HS, nm) - old[nm]).ravel()
        for nm in ("Wq", "Wk", "Wv", "Wo")])
    sizes = [old[nm].size for nm in ("Wq", "Wk", "Wv", "Wo")]
    ofs = np.cumsum([0] + sizes)
    for i, nm in enumerate(("Wq", "Wk", "Wv", "Wo")):
        seg = dtheta[ofs[i]:ofs[i + 1]]
        assert np.allclose(seg, deltas[nm]), (nm, seg[:3])
    # and u_h stays in [0, 1] through real training
    m2 = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                    seed=5, heads_spec=[[1, 2]])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 2))
    yv = X[:, 0] - 2 * X[:, 1]
    for _ in range(30):
        m2.train_step(X, yv)
    for u in m2.u_stats(0):          # DD-7: per-matrix storage;
        assert 0.0 <= u <= 1.0, u    # values identical (doc 32)


def test_t15_learning_smoke():
    """T15: loss decreases; head matrices actually change (pins the
    two-pass Adam write-back); REQUIREMENTS A1 'trainable
    immediately after' end-to-end."""
    m = GrowableAttentionSubstrate(3, 8, d_model=8, n_layers=2,
                                   seed=4, heads_spec=[[1, 3], [2, 1]])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3))
    yv = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    before = {(l, h, nm): getattr(HS, nm).copy()
              for l, layer in enumerate(m.heads)
              for h, HS in enumerate(layer)
              for nm in ("Wq", "Wk", "Wv", "Wo")}
    losses = [m.train_step(X, yv) for _ in range(60)]
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
    moved = sum(
        not np.array_equal(before[(l, h, nm)],
                           getattr(m.heads[l][h], nm))
        for l, layer in enumerate(m.heads)
        for h, HS in enumerate(layer)
        for nm in ("Wq", "Wk", "Wv", "Wo"))
    assert moved == len(before), f"only {moved}/{len(before)} moved"
    # predictions track the law
    pred = m.predict(X)
    mse = float(((pred.ravel() - yv) ** 2).mean())
    assert mse < 0.5, mse


# ---------------------------------------------------------- S3 tests

def _mk_trained(seed=4, heads=[[1, 3], [2, 1]]):
    m = GrowableAttentionSubstrate(3, 8, d_model=8,
                                   n_layers=len(heads),
                                   seed=seed, heads_spec=heads)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3))
    yv = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    for _ in range(15):
        m.train_step(X, yv)
    return m, X, yv


def test_t2_head_add():
    """T2a bitwise preservation + slots/birth_t; T2b step-1
    two-step pattern on the REAL substrate; T2c RED-ARM: an
    unwired head must FAIL (b) — proving (a)+(b) distinguish a
    live head from an unwired one (plan D3)."""
    m, X, yv = _mk_trained()
    Xp = np.random.default_rng(9).normal(size=(6, 3))
    before = m.predict(Xp)
    ev = m.head_add(0)
    # (a) bitwise + metadata
    assert np.array_equal(before, m.predict(Xp))
    hnew = ev["head"]
    HS = m.heads[0][hnew]
    assert HS.birth_t == m._t_att
    for nm in ("Wq", "Wk", "Wv", "Wo"):
        mm, vv = m._adam_h[(0, hnew, nm)]
        assert not mm.any() and not vv.any()
    # (b) step-1: Wo moves, all three generators exactly unmoved
    gen0 = {nm: getattr(HS, nm).copy() for nm in ("Wq", "Wk", "Wv")}
    wo0 = HS.Wo.copy()
    m.train_step(X, yv, sgd_lr=0.05)
    assert not np.array_equal(wo0, HS.Wo), "W_o must move at step 1"
    for nm, W0 in gen0.items():
        assert np.array_equal(W0, getattr(HS, nm)), \
            f"{nm} must be exactly unmoved at step 1"
    # generators live from step 2 (partner lifted off)
    m.train_step(X, yv, sgd_lr=0.05)
    moved2 = any(not np.array_equal(gen0[nm], getattr(HS, nm))
                 for nm in gen0)
    assert moved2, "generators must be live from step 2"
    # (c) RED-ARM: an UNWIRED head (not in the forward loop) must
    # fail the liveness half of (b)
    m2, X2, yv2 = _mk_trained(seed=6)
    ev2 = m2.head_add(0)
    hs2 = m2.heads[0][ev2["head"]]
    ghost = m2.heads[0].pop()          # deliberately unwire it
    for nm in ("Wq", "Wk", "Wv", "Wo"):      # orphan it fully
        m2._adam_h.pop((0, ev2["head"], nm))
    wo_ghost = ghost.Wo.copy()
    m2.train_step(X2, yv2, sgd_lr=0.05)
    assert np.array_equal(wo_ghost, ghost.Wo), \
        "red-arm: the unwired head's Wo must NOT move — liveness " \
        "assertion (b) would catch exactly this fault"


def test_t3_head_widen():
    """T3a preservation <=1e-12 + slots/EMAs extended zeroed +
    d_h_birth unchanged + uniform absorbed raw scale; T3b seeded
    step-1 pattern: new Wk cols / Wo rows move at step 1, new
    Wq/Wv cols exactly unmoved, then move at step 2."""
    m, X, yv = _mk_trained()
    HS = m.heads[1][0]                 # width 2 head
    dh0, birth0 = HS.d_h, HS.d_h_birth
    Xp = np.random.default_rng(9).normal(size=(6, 3))
    before = m.predict(Xp)
    q_old = HS.Wq.copy()
    m.head_widen(1, 0, m=2)
    # (a)
    assert np.abs(before - m.predict(Xp)).max() <= 1e-12
    assert HS.d_h == dh0 + 2 and HS.d_h_birth == birth0
    for nm in ("Wq", "Wk", "Wv", "Wo"):
        mm, vv = m._adam_h[(1, 0, nm)]
        assert mm.shape == getattr(HS, nm).shape
    P_h = 8 * 3 * HS.d_h + HS.d_h * 8
    assert sum(v.size for v in HS.mu.values()) == P_h
    assert sum(v.size for v in HS.nu.values()) == P_h
    # uniform absorbed scale: undoing the birth normalizer leaves
    # old and new Wq columns at the same raw scale
    raw = HS.Wq * np.sqrt(birth0)
    ratio = raw[:, dh0:].std() / raw[:, :dh0].std()
    assert 0.2 < ratio < 5.0, ratio
    # (b) step-1 pattern
    k0 = HS.Wk[:, dh0:].copy(); o0 = HS.Wo[dh0:, :].copy()
    qn0 = HS.Wq[:, dh0:].copy(); vn0 = HS.Wv[:, dh0:].copy()
    m.train_step(X, yv, sgd_lr=0.05)
    assert not np.array_equal(k0, HS.Wk[:, dh0:]), \
        "new W_k cols must move at step 1"
    assert not np.array_equal(o0, HS.Wo[dh0:, :]), \
        "new W_o rows must move at step 1"
    assert np.array_equal(qn0, HS.Wq[:, dh0:]), \
        "new Wq cols must be exactly unmoved at step 1"
    assert np.array_equal(vn0, HS.Wv[:, dh0:]), \
        "new Wv cols must be exactly unmoved at step 1"
    m.train_step(X, yv, sgd_lr=0.05)
    assert (not np.array_equal(qn0, HS.Wq[:, dh0:])
            or not np.array_equal(vn0, HS.Wv[:, dh0:])), \
        "seeded sides must be live from step 2"


def test_t4_saddle_red_arm():
    """T4: an all-zero widen attempt must RAISE; with the guard
    removed (monkeypatched zero seeds), the step-1 gradient on
    every new dim is exactly 0 — the guard guards a real saddle."""
    import pytest
    m, X, yv = _mk_trained()
    HS = m.heads[0][1]
    dh0 = HS.d_h
    # guard fires: force zero seeds by monkeypatching the rng
    class ZeroRng:
        def normal(self, *a, **k):
            size = k.get("size") or (a[2] if len(a) > 2 else None)
            return np.zeros(size)
    import core.substrates.growable_attention as gam
    real_rng = np.random.default_rng
    try:
        np.random.default_rng = lambda seed=None: ZeroRng()
        with pytest.raises(AssertionError):
            m.head_widen(0, 1, m=1)
    finally:
        np.random.default_rng = real_rng
    # guard removed: hand-build the all-zero widen, prove the saddle
    m2, X2, yv2 = _mk_trained(seed=6)
    HS2 = m2.heads[0][1]
    d = m2.d; dh = HS2.d_h
    HS2.Wq = np.hstack([HS2.Wq, np.zeros((d, 1))])
    HS2.Wv = np.hstack([HS2.Wv, np.zeros((d, 1))])
    HS2.Wk = np.hstack([HS2.Wk, np.zeros((d, 1))])
    HS2.Wo = np.vstack([HS2.Wo, np.zeros((1, d))])
    m2._sync_head_slots()
    P_h = d * 3 * HS2.d_h + HS2.d_h * d
    HS2.mu = {nm: np.zeros_like(getattr(HS2, nm))
               for nm in ("Wq", "Wk", "Wv", "Wo")}
    HS2.nu = {nm: np.zeros_like(getattr(HS2, nm)) + 1e-12
               for nm in ("Wq", "Wk", "Wv", "Wo")}
    news = {nm: getattr(HS2, nm).copy() for nm in
            ("Wq", "Wk", "Wv", "Wo")}
    m2.train_step(X2, yv2, sgd_lr=0.05)
    assert np.array_equal(news["Wq"][:, dh:], HS2.Wq[:, dh:])
    assert np.array_equal(news["Wk"][:, dh:], HS2.Wk[:, dh:])
    assert np.array_equal(news["Wv"][:, dh:], HS2.Wv[:, dh:])
    assert np.array_equal(news["Wo"][dh:, :], HS2.Wo[dh:, :]), \
        "all-zero widen: every new dim frozen — the certified " \
        "TOTAL SADDLE the guard exists for"


def test_t8b_buffers_extend_zeroed_on_widen():
    """T8b: mu/nu extend zeroed on widen; u stays in [0,1];
    newborn u = 1.0 (the EMA artifact the age exclusion exists
    for)."""
    m, X, yv = _mk_trained()
    HS = m.heads[0][1]
    dh0 = HS.d_h
    mu_before = {nm: HS.mu[nm].copy() for nm in ("Wq", "Wk", "Wv", "Wo")}
    m.head_widen(0, 1, m=1)
    # POSITION-CORRECT extension (owner fix ruling 2026-07-21,
    # doc 34 5): old values stay at their parameter coordinates,
    # new columns/rows enter zeroed
    for nm in ("Wq", "Wk", "Wv"):
        new_m = HS.mu[nm]
        assert np.array_equal(new_m[:, :dh0], mu_before[nm])
        assert not new_m[:, dh0:].any()
    new_o = HS.mu["Wo"]
    assert np.array_equal(new_o[:dh0, :], mu_before["Wo"])
    assert not new_o[dh0:, :].any()
    # newborn head: u = 1.0 by construction
    ev = m.head_add(1)
    u = m.u_stats(1)[ev["head"]]
    assert abs(u - 1.0) < 1e-9


# ---------------------------------------------------------- S4 tests

def _selfproc_env(policy_over):
    """Context: temporarily override POLICY keys."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        old = {k: POLICY[k] for k in policy_over}
        POLICY.update(policy_over)
        try:
            yield
        finally:
            POLICY.update(old)
    return ctx()


def test_t6a_fd_with_injection():
    """T6a: FD of task + lambda*J_att on Wq/Wk — vector, causal,
    AND with the upper hinge BINDING (cert S2's three configs)."""
    configs = [
        (False, 0.2, 1.0),          # vector, lower hinge
        (True, 0.2, 1.0),           # causal, lower hinge
        (False, 0.01, 0.10),        # upper hinge BINDING
    ]
    for causal, lo, hi in configs:
        with _selfproc_env({"att_warmup": 0, "att_head_age_min": 0,
                            "att_h_lo": lo, "att_h_hi": hi}):
            m = GrowableAttentionSubstrate(
                2, 4, d_model=6, n_layers=2, seed=8,
                heads_spec=[[1, 3], [2, 1]], causal=causal)
            m._selfproc_on = True
            rng = np.random.default_rng(7)
            X = (rng.normal(size=(3, 5, 2)) if causal
                 else rng.normal(size=(5, 2)))
            ys = rng.normal(size=(len(X), 1))
            m._x_mu, m._x_sd = np.zeros(2), np.ones(2)
            m._y_mu, m._y_sd = 0.0, 1.0
            import copy
            m0 = copy.deepcopy(m)

            def head_loss(l_ref, h_ref):
                """SPLIT-semantics reference (COMPUTE C3): task
                (full) + lambda * THIS head's own J_att. The
                cross-layer path (this param -> T -> downstream
                layers' A -> their J_att) is EXCLUDED by design —
                the split stops J_att at the head's own scoring
                pair; cert S2 likewise holds Tn fixed."""
                task = _fd_loss(m0, np.asarray(X, float), ys)
                Xs = m0._stdx(np.asarray(X, float))
                _, _, caches = m0._forward(Xs, cache=True)
                A = caches[1 + l_ref][3][h_ref][3]
                n, F, _ = A.shape
                F_i = (np.arange(1, F + 1) if causal
                       else np.full(F, F))
                tot, cnt = 0.0, 0
                for b in range(n):
                    for i in range(F):
                        if F_i[i] < 2:
                            continue
                        pp = A[b, i, :F_i[i]]
                        pc = np.maximum(pp, 1e-300)
                        Hi = float(-(pp * np.log(pc)).sum())
                        l_b = POLICY["att_h_lo"] * np.log(F_i[i])
                        h_b = POLICY["att_h_hi"] * np.log(F_i[i])
                        tot += max(0.0, Hi - h_b) ** 2 \
                            + max(0.0, l_b - Hi) ** 2
                        cnt += 1
                return task + POLICY["att_lambda"] * tot \
                    / max(cnt, 1)

            lr = 1.0
            m.train_step(X, ys.ravel(), sgd_lr=lr)
            eps = 1e-6
            for l, layer in enumerate(m0.heads):
                for h, HS in enumerate(layer):
                    for nm in ("Wq", "Wk"):
                        g_an = (getattr(HS, nm)
                                - getattr(m.heads[l][h], nm)) / lr
                        W = getattr(HS, nm)
                        g_fd = np.zeros_like(W)
                        it = np.nditer(W, flags=["multi_index"])
                        while not it.finished:
                            i = it.multi_index
                            old = W[i]
                            W[i] = old + eps
                            fp = head_loss(l, h)
                            W[i] = old - eps
                            fm = head_loss(l, h)
                            W[i] = old
                            g_fd[i] = (fp - fm) / (2 * eps)
                            it.iternext()
                        tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
                        assert (np.abs(g_an - g_fd) <= tol).all(), \
                            (causal, lo, hi, l, h, nm,
                             np.abs(g_an - g_fd).max())


def test_t6b_locality_with_positive_control():
    """T6b: task loss zeroed, J_att active — every non-attention
    grad, the same head's Wv/Wo, and EVERY lower-layer parameter
    exactly 0; WITH the positive control that gWq/gWk are NONZERO
    in the same run."""
    with _selfproc_env({"att_warmup": 0, "att_head_age_min": 0,
                        "att_h_lo": 0.6, "att_h_hi": 1.0}):
        m = GrowableAttentionSubstrate(
            2, 4, d_model=6, n_layers=2, seed=8,
            heads_spec=[[2], [2]])
        m._selfproc_on = True
        rng = np.random.default_rng(7)
        X = rng.normal(size=(5, 2))
        m._x_mu, m._x_sd = np.zeros(2), np.ones(2)
        m._y_mu, m._y_sd = 0.0, 1.0
        # zero task: predict the model's own output
        ys = m._forward(m._stdx(X))
        import copy
        m0 = copy.deepcopy(m)
        lr = 1.0
        m.train_step(X, ys.ravel(), sgd_lr=lr)
        # host params: all exactly unmoved
        for k in m0.P:
            assert np.array_equal(m0.P[k], m.P[k]), \
                f"J_att leaked into host param {k}"
        moved_q = moved_k = 0
        for l, layer in enumerate(m0.heads):
            for h, HS in enumerate(layer):
                assert np.array_equal(HS.Wv, m.heads[l][h].Wv), \
                    f"leak into Wv ({l},{h})"
                assert np.array_equal(HS.Wo, m.heads[l][h].Wo), \
                    f"leak into Wo ({l},{h})"
                moved_q += not np.array_equal(
                    HS.Wq, m.heads[l][h].Wq)
                moved_k += not np.array_equal(
                    HS.Wk, m.heads[l][h].Wk)
        assert moved_q > 0 and moved_k > 0, \
            "positive control: an always-off J_att must not pass"


def test_t6c_red_arm_split_violation():
    """T6c RED-ARM: route dS_total into dTn (the exact critical
    defect the ALGO review caught) — locality must then FAIL,
    proving T6b guards something real."""
    with _selfproc_env({"att_warmup": 0, "att_head_age_min": 0,
                        "att_h_lo": 0.6, "att_h_hi": 1.0}):
        m = GrowableAttentionSubstrate(
            2, 4, d_model=6, n_layers=2, seed=8,
            heads_spec=[[2], [2]])
        m._selfproc_on = True
        rng = np.random.default_rng(7)
        X = rng.normal(size=(5, 2))
        m._x_mu, m._x_sd = np.zeros(2), np.ones(2)
        m._y_mu, m._y_sd = 0.0, 1.0
        ys = m._forward(m._stdx(X))
        import copy
        m0 = copy.deepcopy(m)
        # monkeypatch: violate the split inside train_step by
        # re-running the backward with dTn built from dS_total.
        # Simplest faithful fault: wrap _datt_dS so its output is
        # ALSO added into dVh path? No — inject by patching the
        # method that computes dS_task-based dTn is inline; so we
        # emulate the violation at the MATH level: a hand-rolled
        # step where dTn uses dS_total.
        lam = POLICY["att_lambda"]
        Xs = m0._stdx(np.asarray(X, float))
        logits, Pool, caches = m0._forward(Xs, cache=True)
        n = len(Xs)
        dlog = 2 * (logits - ys) / n            # == 0 (zero task)
        dPool = dlog @ m0.P["Wh"].T
        Fn = Xs.shape[1]
        dT = np.repeat(dPool[:, None, :], Fn, axis=1) / Fn
        leaked = {}
        from engine.primitives import gelu_d as _gd
        d = m0.d
        for l in range(m0.L - 1, -1, -1):
            (T, Tn, c1, att_cache_l, T1, Tn2, c2,
             pre, H, inner_in) = caches[1 + l]
            dF = dT
            dH = dF @ m0.P[f"W2_{l}"].T
            dpre = dH * _gd(pre)
            dTn2 = dpre @ m0.P[f"W1_{l}"].T
            from engine.primitives import ln_bwd as _lb
            dT1, dg2, db2 = _lb(dTn2, c2, m0.P[f"g2_{l}"])
            dT1 = dT1 + dT
            dO = dT1
            dTn = np.zeros_like(Tn)
            for h, HS in enumerate(m0.heads[l]):
                (Qh, Kh, Vh, A, Oh) = att_cache_l[h]
                dOh = dO @ HS.Wo.T
                dA = dOh @ np.transpose(Vh, (0, 2, 1))
                dVh = np.transpose(A, (0, 2, 1)) @ dOh
                dS_task = A * (dA - (dA * A).sum(-1, keepdims=True))
                dS_total = dS_task + lam * m0._datt_dS(A)
                dQh = dS_total @ Kh
                dKh = np.transpose(dS_total, (0, 2, 1)) @ Qh
                # THE VIOLATION: dTn from dS_total
                dTn += dQh @ HS.Wq.T + dKh @ HS.Wk.T \
                    + dVh @ HS.Wv.T
            dT0, dg1, db1 = _lb(dTn, c1, m0.P[f"g1_{l}"])
            leaked[f"g1_{l}"] = np.abs(dg1).max()
            dT = dT0 + dT1
        emb_leak = np.abs(np.einsum("nf,nfd->fd",
                                    caches[0][1], dT)).max()
        assert max(max(leaked.values()), emb_leak) > 1e-12, \
            "red-arm must show the leak the split prevents"


def test_t7_gates_with_paired_positive():
    """T7: no injection before att_warmup; none on heads younger
    than att_head_age_min; none outside att_selfproc_heads; AND a
    head past both clocks inside the set DOES receive injection."""
    def q_moved_under_zero_task(m):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(5, 2))
        m._x_mu, m._x_sd = np.zeros(2), np.ones(2)
        m._y_mu, m._y_sd = 0.0, 1.0
        ys = m._forward(m._stdx(X))
        import copy
        m0 = copy.deepcopy(m)
        m.train_step(X, ys.ravel(), sgd_lr=1.0)
        return [not np.array_equal(HS.Wq, m.heads[l][h].Wq)
                for l, layer in enumerate(m0.heads)
                for h, HS in enumerate(layer)]

    # (1) warmup gate
    with _selfproc_env({"att_warmup": 1000, "att_head_age_min": 0,
                        "att_h_lo": 0.6}):
        m = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                       seed=8, heads_spec=[[2, 2]])
        m._selfproc_on = True
        assert not any(q_moved_under_zero_task(m)), \
            "no injection before warmup"
    # (2) age gate: young head silent, old head active
    with _selfproc_env({"att_warmup": 0, "att_head_age_min": 50,
                        "att_h_lo": 0.0, "att_h_hi": 0.10}):
        m = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                       seed=8, heads_spec=[[2]])
        m._selfproc_on = True
        m._t_att = 100                    # aged host + aged head 0
        m.head_add(0)                     # newborn head 1
        moved = q_moved_under_zero_task(m)
        assert moved[0], "aged head must receive injection"
        assert not moved[1], "newborn head must be excluded"
    # (3) per-head switch + paired positive
    with _selfproc_env({"att_warmup": 0, "att_head_age_min": 0,
                        "att_h_lo": 0.0, "att_h_hi": 0.10,
                        "att_selfproc_heads": {1}}):
        m = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                       seed=8, heads_spec=[[2, 2]])
        m._selfproc_on = True
        moved = q_moved_under_zero_task(m)
        assert not moved[0] and moved[1], \
            "switch: only the enabled head is processed"


# ---------------------------------------------------------- S5 tests

def test_t9_decide_branches():
    """T9: localization -> widen(argmax); saturation (H>=2) ->
    add; H=1 -> widen-first then escalation via the event ledger;
    overlap -> widen wins; no trigger -> None; newborns excluded
    from BOTH sides (plan D3)."""
    m, X, yv = _mk_trained(heads=[[2, 2, 2]])
    L = 0
    # scripted evidence (FX3 pattern): hand-set mu/nu
    def set_u(h, u):
        HS = m.heads[L][h]
        HS.nu = {nm: np.ones_like(v) for nm, v in HS.nu.items()}
        HS.mu = {nm: np.ones_like(v) * (1.0 - u)
                 for nm, v in HS.mu.items()}   # u = 1-|mu|/|nu|
    # heads were born at t=0 and the fixture ran 15 steps — age
    # them past the gate for branches (1)-(4); branch (5) tests
    # the exclusion explicitly
    m._t_att += 500
    # (1) localization: head 1 far above kappa*mean
    set_u(0, 0.05); set_u(1, 0.9); set_u(2, 0.05)
    sug = m.decide(L, X)
    assert sug[0] == "widen" and sug[1] == 1, sug
    # (2) no trigger: flat low u, loading spread (PR below band)
    set_u(0, 0.1); set_u(1, 0.1); set_u(2, 0.1)
    import types
    m_load = m.head_loading(L, X)
    # force unbalanced loading -> PR far from ceiling
    m2 = m
    orig = m2.head_loading
    m2.head_loading = types.MethodType(
        lambda self, l, Xq: [10.0, 0.1, 0.1], m2)
    assert m2.decide(L, X) is None
    # (3) saturation: even loading -> PR ~ H -> add
    m2.head_loading = types.MethodType(
        lambda self, l, Xq: [1.0, 1.0, 1.0], m2)
    assert m2.decide(L, X) == ("add",)
    # (4) overlap: localization AND saturation -> widen wins
    set_u(1, 0.9)
    assert m2.decide(L, X)[0] == "widen"
    m2.head_loading = orig
    # (5) newborn exclusion BOTH sides: a fresh head with u=1.0
    # must neither be picked nor inflate the mean
    set_u(0, 0.05); set_u(1, 0.30); set_u(2, 0.05)
    with _selfproc_env({"att_head_age_min": 50}):
        ev = m.head_add(L)                 # newborn u = 1.0
        sug = m.decide(L, X)
        assert sug is not None and sug[0] == "widen" \
            and sug[1] == 1, sug           # 0.30 >= 2*mean(aged)
        m.heads[L].pop()
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            m._adam_h.pop((L, ev["head"], nm))
    # (6) H=1 widen-first + ledger escalation
    m1, X1, _ = _mk_trained(heads=[[2], [2]])
    m1._t_att += 500
    with _selfproc_env({"att_head_age_min": 0}):
        s0 = m1.decide(0, X1, events=[])
        assert s0 == ("widen", 0, POLICY["att_widen_m"])
        recent = [{"event": "head_widen", "layer": 0,
                   "verdict": "accepted", "t": m1._t_att - 10}]
        assert m1.decide(0, X1, events=recent) == ("add",)
        stale = [{"event": "head_widen", "layer": 0,
                  "verdict": "accepted",
                  "t": m1._t_att - POLICY["att_window"] - 1}]
        assert m1.decide(0, X1, events=stale)[0] == "widen"


def test_t10_instruments_read_only_and_values():
    """T10: byte-state hash identical before/after every
    instrument call; values match hand-computed fixtures."""
    import pickle as pkl
    m, X, yv = _mk_trained()
    def state_bytes():
        return pkl.dumps({"P": m.P, "heads": [
            [(HS.Wq, HS.Wk, HS.Wv, HS.Wo, HS.mu, HS.nu)
             for HS in layer] for layer in m.heads]})
    h0 = state_bytes()
    _ = m.u_stats(0)
    _ = m.head_loading(0, X)
    _ = m.decide(0, X)
    _ = m.disposition_head(0, 1)
    _ = m.disposition_unit(0, 2)
    _ = m.row_entropies(0, 1, X)
    _ = m.capacity([1.0, 2.0, 3.0])
    _ = m.j_att_value(0, 1, X)
    assert state_bytes() == h0, "an instrument mutated state"
    # hand-computed checks
    D = m.disposition_head(0, 1)
    assert np.array_equal(D, m.heads[0][1].Wo) and D.base is None
    dj = m.disposition_unit(0, 2)
    assert np.array_equal(dj, m.P["W2_0"][2, :])
    cap = m.capacity([2.0, 2.0])          # even: PR = 2
    assert abs(cap["PR"] - 2.0) < 1e-9
    assert abs(cap["entropy"] - np.log(2)) < 1e-9
    cap1 = m.capacity([5.0, 0.0])         # one-hot: PR = 1, H = 0
    assert abs(cap1["PR"] - 1.0) < 1e-9 and cap1["entropy"] < 1e-9
    # row_entropies vs direct softmax computation on head (0,1)
    Xs = m._stdx(X)
    _, _, caches = m._forward(Xs, cache=True)
    A = caches[1][3][1][3]
    Hrows = m.row_entropies(0, 1, X)
    pc = np.maximum(A[0, 0], 1e-300)
    assert abs(Hrows[0, 0] - float(-(A[0, 0] * np.log(pc)).sum())) \
        < 1e-12


# ---------------------------------------------------------- S6 tests

def test_t12_governance():
    """T12: refused event -> serving model byte-identical (pinned,
    though the deepcopy design gives it by construction); accepted
    event -> the serving model DID change as suggested; rows carry
    the P10 schema incl. t; no_trigger path records honestly."""
    import pickle as pkl
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 3))
    yv = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]

    def mk():
        m = GrowableAttentionSubstrate(3, 8, d_model=8, n_layers=2,
                                       seed=4, heads_spec=[[2], [2]])
        for _ in range(30):
            m.train_step(X, yv)
        m._t_att += 500
        return m

    # (1) accepted: census changes as suggested; row schema
    m = mk()
    census0 = m.shape_record()["heads"]
    events = []
    row = attention_grow_event(m, 0, X, X[:16], yv[:16],
                               probe_X=X[16:], probe_y=yv[16:],
                               events=events)
    assert row["verdict"] == "accepted"
    assert m.shape_record()["heads"] != census0, \
        "accepted event must actually mutate the serving model"
    for key in ("event", "t", "layer", "evidence", "policy",
                "verdict", "heldout_before", "heldout_after"):
        assert key in row, key
    assert row["t"] == events[-1]["t"] and len(events) == 1
    assert "u" in row["evidence"] and "loadings" in row["evidence"]

    # (2) refused: force refusal with an impossible tolerance;
    # serving model byte-identical
    m2 = mk()
    blob0 = pkl.dumps({"P": m2.P, "heads": [
        [(HS.Wq, HS.Wk, HS.Wv, HS.Wo) for HS in layer]
        for layer in m2.heads]})
    ev2 = []
    row2 = attention_grow_event(m2, 0, X, X[:16], yv[:16],
                                probe_X=X[16:], probe_y=yv[16:],
                                events=ev2, tol=0.0)   # unpassable
    assert row2["verdict"] == "refused"
    blob1 = pkl.dumps({"P": m2.P, "heads": [
        [(HS.Wq, HS.Wk, HS.Wv, HS.Wo) for HS in layer]
        for layer in m2.heads]})
    assert blob0 == blob1, "refused event must leave the model " \
        "byte-identical"

    # (3) no trigger -> honest row, no mutation
    m3 = mk()
    for h in range(len(m3.heads[0])):
        HS = m3.heads[0][h]
        HS.nu = {nm: np.ones_like(v) for nm, v in HS.nu.items()}
        HS.mu = {nm: np.ones_like(v) * 0.9
                 for nm, v in HS.mu.items()}       # flat low u
    import types
    m3.head_loading = types.MethodType(
        lambda self, l, Xq: [10.0, 0.1], m3)       # PR far from H
    # H=1? no: heads_spec [[2]] single head layer... use layer 0 has
    # ONE head -> H==1 branch would fire; so pin ledger to escalate
    ev3 = [{"event": "head_widen", "layer": 0,
            "verdict": "accepted", "t": m3._t_att - 1}]
    row3 = attention_grow_event(m3, 0, X, X[:16], yv[:16],
                                probe_X=X[16:], probe_y=yv[16:],
                                events=ev3, tol=0.0)
    # H==1 + recent widen -> add suggested -> refused by tol=0
    assert row3["verdict"] in ("refused", "no_trigger")


def test_t12b_two_lane_contract():
    """Doc 18 (audit B1): the probe epoch trains ONLY on the probe
    lane; the heldout lane is scored, never trained on; growth
    proposed without probe data is refused loudly; probe steps come
    from POLICY (att_probe_steps)."""
    import pickle as pkl
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 3))
    yv = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]

    def mk():
        m = GrowableAttentionSubstrate(3, 8, d_model=8, n_layers=2,
                                       seed=4, heads_spec=[[2], [2]])
        for _ in range(30):
            m.train_step(X, yv)
        m._t_att += 500
        return m

    # (1) spy: every probe batch is the probe lane; the scoring
    # batch never enters train_step
    m = mk()
    hx, hy = X[:16], yv[:16]
    px, py = X[16:], yv[16:]
    seen = []
    orig_ts = GrowableAttentionSubstrate.train_step

    def spy_ts(self, Xb, yb, **kw):
        seen.append(np.asarray(Xb))
        return orig_ts(self, Xb, yb, **kw)

    GrowableAttentionSubstrate.train_step = spy_ts
    try:
        row = attention_grow_event(m, 0, X, hx, hy,
                                   probe_X=px, probe_y=py,
                                   events=[])
    finally:
        GrowableAttentionSubstrate.train_step = orig_ts
    assert row["verdict"] in ("accepted", "refused")
    assert len(seen) == POLICY["att_probe_steps"]
    for Xb in seen:
        assert Xb.shape == np.asarray(px, float).shape
        assert np.array_equal(Xb, np.asarray(px, float))
        assert not np.array_equal(Xb, np.asarray(hx, float))

    # (2) growth proposed, no probe data -> loud refusal,
    # byte-identical substrate
    m2 = mk()
    blob0 = pkl.dumps({"P": m2.P, "heads": [
        [(HS.Wq, HS.Wk, HS.Wv, HS.Wo) for HS in layer]
        for layer in m2.heads]})
    row2 = attention_grow_event(m2, 0, X, hx, hy, events=[])
    assert row2["verdict"] == "refused"
    assert row2["event"] == "none"
    assert "no probe data" in row2["reason"]
    assert row2["suggestion"] in ("widen", "add")
    blob1 = pkl.dumps({"P": m2.P, "heads": [
        [(HS.Wq, HS.Wk, HS.Wv, HS.Wo) for HS in layer]
        for layer in m2.heads]})
    assert blob0 == blob1

    # (3) att_probe_steps=0 -> before == after exactly (growth is
    # exact at application; the probe is the only mover)
    m3 = mk()
    m3._att_policy = {**POLICY, "att_probe_steps": 0}
    row3 = attention_grow_event(m3, 0, X, hx, hy,
                                probe_X=px, probe_y=py, events=[])
    assert row3["heldout_before"] == row3["heldout_after"]
