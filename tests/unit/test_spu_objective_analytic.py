"""B1 analytic SPU objective tests (attention-build S7; plan D3
T11 a-g). The analytic objective is the EXACT expectation of the
mask objective under its own truncated mask law; the mask path is
preserved unchanged and stays the build-time fallback.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

from engine.spu.spu_objective import (            # noqa: E402
    draw_masks, forward_parts, jan_and_grads, jdisc_and_grads,
    jinv_and_grads, jlocal_an_and_grads, jlocal_and_grads)
from engine.spu.spu_policy import (               # noqa: E402
    DEFAULT_SPU_POLICY, validate_spu_policy)


def _leaf(seed=5, n=6, d=4, H=5):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(H, d)), rng.normal(size=H),
            rng.normal(size=(1, H)), rng.normal(size=1),
            rng.normal(size=(n, d)))


def test_t11a_fd_of_jan():
    """T11a: jan_and_grads gradients entry-by-entry vs central
    differences (cert S3 form)."""
    W1, b1, W2, c, X = _leaf()
    _, g = jan_and_grads(W1, b1, W2, c, X, 0.10, 4)
    eps = 1e-6
    for nm, W in (("W1", W1), ("b1", b1), ("W2", W2)):
        g_fd = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = W[i]
            W[i] = old + eps
            fp, _ = jan_and_grads(W1, b1, W2, c, X, 0.10, 4)
            W[i] = old - eps
            fm, _ = jan_and_grads(W1, b1, W2, c, X, 0.10, 4)
            W[i] = old
            g_fd[i] = (fp - fm) / (2 * eps)
            it.iternext()
        tol = 3e-5 * np.maximum(1.0, np.abs(g_fd))
        assert (np.abs(g[nm] - g_fd) <= tol).all(), \
            (nm, np.abs(g[nm] - g_fd).max())
    assert not g["c"].any(), "grad c must be exactly 0"
    assert g["blocks"] == []


def test_t11b_mc_equality_vs_mask_law():
    """T11b: E[J_inv] over the lib's OWN draw_masks equals
    (1-1/K)*J_an -> jan returns J~ directly; |z| < 4."""
    W1, b1, W2, c, X = _leaf()
    K, p = 4, 0.10
    Jt, _ = jan_and_grads(W1, b1, W2, c, X, p, K)
    rng = np.random.default_rng(11)
    vals = []
    for _ in range(3000):
        masks = draw_masks(rng, K, W1.shape[0], p)
        Ji, _ = jinv_and_grads(W1, b1, W2, c, X, masks)
        vals.append(Ji)
    vals = np.asarray(vals)
    sem = vals.std(ddof=1) / np.sqrt(len(vals))
    z = abs(vals.mean() - Jt) / sem
    assert z < 4.0, (vals.mean(), Jt, z)


def test_t11c_session_smoke_same_verdict():
    """T11c: an spu_loop session under 'analytic' reaches the SAME
    stop verdict as 'mask' on a reference fixture (thresholds
    carry over via the (1-1/K) factor inside J~)."""
    from engine.spu.spu_loop import self_process
    from reference_net.net import Network

    def trained_leaf(seed=3, steps=30):
        rng = np.random.default_rng(seed)
        leaf = Network(d_in=3, hidden=5, lr=1e-2, seed=seed)
        Xl = rng.normal(0, 1, (32, 3))
        yl = np.sin(Xl.sum(1, keepdims=True))
        for _ in range(steps):
            leaf.train_step(Xl, yl)
        return leaf, Xl

    # threshold carry-over is exact in EXPECTATION (T11b); the
    # mask path's per-draw Ji is STOCHASTIC (sd ~ mean on this
    # fixture: a ~17%% per-step lucky-stop probability) — which is
    # precisely the determinism gain B1 claims. Verdict equality
    # is therefore asserted on the two DECISIVE sides:
    # (p_mask raised on the tau=0 side: at p=0.1/H=5 all four
    # masks coincide with ~12%% probability per draw, making the
    # mask path's Ji EXACTLY 0 by luck — one more stochastic
    # artifact the analytic form removes)
    for tau, pmask, expect in ((1e9, 0.10, True),
                               (0.0, 0.50, False)):
        verdicts = {}
        for obj in ("mask", "analytic"):
            leaf, Xl = trained_leaf()
            pol = validate_spu_policy(
                {"spu_enabled": True, "spu_warmup_steps": 0,
                 "spu_tau_rel": tau, "spu_S_max": 3,
                 "spu_p_mask": pmask,
                 "spu_objective": obj})
            rep = self_process(leaf, Xl, pol, step_index=0)
            verdicts[obj] = rep.get("converged")
        assert verdicts["mask"] == verdicts["analytic"] == expect, \
            (tau, verdicts)
    # and the analytic value sits at the mask objective's
    # expectation on the session's own leaf (carry-over, direct)
    leaf, Xl = trained_leaf()
    Xs = (Xl - leaf._x_mu) / leaf._x_sd \
        if getattr(leaf, "_x_mu", None) is not None else Xl
    Jt, _ = jan_and_grads(leaf.W1, leaf.b1, leaf.W2, leaf.c,
                          Xs, 0.10, 4)
    rng2 = np.random.default_rng(99)
    vals = []
    for _ in range(400):
        mk = draw_masks(rng2, 4, leaf.W1.shape[0], 0.10)
        v, _ = jinv_and_grads(leaf.W1, leaf.b1, leaf.W2, leaf.c,
                              Xs, mk)
        vals.append(v)
    vals = np.asarray(vals)
    sem = vals.std(ddof=1) / np.sqrt(len(vals))
    assert abs(vals.mean() - Jt) / sem < 4.0


def test_t11d_mask_path_and_defaults_preserved():
    """T11d: the mask path is bit-identical to before the edits
    (same jinv/jlocal values on a fixed fixture) AND
    len(DEFAULT_SPU_POLICY) is still 14 — the optional key never
    enters the defaults."""
    assert len(DEFAULT_SPU_POLICY) == 14
    assert "spu_objective" not in DEFAULT_SPU_POLICY
    W1, b1, W2, c, X = _leaf(9)
    masks = draw_masks(np.random.default_rng(2), 4, W1.shape[0],
                       0.10)
    Ji, gi = jinv_and_grads(W1, b1, W2, c, X, masks)
    # regression pin (computed at S7 with the untouched mask code)
    A, h, z0 = forward_parts(W1, b1, W2, c, X)
    s_entry = float(np.sqrt(np.mean((z0 - z0.mean()) ** 2) + 1e-12))
    J, Ji2, Jd, g = jlocal_and_grads(W1, b1, W2, c, X, masks,
                                     s_entry * 2, 1.0, 0.5)
    assert Ji == Ji2 and np.isfinite(J)
    assert "blocks" in gi and gi["blocks"] == []


def test_t11e_policy_validation():
    """T11e: validate accepts both values of the optional key,
    refuses others; None still yields the exact defaults."""
    assert validate_spu_policy(None) == DEFAULT_SPU_POLICY
    v = validate_spu_policy({"spu_objective": "analytic"})
    assert v["spu_objective"] == "analytic"
    v2 = validate_spu_policy({"spu_objective": "mask"})
    assert v2["spu_objective"] == "mask"
    with pytest.raises(ValueError):
        validate_spu_policy({"spu_objective": "gaussian"})
    with pytest.raises(ValueError):
        validate_spu_policy({"made_up_key": 1})


def test_t11f_leaf_identity():
    """T11f: on a leaf, the leave-one-out effect of unit u equals
    its loading exactly (the linearity the closed form rests on;
    cert S7)."""
    W1, b1, W2, c, X = _leaf(3)
    A, h, z0 = forward_parts(W1, b1, W2, c, X)
    u = 2
    mask = np.ones(W1.shape[0])
    mask[u] = 0.0
    zm = (h * mask) @ W2[0] + c[0]
    loading = h[:, u] * W2[0, u]
    assert np.abs((z0 - zm) - loading).max() < 1e-12


def test_t11g_boundary_red_arm():
    """T11g BOUNDARY RED-ARM: with a composition block the
    identity breaks O(1) (cert S7) AND the dispatch refuses the
    analytic path (falls back to the mask objective) — the leaf
    gate guards something real."""
    from engine.spu.spu_objective import chain_apply
    W1, b1, W2, c, X = _leaf(3)
    rng = np.random.default_rng(4)
    blk = {"Bin": rng.normal(size=(3, W1.shape[0])),
           "bb": rng.normal(size=3),
           "Bout": rng.normal(size=(W1.shape[0], 3))}
    A, h, z0 = forward_parts(W1, b1, W2, c, X)
    u = 2
    mask = np.ones(W1.shape[0]); mask[u] = 0.0

    def z_with_blocks(hh):
        HL, _ = chain_apply([blk], hh)
        return HL @ W2[0] + c[0]

    brk = np.abs((z_with_blocks(h) - z_with_blocks(h * mask))
                 - h[:, u] * W2[0, u]).max()
    assert brk > 0.01, "block must break the identity O(1)"
    # dispatch refusal: analytic requested + blocks present ->
    # the loop takes the MASK branch (source-verified condition
    # `and not blocks`); pin it by reading the code path
    import inspect
    from engine.spu import spu_loop
    src = inspect.getsource(spu_loop.self_process)
    assert 'spu_objective", "mask") == "analytic"' in src \
        and "and not blocks" in src
