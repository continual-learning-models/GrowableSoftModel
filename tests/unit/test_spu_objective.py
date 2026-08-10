"""SPU S1 tests: masks, J_inv, J_disc, hand gradients vs central
finite differences (DEV_PLAN_SPU S1, T1.1-T1.14)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from engine.spu.spu_objective import (   # noqa: E402
    batch_std, draw_masks, forward_chain, forward_parts, jdisc_and_grads, jinv_and_grads, jlocal_and_grads)

H, D, N, K = 5, 3, 12, 4


def leaf(seed=0, w2_scale=0.5):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.5, (H, D)), rng.normal(0, 0.1, H),
            rng.normal(0, w2_scale, (1, H)), rng.normal(0, 0.1, 1))


def data(seed=1):
    return np.random.default_rng(seed).normal(0, 1, (N, D))


def masks(seed=2, k=K):
    return draw_masks(np.random.default_rng(seed), k, H, 0.25)


def fd_check(f, arrays, grads, rel=1e-5, h=1e-6):
    """Central-difference check of every entry of every array."""
    for name, arr in arrays.items():
        g = grads[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + h; Jp = f()
            arr[idx] = old - h; Jm = f()
            arr[idx] = old
            fd = (Jp - Jm) / (2 * h)
            an = g[idx]
            denom = max(abs(fd), abs(an), 1e-8)
            assert abs(fd - an) / denom < rel or abs(fd - an) < 1e-9, (
                f"{name}{idx}: fd={fd:.3e} analytic={an:.3e}")


# T1.1 masks
def test_masks_shape_rate_determinism():
    m1 = draw_masks(np.random.default_rng(7), 200, H, 0.25)
    m2 = draw_masks(np.random.default_rng(7), 200, H, 0.25)
    assert m1.shape == (200, H) and set(np.unique(m1)) <= {0.0, 1.0}
    assert np.array_equal(m1, m2)
    assert abs((1 - m1.mean()) - 0.25) < 0.05          # drop rate


# T1.2 masked outputs equal brute force
def test_masked_outputs_bruteforce_bitwise():
    W1, b1, W2, c = leaf(); X = data(); M = masks()
    A, h, z0 = forward_parts(W1, b1, W2, c, X)
    for i, m in enumerate(M):
        z_i = h @ (m * W2[0]) + c[0]
        z_brute = (m * h) @ W2[0] + c[0]
        assert np.array_equal(z_i, z_brute)


# T1.3 identical masks -> J_inv = 0, zero grads
def test_jinv_zero_under_identical_masks():
    W1, b1, W2, c = leaf(); X = data()
    M = np.ones((K, H)); M[:, 0] = 0                    # identical
    J, g = jinv_and_grads(W1, b1, W2, c, X, M)
    assert J == 0.0
    assert all(np.allclose(g[k], 0) for k in g)


# T1.4 distinct masks -> J_inv > 0
def test_jinv_positive_under_distinct_masks():
    W1, b1, W2, c = leaf(); X = data()
    J, _ = jinv_and_grads(W1, b1, W2, c, X, masks())
    assert J > 0


# T1.5-T1.7 FD of J_inv wrt all params (fixed masks)
def test_jinv_gradients_fd():
    W1, b1, W2, c = leaf(); X = data(); M = masks()
    arrays = {"W1": W1, "b1": b1, "W2": W2, "c": c}
    _, g = jinv_and_grads(W1, b1, W2, c, X, M)
    fd_check(lambda: jinv_and_grads(W1, b1, W2, c, X, M)[0],
             arrays, g)


# T1.8 hinge inactive -> zero
def test_jdisc_inactive_hinge_zero():
    W1, b1, W2, c = leaf(); X = data()
    _, _, z0 = forward_parts(W1, b1, W2, c, X)
    s = batch_std(z0)
    J, g = jdisc_and_grads(W1, b1, W2, c, X, s_entry=s,
                           rho_floor=0.5)               # 0.5s < s
    assert J == 0.0 and all(np.allclose(g[k], 0) for k in g)


# T1.9 penalty grows as output collapses
def test_jdisc_grows_toward_collapse():
    W1, b1, W2, c = leaf(); X = data()
    _, _, z0 = forward_parts(W1, b1, W2, c, X)
    s_entry = batch_std(z0)
    J_half, _ = jdisc_and_grads(W1, b1, 0.4 * W2, c, X, s_entry, 0.9)
    J_tiny, _ = jdisc_and_grads(W1, b1, 0.01 * W2, c, X, s_entry, 0.9)
    assert J_tiny > J_half > 0


# T1.10 FD of J_disc (active hinge)
def test_jdisc_gradients_fd_active():
    W1, b1, W2, c = leaf(w2_scale=0.05); X = data()     # small spread
    _, _, z0 = forward_parts(W1, b1, 10 * W2, c, X)
    s_entry = batch_std(z0)                              # healthy entry
    arrays = {"W1": W1, "b1": b1, "W2": W2, "c": c}
    J, g = jdisc_and_grads(W1, b1, W2, c, X, s_entry, 0.9)
    assert J > 0
    fd_check(lambda: jdisc_and_grads(W1, b1, W2, c, X,
                                     s_entry, 0.9)[0], arrays, g)


# T1.11 combined FD
def test_jlocal_combined_fd():
    W1, b1, W2, c = leaf(w2_scale=0.05); X = data(); M = masks()
    _, _, z0 = forward_parts(W1, b1, 10 * W2, c, X)
    s_entry = batch_std(z0)
    arrays = {"W1": W1, "b1": b1, "W2": W2, "c": c}
    _, _, _, g = jlocal_and_grads(W1, b1, W2, c, X, M, s_entry,
                                  gamma=1.0, rho_floor=0.9)
    fd_check(lambda: jlocal_and_grads(W1, b1, W2, c, X, M, s_entry,
                                      1.0, 0.9)[0], arrays, g)


# T1.12 scale-freeness of the hinge pattern
def test_relative_floor_scale_free():
    W1, b1, W2, c = leaf(); X = data()
    for scale in (1.0, 100.0):
        W2s, cs = scale * W2, scale * c
        _, _, z0 = forward_parts(W1, b1, W2s, cs, X)
        s_entry = batch_std(z0)
        J_ok, _ = jdisc_and_grads(W1, b1, W2s, cs, X, s_entry, 0.5)
        J_bad, _ = jdisc_and_grads(W1, b1, 0.1 * W2s, cs, X,
                                   s_entry, 0.5)
        assert J_ok == 0.0 and J_bad > 0     # same pattern any scale


# T1.13 gradient points away from collapse
def test_jdisc_gradient_restores_spread():
    W1, b1, W2, c = leaf(); X = data()
    _, _, z0 = forward_parts(W1, b1, W2, c, X)
    s_entry = batch_std(z0)
    W2c = 0.05 * W2                                      # collapsed-ish
    _, _, z0c = forward_parts(W1, b1, W2c, c, X)
    s_before = batch_std(z0c)
    _, g = jdisc_and_grads(W1, b1, W2c, c, X, s_entry, 0.9)
    W2n = W2c - 0.05 * g["W2"]
    _, _, z0n = forward_parts(W1, b1, W2n, c, X)
    assert batch_std(z0n) > s_before


# T1.14 n=1 degenerate: no NaN, all zero
def test_single_row_degenerate():
    W1, b1, W2, c = leaf()
    X1 = data()[:1]
    _, _, z0 = forward_parts(W1, b1, W2, c, X1)
    s_entry = batch_std(z0)                              # ~0
    J, g = jdisc_and_grads(W1, b1, W2, c, X1, s_entry, 0.5)
    assert np.isfinite(J) and J == 0.0
    Ji, gi = jinv_and_grads(W1, b1, W2, c, X1, masks())
    assert np.isfinite(Ji)
    assert all(np.all(np.isfinite(gi[k])) for k in gi)


# ---------- P2-S1: chain-aware (deepened-unit) objective ----------

def _rand_blocks(rng, H, m, k):
    return [{"Bin": rng.normal(0, 0.4, (m, H)),
             "bb": rng.normal(0, 0.1, m),
             "Bout": rng.normal(0, 0.4, (H, m))}
            for _ in range(k)]


def _setup_chain(n_blocks, seed=3, n=9, d=4, H=5, m=3):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, (H, d))
    b1 = rng.normal(0, 0.1, H)
    W2 = rng.normal(0, 0.5, (1, H))
    c = rng.normal(0, 0.1, 1)
    X = rng.normal(0, 1, (n, d))
    masks = draw_masks(rng, 4, H, 0.25)
    blocks = _rand_blocks(rng, H, m, n_blocks)
    return W1, b1, W2, c, X, masks, blocks


def _fd_check(fn, arrays, eps=1e-6, tol=2e-5):
    """Central-difference check of every entry of every array
    against the analytic gradient supplied by fn()."""
    J0, grads = fn()
    for arr, g in arrays(grads):
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            keep = arr[idx]
            arr[idx] = keep + eps
            Jp, _ = fn()
            arr[idx] = keep - eps
            Jm, _ = fn()
            arr[idx] = keep
            fd = (Jp - Jm) / (2 * eps)
            assert abs(fd - g[idx]) < tol * max(1.0, abs(fd)), \
                (idx, fd, g[idx])


@pytest.mark.parametrize("nb", [1, 2])
def test_jinv_chain_fd_all_params(nb):
    W1, b1, W2, c, X, masks, blocks = _setup_chain(nb)

    def fn():
        return jinv_and_grads(W1, b1, W2, c, X, masks,
                              blocks=blocks)

    def arrays(g):
        yield W1, g["W1"]
        yield b1, g["b1"]
        yield W2, g["W2"]
        for k in range(nb):
            yield blocks[k]["Bin"], g["blocks"][k]["Bin"]
            yield blocks[k]["bb"], g["blocks"][k]["bb"]
            yield blocks[k]["Bout"], g["blocks"][k]["Bout"]

    _fd_check(fn, arrays)


def test_jdisc_chain_fd_all_params():
    W1, b1, W2, c, X, masks, blocks = _setup_chain(2)
    _, _, _, _, z0 = forward_chain(W1, b1, W2, c, X, blocks)
    s_entry = batch_std(z0) * 4.0          # hinge ACTIVE

    def fn():
        return jdisc_and_grads(W1, b1, W2, c, X, s_entry, 0.5,
                               blocks=blocks)

    def arrays(g):
        yield W1, g["W1"]
        yield b1, g["b1"]
        yield W2, g["W2"]
        for k in range(2):
            yield blocks[k]["Bin"], g["blocks"][k]["Bin"]
            yield blocks[k]["bb"], g["blocks"][k]["bb"]
            yield blocks[k]["Bout"], g["blocks"][k]["Bout"]

    _fd_check(fn, arrays)


def test_grad_c_exactly_zero_with_blocks():
    W1, b1, W2, c, X, masks, blocks = _setup_chain(2)
    _, gi = jinv_and_grads(W1, b1, W2, c, X, masks, blocks=blocks)
    assert np.all(gi["c"] == 0.0)
    _, _, _, _, z0 = forward_chain(W1, b1, W2, c, X, blocks)
    _, gd = jdisc_and_grads(W1, b1, W2, c, X,
                            batch_std(z0) * 4.0, 0.5,
                            blocks=blocks)
    assert np.all(gd["c"] == 0.0)


def test_fresh_zero_bout_block_matches_stage_a():
    """A just-deepened unit (Bout = 0) has the exact Stage-A
    J values: the chain adds zeros."""
    W1, b1, W2, c, X, masks, _ = _setup_chain(0)
    zero_blk = [{"Bin": np.random.default_rng(5).normal(
                     0, 0.4, (3, 5)),
                 "bb": np.zeros(3), "Bout": np.zeros((5, 3))}]
    Ja, _ = jinv_and_grads(W1, b1, W2, c, X, masks)
    Jb, _ = jinv_and_grads(W1, b1, W2, c, X, masks,
                           blocks=zero_blk)
    assert Ja == Jb


def test_identical_masks_zero_jinv_with_blocks():
    W1, b1, W2, c, X, _, blocks = _setup_chain(2)
    same = np.ones((4, 5))
    J, _ = jinv_and_grads(W1, b1, W2, c, X, same, blocks=blocks)
    assert J < 1e-24


def test_chain_n1_degenerate():
    W1, b1, W2, c, X, masks, blocks = _setup_chain(1, n=1)
    J, g = jinv_and_grads(W1, b1, W2, c, X, masks, blocks=blocks)
    assert np.isfinite(J)
    assert all(np.isfinite(g[k]).all() for k in
               ("W1", "b1", "W2", "c"))


def test_all_masked_impossible_by_rejection_sampling():
    """Owner ruling 2026-07-10: all-masked draws are excluded by
    rejection sampling of the truncated Bernoulli. Worst small
    scale (H=2, p=0.5: raw Bernoulli would yield ~25% all-masked
    rows): zero all-masked across 5000 rows; K stays fixed;
    all-KEPT identity rows remain allowed; caller-rng
    determinism preserved."""
    import numpy as np
    from engine.spu.spu_objective import draw_masks
    m = draw_masks(np.random.default_rng(0), 5000, 2, 0.5)
    assert m.shape == (5000, 2)                # K fixed
    assert (m.sum(axis=1) > 0).all()           # never all-masked
    assert (m.sum(axis=1) == 2).any()          # identity allowed
    m2 = draw_masks(np.random.default_rng(0), 5000, 2, 0.5)
    assert np.array_equal(m, m2)               # deterministic
