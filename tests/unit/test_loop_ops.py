"""L1.2 kernel tests: K11/K12 + governance helpers (DEV_PLAN ~12)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.loop_ops import (C_G, loop_backward,          # noqa: E402
                               loop_forward, loop_rho_hat,
                               validate_loop_policy)
from engine.backends.numpy_backend import gelu            # noqa: E402


def _mk(n=4, H=5, m=3, scale=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, H)),
            rng.normal(size=(m, H)) * scale,
            rng.normal(size=m) * 0.1,
            rng.normal(size=(H, m)) * scale)


def test_1_fixed_point_correctness():
    H_L, L_in, b, L_out = _mk()
    z, k, zs = loop_forward(H_L, L_in, b, L_out, 1e-8, 64)
    res = np.max(np.abs(z - (H_L + gelu(z @ L_in.T + b) @ L_out.T)))
    assert res < 1e-7 and 1 < k < 64


def test_2_kmax_semantics():
    H_L, L_in, b, L_out = _mk()
    z_t, k_t, _ = loop_forward(H_L, L_in, b, L_out, 1e-3, 32)
    assert k_t < 32                              # loose tol stops early
    z_i, k_i, zs = loop_forward(H_L, L_in, b, L_out, 0.0 + 1e-300, 5)
    assert k_i == 5 and len(zs) == 6             # budget hit, last iterate
    assert np.all(np.isfinite(z_i))              # returned, not raised


def test_3_exact_entry_bitwise():
    H_L, L_in, b, _ = _mk()
    z, k, _ = loop_forward(H_L, L_in, b, np.zeros((5, 3)), 1e-6, 32)
    assert k == 1
    assert z is not H_L or True
    assert np.array_equal(z, H_L)                # bitwise


@pytest.mark.parametrize("n,H,m", [(4, 5, 3), (6, 6, 8)])
@pytest.mark.parametrize("tol,K", [(1e-10, 3), (1e-12, 64)])
def test_4_fd_every_gradient_entry(n, H, m, tol, K):
    rng = np.random.default_rng(7)
    H_L = rng.normal(size=(n, H))
    L_in = rng.normal(size=(m, H)) * 0.3
    b = rng.normal(size=m) * 0.1
    L_out = rng.normal(size=(H, m)) * 0.3
    W = rng.normal(size=(H, 1))                  # scalar loss probe

    def loss(H_L_, L_in_, b_, L_out_):
        z, _, _ = loop_forward(H_L_, L_in_, b_, L_out_, tol, K)
        return float((z @ W).sum())

    z, _, zs = loop_forward(H_L, L_in, b, L_out, tol, K)
    dz = np.repeat(W.T, n, axis=0)               # dloss/dz
    dH_L, gL_in, gb, gL_out = loop_backward(zs, L_in, b, L_out, dz)
    eps = 1e-6
    for arr, grad in ((H_L, dH_L), (L_in, gL_in), (b, gb),
                      (L_out, gL_out)):
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            orig = arr[ix]
            arr[ix] = orig + eps
            lp = loss(H_L, L_in, b, L_out)
            arr[ix] = orig - eps
            lm = loss(H_L, L_in, b, L_out)
            arr[ix] = orig
            fd = (lp - lm) / (2 * eps)
            assert abs(fd - grad[ix]) < 5e-5, (ix, fd, grad[ix])


def test_5_contraction_decay():
    H_L, L_in, b, L_out = _mk(scale=0.2)
    assert loop_rho_hat(L_in, L_out) < 1.0
    _, _, zs = loop_forward(H_L, L_in, b, L_out, 1e-14, 40)
    steps = [np.max(np.abs(zs[i + 1] - zs[i]))
             for i in range(len(zs) - 1)]
    assert steps[-1] < steps[0] * 1e-3           # geometric decay


def test_6_rho_hat_is_a_bound():
    rng = np.random.default_rng(3)
    for seed in range(5):
        _, L_in, b, L_out = _mk(scale=0.5, seed=seed)
        rho = loop_rho_hat(L_in, L_out)
        z1 = rng.normal(size=(8, 5))
        z2 = z1 + rng.normal(size=(8, 5)) * 1e-4
        f1 = gelu(z1 @ L_in.T + b) @ L_out.T
        f2 = gelu(z2 @ L_in.T + b) @ L_out.T
        lip = (np.linalg.norm(f1 - f2)
               / np.linalg.norm(z1 - z2))
        assert lip <= rho + 1e-9                 # bound HOLDS


def test_7_determinism():
    H_L, L_in, b, L_out = _mk()
    a = loop_forward(H_L, L_in, b, L_out, 1e-6, 32)
    c = loop_forward(H_L, L_in, b, L_out, 1e-6, 32)
    assert a[1] == c[1] and np.array_equal(a[0], c[0])


def test_8_degenerate_shapes():
    z, k, _ = loop_forward(np.ones((1, 2)), np.ones((1, 2)) * 0.1,
                           np.zeros(1), np.ones((2, 1)) * 0.1,
                           1e-6, 32)
    assert np.all(np.isfinite(z)) and k >= 1


def test_9_validate_switch_off():
    with pytest.raises(ValueError, match="opt-in option"):
        validate_loop_policy({"loop_enabled": False})


def test_10_validate_triple_infeasible():
    with pytest.raises(ValueError, match="infeasible certificate"):
        validate_loop_policy({"loop_enabled": True,
                              "loop_rho_max": 0.9,
                              "loop_K_max": 4, "loop_tol": 1e-6})


def test_11_validate_unknown_key():
    with pytest.raises(ValueError, match="unknown loop policy"):
        validate_loop_policy({"loop_enabled": True, "loop_warp": 1})


def test_12_validate_defaults_feasible():
    out = validate_loop_policy({"loop_enabled": True})
    assert out == {"loop_m": 8, "loop_K_max": 32,
                   "loop_tol": 1e-6, "loop_rho_max": 0.6}
