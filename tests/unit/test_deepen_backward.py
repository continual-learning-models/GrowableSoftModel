"""S3 delta-backward tests (DEV_PLAN inventory, 11 tests).

Finite differences compare against Network._grads. Convention note:
the code absorbs the 2/n of d(mean err^2) into the learning rate,
so analytic gradients equal HALF the true derivative of the
standardized MSE: analytic == 0.5 * FD(loss).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network, ETA_TARGET, gelu, gelu_d  # noqa: E402
from tests.fixtures.make_golden import build_reference  # noqa: E402


def _loss(net, Xs, ys):
    _, H0 = net._hidden(Xs)
    HL = net._apply_blocks(H0)
    err = HL @ net.W2.T + net.c - ys
    return float(np.mean(err ** 2))


def _fd(net, Xs, ys, arr, eps=1e-5):
    # eps = 1e-5: with 1e-6 the float64 ROUNDOFF term of the central
    # difference (~1e-16/(2 eps)) dominates entries of magnitude
    # ~1e-6 and shows up as false 2e-5 relative errors; 1e-5 pushes
    # roundoff to ~5e-12 while truncation stays negligible for these
    # smooth functions. Tolerance itself is unchanged (1e-5).
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        keep = arr[idx]
        arr[idx] = keep + eps
        lp = _loss(net, Xs, ys)
        arr[idx] = keep - eps
        lm = _loss(net, Xs, ys)
        arr[idx] = keep
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def _rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(
        np.maximum(np.abs(a), np.abs(b)), 1e-8))


def _deep_net(seed=9, blocks=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(32, 2))
    y = np.sin(2 * X[:, 0]).reshape(-1, 1)
    net = Network(d_in=2, hidden=3, lr=1e-2, seed=seed)
    for _ in range(30):
        net.train_step(X, y)
    for _ in range(blocks):
        net.deepen(m=2)
    for blk in net.blocks:                     # activate the chain
        blk["Bout"] = rng.normal(size=blk["Bout"].shape) * 0.2
    Xs = net._std_x(X)
    ys = (y - net._y_mu) / net._y_sd
    return net, X, y, Xs, ys


def _grad_map(net, Xs, ys):
    grads, aux = net._grads(Xs, ys)
    m = {"W1": grads[0], "b1": grads[1], "W2": grads[2], "c": grads[3]}
    for k in range(len(net.blocks)):
        m[f"Bin{k}"], m[f"bb{k}"], m[f"Bout{k}"] = grads[4 + 3 * k:
                                                         7 + 3 * k]
    return m, aux


def test_fd_all_block_params_two_blocks():
    net, _, _, Xs, ys = _deep_net()
    g, _ = _grad_map(net, Xs, ys)
    for k, blk in enumerate(net.blocks):
        for name, arr in (("Bin", blk["Bin"]), ("bb", blk["bb"]),
                          ("Bout", blk["Bout"])):
            fd = _fd(net, Xs, ys, arr)
            assert _rel(g[f"{name}{k}"], 0.5 * fd) < 1e-5, (name, k)


def test_fd_W1_b1_through_chain():
    net, _, _, Xs, ys = _deep_net()
    g, _ = _grad_map(net, Xs, ys)
    assert _rel(g["W1"], 0.5 * _fd(net, Xs, ys, net.W1)) < 1e-5
    assert _rel(g["b1"], 0.5 * _fd(net, Xs, ys, net.b1)) < 1e-5


def test_fd_W2_c_with_blocks():
    net, _, _, Xs, ys = _deep_net()
    g, _ = _grad_map(net, Xs, ys)
    assert _rel(g["W2"], 0.5 * _fd(net, Xs, ys, net.W2)) < 1e-5
    assert _rel(g["c"], 0.5 * _fd(net, Xs, ys, net.c)) < 1e-5


def test_grads_channel_matches_train_step_path():
    net, X, y, Xs, ys = _deep_net()
    grads, _ = net._grads(Xs, ys)
    params_before = [net.W1.copy(), net.b1.copy(),
                     net.W2.copy(), net.c.copy()]
    blocks_before = [(b["Bin"].copy(), b["bb"].copy(), b["Bout"].copy())
                     for b in net.blocks]
    net.train_step(X, y, sgd_lr=0.1)
    assert np.allclose(net.W1, params_before[0] - 0.1 * grads[0],
                       rtol=0, atol=0)
    for k, blk in enumerate(net.blocks):
        gBin, gbb, gBout = grads[4 + 3 * k: 7 + 3 * k]
        assert np.array_equal(blk["Bin"],
                              blocks_before[k][0] - 0.1 * gBin)
        assert np.array_equal(blk["Bout"],
                              blocks_before[k][2] - 0.1 * gBout)


def test_zero_block_reverse_loop_noop_golden_train():
    net, X, y = build_reference()
    ref = np.load(ROOT / "tests/fixtures/golden_train_gp1.npz")["losses"]
    losses = np.array([net.train_step(X, y) for _ in range(50)])
    assert np.array_equal(losses, ref)


def test_convergence_smoke_deepened_scope():
    net, X, y, _, _ = _deep_net(blocks=1)
    at_deepen = net.train_step(X, y)
    for _ in range(300):
        last = net.train_step(X, y)
    assert last < at_deepen


def test_recursive_deepen_inner_trains():
    net, X, y, _, _ = _deep_net(blocks=0)
    net.grow(1, hidden=3)
    for _ in range(20):
        net.train_step(X, y)
    Xq = np.random.default_rng(3).normal(size=(64, 2))
    before = net.predict(Xq)
    net.grown_body(1).deepen(m=2)
    assert np.array_equal(before, net.predict(Xq))    # exact at attach
    for _ in range(200):
        net.train_step(X, y)
    inner_blk = net.grown_body(1).blocks[0]
    assert np.any(inner_blk["Bout"] != 0.0)           # it trained


def test_target_handing_form_unchanged_with_blocks():
    net, _, _, Xs, ys = _deep_net(blocks=1)
    g, aux = _grad_map(net, Xs, ys)
    blk = net.blocks[0]
    _, H0 = net._hidden(Xs)
    err = (net._apply_blocks(H0) @ net.W2.T + net.c) - ys
    dH = err @ net.W2
    Z = H0 @ blk["Bin"].T + blk["bb"]
    dZ = (dH @ blk["Bout"]) * gelu_d(Z)
    dH0_hand = dH + dZ @ blk["Bin"]
    assert np.allclose(aux["dH0"], dH0_hand, rtol=0, atol=1e-15)


def test_adam_rebuild_on_deepen():
    net, _, _, _, _ = _deep_net(blocks=0)
    assert len(net.opt.m) == 4
    net.deepen(m=2)
    assert len(net.opt.m) == 7


def test_adam_rebuild_on_remove():
    net, _, _, _, _ = _deep_net(blocks=1)
    net.remove_block(0)
    assert len(net.opt.m) == 4


def test_instability_ema_semantics_unchanged():
    net, X, y, _, _ = _deep_net(blocks=2)
    for _ in range(5):
        net.train_step(X, y)
    assert net._ema_dw.shape == net.W1.shape
    u = net.instability()
    assert u.shape == (net.H,)
