"""GSM-I3 S1: the K16 numeric-uncertainty head kernels
(EXEC_PLAN_GSM_I3 S1 tests 1-5, design boxes S1.1-S1.5)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.backends.numpy_backend import NumpyBackend  # noqa: E402
from reference_net.net import gelu                             # noqa: E402

BK = NumpyBackend()

DEVICES = [("cpu", "float32", 2e-3)]
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICES.append(("mps", "float32", 2e-3))
except ImportError:
    torch = None


def _fixture(n=12, d=3, H=6, seed=0):
    rng = np.random.default_rng(seed)
    Xs = rng.normal(size=(n, d))
    W1 = rng.normal(0, 0.5, (H, d))   # host convention: A = X @ W1.T
    b1 = rng.normal(0, 0.1, H)
    W2 = rng.normal(0, 0.3, (2, H))
    c = rng.normal(0, 0.1, 2)
    t = rng.normal(size=n)
    return Xs, W1, b1, W2, c, t


def _loss_of(Xs, W1, b1, W2, c, t):
    A = Xs @ W1.T + b1
    Hact = gelu(A)
    mu, v = BK.nll_forward(W2, c, Hact)
    return BK.nll_loss(t, mu, v)


def _grads(Xs, W1, b1, W2, c, t):
    A = Xs @ W1.T + b1
    Hact = gelu(A)
    mu, v = BK.nll_forward(W2, c, Hact)
    return BK.nll_backward(W1, b1, W2, c, Xs, A, Hact, mu, v, t)


def test_nll_fd_all_four_leaf_grads():
    """Entry-by-entry FD of gW1, gb1, gW2, gc — including entries
    in BOTH the mu row and the v row of W2/c."""
    Xs, W1, b1, W2, c, t = _fixture()
    grads, _ = _grads(Xs, W1, b1, W2, c, t)
    eps = 1e-6
    for arr, g in zip([W1, b1, W2, c], grads):
        it = np.nditer(arr, flags=["multi_index"])
        count = 0
        for _ in it:
            ix = it.multi_index
            keep = arr[ix]
            arr[ix] = keep + eps
            lp = _loss_of(Xs, W1, b1, W2, c, t)
            arr[ix] = keep - eps
            lm = _loss_of(Xs, W1, b1, W2, c, t)
            arr[ix] = keep
            fd = (lp - lm) / (2 * eps)
            assert abs(fd - g[ix]) < 3e-5 * max(1.0, abs(fd)), \
                (ix, fd, g[ix])
            count += 1
            if arr.ndim == 2 and arr.shape[0] == 2:
                continue                 # W2: sweep BOTH rows fully
            if count >= 8:
                break


def test_nll_fd_dH_wrt_Hact():
    """dH's FD axis is Hact, not the parameters."""
    Xs, W1, b1, W2, c, t = _fixture()
    A = Xs @ W1.T + b1
    Hact = gelu(A)
    mu, v = BK.nll_forward(W2, c, Hact)
    _, dH = BK.nll_backward(W1, b1, W2, c, Xs, A, Hact, mu, v, t)

    def loss_at_H(Hm):
        mu2, v2 = BK.nll_forward(W2, c, Hm)
        return BK.nll_loss(t, mu2, v2)

    eps = 1e-6
    n = len(Xs)
    for (i, j) in [(0, 0), (3, 2), (11, 5), (7, 1)]:
        keep = Hact[i, j]
        Hact[i, j] = keep + eps
        lp = loss_at_H(Hact)
        Hact[i, j] = keep - eps
        lm = loss_at_H(Hact)
        Hact[i, j] = keep
        fd = (lp - lm) / (2 * eps)
        # dH is per-sample (K14 convention): dJ/dHact = dH / n
        assert abs(fd - dH[i, j] / n) < 3e-5 * max(1.0, abs(fd)), \
            ((i, j), fd, dH[i, j] / n)


def test_clamp_region_masked_grad_zero():
    """Force saturation (v-row of W2 zeroed, c[1]=12 -> v == 10
    clamped): the v-column gradient is exactly 0, and FD agrees
    (the loss is flat under the clamp)."""
    Xs, W1, b1, W2, c, t = _fixture()
    W2[1, :] = 0.0
    c[1] = 12.0
    A = Xs @ W1.T + b1
    Hact = gelu(A)
    mu, v = BK.nll_forward(W2, c, Hact)
    assert np.all(v == 10.0)                    # clamped exactly
    grads, _ = BK.nll_backward(W1, b1, W2, c, Xs, A, Hact, mu, v, t)
    assert np.all(grads[2][1, :] == 0.0)        # gW2 v-row
    assert grads[3][1] == 0.0                   # gc v-entry
    eps = 1e-6
    keep = c[1]
    c[1] = keep + eps
    lp = _loss_of(Xs, W1, b1, W2, c, t)
    c[1] = keep - eps
    lm = _loss_of(Xs, W1, b1, W2, c, t)
    c[1] = keep
    assert lp == lm                             # flat under clamp


@pytest.mark.parametrize("dev,dt,tol", DEVICES)
def test_torch_kernel_parity(dev, dt, tol):
    """KERNEL-level parity: identical pinned inputs through
    forward/loss/backward, plus a short raw-array SGD loop."""
    if torch is None:
        pytest.skip("torch not installed")
    from engine.backends.torch_backend import TorchBackend
    tbk = TorchBackend(device=dev, dtype=dt)

    Xs, W1, b1, W2, c, t = _fixture()
    lr = 0.05
    # judge loop
    j = [W1.copy(), b1.copy(), W2.copy(), c.copy()]
    losses_j = []
    for _ in range(20):
        A = Xs @ j[0].T + j[1]
        Hact = gelu(A)
        mu, v = BK.nll_forward(j[2], j[3], Hact)
        losses_j.append(BK.nll_loss(t, mu, v))
        grads, _ = BK.nll_backward(j[0], j[1], j[2], j[3],
                                   Xs, A, Hact, mu, v, t)
        j = [p - lr * g for p, g in zip(j, grads)]
    # device loop
    Xt, tt = tbk.ingest(Xs), tbk.ingest(t)
    d = [tbk.ingest(W1), tbk.ingest(b1),
         tbk.ingest(W2), tbk.ingest(c)]
    losses_t = []
    for _ in range(20):
        A = Xt @ d[0].T + d[1]
        Hact = tbk.gelu(A)
        mu, v = tbk.nll_forward(d[2], d[3], Hact)
        losses_t.append(tbk.nll_loss(tt, mu, v))
        grads, _ = tbk.nll_backward(d[0], d[1], d[2], d[3],
                                    Xt, A, Hact, mu, v, tt)
        d = [p - lr * g for p, g in zip(d, grads)]
    assert losses_j[-1] < losses_j[0]           # it learns
    assert abs(losses_j[-1] - losses_t[-1]) < tol
    for pj, pt in zip(j, d):
        scale = max(1.0, float(np.abs(pj).max()))
        assert float(np.abs(pj - tbk.to_numpy(pt)).max()) / scale \
            < tol


def test_loss_hand_computed_3_samples():
    t = np.array([1.0, 0.0, -1.0])
    mu = np.array([0.5, 0.0, 0.0])
    v = np.array([0.2, 0.0, -0.1])
    assert abs(BK.nll_loss(t, mu, v)
               - 0.23497560105752383) < 1e-12
