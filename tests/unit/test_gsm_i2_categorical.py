"""GSM-I2 S1: the categorical head on the kernel contract."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.backends import (resolve_backend,        # noqa: E402
                               set_compute_policy)
from core.substrates.mlp import MLPSubstrate           # noqa: E402

DEVICES = [("torch", "cpu", "float32", 1e-3)]
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICES.append(("torch", "mps", "float32", 1e-3))
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _judge_after():
    yield
    set_compute_policy("numpy", "cpu", None)


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = ["a" if v > 0 else "b" for v in X[:, 0]]
    return X, y


def _run(steps=30):
    X, y = _data()
    org = MLPSubstrate(3, 8, mode="categorical",
                       vocab=["a", "b"])
    ces = [org.train_step(X, y) for _ in range(steps)]
    return org, ces


def test_judge_trajectory_and_probs():
    set_compute_policy("numpy", "cpu", None)
    org, ces = _run()
    assert ces[-1] < ces[0]
    p = org.predict_proba(np.array([[1.5, 0, 0]]))
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0][0] > 0.5


def test_ce_gradients_fd_on_judge():
    """Entry-by-entry FD of the relocated CE gradients."""
    set_compute_policy("numpy", "cpu", None)
    org, _ = _run(steps=5)
    X, y = _data(n=12, seed=3)
    Xnp = np.asarray(X, float)
    labels = [org.vocab.index(v) for v in y]
    n = len(Xnp)
    bk = org._bk

    def ce_of():
        Xs = org._std_x(bk.ingest(Xnp))
        A, H = org._hidden(Xs)
        probs = bk.cat_forward(org.W2, org.c, H)
        onehot = np.zeros((n, 2))
        onehot[np.arange(n), labels] = 1.0
        return bk.cat_ce(probs, bk.ingest(onehot))

    Xs = org._std_x(bk.ingest(Xnp))
    A, H = org._hidden(Xs)
    probs = bk.cat_forward(org.W2, org.c, H)
    onehot = np.zeros((n, 2))
    onehot[np.arange(n), labels] = 1.0
    grads, _ = bk.cat_backward(org.W1, org.b1, org.W2, org.c,
                               Xs, A, H, probs,
                               bk.ingest(onehot))
    eps = 1e-6
    for arr, g in zip([org.W1, org.b1, org.W2, org.c], grads):
        it = np.nditer(arr, flags=["multi_index"])
        count = 0
        for _ in it:
            ix = it.multi_index
            orig = arr[ix]
            arr[ix] = orig + eps
            lp = ce_of()
            arr[ix] = orig - eps
            lm = ce_of()
            arr[ix] = orig
            fd = (lp - lm) / (2 * eps)
            assert abs(fd - g[ix]) < 5e-5, (ix, fd, g[ix])
            count += 1
            if count >= 6:
                break                    # 6 entries per tensor


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_torch_trajectory_parity(bk, dev, dt, tol):
    set_compute_policy("numpy", "cpu", None)
    _, ces_j = _run()
    set_compute_policy(bk, dev, dt,
                       acknowledge_f32_precision=True)
    org_t, ces_t = _run()
    assert abs(ces_j[-1] - ces_t[-1]) < tol
    p = org_t.predict_proba(np.array([[1.5, 0, 0]]))
    assert abs(p.sum() - 1.0) < 1e-5


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_add_class_exact_on_device(bk, dev, dt, tol):
    set_compute_policy(bk, dev, dt,
                       acknowledge_f32_precision=True)
    org, _ = _run()
    X, _ = _data()
    p_before = org.predict_proba(X[:6])
    org.add_class("c")
    p_after = org.predict_proba(X[:6])
    # old-class relative ordering preserved; new class ~silent
    assert np.array_equal(np.argmax(p_before, 1),
                          np.argmax(p_after[:, :2], 1))
    assert p_after[:, 2].max() < 0.05
    org.train_step(X[:10], ["c"] * 10)     # trains after growth


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_device_free_pickle_serves_on_judge(bk, dev, dt, tol):
    set_compute_policy(bk, dev, dt,
                       acknowledge_f32_precision=True)
    org, _ = _run()
    X, _ = _data()
    p_dev = org.predict_proba(X[:5])
    blob = pickle.dumps(org)
    set_compute_policy("numpy", "cpu", None)
    clone = pickle.loads(blob)
    p_j = clone.predict_proba(X[:5])
    assert np.max(np.abs(p_dev - p_j)) < tol


def test_old_numpy_artifact_loads():
    """Back-compat: a categorical organ pickled pre-I2 (raw
    numpy head) must load and serve."""
    set_compute_policy("numpy", "cpu", None)
    org, _ = _run()
    # simulate an old artifact: strip to numpy raw
    org.W2 = np.asarray(org._bk.to_numpy(org.W2))
    org.c = np.asarray(org._bk.to_numpy(org.c))
    blob = pickle.dumps(org)
    clone = pickle.loads(blob)
    p = clone.predict_proba(np.array([[1.0, 0, 0]]))
    assert abs(p.sum() - 1.0) < 1e-9


def test_perturb_on_device():
    set_compute_policy(*DEVICES[0][:3],
                       acknowledge_f32_precision=True)
    org, _ = _run(steps=10)
    rng = np.random.default_rng(0)
    p = org.perturb(rng, 0.05)
    X, _ = _data()
    out = p.predict_proba(X[:4])
    assert np.isfinite(out).all()
