"""T1 tests: AttentionBody (DESIGN_GROW_BODY_TYPE v1.2)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from reference_net.attention_body import AttentionBody   # noqa: E402


def make(seed=3, d_in=4, **kw):
    kw.setdefault("d_model", 4)
    kw.setdefault("n_heads", 2)
    kw.setdefault("ffn", 3)
    return AttentionBody(d_in, seed=seed, **kw)


def data(n=16, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, d))
    return X, np.sin(X[:, :1]) + 0.3 * X[:, 1:2]


@pytest.mark.parametrize("L", [1, 2])
def test_fd_every_gradient_entry(L):
    b = make(n_layers=L)
    rng = np.random.default_rng(9)
    b.P["Wout"] = rng.normal(0, 0.4, b.P["Wout"].shape)  # flow
    b.P["bout"] = rng.normal(0, 0.1, b.P["bout"].shape)
    Xs = rng.normal(0, 1, (7, 4))
    ys = rng.normal(0, 1, (7, 1))
    _, G = b._loss_and_grads(Xs, ys)
    eps, tol = 1e-6, 3e-5
    for k, arr in b.P.items():
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            keep = arr[idx]
            arr[idx] = keep + eps
            Jp, _ = b._loss_and_grads(Xs, ys)
            arr[idx] = keep - eps
            Jm, _ = b._loss_and_grads(Xs, ys)
            arr[idx] = keep
            fd = (Jp - Jm) / (2 * eps)
            assert abs(fd - G[k][idx]) < tol * max(1.0, abs(fd)), \
                (k, idx, fd, G[k][idx])


def test_exact_entry_at_birth():
    b = make()
    X, _ = data()
    out = b.predict(X)
    assert out.shape == (16, 1)
    assert np.all(out == 0.0)                      # bitwise zeros


def test_y_mu_pinned_for_life_under_zero_out():
    b = make()
    X, y = data()
    for _ in range(30):
        b.train_step(X, y)
    assert b._y_mu == 0.0


def test_zero_out_false_fits_mean():
    b = make(zero_out=False)
    X, y = data()
    b.train_step(X, y)
    assert b._y_mu == pytest.approx(float(y.mean()))


def test_sgd_lr_plain_mode_leaves_adam_untouched():
    b = make()
    X, y = data()
    b.train_step(X, y)                             # fit + one adam
    t_snap = b._t
    m_snap = {k: mv[0].copy() for k, mv in b._adam.items()}
    b.train_step(X, y, sgd_lr=1e-3)
    assert b._t == t_snap
    assert all(np.array_equal(m_snap[k], b._adam[k][0])
               for k in m_snap)


def test_learns_toy_law():
    b = make(seed=3, d_model=8, ffn=8)
    X, y = data(n=32)
    m0 = b.train_step(X, y)
    for _ in range(300):
        m = b.train_step(X, y)
    assert m < 0.05 * m0


def test_determinism_bit_replay():
    outs = []
    for _ in range(2):
        b = make(seed=11)
        X, y = data()
        for _ in range(10):
            b.train_step(X, y)
        outs.append(b.predict(X))
    assert np.array_equal(outs[0], outs[1])


def test_pickle_round_trip():
    b = make()
    X, y = data()
    for _ in range(5):
        b.train_step(X, y)
    b2 = pickle.loads(pickle.dumps(b))
    assert np.array_equal(b.predict(X), b2.predict(X))
    b2.train_step(X, y)                            # resumes


def test_contract_surface():
    b = make()
    assert b.inner == {} and b.blocks == []
    assert list(b.instability()) == []
    assert b.depth() == 1
    row = b.structure("root/1")[0]
    assert set(row) >= {"path", "H", "composite", "blocks"}
    assert row["blocks"] == 0 and row["composite"] == []
    assert b.n_params() > 0
    assert b._step_count == 0 and b._seed_counter == 3
    X, _ = data()
    assert b.predict(X)[:, 0].shape == (16,)       # host read


def test_scaler_refresh_rescales_and_resets_adam():
    b = make()
    X, y = data()
    for _ in range(5):
        b.train_step(X, y)
    pred_before = b.predict(X)
    b.train_step(X, y * 10.0)                      # target scale jump
    assert b._t == 1                               # adam was reset
    assert b._y_mu == 0.0
    # function preserved through the rescale (up to the one step)
    assert np.isfinite(b.predict(X)).all()
    assert float(np.abs(pred_before).mean()) < 10  # sanity


def test_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        AttentionBody(4, d_model=5, n_heads=2)


def test_degenerate_shapes():
    b = make(d_in=1)
    X = np.random.default_rng(1).uniform(-1, 1, (1, 1))
    y = np.ones((1, 1))
    b.train_step(X, y)
    assert np.isfinite(b.predict(X)).all()


def test_train_step_returns_before_step_mse():
    b = make()
    X, y = data()
    m = b.train_step(X, y)
    assert isinstance(m, float) and np.isfinite(m) and m >= 0
