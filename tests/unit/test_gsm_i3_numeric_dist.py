"""GSM-I3 S2: the numeric_dist substrate mode
(EXEC_PLAN_GSM_I3 S2 tests 1-8, design boxes S2.1-S2.8)."""
import pickle   # safe: round-trips objects THIS test creates
                # in-process (artifact contract is pickle-based;
                # no untrusted data is ever loaded)
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.backends import set_compute_policy       # noqa: E402
from core.substrates.mlp import MLPSubstrate           # noqa: E402

DEVICES = [("torch", "cpu", "float32", 2e-3)]
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICES.append(("torch", "mps", "float32", 2e-3))
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _judge_after():
    yield
    set_compute_policy("numpy", "cpu", None)


def _hetero_data(n=200, seed=0, y_scale=1.0):
    """y = 2*x0 + noise growing with |x1| (heteroscedastic)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))
    noise = rng.normal(0, 1, n) * (0.05 + 0.5 * np.abs(X[:, 1]))
    y = (2.0 * X[:, 0] + noise) * y_scale
    return X, y


def _homo_data(n=400, eps=0.3, seed=1, y_scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))
    y = (2.0 * X[:, 0] + rng.normal(0, eps, n)) * y_scale
    return X, y


def _train(X, y, steps=400, hidden=16):
    org = MLPSubstrate(X.shape[1], hidden, mode="numeric_dist")
    for _ in range(steps):
        org.train_step(X, y)
    return org


def test_judge_nll_falls_and_shape_record():
    set_compute_policy("numpy", "cpu", None)
    X, y = _hetero_data()
    org = MLPSubstrate(2, 16, mode="numeric_dist")
    nlls = [org.train_step(X, y) for _ in range(120)]
    assert nlls[-1] < nlls[0]
    rec = org.shape_record()
    assert rec["mode"] == "numeric_dist"       # owning box (exec v1.1)
    assert rec["substrate"] == "mlp"


def test_birth_params_zero_and_first_step_loss_half():
    """Design 3.3 birth honesty, observable form (design 9.2)."""
    set_compute_policy("numpy", "cpu", None)
    org = MLPSubstrate(2, 16, mode="numeric_dist")
    assert np.all(org._bk.to_numpy(org.W2) == 0.0)
    assert np.all(org._bk.to_numpy(org.c) == 0.0)
    X, y = _homo_data()
    first = org.train_step(X, y)
    assert abs(first - 0.5) < 1e-6             # closed-form birth loss


@pytest.mark.parametrize("y_scale", [1.0, 1e6])
def test_absolute_calibration(y_scale):
    """Known-epsilon homoscedastic fixture: trained mean sigma in RAW
    units lands in a band around eps — scale-SENSITIVE by design;
    run at unit scale AND 1e6 (the extreme-data slice)."""
    set_compute_policy("numpy", "cpu", None)
    eps = 0.3 * y_scale
    X, y = _homo_data(eps=0.3, y_scale=y_scale)
    org = _train(X, y)
    _, std = org.predict_dist(X)
    assert np.isfinite(std).all()
    mean_sigma = float(std.mean())
    assert 0.5 * eps < mean_sigma < 2.0 * eps, (mean_sigma, eps)


def test_hetero_sigma_monotone():
    """Shape smoke: noise grows with |x1| -> learned sigma higher in
    the high-noise region than the low-noise region."""
    set_compute_policy("numpy", "cpu", None)
    X, y = _hetero_data(n=400)
    org = _train(X, y, steps=600)
    _, std = org.predict_dist(X)
    lo = std[np.abs(X[:, 1]) < 0.5].mean()
    hi = std[np.abs(X[:, 1]) > 1.5].mean()
    assert hi > lo


def test_prefit_predict_dist_refuses():
    set_compute_policy("numpy", "cpu", None)
    org = MLPSubstrate(2, 8, mode="numeric_dist")
    with pytest.raises(ValueError, match="untrained"):
        org.predict_dist(np.zeros((1, 2)))


def test_predict_equals_dist_value_bitwise():
    set_compute_policy("numpy", "cpu", None)
    X, y = _hetero_data()
    org = _train(X, y, steps=60)
    value, _ = org.predict_dist(X[:7])
    p = org.predict(X[:7])
    assert np.array_equal(p[:, 0], value)


@pytest.mark.parametrize("bk,dev,dt,tol", DEVICES)
def test_torch_parity_pickle_backcompat(bk, dev, dt, tol):
    """Trajectory parity per device; device-free pickle serves on the
    judge; and an old plain-numeric artifact still loads (the new
    mode changes nothing for existing artifacts)."""
    X, y = _hetero_data(n=64)

    def run():
        org = MLPSubstrate(2, 8, mode="numeric_dist")
        nlls = [org.train_step(X, y) for _ in range(40)]
        return org, nlls[-1]

    set_compute_policy("numpy", "cpu", None)
    _, nll_j = run()
    set_compute_policy(bk, dev, dt,
                       acknowledge_f32_precision=True)
    org_t, nll_t = run()
    assert abs(nll_j - nll_t) < tol
    v_dev, s_dev = org_t.predict_dist(X[:5])
    blob = pickle.dumps(org_t)
    set_compute_policy("numpy", "cpu", None)
    clone = pickle.loads(blob)
    v_j, s_j = clone.predict_dist(X[:5])
    assert np.max(np.abs(v_dev - v_j)) < tol
    assert np.max(np.abs(s_dev - s_j)) < tol * max(1.0, s_j.max())
    # back-compat: plain numeric organ round-trips untouched
    plain = MLPSubstrate(2, 8, mode="numeric")
    plain.train_step(X, y.reshape(-1, 1))
    p1 = plain.predict(X[:3])
    p2 = pickle.loads(pickle.dumps(plain)).predict(X[:3])
    assert np.array_equal(np.asarray(p1), np.asarray(p2))


def test_growth_and_spu_walk_on_numeric_dist():
    """Inner-net growth trains on; SPU host walk unaffected."""
    from reference_net.spu.spu_network import install_spu_policy
    set_compute_policy("numpy", "cpu", None)
    X, y = _hetero_data(n=100)
    org = MLPSubstrate(2, 8, mode="numeric_dist")
    for _ in range(40):
        org.train_step(X, y)
    sites = org.growth_sites()
    assert sites
    org.grow_site(sites[0][0], hidden=4)
    install_spu_policy(org, {"spu_enabled": True,
                             "spu_warmup_steps": 5,
                             "spu_newborn_steps": 30})
    nlls = [org.train_step(X, y) for _ in range(50)]
    assert np.isfinite(nlls).all()
    v, s = org.predict_dist(X[:5])
    assert np.isfinite(v).all() and np.isfinite(s).all()
