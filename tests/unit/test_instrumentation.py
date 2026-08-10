"""S1 instrumentation tests (DEV_PLAN inventory, 14 tests).

Everything here observes; nothing may alter training numerics —
the golden-fixture guards in test_golden_fixtures.py enforce that
side of the contract.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network  # noqa: E402


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.sin(X[:, 0]).reshape(-1, 1)
    return X, y


def _net(seed=1, hidden=6):
    return Network(d_in=3, hidden=hidden, lr=1e-2, seed=seed)


def test_residual_energy_root_matches_hand_ema():
    net, (X, y) = _net(), _data()
    hand, b = None, None
    for _ in range(20):
        mse = net.train_step(X, y)
        b = net.energy_beta
        hand = mse if hand is None else b * hand + (1.0 - b) * mse
    assert net.residual_energy() == hand


def test_residual_energy_inner_matches_rj_stream():
    """LEGACY rj-stream identity (the eta-handoff residual feeds
    the inner scope's energy instrumentation) — exercised on a
    legacy-shaped composite, the artifact-serving path this
    instrument belongs to. PORT bodies train on raw gradients:
    their health reads through the port instruments (doc 35 D5
    loading/instability); their residual_energy stays None."""
    from reference_net.net import Network as _N
    net, (X, y) = _net(), _data()
    for _ in range(30):
        net.train_step(X, y)
    net.inner[1] = _N(net.d_in, 4, lr=net.lr, seed=11,
                      zero_out=True)          # legacy shape
    inner = net.inner[1]
    for _ in range(25):
        net.train_step(X, y)
    # the inner scope trains once per parent step -> its stream exists
    assert inner.residual_energy() is not None
    assert inner.residual_energy() > 0.0
    assert len(inner.energy_ring) == 25
    # port-body: gradient-trained scopes observe the Reading-B
    # target-equivalent stream t = pred - eta*dU (in-batch
    # disclosed addition, doc 38) — energy serves there too
    net.grow(0, hidden=4)
    for _ in range(5):
        net.train_step(X, y)
    assert net.grown_body(0).residual_energy() is not None
    assert len(net.grown_body(0).energy_ring) == 5
    assert net._port_site.instability()       # D5 instrument serves


def test_energy_ring_maxlen_and_contents():
    net, (X, y) = _net(), _data()
    for _ in range(60):
        net.train_step(X, y)
    assert len(net.energy_ring) == 60
    assert net.energy_ring[-1] == net.residual_energy()
    assert net.energy_ring.maxlen == 2048


def test_gain_ledger_event_fields_on_grow():
    net, (X, y) = _net(), _data()
    for _ in range(10):
        net.train_step(X, y)
    net.grow(0, hidden=4)
    (rec,) = net.gain_ledger
    assert rec["event"] == "refine" and rec["site"] == 0
    assert rec["params_added"] == net.grown_body(0).n_params() \
        + net.out_width * net.H   # body + assembly A_g
    assert rec["E_before"] == net.residual_energy()
    assert rec["E_after"] is None and rec["gain"] is None
    assert rec["due"] == rec["step"] + net.gain_horizon


def test_gain_ledger_horizon_resolution_at_W():
    net, (X, y) = _net(), _data()
    net.gain_horizon = 5
    for _ in range(10):
        net.train_step(X, y)
    net.grow(0, hidden=4)
    for _ in range(4):
        net.train_step(X, y)
    assert net.gain_ledger[0]["E_after"] is None       # not due yet
    net.train_step(X, y)
    rec = net.gain_ledger[0]
    assert rec["E_after"] == net.residual_energy()
    eb = rec["E_before"]
    assert np.isclose(rec["gain"], (eb - rec["E_after"]) / eb)


def test_gain_ledger_overlapping_events_each_resolve():
    net, (X, y) = _net(), _data()
    net.gain_horizon = 6
    for _ in range(5):
        net.train_step(X, y)
    net.grow(0, hidden=3)
    for _ in range(3):
        net.train_step(X, y)
    net.grow(1, hidden=3)
    for _ in range(10):
        net.train_step(X, y)
    assert all(r["E_after"] is not None for r in net.gain_ledger)
    assert net.gain_ledger[0]["due"] != net.gain_ledger[1]["due"]
    assert not net._pending_gain


def test_gain_ledger_pending_survives_save_load():
    net, (X, y) = _net(), _data()
    net.gain_horizon = 8
    for _ in range(5):
        net.train_step(X, y)
    net.grow(0, hidden=3)
    blob = pickle.dumps(net)
    net2 = pickle.loads(blob)
    assert net2._pending_gain == net._pending_gain
    for _ in range(8):
        net2.train_step(X, y)
    assert net2.gain_ledger[0]["E_after"] is not None


def test_window_ring_stores_as_received():
    net, (X, y) = _net(), _data(n=8)
    net.train_step(X, y)
    rows = list(net.window_ring)
    assert len(rows) == 8
    for i, (xr, yr) in enumerate(rows):
        assert np.array_equal(xr, X[i])
        assert np.array_equal(yr, y[i])


def test_window_ring_fifo_maxlen():
    net, (X, y) = _net(), _data(n=64)
    for _ in range(10):                       # 640 rows > 512
        net.train_step(X, y)
    assert len(net.window_ring) == 512
    xr, yr = net.window_ring[-1]
    assert np.array_equal(xr, X[-1]) and np.array_equal(yr, y[-1])


def test_cosine_series_alternating_sign_detection():
    net = _net()
    v = np.ones((6, 3))
    for k in range(6):
        net._record_update_direction(v if k % 2 == 0 else -v)
    cosines = [c for _, c in list(net._cos_series)[1:]]
    assert all(np.isclose(c, -1.0) for c in cosines)


def test_cosine_series_saturation_signature():
    net = _net()
    for _ in range(6):
        net._record_update_direction(np.full((6, 3), 1e-9))
    norms = [n for n, _ in net._cos_series]
    assert all(n < 1e-6 for n in norms)


def test_step_count_increments():
    net, (X, y) = _net(), _data()
    for k in range(7):
        net.train_step(X, y)
    assert net._step_count == 7


def test_old_pickle_without_instrumentation_loads():
    net, (X, y) = _net(), _data()
    net.train_step(X, y)
    state = net.__dict__.copy()
    for key in ("_step_count", "_E", "energy_ring", "window_ring",
                "gain_ledger", "_pending_gain", "_cos_series",
                "_prev_dw", "energy_beta", "gain_horizon"):
        state.pop(key, None)
    revived = Network.__new__(Network)
    revived.__setstate__(state)
    assert revived._step_count == 0 and revived.gain_ledger == []
    revived.train_step(X, y)                   # runs fine post-revival
    assert revived._step_count == 1


def test_instrumentation_zero_numeric_effect_spot():
    # spot check beyond the golden fixtures: two fresh nets, one with
    # rings pre-warmed via observation calls, train identically
    (X, y) = _data()
    a, b = _net(seed=5), _net(seed=5)
    a._record_update_direction(np.ones((6, 3)))   # observe-only calls
    la = [a.train_step(X, y) for _ in range(10)]
    lb = [b.train_step(X, y) for _ in range(10)]
    assert la == lb


def test_cosine_series_survives_shape_change():
    # omega widening / sigma features change W1's shape mid-life; the
    # direction series must continue without a cross-shape cosine
    net = _net()
    net._record_update_direction(np.ones((6, 3)))
    net._record_update_direction(np.ones((8, 3)))     # widened shape
    net._record_update_direction(np.ones((8, 3)))
    norms_cos = list(net._cos_series)
    assert len(norms_cos) == 3
    assert norms_cos[1][1] == 0.0                     # no cross-shape cos
    assert np.isclose(norms_cos[2][1], 1.0)
