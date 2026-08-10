"""L1.3/L1.4 operator + integration tests (DEV_PLAN v1.2, 18)."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
sys.path.insert(0, str(REPO))
from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY as GP  # noqa: E402
from reference_net.net import Network                               # noqa: E402


@pytest.fixture
def loop_on():
    GP["loop_enabled"] = True
    yield
    GP["loop_enabled"] = False


def _shaped(seed=1, H=6, steps=120):
    net = Network(3, H, seed=seed)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    y = X[:, :1] * 2.0 + 1.0
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_1_refused_when_disabled():
    net, _, _ = _shaped()
    with pytest.raises(ValueError, match="opt-in option"):
        net.loop()


def test_2_timing_guard(loop_on):
    fresh = Network(3, 6, seed=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fresh.loop()
    assert any("taken shape" in str(x.message) for x in w)
    assert fresh._scale_events[0]["site"] == "loop"
    GP["grow_scale_guard"] = "refuse"
    try:
        fresh2 = Network(3, 6, seed=3)
        with pytest.raises(ValueError, match="taken shape"):
            fresh2.loop()
        assert fresh2.loop_block is None        # untouched
    finally:
        GP["grow_scale_guard"] = "warn"


def test_3_exact_entry_bitwise(loop_on):
    net, X, _ = _shaped()
    p0 = net._bk.to_numpy(net.predict(X)).copy()
    net.loop()
    assert np.array_equal(p0, net._bk.to_numpy(net.predict(X)))


def test_4_budget_one_and_remedy(loop_on):
    net, _, _ = _shaped()
    net.loop()
    with pytest.raises(ValueError, match="remove_loop first"):
        net.loop()
    net.remove_loop()
    assert net.loop() is not None


def test_5_remove_untrained_restores_bitwise(loop_on):
    net, X, _ = _shaped()
    p0 = net._bk.to_numpy(net.predict(X)).copy()
    net.loop()
    net.remove_loop()
    assert np.array_equal(p0, net._bk.to_numpy(net.predict(X)))


def test_6_implicit_toy_smoke(loop_on):
    # y solving y = cos(x*y): lambda should train fine (smoke,
    # NOT the exam); matched no-lambda control for reference only
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, size=(256, 1))
    y = np.ones_like(x)
    for _ in range(60):                          # reference solver
        y = np.cos(x * y)
    Xf = np.hstack([x, x ** 2, np.ones_like(x)])
    net, _, _ = _shaped(H=8)
    for _ in range(30):
        net.train_step(Xf, y)
    net.loop()
    for _ in range(400):
        m_l = net.train_step(Xf, y)
    assert np.isfinite(m_l) and m_l < 1.0


def test_7_integration_level_fd(loop_on):
    for with_blocks in (False, True):
        net, X, y = _shaped(seed=4, H=5, steps=110)
        if with_blocks:
            net.deepen(4)
        net.loop(3)
        for _ in range(5):
            net.train_step(X, y)
        Xs = net._std_x(net._bk.ingest(X))
        ys = (net._bk.ingest(y) - net._y_mu) / net._y_sd
        grads, _ = net._grads(Xs, ys)
        params = [net.W1, net.b1, net.W2, net.c]
        for blk in net.blocks:
            params += [blk["Bin"], blk["bb"], blk["Bout"]]
        lb = net.loop_block
        params += [lb["L_in"], lb["b_l"], lb["L_out"]]

        def loss():
            g, aux = net._grads(Xs, ys)
            return aux["mse"] / 2.0              # d(mse/2) = grads*?
        # FD on a SAMPLE of entries per param (full sweep is the
        # kernel test's job; here we verify the composition)
        eps = 1e-6
        rng = np.random.default_rng(0)
        for pi, (arr, g) in enumerate(zip(params, grads)):
            flat_idx = rng.choice(arr.size,
                                  size=min(4, arr.size),
                                  replace=False)
            for fi in flat_idx:
                ix = np.unravel_index(fi, arr.shape)
                orig = arr[ix]
                arr[ix] = orig + eps
                lp = net._grads(Xs, ys)[1]["mse"]
                arr[ix] = orig - eps
                lm = net._grads(Xs, ys)[1]["mse"]
                arr[ix] = orig
                fd = (lp - lm) / (2 * eps) / 2.0
                assert abs(fd - g[ix]) < 5e-4, (with_blocks, pi, ix)


def test_8_handoff_alive_under_loop(loop_on):
    net, X, y = _shaped(H=8)
    GP["grow_scale_guard"] = "warn"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
        net.loop()
        for _ in range(30):
            m = net.train_step(X, y)
    child = net.grown_body(0)
    assert child._step_count == 30               # dH0 path live
    assert np.isfinite(m)


def test_9_adam_order_stable(loop_on):
    net, X, y = _shaped()
    net.deepen(4)
    net.loop(3)
    assert len(net._opt_shapes()) == 4 + 3 + 3
    net.train_step(X, y)
    net.remove_loop()
    assert len(net._opt_shapes()) == 4 + 3
    net.train_step(X, y)                         # no shape crash


def test_10_projection(loop_on):
    net, X, y = _shaped()
    net.loop(3)
    lb = net.loop_block
    lb["L_out"] = lb["L_out"] + 5.0              # inflate
    n0 = net._loop_projections
    net.train_step(X, y)
    from engine.loop_ops import loop_rho_hat
    rho = loop_rho_hat(net._bk.to_numpy(lb["L_in"]),
                       net._bk.to_numpy(lb["L_out"]))
    assert net._loop_projections > n0
    assert rho <= GP["loop_rho_max"] + 1e-12


def test_11_serving_purity_red_line(loop_on):
    net, X, y = _shaped()
    net.loop()
    net.train_step(X, y)
    before = pickle.dumps(net)
    net.predict(X)
    assert pickle.dumps(net) == before           # bitwise sweep


def test_12_k_stats_training_only(loop_on):
    net, X, y = _shaped()
    net.loop()
    assert net._loop_k_last is None
    net.predict(X)
    assert net._loop_k_last is None              # predict records nothing
    net.train_step(X, y)
    assert net._loop_k_last is not None
    assert net._loop_k_ema is not None


def test_13_pickle_round_trip_and_old_artifacts(loop_on):
    net, X, y = _shaped()
    net.loop()
    for _ in range(20):
        net.train_step(X, y)
    clone = pickle.loads(pickle.dumps(net))
    assert np.array_equal(net._bk.to_numpy(net.predict(X)),
                          clone._bk.to_numpy(clone.predict(X)))
    old = Network(3, 6, seed=9)                  # never looped
    st = old.__getstate__()
    st.pop("loop_block", None)                   # pre-lambda artifact
    reborn = Network.__new__(Network)
    reborn.__setstate__(st)
    assert reborn.loop_block is None
    reborn.predict(np.zeros((2, 3)))             # serves fine


def test_14_widen_on_looped_scope_extends(loop_on):
    """CONSCIOUS UPDATE (60D D-8): the historical refusal
    ("omega-on-looped-scope integration is a next-round
    item") is LIFTED — 60D IS that next round. Widening a
    looped scope now ZERO-EXTENDS L_in (+k zero columns) and
    L_out (+k zero rows): value-exact serve (full-proof
    boxes: test_aspect_ratio_gate.py T-19/T-20; the 4-ulp
    cross-width BLAS line is the 60D s3 R-3 adjudication)."""
    from core.plasticity.net_ops import widen_net
    net, X, _ = _shaped()
    net.loop()
    pre = np.asarray(net._bk.to_numpy(net.predict(X))).copy()
    out = widen_net(net, 2)
    assert isinstance(out, dict)
    lb = net.loop_block
    assert np.asarray(lb["L_in"]).shape[1] == net.H
    assert np.asarray(lb["L_out"]).shape[0] == net.H
    assert np.all(np.asarray(lb["L_in"])[:, -2:] == 0.0)
    assert np.all(np.asarray(lb["L_out"])[-2:, :] == 0.0)
    post = np.asarray(net._bk.to_numpy(net.predict(X)))
    assert np.max(np.abs(pre - post)) <= \
        4 * np.spacing(np.max(np.abs(pre)))


def test_15_structure_and_audit_fields(loop_on):
    net, X, y = _shaped()
    net.loop()
    net.train_step(X, y)
    row = net.structure()[0]
    assert row["loop"] is True
    assert net._loop_k_last >= 1
    assert isinstance(net._loop_projections, int)


def test_16_ledger_events(loop_on):
    net, _, _ = _shaped(H=6)
    net.loop(3)
    ev = net.gain_ledger[-1]
    assert ev["event"] == "loop"
    assert ev["params_added"] == 2 * 3 * 6 + 3
    net.remove_loop()
    assert net.gain_ledger[-1]["event"] == "remove_loop"


def test_17_accounting(loop_on):
    net, _, _ = _shaped(H=6)
    n0 = net.n_params()
    net.loop(3)
    assert net.n_params() - n0 == 2 * 3 * 6 + 3  # hand computation


def test_18_refound_disclosure(loop_on):
    # the rebuilt organ carries no loop block (disclosed
    # non-preservation) — pinned at the Network level: a fresh
    # Network built from the same data lineage starts loop-free
    net, X, y = _shaped()
    net.loop()
    rebuilt = Network(3, 6, seed=1)
    for _ in range(120):
        rebuilt.train_step(X, y)
    assert rebuilt.loop_block is None
