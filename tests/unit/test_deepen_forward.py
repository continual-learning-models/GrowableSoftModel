"""S2 delta-forward tests (DEV_PLAN inventory, 15 tests)."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network, gelu  # noqa: E402
from tests.fixtures.make_golden import build_reference  # noqa: E402


def _trained(seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 3))
    y = np.cos(X[:, 0] * 1.5).reshape(-1, 1)
    net = Network(d_in=3, hidden=6, lr=1e-2, seed=seed)
    for _ in range(60):
        net.train_step(X, y)
    net.grow(2, hidden=4)
    for _ in range(40):
        net.train_step(X, y)
    return net, X, y


def test_deepen_exact_at_root_maxdiff_zero():
    net, X, _ = _trained()
    Xq = np.random.default_rng(11).normal(size=(512, 3))
    before = net.predict(Xq)
    net.deepen()
    after = net.predict(Xq)
    assert np.array_equal(before, after)


def test_deepen_exact_at_inner_scope():
    net, X, _ = _trained()
    Xq = np.random.default_rng(12).normal(size=(512, 3))
    before = net.predict(Xq)
    net.grown_body(2).deepen()
    after = net.predict(Xq)
    assert np.array_equal(before, after)


def test_deepen_default_width_equals_H():
    net, _, _ = _trained()
    net.deepen()
    blk = net.blocks[0]
    assert blk["Bin"].shape == (net.H, net.H)
    assert blk["Bout"].shape == (net.H, net.H)


def test_deepen_custom_width():
    net, _, _ = _trained()
    net.deepen(m=3)
    assert net.blocks[0]["Bin"].shape == (3, net.H)


def test_deepen_on_H1_scope():
    net = Network(d_in=2, hidden=1, lr=1e-2, seed=1)
    idx = net.deepen()
    assert idx == 0 and net.blocks[0]["Bin"].shape == (1, 1)


def test_deepen_before_first_train_legal():
    net = Network(d_in=2, hidden=4, lr=1e-2, seed=1)
    net.deepen()
    out = net.predict(np.zeros((5, 2)))
    assert out.shape == (5, 1)


def test_serial_depth_counts_blocks():
    net, _, _ = _trained()
    assert net.serial_depth() == 1
    net.deepen(); net.deepen()
    assert net.serial_depth() == 3
    assert net.grown_body(2).serial_depth() == 1


def test_n_params_includes_blocks():
    net, _, _ = _trained()
    base = net.n_params()
    net.deepen(m=4)
    assert net.n_params() == base + (4 * net.H + 4 + net.H * 4)


def test_structure_rows_report_blocks():
    net, _, _ = _trained()
    net.deepen()
    root_row = net.structure()[0]
    assert root_row["blocks"] == 1


def test_remove_block_param_count_falls():
    net, _, _ = _trained()
    base = net.n_params()
    net.deepen(m=4)
    net.remove_block(0)
    assert net.n_params() == base
    assert net.blocks == []


def test_remove_block_logged():
    net, _, _ = _trained()
    net.deepen(m=4)
    net.remove_block(0)
    events = [r["event"] for r in net.gain_ledger]
    assert events[-2:] == ["deepen", "prune_block"]
    assert net.gain_ledger[-1]["params_added"] < 0


def test_forward_hand_computed_one_block():
    net = Network(d_in=2, hidden=3, lr=1e-2, seed=5)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = X[:, :1] * 0.5
    for _ in range(30):
        net.train_step(X, y)
    net.deepen(m=2)
    blk = net.blocks[0]
    blk["Bout"] = rng.normal(size=blk["Bout"].shape) * 0.1
    Xq = rng.normal(size=(16, 2))
    Xs = net._std_x(Xq)
    _, H0 = net._hidden(Xs)
    Z = H0 @ blk["Bin"].T + blk["bb"]
    HL = H0 + gelu(Z) @ blk["Bout"].T
    hand = (HL @ net.W2.T + net.c) * net._y_sd + net._y_mu
    assert np.max(np.abs(hand - net.predict(Xq))) < 1e-14


def test_save_load_with_blocks_roundtrip():
    net, _, _ = _trained()
    net.deepen()
    net.blocks[0]["Bout"][:] = 0.05
    Xq = np.random.default_rng(13).normal(size=(64, 3))
    before = net.predict(Xq)
    net2 = pickle.loads(pickle.dumps(net))
    assert np.array_equal(before, net2.predict(Xq))
    assert net2.serial_depth() == 2


def test_ledger_event_on_deepen():
    net, _, _ = _trained()
    net.deepen(m=4)
    rec = net.gain_ledger[-1]
    assert rec["event"] == "deepen"
    assert rec["params_added"] == 4 * net.H + 4 + net.H * 4
    assert rec["E_before"] == net.residual_energy()


def test_golden_fixtures_bit_identity_after_S2():
    net, X, y = build_reference()
    ref_p = np.load(Path(ROOT / "tests/fixtures/golden_predict_gp1.npz"))
    assert np.array_equal(net.predict(ref_p["Xq"]), ref_p["pred"])
    ref_t = np.load(Path(ROOT / "tests/fixtures/golden_train_gp1.npz"))
    losses = np.array([net.train_step(X, y) for _ in range(50)])
    assert np.array_equal(losses, ref_t["losses"])
