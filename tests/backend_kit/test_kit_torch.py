"""BACKEND CONFORMANCE KIT — BK1..BK10 (spec.md committed B0).
Runs per available device; skips cleanly when torch is absent."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
torch = pytest.importorskip("torch")
from engine.backends import (                        # noqa: E402
    get_default_backend, resolve_backend)
from reference_net.growthpolicy.pricer_zero_attach import (  # noqa: E402
    _fingerprint)
from reference_net.net import Network                       # noqa: E402
from reference_net.spu.spu_network import SPUNetwork        # noqa: E402
from engine.spu.spu_report import build_report       # noqa: E402

warnings.filterwarnings("ignore", message=".*scale-hierarchy.*")

DEVICES = [("cpu", "float64", 1e-5), ("cpu", "float32", 1e-3)]
if torch.backends.mps.is_available():
    DEVICES.append(("mps", "float32", 1e-3))
PARAMS = [pytest.param(d, t, tol, id=f"{d}-{t}")
          for d, t, tol in DEVICES]


def bk_of(device, dtype):
    return resolve_backend("torch", device=device, dtype=dtype)


def data(seed=0, n=48, d=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, d))
    return X, np.sin(3 * X[:, :1]) + 0.3 * X[:, 1:2] * X[:, 2:3]


def lifecycle(backend=None, cls=SPUNetwork, spu=True, steps=40):
    X, y = data()
    net = cls(4, 6, seed=7, backend=backend)
    if spu:
        net.set_spu_policy({"spu_enabled": True,
                            "spu_warmup_steps": 5,
                            "spu_newborn_steps": 80})
    for _ in range(steps):
        net.train_step(X, y)
    net.grow(1)
    for _ in range(20):
        net.train_step(X, y)
    net.grown_body(1).deepen(m=3)
    for _ in range(20):
        net.train_step(X, y)
    net.grow(2, body_type="attention")
    for _ in range(40):
        net.train_step(X, y)
    return net, X, y


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk1_kernel_parity(device, dtype, tol):
    bk = bk_of(device, dtype)
    nb = get_default_backend()
    rng = np.random.default_rng(3)
    W1 = rng.normal(0, 1, (5, 4))
    b1 = rng.normal(0, 0.3, 5)
    X = rng.normal(0, 1, (7, 4))
    A_n, h_n = nb.dense_forward(W1, b1, X)
    A_t, h_t = bk.dense_forward(bk.ingest(W1), bk.ingest(b1),
                                bk.ingest(X))
    assert np.allclose(bk.to_numpy(h_t), h_n, rtol=tol, atol=tol)


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk2_bk3_trajectory_and_integer_structure(device, dtype,
                                                  tol):
    bk = bk_of(device, dtype)
    nt, X, _ = lifecycle(backend=bk)
    nn, _, _ = lifecycle()
    p_t = bk.to_numpy(nt.predict(X)).astype(float)
    p_n = nn.predict(X)
    scale = max(1.0, float(np.abs(p_n).max()))
    assert float(np.abs(p_t - p_n).max()) / scale < tol   # BK2
    ev_t = [(e.get("path"), e.get("skip"), e.get("steps"),
             e.get("blocks")) for e in nt.spu_events]
    ev_n = [(e.get("path"), e.get("skip"), e.get("steps"),
             e.get("blocks")) for e in nn.spu_events]
    assert ev_t == ev_n                                    # BK3
    assert [e["event"] for e in nt.gain_ledger] \
        == [e["event"] for e in nn.gain_ledger]


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk4_bit_replay_same_device(device, dtype, tol):
    bk1 = bk_of(device, dtype)
    bk2 = bk_of(device, dtype)
    n1, X, _ = lifecycle(backend=bk1, steps=20)
    n2, _, _ = lifecycle(backend=bk2, steps=20)
    assert np.array_equal(bk1.to_numpy(n1.predict(X)),
                          bk2.to_numpy(n2.predict(X)))


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk5_serving_purity(device, dtype, tol):
    bk = bk_of(device, dtype)
    net, X, _ = lifecycle(backend=bk, steps=20)
    snap = {k: bk.to_numpy(v).copy() for k, v in
            (("W1", net.W1), ("W2", net.W2))}
    n_ev = len(net.spu_events)
    for _ in range(10):
        net.predict(X)
    assert np.array_equal(snap["W1"], bk.to_numpy(net.W1))
    assert len(net.spu_events) == n_ev


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk6_persistence_device_free(device, dtype, tol):
    bk = bk_of(device, dtype)
    net, X, _ = lifecycle(backend=bk, steps=20)
    blob = pickle.dumps(net)
    clone = pickle.loads(blob)              # loads onto the judge
    assert clone._bk is get_default_backend()
    assert isinstance(clone.W1, np.ndarray)
    p = bk.to_numpy(net.predict(X)).astype(float)
    assert float(np.abs(clone.predict(X) - p).max()) < max(
        tol, 1e-6)
    clone.train_step(X, np.sin(X[:, :1]))   # resumes on judge


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk7_lifecycle_entry_exactness(device, dtype, tol):
    bk = bk_of(device, dtype)
    X, y = data()
    net = Network(4, 6, seed=7, backend=bk)
    for _ in range(20):
        net.train_step(X, y)
    before = bk.to_numpy(net.predict(X)).copy()
    net.grow(1)                              # exact entry: bitwise
    assert np.array_equal(bk.to_numpy(net.predict(X)), before)
    net.grow(2, body_type="attention")
    assert np.array_equal(bk.to_numpy(net.predict(X)), before)
    net.grown_body(1).deepen(m=3)                 # zero block: bitwise
    assert np.array_equal(bk.to_numpy(net.predict(X)), before)
    net.remove_grown(2)                      # removal restores
    assert np.array_equal(bk.to_numpy(net.predict(X)), before)


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk8_boundary_surface(device, dtype, tol):
    bk = bk_of(device, dtype)
    net, X, _ = lifecycle(backend=bk, steps=20)
    fp1 = _fingerprint(net)                  # would crash unrouted
    assert fp1 == _fingerprint(net)
    with torch.no_grad():
        net.W1[0, 0] += 1.0
    assert _fingerprint(net) != fp1          # value-sensitive
    rows = net.structure()
    assert len(rows) == 3
    rep = build_report(net)
    assert rep["processed_steps"] > 0
    assert net.n_params() > 0


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk9_spu_discipline(device, dtype, tol):
    bk = bk_of(device, dtype)
    net, X, y = lifecycle(backend=bk, steps=20)
    m_snap = [bk.to_numpy(a).copy() for a in net.grown_body(1).opt.m]
    evs = [e for e in net.spu_events
           if e.get("skip") is None and "steps" in e]
    assert evs and all(e["steps"] <= 4 for e in evs)
    from engine.spu.spu_loop import self_process
    self_process(net.grown_body(1), bk.ingest(X), 
                 net.get_spu_policy(), 999)
    assert all(np.array_equal(a, bk.to_numpy(b))
               for a, b in zip(m_snap, net.grown_body(1).opt.m))
    unit = net.grown_body(1)
    Xs = bk.standardize(bk.ingest(X), unit._x_mu, unit._x_sd)
    _, g = bk.jinv_and_grads(
        unit.W1, unit.b1, unit.W2, unit.c, Xs,
        np.ones((4, unit.H)), blocks=unit.blocks)
    assert float(bk.to_numpy(g["c"]).max()) == 0.0


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk10_verdict_agreement(device, dtype, tol):
    """Mini pre-registered comparison adjudicated identically by
    both backends: with-SPU vs without at equal steps, 2 seeds."""
    def verdict(backend):
        wins = []
        for seed in (7, 8):
            rng = np.random.default_rng(seed)
            X = rng.uniform(-2, 2, (48, 4))
            y = np.sin(3 * X[:, :1])
            Xe = np.random.default_rng(seed + 100).uniform(
                -2, 2, (96, 4))
            ye = np.sin(3 * Xe[:, :1])
            mses = {}
            for spu in (True, False):
                net = SPUNetwork(4, 6, seed=seed, backend=backend)
                if spu:
                    net.set_spu_policy({"spu_enabled": True,
                                        "spu_warmup_steps": 5,
                                        "spu_newborn_steps": 80})
                for step in range(120):
                    if step == 30:
                        net.grow(1)
                    net.train_step(X, y)
                p = net.predict(Xe)
                if backend is not None:
                    p = backend.to_numpy(p).astype(float)
                mses[spu] = float(((p - ye) ** 2).mean())
            wins.append(mses[True] < mses[False])
        return wins

    assert verdict(bk_of(device, dtype)) == verdict(None)


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk11_transformer_host_numeric(device, dtype, tol):
    """B4: numeric-mode attention host per device — lifecycle
    (train, grow_site, handoff training, SPU host walk) with
    trajectory parity and exact integer structure vs the judge."""
    from core.substrates.transformer import TransformerSubstrate
    from reference_net.spu.spu_network import install_spu_policy

    def run(backend=None):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, (48, 4))
        y = np.sin(3 * X[:, :1]) + 0.3 * X[:, 2:3] * X[:, 3:4]
        h = TransformerSubstrate(4, 6, mode="numeric", seed=7,
                                 backend=backend)
        for _ in range(25):
            h.train_step(X, y)
        h.grow_site("layer0/ffn[2]", hidden=5)
        install_spu_policy(h, {"spu_enabled": True,
                               "spu_warmup_steps": 5,
                               "spu_newborn_steps": 60})
        for _ in range(35):
            loss = h.train_step(X, y)
        p = h.predict(X)
        if backend is not None:
            p = backend.to_numpy(p).astype(float)
        evs = [(e.get("path"), e.get("skip"), e.get("steps"))
               for e in getattr(h, "_spu_events", [])]
        skips = dict(getattr(h, "_spu_skip_counts", {}))
        return p, evs, skips, loss

    bk = bk_of(device, dtype)
    p_t, ev_t, sk_t, _ = run(bk)
    p_n, ev_n, sk_n, _ = run()
    scale = max(1.0, float(np.abs(p_n).max()))
    assert float(np.abs(p_t - p_n).max()) / scale < tol
    assert ev_t == ev_n and sk_t == sk_n     # integer structure


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk14_categorical_and_causal_on_device(device, dtype,
                                               tol):
    """GSM-I2 (the FLIP of the old BK12 refusal test): the B4
    boundary is CLOSED — categorical and causal host modes now
    TRAIN on torch with judge parity."""
    from core.substrates.sequence import SequenceSubstrate
    from core.substrates.transformer import TransformerSubstrate

    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(32, 4))
    yc = ["a" if v > 0 else "b" for v in Xc[:, 0]]
    Xs = rng.normal(size=(32, 8, 2))
    ys = Xs[:, -1, 0:1] * 0.5

    def run(backend):
        tc = TransformerSubstrate(4, 8, mode="categorical",
                                  vocab=["a", "b"], d_model=8,
                                  n_layers=1, n_heads=2,
                                  backend=backend)
        ces = [tc.train_step(Xc, yc) for _ in range(12)]
        sq = SequenceSubstrate(2, 8, d_model=8, n_layers=1,
                               n_heads=2, backend=backend)
        mses = [sq.train_step(Xs, ys) for _ in range(12)]
        pl, conf = tc.predict_label(Xc[:6])
        return ces[-1], mses[-1], pl

    ce_j, mse_j, pl_j = run(None)                    # judge
    ce_t, mse_t, pl_t = run(bk_of(device, dtype))
    assert abs(ce_j - ce_t) < tol
    assert abs(mse_j - mse_t) < tol
    assert pl_j == pl_t                              # labels agree


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk11_transformer_host_numeric(device, dtype, tol):
    """B4: numeric-mode attention host per device — lifecycle
    (train, grow_site, handoff training, SPU host walk) with
    trajectory parity within tolerance (owner ruling: tiny
    library-level differences acceptable) and EXACT integer
    structure vs the judge."""
    from core.substrates.transformer import TransformerSubstrate
    from reference_net.spu.spu_network import install_spu_policy

    def run(backend=None):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, (48, 4))
        y = np.sin(3 * X[:, :1]) + 0.3 * X[:, 2:3] * X[:, 3:4]
        h = TransformerSubstrate(4, 6, mode="numeric", seed=7,
                                 backend=backend)
        for _ in range(25):
            h.train_step(X, y)
        h.grow_site("layer0/ffn[2]", hidden=5)
        install_spu_policy(h, {"spu_enabled": True,
                               "spu_warmup_steps": 5,
                               "spu_newborn_steps": 60})
        for _ in range(35):
            h.train_step(X, y)
        p = h.predict(X)
        if backend is not None:
            p = backend.to_numpy(p).astype(float)
        evs = [(e.get("path"), e.get("skip"), e.get("steps"))
               for e in getattr(h, "_spu_events", [])]
        skips = dict(getattr(h, "_spu_skip_counts", {}))
        return p, evs, skips

    bk = bk_of(device, dtype)
    p_t, ev_t, sk_t = run(bk)
    p_n, ev_n, sk_n = run()
    scale = max(1.0, float(np.abs(p_n).max()))
    assert float(np.abs(p_t - p_n).max()) / scale < tol
    assert ev_t == ev_n and sk_t == sk_n     # integer structure


# (duplicate BK12 refusal test removed with the boundary —
# GSM-I2; BK14 above replaces it)


# ---------- BK13: the loop operator (lambda) lifecycle ----------

def _looped(backend, steps=110):
    from reference_net.net import Network
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY as GP
    X, y = data()
    net = Network(4, 6, seed=7, backend=backend)
    for _ in range(steps):
        net.train_step(X, y)
    GP["loop_enabled"] = True
    try:
        net.loop(3)
    finally:
        GP["loop_enabled"] = False
    return net, X, y


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk13a_exact_entry_bitwise(device, dtype, tol):
    bk = bk_of(device, dtype)
    net, X, y = _looped(bk)
    net.remove_loop()
    p0 = bk.to_numpy(net.predict(X)).copy()
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY as GP
    GP["loop_enabled"] = True
    try:
        net.loop(3)
    finally:
        GP["loop_enabled"] = False
    p1 = bk.to_numpy(net.predict(X))
    assert np.array_equal(p0, p1)                 # bitwise per device


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk13b_trained_trajectory_parity(device, dtype, tol):
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY as GP
    def run(backend):
        net, X, y = _looped(backend)
        for _ in range(60):
            m = net.train_step(X, y)
        return (net._bk.to_numpy(net.predict(X)), m,
                net._loop_k_last, net._loop_projections)
    pj, mj, kj, prj = run(None)                   # judge
    pt, mt, kt, prt = run(bk_of(device, dtype))
    assert np.max(np.abs(pj - pt)) < tol
    assert abs(mj - mt) < tol
    assert prj == prt                             # projection parity


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk13c_k_used_integer_equal_pinned(device, dtype, tol):
    # pinned case away from the stop threshold (DESIGN: asserted
    # on pinned cases at BOTH dtypes)
    from engine.loop_ops import loop_forward as judge_fwd
    rng = np.random.default_rng(11)
    H_L = rng.normal(size=(6, 5))
    L_in = rng.normal(size=(3, 5)) * 0.25
    b = rng.normal(size=3) * 0.1
    L_out = rng.normal(size=(5, 3)) * 0.25
    _, k_j, _ = judge_fwd(H_L, L_in, b, L_out, 1e-6, 32)
    bk = bk_of(device, dtype)
    _, k_t, _ = bk.loop_forward(bk.ingest(H_L), bk.ingest(L_in),
                                bk.ingest(b), bk.ingest(L_out),
                                1e-6, 32)
    assert k_j == k_t                             # INTEGER equal


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk13d_serving_purity_on_device(device, dtype, tol):
    import pickle
    net, X, y = _looped(bk_of(device, dtype))
    net.train_step(X, y)
    before = pickle.dumps(net)
    net.predict(X)
    assert pickle.dumps(net) == before


@pytest.mark.parametrize("device,dtype,tol", PARAMS)
def test_bk13e_device_free_pickle_serves_on_judge(device, dtype,
                                                  tol):
    import pickle
    net, X, y = _looped(bk_of(device, dtype))
    for _ in range(30):
        net.train_step(X, y)
    p_dev = net._bk.to_numpy(net.predict(X))
    clone = pickle.loads(pickle.dumps(net))       # judge on load
    assert type(clone._bk).__name__ == "NumpyBackend"
    assert np.max(np.abs(p_dev
                         - clone.predict(X))) < tol
