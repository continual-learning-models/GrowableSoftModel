"""GA backend-port CPU-vs-GPU parity boxes — W1 set:
P-1..P-4, P-2b, P-3b, P-9, P-12 (docs/system/32 v1.1 9.6/9.6b;
dev plan doc 33 W1). W2/W3 boxes live in this file's sequel
sections added at their steps.

Verification chain (doc 32 9.6): LINK-2 boxes compare each
device against the numpy float64 judge under the declared
tolerance; LINK-3 boxes (P-2b, P-3b) anchor the DEVICE
directly to mathematics with no numpy-implementation
reference in the loop.
"""
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from engine.backends import resolve_backend                # noqa: E402
from core.substrates.growable_attention import (           # noqa: E402
    GrowableAttentionSubstrate as GA, _HMATS)

# doc 32 9.6 device/tolerance matrix (skip-if-unavailable).
# F32_TRAJECTORY_ENVELOPE (doc 61 I-D, owner-ruled semantics
# 2026-07-25): NOT an accuracy pass-line — accuracy is
# certified on the float64 rows only. This is the f32
# HARDWARE/PRECISION ENVELOPE for the full growth script:
# measured deterministic peak 4.038e-3 (5/5 identical runs;
# torch 2.12.1 / macOS 26.5.1 / M1 Max; deviation attributed
# by third-party referee to f32 physics — our kernels match
# PyTorch official at 4.8e-7 on the same mps device, and
# cpu-f32 shows the same per-step deviation as mps); value =
# smallest one-significant-figure number with >= 40% headroom
# over the peak: 4.038e-3 * 1.4 = 5.65e-3 -> 6e-3. A breach
# means the f32 behavior WORSENED (kernel regression or real
# defect) — see test_f32_precision_door.py for the canary
# that fires before this row does.
F32_TRAJECTORY_ENVELOPE = 6e-3
# F32_STATE_ENVELOPE: the same-formula record line for the
# FINAL PER-TENSOR state comparison of the growth script
# (a different judge than the loss checkpoints): measured
# deterministic worst tensor Bf 8.53e-3 (same run/session as
# above); 8.53e-3 * 1.4 = 1.19e-2 -> 1.2e-2 (two significant
# figures). Same semantics: an f32 PRECISION RECORD, not an
# accuracy certification.
F32_STATE_ENVELOPE = 1.2e-2
DEVICES = []
try:
    import torch
    DEVICES.append(("torch", "cpu", "float64", 1e-8))
    if torch.backends.mps.is_available():
        DEVICES.append(("torch", "mps", "float32",
                        F32_TRAJECTORY_ENVELOPE))
    if torch.cuda.is_available():
        DEVICES.append(("torch", "cuda", "float32",
                        F32_TRAJECTORY_ENVELOPE))
except ImportError:
    torch = None


def err(a, b):
    """doc 32 9.6 metric: scaled max-abs difference."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0:
        return 0.0
    return float(np.abs(a - b).max() / max(1.0, np.abs(a).max()))


def mk_f1(bk=None, seed=0):
    return GA(6, 10, mode="numeric", lr=1e-2, seed=seed,
              d_model=16, n_layers=2, heads_spec=[[1, 3], [2, 1]],
              backend=bk)


def mk_f2(bk=None, seed=0):
    return GA(64, 10, mode="categorical",
              vocab=["t3", "t4", "EOS"], lr=1e-2, seed=seed,
              d_model=16, n_layers=1, heads_spec=[[2, 2]],
              causal=True, window=12, backend=bk)


def f1_data(k=3):
    rng = np.random.default_rng(1)
    return [(rng.normal(size=(8, 6)), rng.normal(size=8))
            for _ in range(k)]


def f2_data(k=3):
    rng = np.random.default_rng(2)
    out = []
    for _ in range(k):
        T = int(rng.integers(5, 13))
        X = np.zeros((8, T, 64))
        seq = rng.integers(3, 5, size=(8, T))
        for i in range(8):
            X[i, np.arange(T), seq[i]] = 1.0
        y = np.array([("t3", "t4", "EOS")[int(v)]
                      for v in rng.integers(0, 3, 8)])
        out.append((X, y))
    return out


# ---------------- P-1 construction parity ----------------

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p1_construction_parity(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    for mkr in (mk_f1, mk_f2):
        mj, mt = mkr(), mkr(bk)
        # D5: identical numpy birth state, so device state equals
        # the judge state CAST to the device dtype, exactly
        for k in mj.P:
            got = np.asarray(bk.to_numpy(mt.P[k]))
            assert np.array_equal(got,
                                  np.asarray(mj.P[k], got.dtype)), k
        for lj, lt in zip(mj.heads, mt.heads):
            for HJ, HT in zip(lj, lt):
                for nm in _HMATS:
                    got = np.asarray(bk.to_numpy(getattr(HT, nm)))
                    assert np.array_equal(
                        got, np.asarray(getattr(HJ, nm), got.dtype))


# ---------------- P-2 forward parity ----------------

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p2_forward_parity(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    mj, mt = mk_f1(), mk_f1(bk)
    X, _ = f1_data(1)[0]
    zj = np.asarray(mj._bk.to_numpy(mj._forward(mj._stdx(X))))
    zt = np.asarray(bk.to_numpy(mt._forward(mt._stdx(X))))
    assert err(zj, zt) < tol
    # categorical serving path incl. the pre-scaler uniform prior
    cj, ct = mk_f2(), mk_f2(bk)
    Xc, yc = f2_data(1)[0]
    assert np.array_equal(cj.predict_proba(Xc),
                          ct.predict_proba(Xc))   # uniform prior
    cj.train_step(Xc, yc)
    ct.train_step(Xc, yc)
    assert err(cj.predict_proba(Xc), ct.predict_proba(Xc)) < tol


# ---------------- P-2b hand-computed forward oracle (LINK-3) ----

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p2b_hand_forward_oracle_on_device(bkn, dev, dt, tol):
    """DEVICE vs pencil-and-paper. Pinned 1-layer/1-head organ,
    integer-ish weights; every number below is derivable by hand
    (ln eps = 1e-5 as in the kernel):
      X = [[2, 4]], Wv = I2, Bf = 0  ->  T = [[2,0],[0,4]]
      LN rows: [2,0] -> mean 1, var 1,  xhat = [1,-1]/sqrt(1+1e-5)
               [0,4] -> mean 2, var 4,  xhat = [-2,2]/sqrt(4+1e-5)/2
                        (i.e. [-1,1]*2/sqrt(4+1e-5))
      head: Wq = Wk = 0 -> scores 0 -> A uniform 1/2
            Wv_h = [[1],[0]] -> Vh = xhat[:,0] per position
            AVh: both positions = (Vh0+Vh1)/2
            Wo = [[1,0]] -> adds AVh to channel 0
      FFN: W1 = 0, b2 = 0 -> identity
      Pool = mean over the 2 positions; Wh = [[1],[2]], bh = 0.5
    """
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    m = GA(2, 1, mode="numeric", lr=1e-2, seed=0, d_model=2,
           n_layers=1, heads_spec=[[1]], backend=bk)
    ing = bk.ingest
    m.P["Wv"] = ing(np.eye(2))
    m.P["Bf"] = ing(np.zeros((2, 2)))
    m.P["W1_0"] = ing(np.zeros((2, 1)))
    m.P["b1_0"] = ing(np.zeros(1))
    m.P["W2_0"] = ing(np.zeros((1, 2)))
    m.P["b2_0"] = ing(np.zeros(2))
    m.P["Wh"] = ing(np.array([[1.0], [2.0]]))
    m.P["bh"] = ing(np.array([0.5]))
    HS = m.heads[0][0]
    HS.Wq = ing(np.zeros((2, 1)))
    HS.Wk = ing(np.zeros((2, 1)))
    HS.Wv = ing(np.array([[1.0], [0.0]]))
    HS.Wo = ing(np.array([[1.0, 0.0]]))
    X = np.array([[2.0, 4.0]])
    # ---- pencil derivation (pure literals; no GA/numpy-impl) ----
    s1 = (1.0 + 1e-5) ** 0.5          # LN sd of row [2, 0]
    s2 = (4.0 + 1e-5) ** 0.5          # LN sd of row [0, 4]
    xh1 = (1.0 / s1, -1.0 / s1)       # xhat row 1
    xh2 = (-2.0 / s2, 2.0 / s2)       # xhat row 2
    vh = (xh1[0], xh2[0])             # Vh per position (channel 0)
    avh = (vh[0] + vh[1]) / 2.0       # uniform attention
    t1 = (2.0 + avh, 0.0, 0.0 + avh, 4.0)   # T1 rows (c0, c1)
    pool = ((t1[0] + t1[2]) / 2.0, (t1[1] + t1[3]) / 2.0)
    expected = pool[0] * 1.0 + pool[1] * 2.0 + 0.5
    got = float(np.asarray(
        bk.to_numpy(m._forward(m._stdx(X)))).ravel()[0])
    assert abs(got - expected) / max(1.0, abs(expected)) < tol, \
        (got, expected)


# ---------------- P-3 trajectory parity ----------------

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p3_trajectory_parity(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    for mkr, data in ((mk_f1, f1_data()), (mk_f2, f2_data())):
        mj, mt = mkr(), mkr(bk)
        lj, lt = [], []
        for i in range(200):
            X, y = data[i % len(data)]
            a = mj.train_step(X, y)
            b = mt.train_step(X, y)
            if i % 10 == 0:
                lj.append(a)
                lt.append(b)
        assert lj[-1] < lj[0]              # the judge run learns
        for a, b in zip(lj, lt):
            assert abs(a - b) / max(1.0, abs(a)) < tol
        for k in mj.P:
            assert err(mj._bk.to_numpy(mj.P[k]),
                       bk.to_numpy(mt.P[k])) < tol, k


# ---------------- P-3b finite-difference oracle (LINK-3) --------

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p3b_fd_gradient_oracle_on_device(bkn, dev, dt, tol):
    """The DEVICE's own backward vs numerical differentiation of
    the DEVICE's own forward — calculus, no numpy reference."""
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    m = GA(4, 4, mode="numeric", lr=1e-2, seed=0, d_model=8,
           n_layers=1, heads_spec=[[2]], backend=bk)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 4))
    y = rng.normal(size=6)
    m.train_step(X, y)                  # fixes the scalers
    base = copy.deepcopy(m)             # stays on the device

    def loss_of(mm):
        ys = (np.asarray(y, float).reshape(-1, 1)
              - mm._y_mu) / mm._y_sd
        z = np.asarray(bk.to_numpy(mm._forward(mm._stdx(X))))
        return float(((z - ys) ** 2).mean())

    ga = copy.deepcopy(base)
    ga.train_step(X, y, sgd_lr=1.0)     # theta' = theta - 1.0 * G
    h = 1e-6 if dt == "float64" else 1e-2
    ftol = 1e-5 if dt == "float64" else 8e-2
    checks = [("P", "Wh", (0, 0)), ("P", "W1_0", (1, 2)),
              ("P", "Wv", (2, 3)), ("H", "Wq", (0, 0))]
    for kind, name, idx in checks:
        if kind == "P":
            g = float(np.asarray(bk.to_numpy(base.P[name]))[idx]
                      - np.asarray(bk.to_numpy(ga.P[name]))[idx])
        else:
            g = float(
                np.asarray(bk.to_numpy(
                    getattr(base.heads[0][0], name)))[idx]
                - np.asarray(bk.to_numpy(
                    getattr(ga.heads[0][0], name)))[idx])
        fd = []
        for sgn in (+1, -1):
            mm = copy.deepcopy(base)
            if kind == "P":
                mm.P[name][idx] = mm.P[name][idx] + sgn * h
            else:
                M = getattr(mm.heads[0][0], name)
                M[idx] = M[idx] + sgn * h
            fd.append(loss_of(mm))
        fd_g = (fd[0] - fd[1]) / (2 * h)
        scale = max(1.0, abs(fd_g))
        assert abs(g - fd_g) / scale < ftol, (name, idx, g, fd_g)


# ---------------- P-4 serving boundaries ----------------

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p4_serving_boundaries(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    m = mk_f1(bk)
    assert isinstance(m.predict(np.zeros((2, 6))), np.ndarray)
    c = mk_f2(bk)
    pu = c.predict_proba(f2_data(1)[0][0])
    assert isinstance(pu, np.ndarray)
    assert np.allclose(pu, 1.0 / 3)     # uniform pre-scaler prior
    X, y = f2_data(1)[0]
    c.train_step(X, y)
    labs, conf = c.predict_label(X)
    assert isinstance(labs, list) and isinstance(
        np.asarray(conf), np.ndarray)


# ---------------- P-9 DD-7 equivalence (judge, float64) ---------

def test_p9_dd7_flat_equivalence():
    m = mk_f1()
    data = f1_data()
    flat_mu = {}
    flat_nu = {}
    for l, layer in enumerate(m.heads):
        for h, HS in enumerate(layer):
            p = sum(int(np.prod(getattr(HS, nm).shape))
                    for nm in _HMATS)
            flat_mu[(l, h)] = np.zeros(p)
            flat_nu[(l, h)] = np.zeros(p) + 1e-12
    for i in range(60):
        old = {(l, h, nm): np.array(getattr(HS, nm))
               for l, layer in enumerate(m.heads)
               for h, HS in enumerate(layer) for nm in _HMATS}
        X, y = data[i % len(data)]
        m.train_step(X, y)
        for l, layer in enumerate(m.heads):
            for h, HS in enumerate(layer):
                dtheta = np.concatenate(
                    [(np.asarray(getattr(HS, nm))
                      - old[(l, h, nm)]).ravel()
                     for nm in _HMATS])
                flat_mu[(l, h)] = 0.95 * flat_mu[(l, h)] \
                    + 0.05 * dtheta
                flat_nu[(l, h)] = 0.95 * flat_nu[(l, h)] \
                    + 0.05 * np.abs(dtheta)
        if i == 30:                      # widen mid-stream
            dh0 = m.heads[0][0].d_h
            m.head_widen(0, 0, m=1)
            d = m.d
            # POSITION-CORRECT reference rebuild (owner fix
            # ruling): pad each matrix segment at its true
            # coordinates, then re-flatten
            shapes_old = [(d, dh0)] * 3 + [(dh0, d)]
            pads = [((0, 0), (0, 1))] * 3 + [((0, 1), (0, 0))]
            for buf in (flat_mu, flat_nu):
                fill = 0.0 if buf is flat_mu else 1e-12
                parts, ofs = [], 0
                for sh, pd in zip(shapes_old, pads):
                    a = sh[0] * sh[1]
                    seg = buf[(0, 0)][ofs:ofs + a].reshape(sh)
                    parts.append(np.pad(
                        seg, pd, constant_values=fill).ravel())
                    ofs += a
                buf[(0, 0)] = np.concatenate(parts)
            old = None
        if i % 10 == 9:
            for l in range(m.L):
                got = m.u_stats(l)
                for h, HS in enumerate(m.heads[l]):
                    ref = 1.0 - np.linalg.norm(flat_mu[(l, h)]) \
                        / (np.linalg.norm(flat_nu[(l, h)]) + 1e-12)
                    assert got[h] == ref, (i, l, h)   # BIT-equal


# ---------------- P-12 boundary audit ----------------

class _Counting:
    def __init__(self, inner):
        self._inner = inner
        self.n_to = 0
        self.n_in = 0

    def to_numpy(self, x):
        self.n_to += 1
        return self._inner.to_numpy(x)

    def ingest(self, x):
        self.n_in += 1
        return self._inner.ingest(x)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def test_p12_no_hot_path_round_trips():
    bk = _Counting(resolve_backend("numpy"))
    m = mk_f1(bk)
    X, y = f1_data(1)[0]
    m.train_step(X, y)                  # scaler-fit step
    bk.n_to = bk.n_in = 0
    for _ in range(20):
        m.train_step(X, y)
    # per step: ingest = {X via _stdx, ys} = 2; to_numpy = the
    # loss scalar only (declared exempt boundary)
    assert bk.n_in == 40, bk.n_in
    assert bk.n_to == 20, bk.n_to


# ================= W2 boxes (doc 33): events + serialization ====

def _growth_script(m, data, drill=25):
    """F3: training interleaved with all three growth operators."""
    losses = []
    for i in range(drill):
        X, y = data[i % len(data)]
        losses.append(m.train_step(X, y))
    m.head_widen(0, 0, m=2)
    for i in range(drill):
        X, y = data[i % len(data)]
        losses.append(m.train_step(X, y))
    m.head_add(0)
    for i in range(drill):
        X, y = data[i % len(data)]
        losses.append(m.train_step(X, y))
    m.grow_site(m.growth_sites()[0][0], hidden=4)
    for i in range(drill):
        X, y = data[i % len(data)]
        losses.append(m.train_step(X, y))
    return losses


@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p5_add_class_parity(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    mj, mt = mk_f2(), mk_f2(bk)
    data = f2_data()
    for i in range(60):
        X, y = data[i % len(data)]
        a = mj.train_step(X, y)
        b = mt.train_step(X, y)
        if i in (20, 40):
            mj.add_class(f"t{60 + i}")
            mt.add_class(f"t{60 + i}")
    assert abs(a - b) / max(1.0, abs(a)) < tol
    assert err(mj._bk.to_numpy(mj.P["Wh"]),
               bk.to_numpy(mt.P["Wh"])) < tol
    for m in (mj, mt):
        assert tuple(m._adam["Wh"][0].shape) \
            == tuple(m.P["Wh"].shape)


@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p6_widen_add_preservation_per_backend(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    m = mk_f1(bk)
    data = f1_data()
    for i in range(10):
        m.train_step(*data[i % 3])
    Xp = np.random.default_rng(9).normal(size=(6, 6))
    before = m.predict(Xp)
    HS = m.heads[0][0]
    dh0 = HS.d_h
    # head_widen contract = dev <= 1e-12 (the operator's own
    # on-device assert ran); existing-matrix dims change, so
    # blocked matmul reordering may shift ULPs on torch
    m.head_widen(0, 0, m=2)
    mid = m.predict(Xp)
    assert err(before, mid) < tol
    # head_add contract = BITWISE (fresh zero-side head; the
    # existing heads' matmuls are untouched)
    m.head_add(1)
    assert np.array_equal(mid, m.predict(Xp))
    assert HS.d_h == dh0 + 2
    for nm in _HMATS:
        assert tuple(m._adam_h[(0, 0, nm)][0].shape) \
            == tuple(getattr(HS, nm).shape)
        assert tuple(HS.mu[nm].shape) \
            == tuple(getattr(HS, nm).shape)


@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p7_grow_site_on_device(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    m = mk_f1(bk)
    data = f1_data()
    for i in range(10):
        m.train_step(*data[i % 3])
    Xp = np.random.default_rng(9).normal(size=(6, 6))
    before = m.predict(Xp)
    site = m.growth_sites()[0][0]
    m.grow_site(site, hidden=4)
    inner = m._port_sites[0].bodies[0]["body"]   # fullwidth port
    assert inner._bk is m._bk          # doc 32 D4: same device
    assert err(before, m.predict(Xp)) < tol   # zero-out inner
    drill = [m.train_step(*data[i % 3]) for i in range(30)]
    assert drill[-1] < drill[0]        # inner trains on device
    nested = [s for s, _ in m.growth_sites() if "::" in s]
    assert nested
    m.grow_site(nested[0], hidden=3)   # nested grow applies
    assert m.depth() >= 3


@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p8_handoff_parity_full_script(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)
    mj, mt = mk_f1(), mk_f1(bk)
    data = f1_data()
    lj = _growth_script(mj, data)
    lt = _growth_script(mt, data)
    for a, b in zip(lj[::10] + lj[-1:], lt[::10] + lt[-1:]):
        assert abs(a - b) / max(1.0, abs(a)) < tol
    assert mj.n_params() == mt.n_params()
    assert mj.depth() == mt.depth()
    st_tol = tol if dt == "float64" else F32_STATE_ENVELOPE
    for k in mj.P:
        assert err(mj._bk.to_numpy(mj.P[k]),
                   bk.to_numpy(mt.P[k])) < st_tol, k


def test_p13_serialization_matrix(tmp_path):
    from engine.backends import set_compute_policy
    data = f1_data()
    # (a) save under torch (mps if present, else cpu) -> load
    #     under the DEFAULT numpy policy
    dev = ("mps" if DEVICES and any(d[1] == "mps"
                                    for d in DEVICES) else "cpu")
    dt = "float32" if dev == "mps" else "float64"
    bk = resolve_backend("torch", device=dev, dtype=dt)
    mt = mk_f1(bk)
    for i in range(10):
        mt.train_step(*data[i % 3])
    Xp = np.random.default_rng(9).normal(size=(6, 6))
    want = mt.predict(Xp)
    mt.save(tmp_path / "t")
    m2 = GA.load(tmp_path / "t")
    assert type(m2._bk).__name__ == "NumpyBackend"
    assert err(want, m2.predict(Xp)) < 2e-3
    # (b) save under numpy -> load under a torch policy
    mj = mk_f1()
    for i in range(10):
        mj.train_step(*data[i % 3])
    wantj = mj.predict(Xp)
    mj.save(tmp_path / "j")
    try:
        set_compute_policy(compute_backend="torch",
                           compute_device="cpu",
                           compute_dtype="float64")
        m3 = GA.load(tmp_path / "j")
        assert type(m3._bk).__name__ == "TorchBackend"
        assert err(wantj, m3.predict(Xp)) < 1e-8
        # (d) pickle clone lands on the CURRENT policy backend
        m4 = pickle.loads(pickle.dumps(mj))
        assert type(m4._bk).__name__ == "TorchBackend"
    finally:
        set_compute_policy(compute_backend="numpy",
                           compute_device="cpu")
    # (c) the OLD pre-port artifact loads and serves BITWISE
    fx = REPO / "tests" / "unit" / "fixtures" / \
        "ga_preport_artifact"
    old = GA.load(fx)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 6))
    rng.normal(size=8)                 # consume y draws as built
    expected = np.load(fx / "expected_pred.npy")
    assert np.array_equal(old.predict(X), expected)


# ================= W3 boxes (doc 33): islands + refusals ========

@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p10_instrument_parity_on_identical_state(
        bkn, dev, dt, tol, tmp_path):
    from engine.backends import set_compute_policy
    from core.introspect import inspect as introspect
    mj = mk_f1()
    data = f1_data()
    for i in range(20):
        mj.train_step(*data[i % 3])
    mj.save(tmp_path / "art")
    X = np.random.default_rng(7).normal(size=(6, 6))
    ref_u = [mj.u_stats(l) for l in range(mj.L)]
    ref_load = [mj.head_loading(l, X) for l in range(mj.L)]
    ref_row = mj.row_entropies(0, 0, X)
    ref_j = mj.j_att_value(0, 0, X)
    ref_ins = introspect(mj, X)
    try:
        set_compute_policy(compute_backend=bkn,
                           compute_device=dev,
                           compute_dtype=dt,
                           acknowledge_f32_precision=True)
        mt = GA.load(tmp_path / "art")   # ingests on the device
        for l in range(mt.L):
            assert err(ref_u[l], mt.u_stats(l)) < tol
            assert err(ref_load[l], mt.head_loading(l, X)) < tol
        assert err(ref_row, mt.row_entropies(0, 0, X)) < tol
        assert abs(ref_j - mt.j_att_value(0, 0, X)) \
            / max(1.0, abs(ref_j)) < tol
        got = introspect(mt, X)          # numpy-copy contract
        assert isinstance(got["pooled"], np.ndarray)
        assert err(ref_ins["logits"], got["logits"]) < tol
        assert err(ref_ins["attention"][0][0],
                   got["attention"][0][0]) < tol
    finally:
        set_compute_policy(compute_backend="numpy",
                           compute_device="cpu")


@pytest.mark.parametrize("bkn,dev,dt,tol", DEVICES)
def test_p11_selfproc_island_parity(bkn, dev, dt, tol):
    bk = resolve_backend(bkn, device=dev, dtype=dt)

    def mk_sp(b):
        m = GA(6, 10, mode="numeric", lr=1e-2, seed=0,
               d_model=16, n_layers=1, heads_spec=[[2, 2]],
               selfproc=True, backend=b)
        m._t_att += 500          # past warmup + age gates
        for HS in m.heads[0]:
            HS.birth_t = -500
        return m

    mj, mt = mk_sp(None), mk_sp(bk)
    assert mj.selfproc_active(0, 0) and mt.selfproc_active(0, 0)
    data = f1_data()
    for i in range(30):
        X, y = data[i % 3]
        a = mj.train_step(X, y)
        b = mt.train_step(X, y)
    assert abs(a - b) / max(1.0, abs(a)) < tol
    for k in mj.P:
        assert err(mj._bk.to_numpy(mj.P[k]),
                   bk.to_numpy(mt.P[k])) < tol, k
    # OFF switch: no island in the path
    m_off = mk_f1(bk)
    assert not m_off.selfproc_active(0, 0)


def test_p14_refusals():
    with pytest.raises(ValueError, match="backend must be"):
        GA(4, 4, backend=123)
    with pytest.raises(ValueError):
        GA(4, 4, backend="no_such_backend")
    if torch is not None and torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="float64"):
            GA(4, 4, backend=resolve_backend(
                "torch", device="mps", dtype="float64"))
