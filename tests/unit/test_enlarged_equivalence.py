"""Part B boxes (doc 37 v1.3): GROWN-STATE verification against
the CLASSICAL equations of the ENLARGED network.

The reference for a fullwidth-grown network is the equivalent
textbook network built INDEPENDENTLY here: the body's hidden
stack appended as additional hidden units, the coupling embedded
in the enlarged outgoing weight matrix (block-sparse embedding,
doc 47 5d). SCOPE (doc 37): function map + gradient map at
matched states; Adam trajectories are legitimately different
(slot partitioning) and NOT claimed equal — plain-SGD
trajectories ARE claimed equal (T-14b). The defective legacy
scalar method is NEVER a reference for anything.

Box index: T-13 (a-d) forward + n_params; T-14 gradient map;
T-14b SGD trajectory; T-15 instrument pins; T-16 backend rows;
VM-4 birth preservation.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from reference_net.net import Network                    # noqa: E402

RNG = np.random.default_rng(0)


def _gelu(a):
    return 0.5 * a * (1.0 + np.tanh(
        0.7978845608 * (a + 0.044715 * a ** 3)))


def _gelu_d(a):
    t = np.tanh(0.7978845608 * (a + 0.044715 * a ** 3))
    return 0.5 * (1.0 + t) + 0.5 * a * (1.0 - t ** 2) \
        * 0.7978845608 * (1.0 + 3 * 0.044715 * a ** 2)


def _np(x):
    return np.asarray(x, dtype=float)


# ---------- the independent enlarged-net reference ----------

def _node_state(node, port_slots):
    """Plain-numpy parameter tree of one Network scope + its
    fullwidth port slots (recursive)."""
    st = {"W1": _np(node.W1), "b1": _np(node.b1),
          "W2": _np(node.W2), "c": _np(node.c), "bodies": []}
    for slot in port_slots(node):
        st["bodies"].append(
            {"A": _np(slot["A"]),
             "node": _node_state(slot["body"], port_slots)})
    return st


def _slots(node):
    port = getattr(node, "_port_site", None)
    return port.bodies if port is not None else []


def _raw_out(st, Xh):
    """RECURSIVE composite textbook forward: the enlarged network
    written with the factorized coupling — H' = gelu(W1 x + b1)
    + sum_g u_g A_g; raw = H' W2^T + c. Nothing outside classical
    NN math (doc 47 5d)."""
    Hp = _gelu(Xh @ st["W1"].T + st["b1"])
    for b in st["bodies"]:
        Hp = Hp + _raw_out(b["node"], Xh) @ b["A"]
    return Hp @ st["W2"].T + st["c"]


def _serve_mlp(m, X):
    """Full served output from the extracted state (host scalers
    applied outside the enlarged core)."""
    st = _node_state(m, _slots)
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd)
    raw = _raw_out(st, Xh)
    return raw * m._y_sd + m._y_mu


def _dense_embedding(st):
    """ONE-BODY dense block-sparse embedding (the literal
    'enlarged textbook net'): hidden = [host H; body Hb],
    W2e = [W2 | W2 A^T W2b], ce = c + cb A W2^T."""
    b = st["bodies"][0]
    bn, A = b["node"], b["A"]
    W1e = np.vstack([st["W1"], bn["W1"]])
    b1e = np.concatenate([st["b1"], bn["b1"]])
    W2e = np.hstack([st["W2"], st["W2"] @ A.T @ bn["W2"]])
    ce = st["c"] + bn["c"] @ A @ st["W2"].T
    return W1e, b1e, W2e, ce


def _tensor_count(st):
    n = sum(int(v.size) for k, v in st.items() if k != "bodies")
    for b in st["bodies"]:
        n += int(b["A"].size) + _tensor_count(b["node"])
    return n


def _mk_grown(seed=3, k=3, hidden=6, steps=40, d_in=5, H=8):
    X = RNG.normal(size=(24, d_in))
    y = RNG.normal(size=(24, 1))
    m = Network(d_in, H, lr=1e-2, seed=seed)
    m._growth_policy = {"grow_body_out_width": k}
    for _ in range(steps):
        m.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # toy-scale guard
        m.grow(2, hidden=hidden)
    for _ in range(30):                     # A leaves zero
        m.train_step(X, y)
    return m, X, y


# ================= T-13: FORWARD equivalence =================

def test_t13a_network_one_body_forward_and_nparams():
    m, X, y = _mk_grown()
    served = _np(m.predict(X))
    # (1) recursive composite textbook form
    book = _serve_mlp(m, X)
    assert np.abs(served - book).max() < 1e-12
    # (2) the LITERAL dense enlarged textbook net (block-sparse
    # embedding) — an independent second construction
    st = _node_state(m, _slots)
    W1e, b1e, W2e, ce = _dense_embedding(st)
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd)
    dense = (_gelu(Xh @ W1e.T + b1e) @ W2e.T + ce) \
        * m._y_sd + m._y_mu
    assert np.abs(served - dense).max() < 1e-12
    # (3) n_params identity: the reported count equals the sum of
    # every parameter tensor in the grown system (T-18 folded)
    assert m.n_params() == _tensor_count(st)


def test_t13b_network_nested_bodies_forward():
    m, X, y = _mk_grown()
    body = m.grown_body(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        body.grow(1, hidden=4)              # nested once
        m.grow(5, hidden=4)                 # second root body
    for _ in range(25):
        m.train_step(X, y)
    served = _np(m.predict(X))
    book = _serve_mlp(m, X)
    assert np.abs(served - book).max() < 1e-12
    st = _node_state(m, _slots)
    assert len(st["bodies"]) == 2
    assert len(st["bodies"][0]["node"]["bodies"]) == 1
    assert m.n_params() == _tensor_count(st)


def _ga_enlarged_check(mode):
    from tests.unit.test_classical_conformance import (
        textbook_ga_forward, _cap)
    name = {"numeric": "F3", "categorical": "F4"}[mode]
    spec = _cap.build_configs()[name]
    m = spec["make"]()
    X, y = spec["data"]()
    for _ in range(10):
        m.train_step(X, y)
    m.grow_site("layer0/ffn[2]", hidden=5)
    for _ in range(15):
        m.train_step(X, y)
    served = _np(m.predict(X) if mode == "numeric"
                 else m.predict_proba(X))
    slot = m._port_sites[0].bodies[0]
    bn = {"W1": _np(slot["body"].W1), "b1": _np(slot["body"].b1),
          "W2": _np(slot["body"].W2), "c": _np(slot["body"].c)}
    A = _np(slot["A"])

    class Shim:
        pass
    s = Shim()
    s.P = {k: _np(v) for k, v in m.P.items()}
    # block-sparse FFN embedding at the grown layer 0:
    # W1e = [W1 | W1b^T]; W2e = [[W2],[W2b^T A W2]];
    # b2e = b2 + cb A W2
    s.P["W1_0"] = np.hstack([s.P["W1_0"], bn["W1"].T])
    s.P["b1_0"] = np.concatenate([s.P["b1_0"], bn["b1"]])
    s.P["W2_0"] = np.vstack(
        [s.P["W2_0"], bn["W2"].T @ A @ s.P["W2_0"]])
    s.P["b2_0"] = s.P["b2_0"] + bn["c"] @ A @ _np(m.P["W2_0"])
    s.heads = m.heads
    s.CAUSAL, s.L = m.CAUSAL, m.L
    s._x_mu, s._x_sd = m._x_mu, m._x_sd
    s._y_mu, s._y_sd = m._y_mu, m._y_sd
    book = _np(textbook_ga_forward(s, X, mode))
    assert np.abs(served - book).max() < 1e-12
    # n_params identity for the attention host
    own = m.n_params()
    manual = sum(int(np.prod(np.asarray(v).shape))
                 for v in m.P.values())
    manual += sum(HS.n_params() for layer in m.heads
                  for HS in layer)
    manual += sum(int(_np(sl["A"]).size) + sl["body"].n_params()
                  for site in m._port_sites.values()
                  for sl in site.bodies)
    assert own == manual


def test_t13c_ga_ffn_body_vector_numeric():
    _ga_enlarged_check("numeric")


def test_t13c_ga_ffn_body_causal_categorical():
    _ga_enlarged_check("categorical")


def test_t13d_mlp_host_forward():
    from tests.unit.test_classical_conformance import _cap
    spec = _cap.build_configs()["F1"]
    m = spec["make"]()
    X, y = spec["data"]()
    for _ in range(20):
        m.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.grow(1, hidden=5)
    for _ in range(20):
        m.train_step(X, y)
    served = _np(m.predict(X))
    book = _serve_mlp(m, X)
    assert np.abs(served - book).max() < 1e-12


# ============= T-14: GRADIENT-map equivalence =============

def _composite_grads(st, Xh, ys):
    """FULL closed-form backprop of the enlarged network (the
    factorized composite), kernel-documented objective
    L = (1/2n) sum err^2 — every parameter including A_g and all
    body internals, recursively. dH timing = the state itself
    (pre-step, pin P-4)."""
    n = len(Xh)
    caches = {}

    def fwd(node, key):
        A1 = Xh @ node["W1"].T + node["b1"]
        Z = _gelu(A1)
        Hp = Z.copy()
        subs = []
        for i, b in enumerate(node["bodies"]):
            u = fwd(b["node"], key + (i,))
            subs.append(u)
            Hp = Hp + u @ b["A"]
        raw = Hp @ node["W2"].T + node["c"]
        caches[key] = (A1, Z, Hp, subs, raw)
        return raw

    pred = fwd(st, ())
    err = pred - ys
    grads = {}

    def bwd(node, key, dRaw):
        A1, Z, Hp, subs, raw = caches[key]
        grads[key + ("W2",)] = dRaw.T @ Hp
        grads[key + ("c",)] = dRaw.sum(0)
        dHp = dRaw @ node["W2"]
        dZ = dHp * _gelu_d(A1)
        grads[key + ("W1",)] = dZ.T @ Xh
        grads[key + ("b1",)] = dZ.sum(0)
        for i, b in enumerate(node["bodies"]):
            grads[key + (i, "A")] = subs[i].T @ dHp
            bwd(b["node"], key + (i,), dHp @ b["A"].T)

    bwd(st, (), err / n)
    return grads, pred


def test_t14_gradient_map_all_parameters():
    m, X, y = _mk_grown()
    st0 = _node_state(m, _slots)
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd)
    ys = (_np(y) - m._y_mu) / m._y_sd
    book, _ = _composite_grads(st0, Xh, ys)
    m.train_step(X, y, sgd_lr=1.0)          # delta trick
    st1 = _node_state(m, _slots)

    def check(n0, n1, key):
        for p in ("W1", "b1", "W2", "c"):
            applied = n0[p] - n1[p]
            g = book[key + (p,)]
            scale = max(1.0, float(np.abs(g).max()))
            assert np.abs(applied - g).max() / scale < 1e-10, \
                (key, p)
        for i, (b0, b1_) in enumerate(zip(n0["bodies"],
                                          n1["bodies"])):
            gA = book[key + (i, "A")]
            appliedA = b0["A"] - b1_["A"]
            scale = max(1.0, float(np.abs(gA).max()))
            assert np.abs(appliedA - gA).max() / scale < 1e-10, \
                (key, i, "A")
            check(b0["node"], b1_["node"], key + (i,))

    check(st0, st1, ())


# ========== T-14b: plain-SGD TRAJECTORY equivalence ==========

def test_t14b_sgd_trajectory_10_steps():
    """Plain SGD has no optimizer state, hence no slot-partition
    freedom: the grown net and the independently implemented
    enlarged textbook net must follow the SAME trajectory —
    every parameter tensor within 1e-12 for 10 consecutive
    steps (update APPLICATION verified end-to-end)."""
    m, X, y = _mk_grown()
    st = _node_state(m, _slots)             # textbook mirror
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd)
    ys = (_np(y) - m._y_mu) / m._y_sd
    lr = 0.01

    def apply_sgd(node, key, grads):
        for p in ("W1", "b1", "W2", "c"):
            node[p] = node[p] - lr * grads[key + (p,)]
        for i, b in enumerate(node["bodies"]):
            b["A"] = b["A"] - lr * grads[key + (i, "A")]
            apply_sgd(b["node"], key + (i,), grads)

    def compare(n0, n1, key):
        for p in ("W1", "b1", "W2", "c"):
            assert np.abs(n0[p] - n1[p]).max() < 1e-12, (key, p)
        for i, (a, b) in enumerate(zip(n0["bodies"],
                                       n1["bodies"])):
            assert np.abs(a["A"] - b["A"]).max() < 1e-12
            compare(a["node"], b["node"], key + (i,))

    for step in range(10):
        grads, _ = _composite_grads(st, Xh, ys)
        apply_sgd(st, (), grads)
        m.train_step(X, y, sgd_lr=lr)
        compare(st, _node_state(m, _slots), (step,))


# ================= T-15: instrument pins =================

def test_t15_loading_hand_literal():
    """loading = ||u A||_F pinned by a hand literal:
    u = [[1,2],[3,4]], A = [[1,0,1],[0,1,1]] ->
    uA = [[1,2,3],[3,4,7]], ||uA||_F = sqrt(1+4+9+9+16+49)
    = sqrt(88)."""
    from reference_net.growth_port import PortSite, make_port_body
    site = PortSite(3)
    body = make_port_body(2, 4, 2, lr=1e-2, seed=1)
    site.add_body(body)
    site.bodies[0]["A"] = np.array([[1.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0]])
    body.predict = lambda X: np.array([[1.0, 2.0], [3.0, 4.0]])
    (got,) = site.loading(np.zeros((2, 2)))
    assert abs(got - np.sqrt(88.0)) < 1e-12
    # instability recursion literals: owned by the D5 box in
    # test_growth_port.py (same box index entry)


# ============ VM-4: birth preservation, every host ============

def test_vm4_birth_preservation_all_hosts():
    from tests.unit.test_classical_conformance import _cap
    for name in ("F1", "F2", "F3", "F4", "F6"):
        spec = _cap.build_configs()[name]
        m = spec["make"]()
        X, y = spec["data"]()
        for _ in range(10):
            m.train_step(X, y)
        before = _np(m.predict(X) if spec["kind"] == "numeric"
                     else m.predict_proba(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(m, "P"):              # per-layer 3-D hosts
                m.grow_site("layer0/ffn[1]", hidden=4)
            else:                            # Network-family hosts
                m.grow(1, hidden=4)
        after = _np(m.predict(X) if spec["kind"] == "numeric"
                    else m.predict_proba(X))
        assert np.array_equal(before, after), name


# ================= T-16: backend rows =================

def _torch_backends():
    rows = []
    try:
        import torch                         # noqa: F401
        from engine.backends import resolve_backend
        rows.append(("torch-cpu-f64", resolve_backend(
            "torch", device="cpu", dtype="float64"), 1e-8))
        if torch.backends.mps.is_available():
            rows.append(("torch-mps-f32", resolve_backend(
                "torch", device="mps", dtype="float32"), 2e-3))
    except Exception:
        pass
    return rows


@pytest.mark.parametrize(
    "label,bk,tol",
    _torch_backends() or [("unavailable", None, 0.0)])
def test_t16_backend_rows(label, bk, tol):
    if bk is None:
        pytest.skip("torch unavailable")
    X = np.random.default_rng(1).normal(size=(24, 5))
    y = np.random.default_rng(2).normal(size=(24, 1))
    m = Network(5, 8, lr=1e-2, seed=3, backend=bk)
    for _ in range(40):
        m.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.grow(2, hidden=6)
    for _ in range(30):
        m.train_step(X, y)
    bslots = m._port_site.bodies

    def n_(v):
        return np.asarray(bk.to_numpy(v), dtype=float)
    st = {"W1": n_(m.W1), "b1": n_(m.b1), "W2": n_(m.W2),
          "c": n_(m.c),
          "bodies": [{"A": n_(bslots[0]["A"]),
                      "node": {"W1": n_(bslots[0]["body"].W1),
                               "b1": n_(bslots[0]["body"].b1),
                               "W2": n_(bslots[0]["body"].W2),
                               "c": n_(bslots[0]["body"].c),
                               "bodies": []}}]}
    Xh = (X - n_(m._x_mu)) / n_(m._x_sd)
    served = n_(m.predict(X))
    book = _raw_out(st, Xh) * float(m._y_sd) + float(m._y_mu)
    scale = max(1.0, float(np.abs(book).max()))
    assert np.abs(served - book).max() / scale < tol
    # gradient row: sgd delta vs closed form, 4+ coords/family
    ys = (y - float(m._y_mu)) / float(m._y_sd)
    grads, _ = _composite_grads(st, Xh, ys)
    m.train_step(X, y, sgd_lr=1.0)
    for key, pick in ((("W1",), None), (("W2",), None),
                      ((0, "A"), None), ((0, "W1"), None),
                      ((0, "W2"), None), ((0, "c"), None)):
        g = grads[key]
        if key == ("W1",):
            applied = st["W1"] - n_(m.W1)
        elif key == ("W2",):
            applied = st["W2"] - n_(m.W2)
        elif key == (0, "A"):
            applied = st["bodies"][0]["A"] \
                - n_(m._port_site.bodies[0]["A"])
        else:
            body = m._port_site.bodies[0]["body"]
            applied = st["bodies"][0]["node"][key[1]] \
                - n_(getattr(body, key[1]))
        scale = max(1.0, float(np.abs(g).max()))
        assert np.abs(applied - g).max() / scale < tol, key
