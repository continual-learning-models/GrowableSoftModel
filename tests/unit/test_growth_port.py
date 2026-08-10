"""Growth Port W1 boxes (docs/system/37 v1.3): T-1..T-6,
VM-3(a)(b), VM-5, plus the Network out_width / train_from_grad
contract. Pure-addition stage: no host consumes the port yet."""
import pickle   # safe: in-process round-trips of our own objects
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from reference_net.growth_port import (            # noqa: E402
    PortSite, LegacyScalarPort, make_port_body)
from reference_net.net import Network              # noqa: E402

RNG = np.random.default_rng(0)


def make_site(C=5, k=3, d_in=4, hidden=6, seed=1):
    site = PortSite(C)
    body = make_port_body(d_in, hidden, k, lr=1e-2, seed=seed)
    site.add_body(body)
    return site, body


# ---------------- T-1 port algebra ----------------

def test_t1_forward_matches_manual_assembly():
    site, body = make_site()
    H = RNG.normal(size=(8, 5))
    Xb = RNG.normal(size=(8, 4))
    # give the assembly nonzero values to exercise the algebra
    A = RNG.normal(size=(3, 5))
    site.bodies[0]["A"] = A.copy()
    got = site.forward(H, Xb)
    manual = H + body.predict(Xb) @ A
    assert np.array_equal(got, manual)


# ---------------- T-2 zero-birth preservation ----------------

def test_t2_birth_is_exact_no_op():
    """Zero-SIDE doctrine (doc 35 R4, same as head_add): the ONE
    zero side is the assembly A_g; the body is born LIVE. A = 0
    alone gives exact preservation — and the live body is what
    makes dA = u^T dH nonzero from step 1 (a zero body AND zero
    assembly would deadlock both chain-rule gradients)."""
    site, body = make_site()
    H = RNG.normal(size=(8, 5))
    Xb = RNG.normal(size=(8, 4))
    got = site.forward(H, Xb)          # A = 0 -> u @ A == 0
    assert np.array_equal(got, H)
    # the body is LIVE at birth (u != 0): gradient flow can start
    assert np.asarray(body.predict(Xb)).any()


# ---------------- T-3 / VM-3(a) hand literals ----------------

def test_t3_vm3a_chain_rule_hand_literals():
    """Hand-checked 2x3 case (derivation in comments):
       u = [[1, 2],          A = [[1, 0, 2],
            [0, 3]]               [0, 1, 1]]
       dH = [[1, 0, 1],
             [2, 1, 0]]
       dA = u^T dH = [[1*1+0*2, 1*0+0*1, 1*1+0*0],   = [[1,0,1],
                      [2*1+3*2, 2*0+3*1, 2*1+3*0]]      [8,3,2]]
       dU = dH A^T = [[1*1+0*0+1*2, 1*0+0*1+1*1],    = [[3,1],
                      [2*1+1*0+0*2, 2*0+1*1+0*1]]       [2,1]]"""
    u = np.array([[1.0, 2.0], [0.0, 3.0]])
    A = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    dH = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    dA = u.T @ dH
    dU = dH @ A.T
    assert np.array_equal(dA, np.array([[1, 0, 1], [8, 3, 2]]))
    assert np.array_equal(dU, np.array([[3, 1], [2, 1]]))
    # and the PORT applies exactly these (sgd step, lr 1):
    site = PortSite(3)
    body = make_port_body(2, 4, 2, lr=1e-2, seed=3)
    site.add_body(body)
    site.bodies[0]["A"] = A.copy()
    grads_seen = {}
    body.train_from_grad = lambda X, dU_, sgd_lr=None: \
        grads_seen.update(dU=np.asarray(dU_))
    site._u_cache = [u]
    site.bodies[0]["A"] = A.copy()
    site.backward_step(dH, np.zeros((2, 2)), lr=1e-2, sgd_lr=1.0)
    assert np.array_equal(A - dA, site.bodies[0]["A"])
    assert np.array_equal(grads_seen["dU"], dU)


def test_t3_vm3b_central_differences():
    """dL/dA and dL/du for L = sum(W * (H + u A)) checked by
    central finite differences at float64."""
    site, body = make_site()
    H = RNG.normal(size=(6, 5))
    Xb = RNG.normal(size=(6, 4))
    A = RNG.normal(size=(3, 5))
    W = RNG.normal(size=(6, 5))        # dL/dH' = W
    # train the body a little so u is nonzero
    for _ in range(5):
        body.train_from_grad(Xb, RNG.normal(size=(6, 3)) * 0.1,
                             sgd_lr=0.05)
    u = np.asarray(body.predict(Xb))
    dA = u.T @ W
    h = 1e-6
    for idx in ((0, 0), (1, 3), (2, 4)):
        Ap, Am = A.copy(), A.copy()
        Ap[idx] += h
        Am[idx] -= h
        fd = (np.sum(W * (H + u @ Ap))
              - np.sum(W * (H + u @ Am))) / (2 * h)
        assert abs(fd - dA[idx]) < 1e-6, idx
    dU = W @ A.T
    for idx in ((0, 0), (3, 1), (5, 2)):
        up, um = u.copy(), u.copy()
        up[idx] += h
        um[idx] -= h
        fd = (np.sum(W * (H + up @ A))
              - np.sum(W * (H + um @ A))) / (2 * h)
        assert abs(fd - dU[idx]) < 1e-6, idx


# ---------------- T-4 no-body structural no-op ----------------

def test_t4_no_body_returns_same_object():
    site = PortSite(5)
    H = RNG.normal(size=(8, 5))
    assert site.forward(H, None) is H          # zero arithmetic
    site.backward_step(np.zeros((8, 5)), None, lr=1e-2)  # no-op


# ---------------- T-5 refusals ----------------

def test_t5_refusals():
    site = PortSite(5)
    with pytest.raises(ValueError, match="load-only|refused"):
        site.add_body(make_port_body(4, 6, 2, 1e-2, 1),
                      port_type="legacy_scalar")
    with pytest.raises(ValueError, match="port ownership"):
        site.add_body(Network(4, 6, seed=1))
    with pytest.raises(ValueError, match="load-only"):
        LegacyScalarPort().add_body(None)
    with pytest.raises(ValueError, match="out_width"):
        Network(4, 6, seed=1, out_width=0)
    m = Network(4, 6, seed=1)                  # not port-owned
    with pytest.raises(ValueError, match="port-owned"):
        m.train_from_grad(np.zeros((2, 4)), np.zeros((2, 1)))


# ---------------- T-6 serialization round trip ----------------

def test_t6_port_pickle_round_trip():
    site, body = make_site()
    Xb = RNG.normal(size=(6, 4))
    for _ in range(3):
        H = RNG.normal(size=(6, 5))
        site.forward(H, Xb)
        site.backward_step(RNG.normal(size=(6, 5)), Xb, lr=1e-2)
    site2 = pickle.loads(pickle.dumps(site))
    H = RNG.normal(size=(6, 5))
    assert np.allclose(site.forward(H, Xb),
                       site2.forward(H, Xb), atol=1e-12)
    assert site2.n_params() == site.n_params()


# ---------------- VM-5 reduction (informational) ----------------

def test_vm5_reduction_to_legacy_expression_informational():
    """The defective v1 attach is the k=1 / frozen-one-hot
    special case of PORT-FWD — documented; NEVER a reference."""
    site = PortSite(5)
    body = make_port_body(4, 6, 1, lr=1e-2, seed=2)
    site.add_body(body)
    onehot = np.zeros((1, 5))
    onehot[0, 3] = 1.0
    site.bodies[0]["A"] = onehot
    H = RNG.normal(size=(8, 5))
    Xb = RNG.normal(size=(8, 4))
    for _ in range(4):                          # non-zero body
        body.train_from_grad(Xb, RNG.normal(size=(8, 1)) * 0.1,
                             sgd_lr=0.05)
    got = site.forward(H, Xb)
    legacy = H.copy()
    legacy[:, 3] = legacy[:, 3] + np.asarray(
        body.predict(Xb))[:, 0]
    assert np.allclose(got, legacy, atol=0)


# ---------------- train_from_grad exactness ----------------

def test_train_from_grad_delivers_exact_raw_gradient():
    """The doc 35 D4 plan-B identity is EXACT, so it is held to
    exact standards (owner hardening ruling): (1) applied
    updates == CLOSED-FORM analytic gradients of
    L = <dU, pred> at <= 1e-12 (formula vs formula; measured
    actuals are machine-epsilon, 1.7e-17..8.9e-16); (2) an
    independent FD cross-check at the FD instrument's own
    limit (rel 1e-6)."""
    from engine.primitives import gelu, gelu_d
    body = make_port_body(3, 5, 2, lr=1e-2, seed=4)
    Xb = RNG.normal(size=(6, 3))
    for _ in range(6):                          # leave zero init
        body.train_from_grad(Xb, RNG.normal(size=(6, 2)) * 0.1,
                             sgd_lr=0.05)
    dU = RNG.normal(size=(6, 2))
    before = {k: np.array(getattr(body, k))
              for k in ("W1", "b1", "W2", "c")}
    body.train_from_grad(Xb, dU, sgd_lr=1.0)
    applied = {k: before[k] - np.asarray(getattr(body, k))
               for k in before}

    # (1) closed forms: pred = gelu(Xb W1^T + b1) W2^T + c
    A = Xb @ before["W1"].T + before["b1"]
    Z = gelu(A)
    gW2 = dU.T @ Z                       # dL/dW2
    gc = dU.sum(0)                       # dL/dc
    dA = (dU @ before["W2"]) * gelu_d(A)
    gW1 = dA.T @ Xb                      # dL/dW1
    gb1 = dA.sum(0)                      # dL/db1
    for name, g in (("W2", gW2), ("c", gc),
                    ("W1", gW1), ("b1", gb1)):
        err = np.abs(applied[name] - g).max()
        assert err / max(1.0, np.abs(g).max()) < 1e-12, \
            (name, err)

    # (2) independent FD cross-check at the instrument's limit
    h = 1e-6

    def loss_at(name, idx, delta):
        m2 = pickle.loads(pickle.dumps(body))
        for nm in ("W1", "b1", "W2", "c"):
            setattr(m2, nm, before[nm].copy())
        getattr(m2, name)[idx] = before[name][idx] + delta
        return float(np.sum(dU * m2.predict(Xb)))

    for name, idx in (("W2", (0, 2)), ("W2", (1, 4)),
                      ("c", (0,)), ("W1", (2, 1))):
        fd = (loss_at(name, idx, +h)
              - loss_at(name, idx, -h)) / (2 * h)
        got = float(applied[name][idx])
        assert abs(got - fd) / max(1.0, abs(fd)) < 1e-6, \
            (name, idx, got, fd)


# ---------------- out_width construction contract ----------------

def test_out_width_contract():
    m1 = Network(4, 6, seed=9)
    mk = Network(4, 6, seed=9, out_width=3, zero_out=True)
    assert m1.W2.shape == (1, 6) and mk.W2.shape == (3, 6)
    # default construction draws are UNCHANGED (bitwise)
    m1b = Network(4, 6, seed=9)
    assert np.array_equal(m1.W1, m1b.W1)
    assert np.array_equal(m1.W2, m1b.W2)


# ============ W2(e): attention-body port compliance ============

def test_w2e_attention_body_port_compliance():
    """Attention bodies are full port citizens: vector output,
    zero-birth exactness, and EXACT raw-gradient training (their
    backward is natively gradient-seeded)."""
    from reference_net.growth_port import make_port_body
    pol = {"grow_attention_d_model": 8, "grow_attention_layers": 1,
           "grow_attention_heads": 2, "grow_attention_ffn": 8}
    body = make_port_body(4, 6, 3, lr=1e-2, seed=5,
                          body_type="attention", policy=pol)
    Xb = RNG.normal(size=(6, 4))
    assert body.out_width == 3
    out0 = np.asarray(body.predict(Xb))
    assert out0.shape == (6, 3) and out0.any()   # LIVE at birth
    # port integration: A = 0 -> exact no-op through a site
    # (zero-side doctrine: the assembly is the zero side)
    site = PortSite(5)
    site.add_body(body)
    H = RNG.normal(size=(6, 5))
    assert np.array_equal(site.forward(H, Xb), H)
    # raw-gradient exactness on the readout (closed form):
    # dL/dWout = Pool^T dU — extract via sgd_lr = 1
    for _ in range(5):
        body.train_from_grad(Xb, RNG.normal(size=(6, 3)) * 0.1,
                             sgd_lr=0.05)
    dU = RNG.normal(size=(6, 3))
    Wout_before = np.array(body.P["Wout"])
    raw, Pool, _ = body._forward(body._std_x(
        body._bk.ingest(Xb)), cache=True)
    body.train_from_grad(Xb, dU, sgd_lr=1.0)
    applied = Wout_before - np.asarray(body.P["Wout"])
    closed = np.asarray(Pool).T @ dU
    assert np.abs(applied - closed).max() < 1e-12
    # FD cross-check on one coordinate at the instrument's limit
    h = 1e-6
    import pickle as pk
    base_state = pk.dumps(body)

    def loss_at(delta):
        b2 = pk.loads(base_state)
        W = np.array(b2.P["Wout"]); W[1, 2] += delta
        b2.P["Wout"] = b2._bk.ingest(W)
        return float(np.sum(dU * np.asarray(b2.predict(Xb))))

    fd = (loss_at(h) - loss_at(-h)) / (2 * h)
    # note: applied refers to the PRE-step state; recompute grad
    b3 = pk.loads(base_state)
    raw3, Pool3, c3 = b3._forward(b3._std_x(
        b3._bk.ingest(Xb)), cache=True)
    g3 = np.asarray(Pool3).T @ dU
    assert abs(g3[1, 2] - fd) / max(1.0, abs(fd)) < 1e-6


# ============ D5 instability instrument (review gap-fill) ======

def test_d5_port_instability_instrument_hand_literals():
    """doc 35 D5: per-body assembly instability from update
    EMAs. Hand-checked recursion (fresh slot: edw = 0,
    eadw = 1e-12): after ONE sgd step with known update
    upd = u^T dH,
        edw  == 0.05 * upd            (exactly)
        eadw == 0.95e-12 + 0.05*|upd| (exactly)
    and instability = 1 - ||edw||_F / (||eadw||_F + 1e-12)
    is ~0 for a steady drift (repeat same direction) and
    strictly larger under sign-alternating updates."""
    u = np.array([[1.0, 2.0], [0.0, 3.0]])
    dH = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    dA = u.T @ dH
    site = PortSite(3)
    body = make_port_body(2, 4, 2, lr=1e-2, seed=8)
    site.add_body(body)
    body.train_from_grad = lambda X, dU_, sgd_lr=None: 0.0
    site._u_cache = [u]
    site.backward_step(dH, np.zeros((2, 2)), lr=1e-2, sgd_lr=1.0)
    slot = site.bodies[0]
    assert np.array_equal(slot["edw"], 0.05 * dA)
    assert np.array_equal(slot["eadw"],
                          0.95e-12 + 0.05 * np.abs(dA))
    # eadw >= |edw| pointwise (induction invariant) -> in [0,1]
    inst = site.instability()
    assert len(inst) == 1 and 0.0 <= inst[0] <= 1.0
    # steady drift: same-direction updates -> low instability
    for _ in range(30):
        site._u_cache = [u]
        site.backward_step(dH, np.zeros((2, 2)), lr=1e-2,
                           sgd_lr=1.0)
    steady = site.instability()[0]
    # oscillation: alternate the sign every step -> higher
    site2 = PortSite(3)
    body2 = make_port_body(2, 4, 2, lr=1e-2, seed=9)
    site2.add_body(body2)
    body2.train_from_grad = lambda X, dU_, sgd_lr=None: 0.0
    for i in range(30):
        site2._u_cache = [u]
        site2.backward_step(dH if i % 2 == 0 else -dH,
                            np.zeros((2, 2)), lr=1e-2,
                            sgd_lr=1.0)
    oscillating = site2.instability()[0]
    assert steady < 0.2 < oscillating
    # instrument state survives serialization (drop the
    # instance-level test mock first — lambdas don't pickle)
    del body2.__dict__["train_from_grad"]
    site3 = pickle.loads(pickle.dumps(site2))
    assert np.allclose(site3.instability(), [oscillating],
                       atol=1e-15)


def test_w2e_attention_body_through_port_site_backward():
    """W2(e) integration: an ATTENTION body coupled through the
    full PortSite forward/backward — assembly step A - u^T dH
    exact; the body's readout receives EXACTLY the chain-rule
    gradient Pool^T (dH A^T)."""
    from reference_net.growth_port import make_port_body
    pol = {"grow_attention_d_model": 8, "grow_attention_layers": 1,
           "grow_attention_heads": 2, "grow_attention_ffn": 8}
    body = make_port_body(4, 6, 2, lr=1e-2, seed=7,
                          body_type="attention", policy=pol)
    site = PortSite(5)
    site.add_body(body)
    Xb = RNG.normal(size=(6, 4))
    for _ in range(5):                          # leave zero init
        body.train_from_grad(Xb, RNG.normal(size=(6, 2)) * 0.1,
                             sgd_lr=0.05)
    A = RNG.normal(size=(2, 5))
    site.bodies[0]["A"] = A.copy()
    pre = pickle.dumps(body)                    # pre-step state
    H = RNG.normal(size=(6, 5))
    dH = RNG.normal(size=(6, 5))
    out = site.forward(H, Xb)
    u = np.asarray(pickle.loads(pre).predict(Xb))
    assert np.array_equal(out, H + u @ A)       # PORT-FWD exact
    Wout_before = np.array(body.P["Wout"])
    b2 = pickle.loads(pre)
    _, Pool, _ = b2._forward(b2._std_x(b2._bk.ingest(Xb)),
                             cache=True)
    site.backward_step(dH, Xb, lr=1e-2, sgd_lr=1.0)
    assert np.abs(np.asarray(site.bodies[0]["A"])
                  - (A - u.T @ dH)).max() < 1e-12   # PORT-BWD-A
    dU = dH @ A.T                                   # PORT-BWD-U
    applied = Wout_before - np.asarray(body.P["Wout"])
    assert np.abs(applied - np.asarray(Pool).T @ dU).max() < 1e-12


# ========= W3: coupling LIVENESS (anti-deadlock regression) =====

def test_w3_grown_coupling_actually_trains():
    """The regression that caught the double-zero deadlock: after
    growth + training, the port coupling must move — loading
    ||u A||_F becomes nonzero and the host's fit on the residual
    improves relative to birth. Guards the zero-SIDE doctrine
    end-to-end on a real host."""
    import warnings
    from reference_net.net import Network
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 5))
    y = rng.normal(size=(24, 1))
    m = Network(5, 8, lr=1e-2, seed=3)
    for _ in range(30):
        m.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # toy-scale guard
        m.grow(2, hidden=6)
    site = m._port_site
    assert site.loading(m._std_x(m._bk.ingest(X))) == [0.0]
    for _ in range(50):
        m.train_step(X, y)
    load = site.loading(m._std_x(m._bk.ingest(X)))[0]
    assert load > 0.0, "port coupling never moved (deadlock)"
    A = np.asarray(site.bodies[0]["A"])
    assert np.abs(A).max() > 0.0
    inst = site.instability()
    assert len(inst) == 1 and 0.0 <= inst[0] < 1.0
