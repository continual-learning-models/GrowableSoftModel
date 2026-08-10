"""A3 boxes (plan 53 v1.2 step A3; 54 v1.10 section 3):
worked examples for every NEWLY HONORED spec value, plus the
permanent root-neutrality static scan.

 V-A3-1 (a) subset-span growth END-TO-END on a live Network
            (grown coupling reads span_in of its body, writes
            span_out of the trunk stream): hand-checked
            contribution placement, exact zeros outside the
            span, textbook A-update, dU scattered to the
            body's coordinates; serialization round-trip
            preserves spans (ADDITIVE keys only).
        (b) COUPLING-ONLY event (kind="none", the standalone
            cross-layer connection — "skip = one more
            coupling"): zero contribution at birth (bitwise),
            trainable A, ledgered with verbatim specs.
        (c) INTERIOR-POSITION insertion into the block chain:
            a zero-Bout block inserted at ANY seam leaves
            predict() bitwise unchanged (the delta-operator
            preservation property at arbitrary p).
        (d) copy/interleave recipes through the birth() stage
            on a shape_map structure.
        Verdict citations (PR-2, math-research/02): span
        widening = (degree +0, rank +k) addition move; skip
        coupling repairs junction rank cuts (T8); interior
        delta = (+1, rank unchanged).
 V-A3-2 root-neutrality static scan (AC-1, permanent): no
        growth-policy tokens anywhere in foundation/*; the
        mode-determining spec fields have NO defaults; the
        neutral defaults are exactly {span=ALL, position=END,
        chain=NONE, recipe="random"}.
"""
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RN = REPO / "modules" / "ReferenceNet" / "reference_net"
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference_net.foundation import compose                # noqa: E402
from reference_net.foundation.coupling import (             # noqa: E402
    Coupling, IdentitySource)
from reference_net.foundation.specs import (                # noqa: E402
    ALL, END, NONE, BirthSpec, PlacementSpec, StructureSpec,
    Tap, WiringSpec)
from reference_net.net import Network                       # noqa: E402
import reference_net.growth_port  # noqa: F401, E402 (builders)


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _warm_net(seed=7, steps=120):
    X, y = _data()
    net = Network(3, 8, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


# ---------------- V-A3-1 (a): subset spans end-to-end ----------

def test_a3_span_growth_end_to_end():
    net, X, y = _warm_net()
    base = np.asarray(net._bk.to_numpy(net.predict(X)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from reference_net.growth_port import make_port_body
        body = make_port_body(3, 4, 2, net.lr, 99,
                              backend=net._bk)
        site = net._ensure_port_site()
        site.add_body(body, key="span-demo",
                      span_in=np.array([0]),
                      span_out=np.array([2, 5]))
    c = site.bodies[-1]
    assert c.A.shape == (1, 2)          # |span_in| x |span_out|
    # zero-born: prediction bitwise unchanged at birth
    assert np.array_equal(
        np.asarray(net._bk.to_numpy(net.predict(X))), base)
    # hand-set A and verify the forward placement exactly:
    # contribution lands ONLY in trunk columns [2, 5]
    c.A = net._bk.ingest(np.array([[10.0, 20.0]]))
    H = net._bk.ingest(np.zeros((len(X), 8)))
    out = np.asarray(net._bk.to_numpy(site.forward(H, X)))
    u = np.asarray(net._bk.to_numpy(body.predict(X)))[:, [0]]
    expect = np.zeros((len(X), 8))
    expect[:, [2, 5]] = u @ np.array([[10.0, 20.0]])
    assert np.max(np.abs(out - expect)) <= 1e-12
    zero_cols = [0, 1, 3, 4, 6, 7]
    assert np.array_equal(out[:, zero_cols],
                          np.zeros((len(X), 6)))   # exact zeros
    # backward: dU scattered to the body's own coordinates
    dH = net._bk.ingest(np.ones((len(X), 8)))
    dA, dU = c.gradients(np.asarray(net._bk.to_numpy(dH)), u)
    assert np.asarray(dA).shape == (1, 2)
    dU = np.asarray(dU)
    assert dU.shape == (len(X), 2)      # body out_width = 2
    assert np.array_equal(dU[:, 1], np.zeros(len(X)))
    # serialization round-trip preserves spans (additive keys)
    slot = c.to_slot_dict(net._bk.to_numpy)
    assert "span_in" in slot and "span_out" in slot
    c2 = Coupling.from_slot_dict(slot, net._bk)
    assert np.array_equal(np.asarray(c2.span_out), [2, 5])
    default_slot = site.bodies[0].to_slot_dict(net._bk.to_numpy) \
        if len(site.bodies) > 1 else None
    if default_slot is not None:
        assert "span_in" not in default_slot   # bytes unchanged


# ---------------- V-A3-1 (b): coupling-only event ----------

def test_a3_coupling_only_event():
    net, X, y = _warm_net(seed=11)
    base = np.asarray(net._bk.to_numpy(net.predict(X)))
    n0 = net.n_params()
    h = compose.grow(
        net,
        StructureSpec(kind="none", params={"width": 3}),
        WiringSpec(reads=[Tap("scope_input", role="skip")],
                   write={"target": "stream", "span": ALL}),
        PlacementSpec(chain=NONE),
        BirthSpec(), key="skip-1")
    assert isinstance(h.body, IdentitySource)
    # zero-born: bitwise unchanged at birth
    assert np.array_equal(
        np.asarray(net._bk.to_numpy(net.predict(X))), base)
    assert net.n_params() == n0 + 3 * 8      # A only (3x8)
    rec = net.gain_ledger[-1]
    assert rec["event"] == "grow[none]"
    assert rec["specs"]["wiring"]["reads"][0]["role"] == "skip"
    # the coupling trains: A moves off zero under real steps
    for _ in range(3):
        net.train_step(X, y)
    assert float(np.max(np.abs(
        np.asarray(net._bk.to_numpy(h.A))))) > 0.0


# ---------------- V-A3-1 (c): interior position ----------

def test_a3_interior_insert_preserves_function_bitwise():
    net, X, y = _warm_net(seed=21)
    net.deepen(); net.deepen()
    for _ in range(5):
        net.train_step(X, y)
    base = np.asarray(net._bk.to_numpy(net.predict(X)))
    m = net.H
    rng = np.random.default_rng(123)
    blk = {"Bin": net._bk.ingest(
               rng.normal(0, np.sqrt(2.0 / net.H), (m, net.H))),
           "bb": net._bk.ingest(np.zeros(m)),
           "Bout": net._bk.ingest(np.zeros((net.H, m)))}
    w = WiringSpec(reads=[Tap("stream")],
                   write={"target": "stream"})
    r = compose.resolve(net, w,
                        PlacementSpec(chain="blocks", position=1))
    k = compose.place(net, blk, r)
    net._rebuild_opt()
    assert k == 1 and len(net.blocks) == 3
    out = np.asarray(net._bk.to_numpy(net.predict(X)))
    assert np.array_equal(out, base)     # preservation at p=1
    net.train_step(X, y)                 # and it trains


# ---------------- V-A3-1 (d): recipes through birth() ----------

def test_a3_recipes_through_birth_stage():
    class Struct:
        def __init__(self):
            self.W = None
        def shape_map(self):
            return {"W": (2, 2)}
        def n_params(self):
            return 4
        def predict(self, X):
            return X
        def train_from_grad(self, X, dU, sgd_lr=None):
            pass
        def __getstate__(self):
            return {"W": self.W}
        def __setstate__(self, st):
            self.W = st["W"]
    src = {"W": np.arange(4.0).reshape(2, 2)}
    s = compose.birth(Struct(),
                      BirthSpec(recipe="copy_layer"),
                      context={"source": src})
    assert np.array_equal(s.W, src["W"])
    s = compose.birth(Struct(),
                      BirthSpec(recipe="interleave_neighbors"),
                      context={"left": {"W": np.zeros((2, 2))},
                               "right": {"W": 4 * np.ones((2, 2))}})
    assert np.array_equal(s.W, 2 * np.ones((2, 2)))


# ---------------- V-A3-2: root-neutrality static scan ----------

_FORBIDDEN_TOKENS = (
    "grow_port_type", "grow_body_out_width", "grow_body_type",
    "grow_scale_guard", "grow_min_host", "DEFAULT_GROWTH_POLICY",
    "growthpolicy", "INNER_LR_FACTOR", "eta_target",
)


def test_a3_neutrality_scan_foundation_tokens():
    for f in sorted((RN / "foundation").glob("*.py")):
        text = f.read_text()
        for tok in _FORBIDDEN_TOKENS:
            assert tok not in text, f"{f.name}: {tok}"


def test_a3_neutrality_mode_fields_have_no_defaults():
    sig = inspect.signature(WiringSpec.__init__)
    assert sig.parameters["reads"].default \
        is inspect.Parameter.empty
    assert sig.parameters["write"].default \
        is inspect.Parameter.empty


def test_a3_neutrality_defaults_are_exactly_the_neutral_set():
    assert ALL is None and END == "end" and NONE == "none"
    p = inspect.signature(PlacementSpec.__init__).parameters
    assert p["chain"].default == NONE
    assert p["position"].default == END
    b = inspect.signature(BirthSpec.__init__).parameters
    assert b["recipe"].default == "random"
    t = inspect.signature(Tap.__init__).parameters
    assert t["span"].default is ALL
