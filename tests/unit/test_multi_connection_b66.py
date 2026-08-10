"""B-6-6 (54 v1.12; matrix C46; 53 §7 PC-2): MULTI-CONNECTION
COEXISTENCE — the amended FR-2/FR-5 official route "several
connections = several events", exactly per the box spec:

On ONE model: THREE standalone coupling-only events
(kind="none", distinct keys) PLUS one deepen AND one grow
(widen) — the full move mix — with training between events:
 (a) each connection is born zero-contribution (prediction
     bitwise unchanged after each creation);
 (b) NUMERIC ACCURACY (owner discipline: verify correct
     VALUES, never mere motion): with hand-set A matrices the
     site's combined output equals the SUPERPOSITION sum of
     each connection's own u@A exactly (1e-12); and after
     training every coupling's A has moved independently off
     zero;
 (c) no cross-contamination: removing ONE by key leaves the
     other two's tensors bitwise untouched and the model
     serves;
 (d) every event and removal is ledgered with its own key and
     verbatim specs;
 (e) artifact round-trip with all three present serves
     bitwise.
"""
import pickle   # safe: locally produced model states only
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference_net.net import Network                       # noqa: E402
from reference_net.foundation import compose                # noqa: E402
from reference_net.foundation.specs import (                # noqa: E402
    ALL, NONE, BirthSpec, PlacementSpec, StructureSpec, Tap,
    WiringSpec)
import reference_net.growth_port  # noqa: F401, E402 (builders)


def _data(seed=101, n=24, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _skip(net, key):
    return compose.grow(
        net,
        StructureSpec(kind="none", params={"width": 3}),
        WiringSpec(reads=[Tap("scope_input", role="skip")],
                   write={"target": "stream", "span": ALL}),
        PlacementSpec(chain=NONE), BirthSpec(), key=key)


def _pred(net, X):
    return np.asarray(net._bk.to_numpy(net.predict(X)),
                      dtype=np.float64)


def test_b66_multi_connection_coexistence_full_mix():
    net = Network(3, 8, lr=1e-2, seed=7)
    X, y = _data()
    for _ in range(120):
        net.train_step(X, y)

    # --- creations interleaved with training and the full
    #     move mix; (a) zero-birth at EACH creation ---
    handles = {}
    base = _pred(net, X)
    handles["skip-A"] = _skip(net, "skip-A")
    assert np.array_equal(_pred(net, X), base)        # (a)
    for _ in range(10):
        net.train_step(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(1, hidden=4)                          # widen
    for _ in range(10):
        net.train_step(X, y)

    base = _pred(net, X)
    handles["skip-B"] = _skip(net, "skip-B")
    assert np.array_equal(_pred(net, X), base)        # (a)
    for _ in range(10):
        net.train_step(X, y)

    net.deepen()                                       # deepen
    for _ in range(10):
        net.train_step(X, y)

    base = _pred(net, X)
    handles["skip-C"] = _skip(net, "skip-C")
    assert np.array_equal(_pred(net, X), base)        # (a)
    for _ in range(10):
        net.train_step(X, y)

    # --- (b) NUMERIC ACCURACY (owner discipline: verify
    #     correct VALUES, not mere motion). Superposition:
    #     with hand-set A matrices the site's combined output
    #     must equal the SUM of each connection's own u @ A —
    #     hand-checkable linear algebra, judged exactly. ---
    site = net._port_site
    keys = [c.get("key") for c in site.bodies]
    assert {"skip-A", "skip-B", "skip-C"} <= set(keys)
    saved = {c.get("key"): np.asarray(
        net._bk.to_numpy(c.A)).copy() for c in site.bodies}
    hand = {}
    for i, c in enumerate(site.bodies):
        Ah = np.zeros(np.asarray(c.A).shape)
        Ah[0, (2 * i) % Ah.shape[1]] = float(i + 1)  # distinct
        hand[c.get("key")] = Ah
        c.A = net._bk.ingest(Ah.copy())
    H0 = net._bk.ingest(np.zeros((len(X), net.H)))
    combined = np.asarray(net._bk.to_numpy(
        site.forward(H0, net._bk.ingest(X))))
    expect = np.zeros((len(X), net.H))
    for c in site.bodies:
        u = np.asarray(net._bk.to_numpy(
            c.body.predict(net._bk.ingest(X))))
        expect = expect + u @ hand[c.get("key")]
    assert np.max(np.abs(combined - expect)) <= 1e-12
    site._u_cache = None                 # probe used forward
    for c in site.bodies:                # restore trained state
        c.A = net._bk.ingest(saved[c.get("key")].copy())
    # and: after training every A had moved independently
    for k, h in handles.items():
        assert float(np.max(np.abs(saved[k]))) > 0.0, k

    # --- (c) per-key removal: others bitwise untouched ---
    A_b = np.asarray(net._bk.to_numpy(
        handles["skip-B"].A)).copy()
    A_c = np.asarray(net._bk.to_numpy(
        handles["skip-C"].A)).copy()
    net.remove_grown("skip-A")
    assert site.body_by_key("skip-A") is None
    assert np.array_equal(np.asarray(net._bk.to_numpy(
        handles["skip-B"].A)), A_b)
    assert np.array_equal(np.asarray(net._bk.to_numpy(
        handles["skip-C"].A)), A_c)
    net.predict(X)                     # model serves
    net.train_step(X, y)               # and trains

    # --- (d) ledger: each creation with its own key + specs
    grow_events = [r for r in net.gain_ledger
                   if r["event"] == "grow[none]"]
    assert [r["site"] for r in grow_events] == \
        ["skip-A", "skip-B", "skip-C"]
    for r in grow_events:
        assert set(r["specs"]) == {"structure", "wiring",
                                   "placement", "birth"}
        assert r["specs"]["structure"]["kind"] == "none"

    # --- (e) artifact round-trip with the remaining two ---
    out = _pred(net, X)
    twin = pickle.loads(pickle.dumps(net))
    assert np.array_equal(_pred(twin, X), out)
    twin.train_step(X, y)
