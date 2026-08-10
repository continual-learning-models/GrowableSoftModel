"""B3 boxes — history & content management (docs 55 s3-B5/B7,
54 s7).

 B-5-1 snapshot/rollback: auto-snapshot precedes a growth
       event (policy-gated, default ON); after further
       training, rollback restores the EXACT captured state
       (bitwise incl. optimizer slots and counters); the
       restored model then trains AND grows — the next move
       is solely the CALLER's (nothing fires automatically);
       retention window drops oldest; rollback beyond the
       in-memory window refuses (finite, FR-18) while a
       DISK-persisted snapshot of the same vintage restores.
 B-5-2 shrink governance: removal is preceded by a snapshot
       and is reversible within the window (post-rollback
       state hash equals pre-removal, bitwise).
 B-6-2 bounded trial: consumes <= budget steps; report carries
       losses and realized gain; state after trial EQUALS the
       pre-trial state (unconditional rollback, bitwise);
       ledger provenance kind='trial'; applying for real
       afterwards succeeds; abortable (C-5).
 B-7-1 GrowthStore: write-once refusal; events.jsonl
       append-only; REINDEX EQUIVALENCE (drop index.db content
       -> rebuild from files -> identical query answers);
       snapshot disk round-trip restores bitwise; queries
       filter by kind/trigger.
 HOST-PARITY: snapshot/rollback on an attention host.
"""
import hashlib
import json
import pickle   # safe: locally produced model states only
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference_net.net import Network                       # noqa: E402
from reference_net.growth_store import (                    # noqa: E402
    GrowthStore, rollback, snapshot, trial)
from reference_net.growthpolicy import \
    DEFAULT_GROWTH_POLICY                                   # noqa: E402
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                              # noqa: E402


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _hash(net):
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if str(k) == "gain_ledger":
                    continue
                h.update(str(k).encode()); walk(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"["); [walk(v) for v in o]; h.update(b"]")
        elif isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
        elif hasattr(o, "__getstate__") and not isinstance(
                o, (str, bytes, int, float, bool, type(None))):
            walk(o.__getstate__())
        else:
            h.update(repr(o).encode())
    walk(net.__getstate__())
    return h.hexdigest()


# ---------------- B-5-1 ----------------

def test_b3_auto_snapshot_and_bitwise_rollback():
    net = Network(3, 8, lr=1e-2, seed=7)
    X, y = _data()
    for _ in range(20):
        net.train_step(X, y)
    pre = _hash(net)
    net.deepen()                       # auto-snapshot precedes
    assert len(net._snapshots) == 1
    for _ in range(10):
        net.train_step(X, y)
    rollback(net, net._snapshots[-1])
    assert _hash(net) == pre           # bitwise incl. opt state
    assert net.gain_ledger[-1]["event"] == "rollback"
    assert net.gain_ledger[-1]["provenance"][
        "events_discarded"] == 1
    # fully LIVE afterwards; next move is the caller's
    net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)          # e.g. widen next
    assert net.grown_body(0) is not None


def test_b3_retention_window_finite():
    net = Network(3, 8, lr=1e-2, seed=9)
    X, y = _data()
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              growth_snapshot_keep=2)
    for _ in range(5):
        net.train_step(X, y)
    first = None
    for i in range(4):
        net.deepen()                  # 4 auto-snapshots
        if first is None:
            first = net._snapshots[0]["id"]
    assert len(net._snapshots) == 2   # window holds, oldest gone
    assert all(r["id"] != first for r in net._snapshots)


def test_b3_beyond_window_disk_restores(tmp_path):
    net = Network(3, 8, lr=1e-2, seed=11)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    rec = snapshot(net, tag="keepsake")
    store = GrowthStore(tmp_path / "gs")
    store.save_snapshot(rec)
    pre = _hash(net)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              growth_snapshot_keep=1)
    net.deepen(); net.deepen()        # pushes rec out of ring
    assert all(r["id"] != rec["id"] for r in net._snapshots)
    disk = store.load_snapshot(rec["id"])
    twin = pickle.loads(disk["blob"])
    assert _hash(twin) == pre         # disk vintage restores


# ---------------- B-5-2 shrink governance ----------------

def test_b3_removal_snapshot_and_reversibility():
    net = Network(3, 8, lr=1e-2, seed=13)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    net.deepen()
    for _ in range(5):
        net.train_step(X, y)
    pre_removal = _hash(net)
    n_snaps = len(net._snapshots)
    net.remove_block(0)
    assert len(net._snapshots) == n_snaps + 1  # pre-event snap
    assert net.blocks == []
    rollback(net, net._snapshots[-1])
    assert _hash(net) == pre_removal   # reversible in window


# ---------------- B-6-2 bounded trial ----------------

def test_b3_trial_budget_rollback_and_report():
    net = Network(3, 8, lr=1e-2, seed=17)
    X, y = _data()
    for _ in range(20):
        net.train_step(X, y)
    pre = _hash(net)
    rep = trial(net, lambda h: h.deepen(), X, y,
                budget_steps=6)
    assert rep["steps_run"] == 6 and len(rep["losses"]) == 6
    assert rep["realized_gain"] is not None
    assert _hash(net) == pre           # unconditional rollback
    assert net.gain_ledger[-1]["event"] == "rollback"
    assert net.gain_ledger[-1]["provenance"]["kind"] == "trial"
    net.deepen()                       # applying for real works
    assert len(net.blocks) == 1


def test_b3_trial_abortable():
    net = Network(3, 8, lr=1e-2, seed=19)
    X, y = _data()
    for _ in range(5):
        net.train_step(X, y)
    pre = _hash(net)
    calls = {"n": 0}

    def abort():
        calls["n"] += 1
        return calls["n"] > 2
    rep = trial(net, lambda h: h.deepen(), X, y,
                budget_steps=50, abort=abort)
    assert rep["aborted"] is True and rep["steps_run"] == 2
    assert _hash(net) == pre           # clean rollback


# ---------------- B-7-1 the store ----------------

def test_b3_store_write_once_and_reindex_equivalence(tmp_path):
    store = GrowthStore(tmp_path / "gs")
    net = Network(3, 8, lr=1e-2, seed=23)
    rec = snapshot(net)
    store.save_snapshot(rec)
    with pytest.raises(ValueError, match="write-once"):
        store.save_snapshot(rec)
    store.append_event("deepen", "caller", "scope", 100,
                       {"note": "e1"})
    store.append_event("deepen", "policy", "scope", 100,
                       {"note": "e2"})
    store.append_event("refine", "caller", 0, 50,
                       {"note": "e3"})
    sha = store.save_plan({"steps": []})
    assert store.save_plan({"steps": []}) == sha  # same content
    q0 = sorted(json.dumps(e, sort_keys=True) for e in
                store.query_events(kind="deepen"))
    assert len(q0) == 2
    assert len(store.query_events(trigger="policy")) == 1
    # REINDEX EQUIVALENCE: rebuild from files, answers identical
    store.reindex()
    q1 = sorted(json.dumps(e, sort_keys=True) for e in
                store.query_events(kind="deepen"))
    assert q1 == q0
    # events.jsonl is append-only text (truth)
    lines = (tmp_path / "gs" / "events.jsonl") \
        .read_text().splitlines()
    assert len(lines) == 3


# ---------------- host parity ----------------

def test_b3_host_parity_snapshot_rollback_on_attention_host():
    X, y = _data(0, 6, 3)
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])
    for _ in range(5):
        ga.train_step(X, y)
    rec = snapshot(ga)
    base = np.asarray(ga.predict(X))
    ga.insert_layer(1)
    for _ in range(3):
        ga.train_step(X, y)
    rollback(ga, rec)
    assert ga.L == 2
    assert np.array_equal(np.asarray(ga.predict(X)), base)
    ga.train_step(X, y)                # live afterwards


def test_b3_rollback_preserves_observer_state():
    """E2E find (close-out F8): the snapshot ring and monitor
    rig are SESSION state — a rollback restores the MODEL and
    must not silence the instruments or shrink the remaining
    history window."""
    from reference_net.instrument import monitor_configure
    net = Network(3, 8, lr=1e-2, seed=29)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    monitor_configure(net, cadence=2, window=10)
    rec = snapshot(net, tag="obs")
    for _ in range(6):
        net.train_step(X, y)
    n_recs = len(net._monitor["records"])
    assert n_recs >= 2
    rollback(net, rec)
    assert net._monitor["records"] and \
        len(net._monitor["records"]) == n_recs   # rig survives
    assert any(r["id"] == rec["id"]
               for r in net._snapshots)          # ring survives
    for _ in range(4):
        net.train_step(X, y)                     # still ticking
    assert len(net._monitor["records"]) > n_recs


# ---- B-5-2 COMPLETION (54 s7: "remove_block AND remove_by_key
# and successors" — review find R1: only remove_block was
# boxed; these two boxes cover the other removal operations
# under the SAME FR-18 governance) ----

def test_b3_remove_grown_snapshot_and_reversibility():
    """FR-18: removing a grown body is preceded by an automatic
    snapshot and reversible within the window."""
    net = Network(3, 8, lr=1e-2, seed=31)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
    for _ in range(5):
        net.train_step(X, y)
    pre_removal = _hash(net)
    n_snaps = len(net._snapshots)
    net.remove_grown(0)
    assert len(net._snapshots) == n_snaps + 1  # pre-event snap
    assert net.grown_body(0) is None
    rollback(net, net._snapshots[-1])
    assert _hash(net) == pre_removal           # reversible
    assert net.grown_body(0) is not None


def test_b3_remove_loop_snapshot_and_reversibility():
    """FR-18: removing the loop block is preceded by an
    automatic snapshot and reversible within the window."""
    net = Network(3, 8, lr=1e-2, seed=37)
    X, y = _data()
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    for _ in range(10):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.loop(4)
    for _ in range(5):
        net.train_step(X, y)
    pre_removal = _hash(net)
    n_snaps = len(net._snapshots)
    net.remove_loop()
    assert len(net._snapshots) == n_snaps + 1  # pre-event snap
    assert net.loop_block is None
    rollback(net, net._snapshots[-1])
    assert _hash(net) == pre_removal           # reversible
    assert net.loop_block is not None


# ---- B-5-2 COMPLETION part 2 (R11; 55 B5 REMOVAL LEDGERING;
# the box spec's 'ledgered with trigger provenance' phrase,
# previously unasserted — now assertions, per the owner's
# value-verification discipline) ----

def test_b3_removals_are_ledgered_with_trigger():
    """FR-18: every removal carrier writes a ledger record —
    remove_grown with NEGATIVE accounting mirroring its grow
    event (sibling semantics), and ALL removal records carry
    trigger provenance."""
    net = Network(3, 8, lr=1e-2, seed=41)
    X, y = _data()
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    for _ in range(10):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
        net.deepen()
        net.loop(4)
    grow_rec = [r for r in net.gain_ledger
                if r["event"] == "refine"][0]
    # -- remove_grown: record exists, negative mirror of grow
    net.remove_grown(0)
    rec = net.gain_ledger[-1]
    assert rec["event"] == "remove_grown"
    assert rec["site"] == 0
    assert rec["params_added"] == -grow_rec["params_added"]
    assert rec["trigger"] == "caller"
    # -- remove_block: record exists WITH trigger
    net.remove_block(0)
    rec = net.gain_ledger[-1]
    assert rec["event"] == "prune_block"
    assert rec["params_added"] < 0
    assert rec["trigger"] == "caller"
    # -- remove_loop: record exists WITH trigger
    net.remove_loop()
    rec = net.gain_ledger[-1]
    assert rec["event"] == "remove_loop"
    assert rec["params_added"] < 0
    assert rec["trigger"] == "caller"


# ============ G7: rollback backend faithfulness (doc 61 I-A;
# owner protocol: box FIRST, RED at the exact gap) ============

def _g7_hosts(bk):
    """The three classes of the G7 spec, each carrying grown
    structure: Network grown+deepened+looped; GA and TR each
    with a grow_site body."""
    import warnings as _w
    from reference_net.net import Network
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    from core.substrates.growable_attention import \
        GrowableAttentionSubstrate as GA
    from core.substrates.transformer import TransformerSubstrate
    rng = np.random.default_rng(5)
    X2 = rng.normal(size=(16, 3))
    y2 = (X2[:, 0] * X2[:, 1]).reshape(-1, 1)
    out = []
    net = Network(3, 8, lr=1e-2, seed=7, backend=bk)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    for _ in range(30):
        net.train_step(X2, y2)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        net.grow(1, hidden=4)
        net.deepen(m=4)
        net.loop(m=4)
    out.append(("network", net, X2, y2))
    ga = GA(6, 10, mode="numeric", lr=1e-2, seed=0,
            d_model=16, n_layers=2, heads_spec=[[1, 3], [2, 1]],
            backend=bk)
    Xg = rng.normal(size=(8, 6))
    yg = rng.normal(size=8)
    for _ in range(10):
        ga.train_step(Xg, yg)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        ga.grow_site(ga.growth_sites()[0][0], hidden=4)
    out.append(("ga", ga, Xg, yg))
    tr = TransformerSubstrate(3, 8, mode="numeric", d_model=8,
                              n_layers=1, n_heads=1, seed=3,
                              backend=bk)
    Xt = rng.normal(size=(8, 3))
    yt = rng.normal(size=(8, 1))
    for _ in range(10):
        tr.train_step(Xt, yt)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        tr.grow_site(tr.growth_sites()[0][0], hidden=4)
    out.append(("tr", tr, Xt, yt))
    return out


_G7_BACKENDS = [("numpy", None)]
try:
    import torch as _torch
    _G7_BACKENDS.append(("torch-cpu-f64",
                         ("torch", "cpu", "float64")))
    if _torch.backends.mps.is_available():
        _G7_BACKENDS.append(("mps-f32",
                             ("torch", "mps", "float32")))
except ImportError:
    _torch = None


@pytest.mark.parametrize("bname,spec",
                         _G7_BACKENDS,
                         ids=[b[0] for b in _G7_BACKENDS])
def test_rollback_backend_faithful(bname, spec):
    """G7 (doc 61 I-A/I-B, value-verified): for each class x
    backend: serve -> snapshot -> train one step -> rollback ->
    (i) serve BITWISE equal to the pre-snapshot serve; (ii) the
    host's _bk is the SAME backend object as before (no silent
    migration); (iii) a parameter tensor's runtime type matches
    that backend (torch tensor on torch); (iv) one further
    train_step runs on that backend. Same-backend determinism
    makes (i) BITWISE on every row incl. mps-f32."""
    from engine.backends import resolve_backend
    from reference_net.growth_store import snapshot, rollback
    bk = None if spec is None else resolve_backend(
        spec[0], device=spec[1], dtype=spec[2])
    for cname, host, X, y in _g7_hosts(bk):
        bk_before = host._bk
        pre = np.asarray(host._bk.to_numpy(host.predict(X)))
        snapshot(host, tag="g7")
        host.train_step(X, y)
        rollback(host, host._snapshots[-1])
        post = np.asarray(host._bk.to_numpy(host.predict(X)))
        assert np.array_equal(pre, post), (cname, bname)  # (i)
        assert host._bk is bk_before, (cname, bname)      # (ii)
        if spec is not None:                              # (iii)
            probe = (host.W1 if cname == "network"
                     else host.P["Wh"] if cname == "ga"
                     else host.P["Wh"])
            assert isinstance(probe, _torch.Tensor), \
                (cname, bname, type(probe).__name__)
        loss = host.train_step(X, y)                      # (iv)
        assert np.isfinite(float(loss)), (cname, bname)
