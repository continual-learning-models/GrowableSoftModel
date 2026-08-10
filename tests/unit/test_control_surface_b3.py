"""B3 boxes — the control surface (docs 55 s3-B3/B6, 54 s7):
assess / propose / plans / monitor / describe / gate modes /
mixed provenance / complete controllability.

 B-3-1 gate enforcement modes: the same failing state under
       "advise" records only, "warn" warns and proceeds,
       "refuse" raises; force=True (C-5 unconditional
       override) executes past refuse.
 B-3-2 junction-width facts exposed for the seam gate (T5).
 B-3-4 assess_growth: every FR-11 field; repeated call leaves
       the state hash unchanged (read-only); single source
       with the gates.
 B-3-5 mixed provenance: manual events trigger="caller",
       plan events trigger="policy"; both interleave.
 B-3-6 controllability audit (C-5): with ALL automation off a
       full manual session yields ZERO policy-trigger events;
       a running plan pauses / aborts; manual override during
       a plan works.
 B-6-1 propose: no mutation; refuse-verdict matches the real
       attempt.
 B-6-3 monitor: off by default; cadence honored; ring bounded;
       JSON export.
 B-6-4 plans: unknown rule type refused LOUDLY; violating step
       flagged; parameter-FILE round trip = identical events;
       limit rules halt exactly; manual bypasses limits.
 B-6-5 describe: complete anatomy, JSON-safe, read-only;
       ADDRESSABILITY closure — a removal and an insertion
       formulated ONLY from report names execute.
 HOST-PARITY: assess + describe on an attention host.
"""
import hashlib
import json
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
from reference_net.method.gates import (                    # noqa: E402
    assess_growth, propose, run_plan, validate_plan,
    width_demand)
from reference_net.instrument import (                      # noqa: E402
    describe, monitor_configure, monitor_export)
from reference_net.growthpolicy import \
    DEFAULT_GROWTH_POLICY                                   # noqa: E402
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                              # noqa: E402


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _net(seed=7, steps=30, grown=True):
    net = Network(3, 8, lr=1e-2, seed=seed)
    X, y = _data()
    for _ in range(steps):
        net.train_step(X, y)
    if grown:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.grow(0, hidden=4)
        net.deepen()
    return net, X, y


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


# ---------------- B-3-4 assess ----------------

def test_b3_assess_fields_and_read_only():
    net, X, y = _net()
    pre = _hash(net)
    rep = assess_growth(net)
    assert set(rep) == {"width_demand", "seam_margins",
                        "event_gains", "instability", "census"}
    assert rep["census"]["blocks"] == 1
    assert rep["seam_margins"][0]["junction_width"] >= 1
    json.dumps(rep)
    assess_growth(net)
    assert _hash(net) == pre              # read-only


# ---------------- B-3-1 gate modes + override ----------------

def _forced_narrow(net):
    """Synthetic failing state: pretend the gradient span
    exceeds the width by inflating the EMA instrument."""
    rng = np.random.default_rng(0)
    net._ema_dw = net._bk.ingest(rng.normal(size=(8, 3)) * 0 +
                                 np.eye(8, 3))
    # full-rank 8x3 -> r_hat = 3 <= 8: NOT failing. Use a fat
    # matrix instead: (8, 20) rank 8 -> r_hat = 8, width 8 ->
    # met. To force failure we shrink the WIDTH view: cheat by
    # a subclass is overkill — instead assert modes fire on a
    # synthetic width_demand result via monkeypatching.
    return net


def test_b3_gate_modes_and_forced_override(monkeypatch):
    import reference_net.method.presets as presets
    failing = {"r_hat": 99, "width": 8, "met": False}
    monkeypatch.setattr("reference_net.method.gates."
                        "width_demand", lambda h: failing)
    net, X, y = _net(grown=False)
    gp_refuse = dict(DEFAULT_GROWTH_POLICY,
                     gate_deepen_mode="refuse")
    net._growth_policy = gp_refuse
    with pytest.raises(ValueError, match="G-DEEPEN"):
        net.deepen(position=0)
    # C-5 unconditional override: force executes past refuse
    k = net.deepen(position=0, force=True)
    assert k == 0 and len(net.blocks) == 1
    # warn mode proceeds with a warning
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              gate_deepen_mode="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.deepen(position=0)
        assert any("G-DEEPEN" in str(x.message) for x in w)
    # advise (default) records nothing and proceeds
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY)
    net.deepen(position=0)
    assert len(net.blocks) == 3


# ---------------- B-6-1 propose ----------------

def test_b3_propose_no_mutation_and_verdict_matches():
    net, X, y = _net()
    pre = _hash(net)
    rep = propose(net, "grow", DEFAULT_GROWTH_POLICY, j=0,
                  hidden=4)
    assert _hash(net) == pre
    assert rep["would_refuse"] is True     # j=0 already grown
    with pytest.raises(ValueError, match="already composite"):
        net.grow(0, hidden=4)              # matches reality
    rep2 = propose(net, "grow", DEFAULT_GROWTH_POLICY, j=1,
                   hidden=4)
    assert rep2["would_refuse"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(1, hidden=4)              # and this succeeds
    with pytest.raises(ValueError, match="unknown move"):
        propose(net, "teleport", DEFAULT_GROWTH_POLICY)


# ---------------- B-6-3 monitor ----------------

def test_b3_monitor_cadence_ring_export():
    net, X, y = _net(grown=False)
    assert getattr(net, "_monitor", None) is None  # off default
    monitor_configure(net, cadence=5, window=3)
    for _ in range(26):
        net.train_step(X, y)
    recs = net._monitor["records"]
    assert len(recs) == 3                  # ring bounded (5 of
    #                                        them happened)
    assert recs[-1]["step"] % 5 == 0
    json.loads(monitor_export(net))
    st = net.__getstate__()
    assert "_monitor" not in st            # never artifact


# ---------------- B-6-4 plans ----------------

def test_b3_plan_validation_and_limits(tmp_path):
    net, X, y = _net(grown=False)
    plan = {"name": "demo",
            "steps": [
                {"rule": "schedule", "move": "deepen",
                 "args": {"position": 0}},
                {"rule": "threshold", "move": "deepen",
                 "when": {"metric": "blocks", "op": "<",
                          "value": 5}, "args": {}},
                {"rule": "schedule", "move": "deepen",
                 "args": {}}],
            "limits": {"max_events": 2}}
    v = validate_plan(net, plan, DEFAULT_GROWTH_POLICY)
    assert v["cumulative_cost_params"] > 0
    with pytest.raises(ValueError, match="unknown rule type"):
        validate_plan(net, {"steps": [{"rule": "astral",
                                       "move": "deepen"}]},
                      DEFAULT_GROWTH_POLICY)
    # parameter-FILE round trip: identical event sequence
    f = tmp_path / "plan.json"
    f.write_text(json.dumps(plan))
    from reference_net.method.gates import load_plan
    plan2 = load_plan(f)
    assert plan2 == plan
    r = run_plan(net, plan2, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=2)
    assert r["halted"] == "limit:max_events"     # halts exactly
    assert len(r["events"]) == 2 and len(net.blocks) == 2
    # every plan GROWTH event carries policy provenance
    # (plan_adopted/plan_halted audit records also carry it
    # BY DESIGN, 58 D-4.2 — excluded from this count)
    pol = [e for e in net.gain_ledger
           if e.get("trigger") == "policy"
           and not e["event"].startswith("plan_")]
    assert len(pol) == 2
    # manual override bypasses the plan's limits (C-5)
    net.deepen()
    assert len(net.blocks) == 3
    assert net.gain_ledger[-1]["trigger"] == "caller"


# ---------------- B-3-5 mixed provenance ----------------

def test_b3_mixed_manual_auto_provenance():
    net, X, y = _net(grown=False)
    net.deepen()                                   # manual
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}]}
    run_plan(net, plan, DEFAULT_GROWTH_POLICY, X, y,
             steps_between=1)                      # auto
    net.deepen()                                   # manual
    trig = [e["trigger"] for e in net.gain_ledger
            if e["event"] == "deepen"]
    assert trig == ["caller", "policy", "caller"]


# ---------------- B-3-6 controllability ----------------

def test_b3_all_automation_off_full_manual_session():
    net, X, y = _net(grown=False)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              growth_auto_snapshot=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
    net.deepen()
    assess_growth(net); describe(net)
    for _ in range(5):
        net.train_step(X, y)
    assert getattr(net, "_snapshots", None) in (None, [])
    assert all(e.get("trigger") != "policy"
               for e in net.gain_ledger)


def test_b3_plan_pause_and_abort():
    net, X, y = _net(grown=False)
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}] * 3}
    ctl = {"pause": True, "abort": False}
    r = run_plan(net, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1, control=ctl)
    assert r["halted"] == "paused" and net.blocks == []
    ctl = {"pause": False, "abort": True}
    r = run_plan(net, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1, control=ctl)
    assert r["halted"] == "aborted" and net.blocks == []


# ---------------- B-6-5 describe + closure ----------------

def test_b3_describe_anatomy_and_addressability_closure():
    net, X, y = _net()
    pre = _hash(net)
    rep = describe(net)
    assert _hash(net) == pre               # read-only
    kinds = {c["kind"] for c in rep["components"]}
    assert {"trunk", "block"} <= kinds
    assert rep["couplings"] and \
        rep["couplings"][0]["A_shape"][1] == 8
    blk = next(c for c in rep["components"]
               if c["kind"] == "block")
    assert blk["id"].startswith("block#")  # permanent id
    # CLOSURE: formulate operations ONLY from report names
    pos = blk["position"]
    net.deepen(position=pos)               # insert at seen seam
    import ast
    key = ast.literal_eval(               # literal-safe parse
        rep["couplings"][0]["key"])       # of the report's key
    net.remove_grown(key)
    assert net.grown_body(key) is None


def test_b3_host_parity_assess_describe_on_attention_host():
    X, y = _data(0, 6, 3)
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])
    for _ in range(5):
        ga.train_step(X, y)
    ga.grow_site("layer0/ffn[1]", hidden=4)
    rep = describe(ga)
    assert sum(c["kind"] == "attn_layer"
               for c in rep["components"]) == 2
    assert rep["couplings"][0]["site"] == "0"
    a = assess_growth(ga)
    assert a["seam_margins"] and a["census"]["params"] > 0
    json.dumps(rep); json.dumps(a)


# ================= T-1 (58 v1.3 D-1): event cost columns ====
# FR-16 OPTION B: every growth/removal record carries
# step_at_event / steps_since_prev (hand-counted judges) and
# wall_ms; trial reports wall_ms; the chain survives
# serialization AND rollback (D-1.3b) with I-8 intact (all
# bookkeeping ledger-resident, nothing in the hashed tree).

def test_t1_cost_columns_hand_counted():
    """Scripted scenario, steps hand-counted: 30 (birth setup)
    -> deepen -> 7 -> grow -> 4 -> remove_block -> 3 ->
    remove_grown. steps_since_prev must EQUAL [30, 7, 4, 3, 0]
    exactly; step_at_event strictly increasing; wall_ms sane
    on every record."""
    net = Network(3, 8, lr=1e-2, seed=7)
    X, y = _data()
    for _ in range(30):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.deepen()
        for _ in range(7):
            net.train_step(X, y)
        net.grow(0, hidden=4)
        for _ in range(4):
            net.train_step(X, y)
        net.remove_block(0)
        for _ in range(3):
            net.train_step(X, y)
        net.remove_grown(0)
    recs = [r for r in net.gain_ledger
            if "step_at_event" in r]
    assert [r["steps_since_prev"] for r in recs] == \
        [30, 7, 4, 3]
    assert [r["step_at_event"] for r in recs] == \
        [30, 37, 41, 44]
    for r in recs:
        assert 0 < r["wall_ms"] < 60000
        assert isinstance(r["steps_since_prev"], int)
        assert isinstance(r["step_at_event"], int)


def test_t1_all_six_carriers_carry_fields():
    """All SIX Network carriers (grow/deepen/loop +
    remove_grown/remove_block/remove_loop) produce records
    with both cost columns."""
    net = Network(3, 8, lr=1e-2, seed=11)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    X, y = _data()
    for _ in range(120):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.deepen()
        net.grow(0, hidden=4)
        net.loop(4)
        net.remove_loop()
        net.remove_block(0)
        net.remove_grown(0)
    events = [r["event"] for r in net.gain_ledger
              if "step_at_event" in r]
    # grow's ledger event is historically named "refine"
    # (S1.2 evidence; growth-interface lineage)
    for kind in ("deepen", "remove_loop", "prune_block",
                 "remove_grown", "loop", "refine"):
        assert any(kind in e for e in events), kind
    for r in net.gain_ledger:
        if "step_at_event" in r:
            assert "steps_since_prev" in r
            assert "wall_ms" in r and r["wall_ms"] > 0


def test_t1_trial_reports_wall_ms():
    from reference_net.growth_store import trial
    net = Network(3, 8, lr=1e-2, seed=13)
    X, y = _data()
    for _ in range(30):
        net.train_step(X, y)
    rep = trial(net, lambda h: h.deepen(), X, y,
                budget_steps=3)
    assert rep["wall_ms"] > 0


def test_t1_serialization_continuity():
    """Save/load then train k more steps then one more event:
    steps_since_prev == k exactly (the anchor rides in the
    ledger, D-1.3)."""
    import pickle
    net = Network(3, 8, lr=1e-2, seed=17)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    net.deepen()                       # anchor at step 10
    twin = pickle.loads(pickle.dumps(net))
    for _ in range(5):
        twin.train_step(X, y)
    twin.deepen()
    last = [r for r in twin.gain_ledger
            if "step_at_event" in r][-1]
    assert last["steps_since_prev"] == 5
    assert last["step_at_event"] == 15


def test_t1_rollback_chain_consistency():
    """D-1.3b: event -> snapshot -> train -> event -> rollback
    -> train k -> event. The final record counts from the
    SURVIVING (pre-snapshot) event; never negative."""
    from reference_net.growth_store import rollback, snapshot
    net = Network(3, 8, lr=1e-2, seed=19)
    X, y = _data()
    for _ in range(10):
        net.train_step(X, y)
    net.deepen()                       # anchor: step 10
    rec = snapshot(net, tag="pre")
    for _ in range(6):
        net.train_step(X, y)
    net.deepen()                       # branch event (step 16)
    rollback(net, rec)                 # back to step 10
    for _ in range(4):
        net.train_step(X, y)
    net.deepen()                       # step 14
    last = [r for r in net.gain_ledger
            if "step_at_event" in r][-1]
    assert last["step_at_event"] == 14
    assert last["steps_since_prev"] == 4
    assert all(r["steps_since_prev"] >= 0
               for r in net.gain_ledger
               if "steps_since_prev" in r)


# ========== T-2/T-3/T-4/T-7 (58 v1.3 D-2/D-3/D-4/D-7) ======
# propose completion + plan provenance + plan-move vocabulary.

def test_t2_propose_cost_fields_hand_formulas():
    """D-2: cost_mem_bytes == cost_params*8*3 EXACTLY;
    cost_step_frac == cost_params/n_params to 1e-15; formulas
    self-documented; no mutation."""
    net, X, y = _net()
    pre = _hash(net)
    for move, kw in (("deepen", {}), ("grow", {"j": 1,
                                              "hidden": 4})):
        rep = propose(net, move, DEFAULT_GROWTH_POLICY, **kw)
        assert rep["cost_mem_bytes"] == \
            rep["cost_params"] * 8 * 3
        assert abs(rep["cost_step_frac"]
                   - rep["cost_params"] / net.n_params()) \
            <= 1e-15
        assert "cost_formulas" in rep
    assert _hash(net) == pre


def test_t3_propose_removal_moves_verdict_and_cost():
    """D-3.1-3.3: removal proposes — negated hand formulas;
    verdict matches the real call for success AND refusal;
    no mutation."""
    net, X, y = _net()          # grown=True: 1 body at j=0,
    #                             1 block
    pre = _hash(net)
    # remove_block valid: m=8 block -> -(8*8+8+8*8)=-136
    rep = propose(net, "remove_block", DEFAULT_GROWTH_POLICY,
                  k=0)
    assert rep["cost_params"] == -136
    assert rep["would_refuse"] is False
    # remove_block invalid index -> refuse, matches reality
    rep = propose(net, "remove_block", DEFAULT_GROWTH_POLICY,
                  k=5)
    assert rep["would_refuse"] is True
    with pytest.raises(Exception):
        net.remove_block(5)
    # remove_grown valid: body Network(3,4,out_w=1): 3*4+4+4+1
    # = 21; A = 1x8 = 8 -> -(29)
    rep = propose(net, "remove_grown", DEFAULT_GROWTH_POLICY,
                  key=0)
    assert rep["cost_params"] == -29
    assert rep["would_refuse"] is False
    # remove_grown unknown key -> refuse, matches reality
    rep = propose(net, "remove_grown", DEFAULT_GROWTH_POLICY,
                  key=99)
    assert rep["would_refuse"] is True
    with pytest.raises(Exception):
        net.remove_grown(99)
    # remove_loop with no loop -> refuse, matches reality
    rep = propose(net, "remove_loop", DEFAULT_GROWTH_POLICY)
    assert rep["would_refuse"] is True
    with pytest.raises(ValueError):
        net.remove_loop()
    assert _hash(net) == pre               # nothing mutated
    # with a loop present: m=4 -> -(2*4*8+4) = -68
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.loop(4)
    rep = propose(net, "remove_loop", net._growth_policy)
    assert rep["cost_params"] == -68
    assert rep["would_refuse"] is False
    # reality agrees: the real removal returns m=4
    assert net.remove_loop() == 4


def _ga(seed=99):
    X, y = _data(0, 6, 3)
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=seed,
        heads_spec=[[4, 4], [4, 4]])
    for _ in range(5):
        ga.train_step(X, y)
    return ga, X, y


def test_t3_propose_insert_and_grow_site_on_host():
    """D-3.4/3.5: host moves — cost equals the real call's
    n_params delta (judged on the same state right after the
    no-mutation check); grow_site ALSO equals the hand
    formula body(8x4+4+4+1=41) + A(1x8) = 49."""
    ga, X, y = _ga()
    p0 = ga.n_params()
    pre = _hash(ga)
    rep = propose(ga, "insert_layer", DEFAULT_GROWTH_POLICY,
                  position=1)
    assert _hash(ga) == pre                # dry
    ga.insert_layer(1)
    assert rep["cost_params"] == ga.n_params() - p0
    # bad position -> refuse matches reality
    rep = propose(ga, "insert_layer", DEFAULT_GROWTH_POLICY,
                  position=99)
    assert rep["would_refuse"] is True
    with pytest.raises(Exception):
        ga.insert_layer(99)
    # grow_site: hand formula AND reality; no mutation
    p1 = ga.n_params()
    pre2 = _hash(ga)
    rep = propose(ga, "grow_site", DEFAULT_GROWTH_POLICY,
                  site_path="layer0/ffn[2]", hidden=4)
    assert _hash(ga) == pre2               # dry (audit GAP-2)
    assert rep["cost_params"] == 49        # hand-counted
    assert rep["would_refuse"] is False
    # BAD site_path (out of range) -> refuse, matches reality
    rep_bad = propose(ga, "grow_site", DEFAULT_GROWTH_POLICY,
                      site_path="layer9/ffn[0]", hidden=4)
    assert rep_bad["would_refuse"] is True  # audit GAP-1
    with pytest.raises(Exception):
        ga.grow_site("layer9/ffn[0]", hidden=4)
    ga.grow_site("layer0/ffn[2]", hidden=4)
    assert ga.n_params() - p1 == 49        # reality agrees
    # already-composite -> refuse matches reality
    rep = propose(ga, "grow_site", DEFAULT_GROWTH_POLICY,
                  site_path="layer0/ffn[2]", hidden=4)
    assert rep["would_refuse"] is True
    with pytest.raises(Exception):
        ga.grow_site("layer0/ffn[2]", hidden=4)
    # capability law: removals refuse LOUDLY on hosts
    with pytest.raises(ValueError, match="rollback"):
        propose(ga, "remove_block", DEFAULT_GROWTH_POLICY,
                k=0)
    # and host moves refuse LOUDLY on the reference Network
    net, _, _ = _net(grown=False)
    with pytest.raises(ValueError, match="insert_layer"):
        propose(net, "insert_layer", DEFAULT_GROWTH_POLICY,
                position=0)


def test_t4_plan_provenance_records(tmp_path):
    """D-4: plan_adopted precedes the first event; plan_halted
    carries exact reason + events_run; sha == GrowthStore.
    save_plan's sha (same formula); audit-only (pending-gain
    untouched)."""
    from reference_net.method import gates as gates_mod
    from reference_net.growth_store import GrowthStore
    net, X, y = _net(grown=False)
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}] * 3,
            "limits": {"max_events": 2}}
    n_pending = len(net._pending_gain)
    r = run_plan(net, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1)
    led = net.gain_ledger
    i_adopt = next(i for i, x in enumerate(led)
                   if x["event"] == "plan_adopted")
    i_first = next(i for i, x in enumerate(led)
                   if x["event"] == "deepen")
    i_halt = next(i for i, x in enumerate(led)
                  if x["event"] == "plan_halted")
    assert i_adopt < i_first < i_halt
    assert led[i_halt]["halted"] == "limit:max_events"
    assert led[i_halt]["events_run"] == 2 == len(r["events"])
    sha = gates_mod.plan_sha(plan)
    assert led[i_adopt]["plan_sha"] == sha
    assert led[i_halt]["plan_sha"] == sha
    store = GrowthStore(tmp_path / "gs")
    assert store.save_plan(plan) == sha    # SAME formula
    # audit-only: the two records never enter the gain queue
    assert len(net._pending_gain) - n_pending == 2  # 2 deepens
    # completed plans stamp "completed"
    net2, _, _ = _net(grown=False)
    run_plan(net2, {"steps": [{"rule": "schedule",
                               "move": "deepen",
                               "args": {}}]},
             DEFAULT_GROWTH_POLICY, X, y, steps_between=1)
    halt = [x for x in net2.gain_ledger
            if x["event"] == "plan_halted"][-1]
    assert halt["halted"] == "completed"
    assert halt["events_run"] == 1


def test_t7_plan_mixed_moves_signed_limits_and_triggers():
    """D-7: mixed deepen+remove_block plan; live-n_params
    limit (hand math, per-block 136); removal records carry
    trigger='policy'."""
    net, X, y = _net(grown=False)
    p0 = net.n_params()
    plan = {"steps": [
        {"rule": "schedule", "move": "deepen", "args": {}},
        {"rule": "schedule", "move": "deepen", "args": {}},
        {"rule": "schedule", "move": "remove_block",
         "args": {"k": 0}},
        {"rule": "schedule", "move": "deepen", "args": {}},
        {"rule": "schedule", "move": "deepen", "args": {}},
        {"rule": "schedule", "move": "deepen", "args": {}}],
        "limits": {"max_params": p0 + 300}}
    r = run_plan(net, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1)
    # hand walk: +136,+272,-136->136,+272,+408 >= p0+300 halt
    assert r["halted"] == "limit:max_params"
    assert len(r["events"]) == 5
    assert len(net.blocks) == 3
    assert net.n_params() == p0 + 3 * 136
    prune = [x for x in net.gain_ledger
             if x["event"] == "prune_block"][-1]
    assert prune["trigger"] == "policy"


def test_t7_host_plan_insert_layer_policy_trigger():
    """D-7.2: insert_layer plan step on a GA host executes
    with trigger='policy' stamped into growth_events (the
    D-4.4 dispatch reaches the hosts' record home)."""
    ga, X, y = _ga(seed=101)
    plan = {"steps": [{"rule": "schedule",
                       "move": "insert_layer",
                       "args": {"position": 1}}]}
    r = run_plan(ga, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1)
    assert len(r["events"]) == 1
    # host insertion's real event name (GA source, :859):
    rec = [e for e in ga.growth_events
           if e["event"] == "deepen_layer[attn]"][-1]
    assert rec["trigger"] == "policy"
    # provenance lands in growth_events too (no gain ledger)
    kinds = [e["event"] for e in ga.growth_events]
    assert "plan_adopted" in kinds and "plan_halted" in kinds


def test_t7_host_plan_grow_site_policy_trigger():
    """grow_site plan step executes; its RECORD (and stamp)
    arms at W5 with D-6.1 — until then assert execution +
    n_params delta only."""
    ga, X, y = _ga(seed=103)
    p0 = ga.n_params()
    plan = {"steps": [{"rule": "schedule", "move": "grow_site",
                       "args": {"site_path": "layer1/ffn[3]",
                                "hidden": 4}}]}
    r = run_plan(ga, plan, DEFAULT_GROWTH_POLICY, X, y,
                 steps_between=1)
    assert len(r["events"]) == 1
    assert ga.n_params() - p0 == 49
    rec = [e for e in ga.growth_events
           if e.get("event") == "grow_site"]
    if not rec:
        pytest.skip("grow_site record lands at W5 with D-6.1")
    assert rec[-1]["trigger"] == "policy"


# ============ T-8 (58 v1.3 D-8): the force catalog ==========
# C-5: force accepted on EVERY entry point, recorded as
# forced=True WHEN AND ONLY WHEN set.

def test_t8_force_catalog_every_entry_point():
    net, X, y = _net(grown=False)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4, force=True)
        net.deepen(force=True)
        net.loop(4, force=True)
        net.remove_loop(force=True)
        net.remove_block(0, force=True)
        net.remove_grown(0, force=True)
    forced = [r for r in net.gain_ledger
              if r.get("forced") is True]
    assert len(forced) == 6                # all six carriers
    # unforced calls carry NO forced key (absence checked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(1, hidden=4)
        net.deepen()
    assert all("forced" not in r
               for r in net.gain_ledger[-2:])


def test_t8_force_on_host_insert_layer_and_grow_site():
    ga, X, y = _ga(seed=107)
    ga.insert_layer(1, force=True)
    rec = [e for e in ga.growth_events
           if e["event"] == "deepen_layer[attn]"][-1]
    assert rec.get("forced") is True
    ga.insert_layer(1)
    rec2 = [e for e in ga.growth_events
            if e["event"] == "deepen_layer[attn]"][-1]
    assert "forced" not in rec2
    # grow_site accepts the flag NOW; its event RECORD (and
    # the marker on it) lands at W5 with D-6.1
    ga.grow_site("layer0/ffn[5]", hidden=4, force=True)
    rec3 = [e for e in ga.growth_events
            if e.get("event") == "grow_site"]
    if not rec3:
        pytest.skip("grow_site record lands at W5 (D-6.1)")
    assert rec3[-1].get("forced") is True


# ======= T-5 (58 v1.3 D-5): the gate family (FR-12/OD-3) ====
# Hand-set widths vs hand-set thresholds; all three modes per
# gate; force past refuse; nesting witness pair; permissive
# default = zero behavior change; the E6-1 tier law.

def _gp5(**kw):
    return dict(DEFAULT_GROWTH_POLICY, **kw)


def test_t5_gseam_three_modes_and_force():
    """G-SEAM at grow: junction width = grow_body_out_width
    (hand 1) vs hand threshold 2."""
    for mode, expect in (("refuse", "raises"),
                         ("warn", "warns"),
                         ("advise", "silent")):
        net, X, y = _net(grown=False)
        net._growth_policy = _gp5(gate_seam_min_width=2,
                                  gate_seam_mode=mode)
        if expect == "raises":
            with pytest.raises(ValueError, match=
                               r"G-SEAM.*1 < threshold 2.*"
                               r"gate_seam_min_width"):
                net.grow(0, hidden=4)
            assert net.grown_body(0) is None   # nothing grew
            # C-5: force executes past refuse, recorded
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.grow(0, hidden=4, force=True)
            assert net.gain_ledger[-1]["forced"] is True
            assert net.grown_body(0) is not None
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                net.grow(0, hidden=4)
            seam = [x for x in w
                    if "G-SEAM" in str(x.message)]
            assert (len(seam) == 1) == (expect == "warns")
            assert net.grown_body(0) is not None
        # threshold met (width 1 >= 1 permissive): admitted
    net2, _, _ = _net(grown=False)
    net2._growth_policy = _gp5(gate_seam_min_width=1,
                               gate_seam_mode="refuse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net2.grow(0, hidden=4)                 # 1 >= 1 admits
    assert net2.grown_body(0) is not None


def test_t5_nesting_witness_pair():
    """OD-3 nesting gate: growing INSIDE a grown body checks
    min(new junction, mouth). Witness pair: mouth-1 body under
    threshold 2 -> inner grow refused; wide-mouth body (4) ->
    admitted (min(4,4)=4 >= 2)."""
    net, X, y = _net(grown=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)                  # mouth = 1
    body = net.grown_body(0)
    assert body.gain_ledger[0]["event"] == "nested_birth"
    body._growth_policy = _gp5(gate_seam_min_width=2,
                               gate_nest_mode="refuse")
    with pytest.raises(ValueError,
                       match=r"G-NEST.*path bottleneck.*"
                             r"1 < threshold 2"):
        body.grow(0, hidden=4)
    # force overrides even the nesting refusal (C-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        body.grow(0, hidden=4, force=True)
    assert body.gain_ledger[-1]["forced"] is True
    # warn mode warns and proceeds (audit GAP-3: third mode)
    net3, _, _ = _net(grown=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net3.grow(0, hidden=4)
    b3 = net3.grown_body(0)
    b3._growth_policy = _gp5(gate_seam_min_width=2,
                             gate_nest_mode="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b3.grow(0, hidden=4)               # proceeds
    assert any("G-NEST" in str(x.message) for x in w)
    assert b3.grown_body(0) is not None
    # wide mouth: grown with out_width 4 -> min(4,4) admitted
    net2, _, _ = _net(grown=False)
    net2._growth_policy = _gp5(grow_body_out_width=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net2.grow(0, hidden=4)                 # mouth = 4
    b2 = net2.grown_body(0)
    assert b2.gain_ledger[0]["mouth"] == 4
    b2._growth_policy = _gp5(grow_body_out_width=4,
                             gate_seam_min_width=2,
                             gate_nest_mode="refuse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b2.grow(0, hidden=4)                   # ADMITTED
    assert b2.grown_body(0) is not None


def test_t5_scope_gate_three_modes():
    """G-SCOPE at scoped deepen: |scope| (hand 1) vs hand
    threshold 2; global deepen (scope=None) is NEVER gated."""
    net, X, y = _net(grown=False)
    net._growth_policy = _gp5(gate_scope_min_width=2,
                              gate_scope_mode="refuse")
    with pytest.raises(ValueError,
                       match=r"G-SCOPE.*scope width 1 < "
                             r"threshold 2"):
        net.deepen(scope=[3])
    assert len(net.blocks) == 0
    net.deepen(scope=[3], force=True)          # C-5
    assert net.gain_ledger[-1]["forced"] is True
    net.deepen()                               # global ungated
    assert len(net.blocks) == 2
    net._growth_policy = _gp5(gate_scope_min_width=2,
                              gate_scope_mode="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.deepen(scope=[0, 5])               # width 2 >= 2:
        #                                        admitted quiet
        net._growth_policy = _gp5(gate_scope_min_width=3,
                                  gate_scope_mode="warn")
        net.deepen(scope=[0, 5])               # 2 < 3: warns
    assert sum("G-SCOPE" in str(x.message) for x in w) == 1
    assert len(net.blocks) == 4                # all proceeded


def test_t5_gwiden_advisory(monkeypatch):
    """G-WIDEN: when width_demand.met is True (no demand) the
    advisory fires per gate_widen_mode; force overrides."""
    import reference_net.method.gates as gates_mod
    met = {"r_hat": 2, "width": 8, "met": True}
    monkeypatch.setattr(gates_mod, "width_demand",
                        lambda h: met)
    net, X, y = _net(grown=False)
    net._growth_policy = _gp5(gate_widen_mode="refuse")
    with pytest.raises(ValueError, match="G-WIDEN"):
        net.grow(0, hidden=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4, force=True)      # C-5
    assert net.gain_ledger[-1]["forced"] is True
    net2, _, _ = _net(grown=False)
    net2._growth_policy = _gp5(gate_widen_mode="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net2.grow(0, hidden=4)
    assert any("G-WIDEN" in str(x.message) for x in w)
    assert net2.grown_body(0) is not None      # proceeded


def test_t5_host_seam_gate_via_grow_site():
    """D-5.1 consult point on the host preset (preset_site)."""
    ga, X, y = _ga(seed=109)
    ga._growth_policy = _gp5(gate_seam_min_width=2,
                             gate_seam_mode="refuse")
    with pytest.raises(ValueError, match="G-SEAM"):
        ga.grow_site("layer0/ffn[6]", hidden=4)
    ga.grow_site("layer0/ffn[6]", hidden=4, force=True)
    assert ga.grown_body(0, 6) is not None     # C-5 through
    #                                            the chain


def test_t5_permissive_default_no_behavior_change():
    """Factory policy: an entire grow/deepen/scoped/nested
    session proceeds with ZERO gate messages (the six keys
    default permissive/advise)."""
    net, X, y = _net(grown=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.grow(0, hidden=4)
        net.deepen()
        net.deepen(scope=[0, 2])
        net.grown_body(0).grow(0, hidden=4)    # nested
    gate_msgs = [x for x in w
                 if any(g in str(x.message) for g in
                        ("G-SEAM", "G-NEST", "G-SCOPE",
                         "G-WIDEN"))]
    assert gate_msgs == []                     # NOTHING fired
    assert len(net.blocks) == 2


def test_t5_tier_law_direct_compose_ungated():
    """E6-1: a direct compose.grow coupling-only event under
    refuse-mode policy still executes — L1 is the ungated
    expert manual surface BY DESIGN."""
    net, X, y = _net(grown=False)
    net._growth_policy = _gp5(gate_seam_min_width=99,
                              gate_seam_mode="refuse")
    h = _skip_event(net, "tier-law")
    assert h is not None                       # executed
    assert net.predict(X) is not None


def _skip_event(net, key):
    from reference_net.foundation import compose as _c
    from reference_net.foundation.specs import (
        ALL, NONE, BirthSpec, PlacementSpec, StructureSpec,
        Tap, WiringSpec)
    import reference_net.growth_port  # noqa: F401 (builders)
    return _c.grow(
        net,
        StructureSpec(kind="none", params={"width": 3}),
        WiringSpec(reads=[Tap("scope_input", role="skip")],
                   write={"target": "stream", "span": ALL}),
        PlacementSpec(chain=NONE), BirthSpec(), key=key)


# ====== T-6 (58 v1.3 D-6): host-side governance (FR-7/13/16) =

def test_t6_grow_site_four_spec_record():
    """D-6.1: grow_site appends the FR-7 birth certificate to
    growth_events — four verbatim specs, hand-counted
    n_params (body 41 + A 8 = 49), wall_ms, caller trigger;
    forced marker when forced."""
    ga, X, y = _ga(seed=113)
    ga.grow_site("layer0/ffn[0]", hidden=4)
    rec = [e for e in ga.growth_events
           if e["event"] == "grow_site"][-1]
    assert rec["site"] == (0, 0)
    assert rec["n_params"] == 49               # hand count
    assert rec["wall_ms"] > 0
    assert rec["trigger"] == "caller"
    sp = rec["specs"]
    assert set(sp) == {"structure", "wiring", "placement",
                       "birth"}
    assert sp["structure"]["kind"] == "reference"
    assert sp["structure"]["params"]["hidden"] == 4
    assert sp["structure"]["params"]["out_width"] == 1
    assert sp["birth"]["zero_side"] == "coupling"
    assert "forced" not in rec
    ga.grow_site("layer1/ffn[0]", hidden=4, force=True)
    rec2 = [e for e in ga.growth_events
            if e["event"] == "grow_site"][-1]
    assert rec2.get("forced") is True
    # insert_layer records carry wall_ms too (D-1.5)
    ga.insert_layer(1)
    rec3 = [e for e in ga.growth_events
            if e["event"] == "deepen_layer[attn]"][-1]
    assert rec3["wall_ms"] > 0


def test_t6_host_auto_snapshot_policy_gated():
    """D-6.2: pre-event capture on the host carriers, policy
    ON by default; OFF when the user switches it off."""
    ga, X, y = _ga(seed=127)
    assert getattr(ga, "_snapshots", None) is None
    ga.insert_layer(1)
    assert len(ga._snapshots) == 1
    assert ga._snapshots[-1]["tag"] == "auto:pre-growth"
    ga.grow_site("layer0/ffn[1]", hidden=4)
    assert len(ga._snapshots) == 2
    ga2, _, _ = _ga(seed=127)
    ga2._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              growth_auto_snapshot=False)
    ga2.insert_layer(1)
    ga2.grow_site("layer0/ffn[1]", hidden=4)
    assert getattr(ga2, "_snapshots", None) is None


def test_t6_host_rollback_bitwise():
    """FR-13 on a host: rollback restores predict + n_params
    bitwise (numpy compare on served outputs)."""
    from reference_net.growth_store import rollback, snapshot
    ga, X, y = _ga(seed=131)
    out0 = np.asarray(ga.predict(X), dtype=np.float64).copy()
    p0 = ga.n_params()
    rec = snapshot(ga, tag="pre")
    ga.insert_layer(1)
    ga.grow_site("layer0/ffn[2]", hidden=4)
    rollback(ga, rec)
    assert ga.n_params() == p0
    assert np.array_equal(
        np.asarray(ga.predict(X), dtype=np.float64), out0)


def test_t6_host_monitor_tick_and_serialization_guard():
    """D-6.3: monitor armed on a host ticks at the cadence
    (ring hand-computed: 26 steps, cadence 5 -> 5 fires, ring
    3); off-default stays off; D-6.4: observer state never
    serialized — pickled bytes identical with and without a
    populated ring."""
    import pickle
    from reference_net.instrument import monitor_configure
    ga, X, y = _ga(seed=137)
    assert getattr(ga, "_monitor", None) is None   # off
    base_bytes = pickle.dumps(ga)
    monitor_configure(ga, cadence=5, window=3)
    for _ in range(26):
        ga.train_step(X, y)
    recs = ga._monitor["records"]
    assert len(recs) == 3                          # ring
    assert recs[-1]["step"] % 5 == 0               # cadence
    # serialization guard: monitor+snapshot state excluded ->
    # BUT training moved parameters; so compare a fresh pair:
    ga2, _, _ = _ga(seed=139)
    b1 = pickle.dumps(ga2)
    from reference_net.growth_store import snapshot
    snapshot(ga2, tag="x")
    monitor_configure(ga2, cadence=5, window=2)
    b2 = pickle.dumps(ga2)
    assert b1 == b2                # observer state invisible
    twin = pickle.loads(b2)
    assert getattr(twin, "_snapshots", None) is None
    assert getattr(twin, "_monitor", None) is None


def test_t7_plan_step_force_passthrough():
    """T-7 sub-box (58 R4-2, written late — caught by the W7
    coverage-index audit): a plan step carrying "force": true
    reaches the entry point verbatim and the event record
    carries BOTH forced=True (C-5) and trigger='policy'
    (automatic mode) — the override channel works inside
    rule-driven execution."""
    net, X, y = _net(grown=False)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              gate_scope_min_width=3,
                              gate_scope_mode="refuse")
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {"scope": [0, 1],
                                "force": True}}]}
    r = run_plan(net, plan, net._growth_policy, X, y,
                 steps_between=1)
    assert len(r["events"]) == 1           # executed past the
    #                                        refusing gate
    rec = [x for x in net.gain_ledger
           if x["event"] == "deepen"][-1]
    assert rec["forced"] is True           # C-5 recorded
    assert rec["trigger"] == "policy"      # automatic mode
    # control: the same plan WITHOUT force refuses loudly
    net2, _, _ = _net(grown=False)
    net2._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                               gate_scope_min_width=3,
                               gate_scope_mode="refuse")
    plan2 = {"steps": [{"rule": "schedule", "move": "deepen",
                        "args": {"scope": [0, 1]}}]}
    with pytest.raises(ValueError, match="G-SCOPE"):
        run_plan(net2, plan2, net2._growth_policy, X, y,
                 steps_between=1)


# ========= T-A (59 v1.3 R-1): scoped-deepen advisory ========

def test_ta1_scoped_advisory_fires_and_executes():
    """R-1.1/R-1.2: ONE reminder on scoped deepen, theory
    terms present; the deepen EXECUTES (not a gate)."""
    net, X, y = _net(grown=False)
    base = np.asarray(net._bk.to_numpy(net.predict(X)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.deepen(scope=[0, 2])
    adv = [x for x in w
           if "functionally CLOSED" in str(x.message)]
    assert len(adv) == 1                       # exactly one
    msg = str(adv[0].message)
    assert "interface bandwidth" in msg
    assert "lifetime-plastic" in msg
    assert len(net.blocks) == 1                # executed
    assert np.array_equal(np.asarray(
        net._bk.to_numpy(net.predict(X))), base)  # zero-birth


def test_ta2_global_deepen_silent():
    """R-1.1: global deepening (default AND positioned)
    produces ZERO advisory warnings."""
    net, X, y = _net(grown=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.deepen()
        net.deepen(position=0)
    assert not any("functionally CLOSED" in str(x.message)
                   for x in w)
    assert len(net.blocks) == 2


def test_ta3_propose_advisory_field():
    """R-1.3: propose(deepen, scope=...) carries the SAME
    single-source text; global propose has no field; dry."""
    from reference_net.method import presets as presets_mod
    net, X, y = _net(grown=False)
    pre = _hash(net)
    rep = propose(net, "deepen", DEFAULT_GROWTH_POLICY,
                  scope=[0, 1])
    assert rep["advisory"] == presets_mod._SCOPE_ADVISORY
    rep2 = propose(net, "deepen", DEFAULT_GROWTH_POLICY)
    assert "advisory" not in rep2
    assert _hash(net) == pre                   # no mutation


# ====== T-B (59 v1.3 R-2): per-region learning-rate scales ==
# Value-verification: exact identities by construction; the
# default is guarded (key absent -> branch skipped -> bitwise
# current behavior).

import copy as _copy


def _np_all(net, a):
    return np.asarray(net._bk.to_numpy(a), dtype=np.float64)


def test_tb1_network_half_speed_hand_judge():
    """R-2.4: with s=0.5 on encoder/block:0/loop, each scaled
    tensor lands EXACTLY at old + 0.5*(full - old); unscaled
    readout lands BITWISE on the full-speed twin's value."""
    net, X, y = _net(grown=False)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    net.deepen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.loop(4)
    for _ in range(5):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)
    old = {"W1": _np_all(net, net.W1).copy(),
           "b1": _np_all(net, net.b1).copy(),
           "Bin": _np_all(net, net.blocks[0]["Bin"]).copy(),
           "bb": _np_all(net, net.blocks[0]["bb"]).copy(),
           "Bout": _np_all(net, net.blocks[0]["Bout"]).copy(),
           "L_in": _np_all(net, net.loop_block["L_in"]).copy(),
           "b_l": _np_all(net, net.loop_block["b_l"]).copy()}
    twin.train_step(X, y)                     # full speed
    net._growth_policy = dict(
        net._growth_policy,
        train_lr_scales={"encoder": 0.5, "block:0": 0.5,
                         "loop": 0.5})
    net.train_step(X, y)                      # scaled
    for nm, get_s, get_f in (
            ("W1", lambda: _np_all(net, net.W1),
             lambda: _np_all(twin, twin.W1)),
            ("b1", lambda: _np_all(net, net.b1),
             lambda: _np_all(twin, twin.b1)),
            ("Bin", lambda: _np_all(net, net.blocks[0]["Bin"]),
             lambda: _np_all(twin, twin.blocks[0]["Bin"])),
            ("bb", lambda: _np_all(net, net.blocks[0]["bb"]),
             lambda: _np_all(twin, twin.blocks[0]["bb"])),
            ("Bout", lambda: _np_all(net,
                                     net.blocks[0]["Bout"]),
             lambda: _np_all(twin, twin.blocks[0]["Bout"])),
            ("L_in", lambda: _np_all(net,
                                     net.loop_block["L_in"]),
             lambda: _np_all(twin,
                             twin.loop_block["L_in"])),
            ("b_l", lambda: _np_all(net,
                                    net.loop_block["b_l"]),
             lambda: _np_all(twin,
                             twin.loop_block["b_l"]))):
        expect = old[nm] + 0.5 * (get_f() - old[nm])
        assert np.array_equal(get_s(), expect), nm  # exact
    # unscaled readout == twin bitwise (same kernel output)
    assert np.array_equal(_np_all(net, net.W2),
                          _np_all(twin, twin.W2))
    assert np.array_equal(_np_all(net, net.c),
                          _np_all(twin, twin.c))


def test_tb2_zero_value_bitwise_still_mv_evolve():
    """R-2.6 + P3-F2: s=0 -> params bitwise still (same-object
    short-circuit); Adam m/v keep evolving (release-kick
    premise)."""
    net, X, y = _net(grown=False)
    for _ in range(3):
        net.train_step(X, y)
    W1_0 = _np_all(net, net.W1).copy()
    b1_0 = _np_all(net, net.b1).copy()
    m_0 = _np_all(net, net.opt.m[0]).copy()
    W2_0 = _np_all(net, net.W2).copy()
    c_0 = _np_all(net, net.c).copy()
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"encoder": 0.0})
    for _ in range(5):
        net.train_step(X, y)
    assert np.array_equal(_np_all(net, net.W1), W1_0)  # still
    assert np.array_equal(_np_all(net, net.b1), b1_0)
    assert not np.array_equal(_np_all(net, net.opt.m[0]), m_0)
    # readout MOVED (real judge: differs from its pre value —
    # the earlier |W2|>0 form was vacuously true, audit gap 2)
    assert not np.array_equal(_np_all(net, net.W2), W2_0)
    assert not np.array_equal(_np_all(net, net.c), c_0)


def test_tb3_new_layer_full_speed_in_window():
    """R-2.1 core use: slow encoder+block:0; the NEW block:1
    trains BITWISE identically to the unscaled twin (one
    step)."""
    net, X, y = _net(grown=False)
    net.deepen()
    for _ in range(5):
        net.train_step(X, y)
    net.deepen()                               # the NEW layer
    twin = _copy.deepcopy(net)
    twin.train_step(X, y)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"encoder": 0.1,
                                               "block:0": 0.1})
    net.train_step(X, y)
    for k in ("Bin", "bb", "Bout"):
        assert np.array_equal(
            _np_all(net, net.blocks[1][k]),
            _np_all(twin, twin.blocks[1][k])), k


def test_tb4_default_guard_branch_never_runs(monkeypatch):
    """I-3: key absent -> the scales helper is NEVER invoked
    (guard by absence, not by arithmetic)."""
    calls = {"n": 0}
    from reference_net.net import Network as _N
    real = _N._apply_lr_scales

    def spy(self, params, new):
        calls["n"] += 1
        return real(self, params, new)
    monkeypatch.setattr(_N, "_apply_lr_scales", spy)
    net, X, y = _net(grown=False)
    for _ in range(5):
        net.train_step(X, y)
    assert calls["n"] == 0                     # never entered
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"encoder": 0.5})
    net.train_step(X, y)
    assert calls["n"] == 1                     # and now it is


def test_tb5_grammar_refusal_and_inert_dead_name():
    """R-2.3b: union-grammar typo refused loudly, state
    untouched; grammar-valid dead name is INERT (bitwise
    equal to the no-table twin)."""
    net, X, y = _net(grown=False)
    for _ in range(3):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"blck:0": 0.5})
    pre = _hash(net)
    with pytest.raises(ValueError, match="blck:0"):
        net.train_step(X, y)
    assert _hash(net) == pre                   # pre-write
    # dead-but-grammatical name: inert
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"block:99": 0.5,
                                               "layer:0": 0.5})
    net.train_step(X, y)
    twin.train_step(X, y)
    for a, b in zip([net.W1, net.b1, net.W2, net.c],
                    [twin.W1, twin.b1, twin.W2, twin.c]):
        assert np.array_equal(_np_all(net, a),
                              _np_all(twin, b))


def test_tb6_sgd_branch_covered():
    """R-2.5: the sgd write-back honors the table too (half-
    speed identity on W1 vs an sgd twin)."""
    net, X, y = _net(grown=False)
    for _ in range(3):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)
    old = _np_all(net, net.W1).copy()
    twin.train_step(X, y, sgd_lr=1e-2)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"encoder": 0.5})
    net.train_step(X, y, sgd_lr=1e-2)
    expect = old + 0.5 * (_np_all(twin, twin.W1) - old)
    assert np.array_equal(_np_all(net, net.W1), expect)


def test_tb9_parameter_file_round_trip(tmp_path):
    """R-2.2: table loaded from a JSON parameter file behaves
    identically to the programmatic table."""
    f = tmp_path / "policy.json"
    f.write_text(json.dumps({"train_lr_scales":
                             {"encoder": 0.5}}))
    loaded = json.loads(f.read_text())
    net, X, y = _net(grown=False)
    for _ in range(3):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY, **loaded)
    twin._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                               train_lr_scales={"encoder": 0.5})
    net.train_step(X, y)
    twin.train_step(X, y)
    for a, b in zip([net.W1, net.b1, net.W2, net.c],
                    [twin.W1, twin.b1, twin.W2, twin.c]):
        assert np.array_equal(_np_all(net, a),
                              _np_all(twin, b))


def test_tb12_loop_lout_certificate_exception():
    """Probe-established fact (59 close-out addendum): the
    half-speed identity holds for L_out MODULO the loop's
    pre-existing rho-certificate projection (net.py DESIGN
    s4) — the projection correctly applies AFTER the blend
    so the certificate survives scaling.
    Arm 1 (mild, projection silent): identity bitwise.
    Arm 2 (binding): projection fires and rho_hat lands on
    the cap EXACTLY — governance verified on the blended
    value."""
    from engine.loop_ops import loop_rho_hat
    # arm 1: small lr -> certificate never binds
    net = Network(3, 8, lr=1e-3, seed=31)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.loop(4)
    for _ in range(3):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)
    old = _np_all(net, net.loop_block["L_out"]).copy()
    p0 = net._loop_projections
    twin.train_step(X, y)
    net._growth_policy = dict(net._growth_policy,
                              train_lr_scales={"loop": 0.5})
    net.train_step(X, y)
    assert net._loop_projections == p0          # silent
    assert twin._loop_projections == p0
    exp = old + 0.5 * (_np_all(twin,
                               twin.loop_block["L_out"]) - old)
    assert np.array_equal(
        _np_all(net, net.loop_block["L_out"]), exp)
    # arm 2: normal lr -> certificate binds; verify it holds
    # ON THE BLENDED VALUE (rho == cap exactly)
    net2 = Network(3, 8, lr=1e-2, seed=7)
    net2._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                               loop_enabled=True)
    net2.deepen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net2.loop(4)
    for _ in range(5):
        net2.train_step(X, y)
    q0 = net2._loop_projections
    net2._growth_policy = dict(net2._growth_policy,
                               train_lr_scales={"loop": 0.5,
                                                "block:0": 0.5})
    net2.train_step(X, y)
    assert net2._loop_projections == q0 + 1     # fired
    rho = loop_rho_hat(
        _np_all(net2, net2.loop_block["L_in"]),
        _np_all(net2, net2.loop_block["L_out"]))
    cap = float(net2._growth_policy.get("loop_rho_max", 0.6))
    assert abs(rho - cap) <= 1e-12              # ON the cap


def test_tb11_propagation_semantics():
    """R-2.3c (documented S2 behavior, three arms): shared
    dict reaches the body; host REBIND does not; body's own
    policy does."""
    net, X, y = _net(grown=False)
    for _ in range(120):
        net.train_step(X, y)
    # policy set BEFORE growth: body shares the dict object
    shared = dict(DEFAULT_GROWTH_POLICY)
    net._growth_policy = shared
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
    body = net.grown_body(0)
    assert body._growth_policy is shared       # S2 fact
    # arm 1: MUTATE the shared dict -> body's encoder stills
    shared["train_lr_scales"] = {"encoder": 0.0}
    bW1_0 = _np_all(body, body.W1).copy()
    for _ in range(3):
        net.train_step(X, y)                   # trains body via
        #                                        the port handoff
    assert np.array_equal(_np_all(body, body.W1), bW1_0)
    # arm 2: REBIND host policy -> body unaffected by the new
    # dict (still holds the old shared one)
    del shared["train_lr_scales"]
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              train_lr_scales={"encoder": 0.0})
    bW1_1 = _np_all(body, body.W1).copy()
    net.train_step(X, y)
    assert not np.array_equal(_np_all(body, body.W1), bW1_1)
    # arm 3: body's own policy works directly
    body._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                               train_lr_scales={"encoder": 0.0})
    bW1_2 = _np_all(body, body.W1).copy()
    net.train_step(X, y)
    assert np.array_equal(_np_all(body, body.W1), bW1_2)


# ============ G1 (doc 61 I-B): plan removal-move dispatch ====

def test_g1_plan_removal_moves_dispatch():
    """A plan whose steps include remove_grown and remove_loop
    EXECUTES on a live model: removal really happens
    (hand-verified params delta: -29 body / -68 loop per the
    established formulas), records carry trigger='policy',
    events count toward max_events."""
    import warnings as _w
    X, y = _data()
    net = Network(3, 8, lr=1e-2, seed=7)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    for _ in range(120):
        net.train_step(X, y)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        net.grow(1, hidden=4)          # +29 = 3*4+4+4*1+1+1*8
        net.loop(m=4)                  # +68 = 2*4*8+4
    p0 = net.n_params()
    plan = {"steps": [
        {"move": "remove_grown", "args": {"key": 1}},
        {"move": "remove_loop"}],
        "limits": {"max_events": 2}}
    rep = run_plan(net, plan, net._growth_policy, X, y,
                   steps_between=0)
    assert len(rep["events"]) == 2         # counted events
    assert net.n_params() == p0 - 29 - 68  # hand deltas
    removals = [r for r in net.gain_ledger
                if r["event"] in ("remove_grown",
                                  "remove_loop")]
    assert len(removals) == 2
    for r in removals:
        assert r["trigger"] == "policy"
    assert removals[0]["params_added"] == -29
    assert removals[1]["params_added"] == -68


# ============ G5 (doc 61 I-B): second training entry ========

def test_tb14_train_from_grad_refusal():
    """A port-owned body with {"blck:0": 0.5}:
    train_from_grad raises ValueError naming the key; body
    params bitwise unchanged (pre-mutation refusal on THIS
    entry too)."""
    import warnings as _w
    X, y = _data()
    net = Network(3, 8, lr=1e-2, seed=7)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY)
    for _ in range(120):
        net.train_step(X, y)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        net.grow(1, hidden=4)
    body = net._port_site.bodies[0]["body"]   # port-owned
    body._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                               train_lr_scales={"blck:0": 0.5})
    W1_pre = np.asarray(body.W1).copy()
    dU = np.full((len(X), 1), 0.01)
    with pytest.raises(ValueError, match="blck:0"):
        body.train_from_grad(X, dU)
    assert np.array_equal(np.asarray(body.W1), W1_pre)
