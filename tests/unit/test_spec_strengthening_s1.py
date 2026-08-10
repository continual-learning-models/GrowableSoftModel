"""S1 (57 v1.0 Group S1): spec-strengthening boxes — the
assertions the 54 box SPECS demand but the implemented boxes
omitted (each item cites its parent box). PURE TEST branch:
expected GREEN; a RED here is a real defect and stops the
item (defect protocol).

 S1.1 (N1 / B-3-4)  assess<->gate single source
 S1.2 (N2 / B-3-5)  symmetric mixed scenarios + mode switch
 S1.3 (N3 / B-6-3)  monitoring overhead smoke
 S1.4 (N4 / B-6-4)  parameter FILE is the authority
 S1.5 (N9 / B-5-1)  zero automatic follow-up after rollback
 S1.6 (N10 / B-1-5) lifetime scope-mix schedule
 S1.7 (R9a / B-6-4) max_params ceiling halts exactly

Owner discipline in force: every box verifies FUNCTION
implemented + RESULT VALUES correct (hand-checkable judges);
"runs"/"changed" never suffice.
"""
import json
import sys
import time
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
import reference_net.method.gates as gates                  # noqa: E402
from reference_net.method.gates import (                    # noqa: E402
    assess_growth, load_plan, run_plan, validate_plan)
from reference_net.instrument import monitor_configure      # noqa: E402
from reference_net.growth_store import rollback, snapshot   # noqa: E402
from reference_net.growthpolicy import \
    DEFAULT_GROWTH_POLICY as GP                             # noqa: E402


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _net(seed=7, steps=30):
    net = Network(3, 8, lr=1e-2, seed=seed)
    X, y = _data()
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def _np_of(net, a):
    return np.asarray(net._bk.to_numpy(a))


# ---------------- S1.1 single source ----------------

def test_s1_1_assess_and_gate_share_the_estimator(monkeypatch):
    """B-3-4 spec: 'values consistent with gate verdicts on
    the same state (report says refuse -> gate refuses,
    single-source check)'. Proof: patch THE estimator symbol
    once; BOTH the assessment report and the admission gate
    see the same patched verdict — one function, two
    consumers."""
    net, X, y = _net()
    failing = {"r_hat": 99, "width": 8, "met": False}
    monkeypatch.setattr(gates, "width_demand",
                        lambda h: failing)
    rep = assess_growth(net)
    assert rep["width_demand"] is failing        # same symbol
    net._growth_policy = dict(GP, gate_deepen_mode="refuse")
    with pytest.raises(ValueError, match="G-DEEPEN"):
        net.deepen(position=0)                   # gate agrees
    passing = {"r_hat": 2, "width": 8, "met": True}
    monkeypatch.setattr(gates, "width_demand",
                        lambda h: passing)
    assert assess_growth(net)["width_demand"] is passing
    net.deepen(position=0)                       # gate admits
    assert len(net.blocks) == 1


# ---------------- S1.2 symmetric mixing ----------------

def test_s1_2a_auto_widen_manual_deepen():
    net, X, y = _net(seed=11)
    plan = {"steps": [{"rule": "schedule", "move": "grow",
                       "args": {"j": 0, "hidden": 4}}]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_plan(net, plan, GP, X, y, steps_between=2)
        net.deepen()                              # manual
        plan2 = {"steps": [{"rule": "schedule", "move": "grow",
                            "args": {"j": 2, "hidden": 4}}]}
        run_plan(net, plan2, GP, X, y, steps_between=2)
    trig = [(r["event"], r.get("trigger"))
            for r in net.gain_ledger
            if r["event"] in ("refine", "deepen")]
    assert trig == [("refine", "policy"), ("deepen", "caller"),
                    ("refine", "policy")]


def test_s1_2b_manual_widen_auto_deepen_and_mode_switch():
    net, X, y = _net(seed=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)                     # manual widen
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}] * 3,
            "limits": {"max_events": 1}}
    r = run_plan(net, plan, GP, X, y, steps_between=2)
    assert r["halted"] == "limit:max_events"      # auto phase
    net.deepen()                                  # SWITCH to
    #                                               manual, same
    #                                               move
    trig = [(rec["event"], rec.get("trigger"))
            for rec in net.gain_ledger
            if rec["event"] in ("refine", "deepen")]
    assert trig == [("refine", "caller"),
                    ("deepen", "policy"),
                    ("deepen", "caller")]


# ---------------- S1.3 overhead smoke ----------------

def test_s1_3_monitor_overhead_within_bound():
    """B-6-3 spec: 'overhead smoke (monitoring on vs off)
    within noise' — informational bound: armed at a probe
    cadence must stay within 2x of unarmed on the same fixed
    scenario."""
    X, y = _data(seed=21)
    steps = 300

    def timed(armed):
        net = Network(3, 8, lr=1e-2, seed=17)
        if armed:
            monitor_configure(net, cadence=100, window=8)
        for _ in range(20):                       # warm
            net.train_step(X, y)
        t0 = time.perf_counter()
        for _ in range(steps):
            net.train_step(X, y)
        return time.perf_counter() - t0

    t_off = min(timed(False), timed(False))
    t_on = min(timed(True), timed(True))
    ratio = t_on / t_off
    print(f"monitor overhead ratio: {ratio:.3f}")
    assert ratio <= 2.0


# ---------------- S1.4 file authority ----------------

def test_s1_4_plan_file_value_change_changes_behavior(tmp_path):
    """B-6-4 spec: 'a rule value changed in the FILE alone
    changes behavior accordingly (nothing hardcoded)'."""
    f = tmp_path / "plan.json"
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}] * 4,
            "limits": {"max_events": 3}}
    f.write_text(json.dumps(plan))
    net, X, y = _net(seed=23)
    r = run_plan(net, load_plan(f), GP, X, y, steps_between=1)
    assert len(r["events"]) == 3 and len(net.blocks) == 3
    # change ONE value in the FILE only
    plan["limits"]["max_events"] = 1
    f.write_text(json.dumps(plan))
    net2, _, _ = _net(seed=23)
    r2 = run_plan(net2, load_plan(f), GP, X, y,
                  steps_between=1)
    assert len(r2["events"]) == 1 and len(net2.blocks) == 1


# ---------------- S1.5 no auto follow-up ----------------

def test_s1_5_rollback_zero_automatic_followup():
    net, X, y = _net(seed=29)
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}]}
    run_plan(net, plan, GP, X, y, steps_between=2)  # policy ev
    rec = snapshot(net, tag="pre")
    net.deepen()
    rollback(net, rec)
    n_ledger = len(net.gain_ledger)
    assert net.gain_ledger[-1]["event"] == "rollback"
    for _ in range(20):                     # plenty of steps —
        net.train_step(X, y)                # nothing may fire
    assert len(net.gain_ledger) == n_ledger
    assert all(r.get("trigger") != "policy"
               for r in net.gain_ledger[n_ledger - 1:])


# ---------------- S1.6 lifetime scope-mix ----------------

def test_s1_6_lifetime_scope_mix_schedule():
    """B-1-5 (54): 2 global -> train -> 2 scoped (distinct
    subsets) -> train -> 1 global; spans recorded; earlier
    global layers STILL TRAIN after the local phase (total
    plasticity, value-checked); preservation bitwise at every
    birth."""
    net, X, y = _net(seed=31, steps=60)
    base = _np_of(net, net.predict(X))
    net.deepen(); net.deepen()                    # 2 global
    assert np.array_equal(_np_of(net, net.predict(X)), base)
    for _ in range(15):
        net.train_step(X, y)
    base = _np_of(net, net.predict(X))
    net.deepen(scope=[0, 1, 2, 3])                # scoped 1
    net.deepen(scope=[0, 2, 5])                   # scoped 2
    assert np.array_equal(_np_of(net, net.predict(X)), base)
    for _ in range(15):
        net.train_step(X, y)
    base = _np_of(net, net.predict(X))
    net.deepen()                                  # global again
    assert np.array_equal(_np_of(net, net.predict(X)), base)
    spans = [r["specs"]["wiring"]["reads"][0]["span"]
             for r in net.gain_ledger
             if r["event"] == "deepen"]
    assert spans == [None, None, [0, 1, 2, 3], [0, 2, 5], None]
    # earlier GLOBAL layer keeps training after the local phase
    g0_before = _np_of(net, net.blocks[0].Bin).copy()
    for _ in range(10):
        net.train_step(X, y)
    delta = float(np.max(np.abs(
        _np_of(net, net.blocks[0].Bin) - g0_before)))
    assert delta > 0.0                            # never frozen


# ---------------- S1.7 max_params ceiling ----------------

def test_s1_7_max_params_ceiling_halts_exactly():
    net, X, y = _net(seed=37)
    p0 = net.n_params()
    per_block = 8 * 8 + 8 + 8 * 8                 # m=H=8, hand
    ceiling = p0 + 2 * per_block                  # allows 2
    plan = {"steps": [{"rule": "schedule", "move": "deepen",
                       "args": {}}] * 5,
            "limits": {"max_params": ceiling}}
    r = run_plan(net, plan, GP, X, y, steps_between=1)
    assert r["halted"] == "limit:max_params"
    assert len(net.blocks) == 2                   # exactly two
    assert net.n_params() == p0 + 2 * per_block   # hand math
