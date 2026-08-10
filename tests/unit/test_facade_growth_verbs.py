"""59C stage 1 (59B s2): the 9 growth-control verbs on the
served facade (T2a). Boxes written FIRST from the design text;
RED today = AttributeError (no verb) on F-1..F-10, tuple
mismatch on F-11."""
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

from core.facade import System                       # noqa: E402
from core.wiring import Config                       # noqa: E402

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]

ATT_SP = {"d_model": 8, "n_layers": 1, "heads_spec": [[1]],
          "seed": 3}


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


def _mk(s, mid, substrate=None, policy=None):
    kw = {"substrate": substrate} if substrate else {}
    if policy:
        kw["policy"] = policy
    out = s.create_model(mid, description="f", **kw)
    assert "refusal" not in out, out
    r = s.study(mid, ROWS, steps=20)
    assert "refusal" not in r, r
    return s.lc._load_working(mid)[0]


def _hash(net):
    """State fingerprint (b3 precedent): every array/scalar in
    the object tree except the audit ledger."""
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if str(k) in ("gain_ledger", "growth_events",
                              "_growth_policy"):
                    continue
                h.update(str(k).encode())
                walk(o[k])
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, np.ndarray):
            h.update(o.tobytes())
        elif hasattr(o, "__dict__"):
            walk(vars(o))
        else:
            h.update(repr(o).encode())
    walk(vars(net))
    return h.hexdigest()


def _events(s, mid):
    return [e["event"] for e in s.lc.events(mid)]


# ---------------- F-1 deepen / mlp ----------------

def test_f1_deepen_mlp_exact_delta(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    H, p0 = organ.H, organ.n_params()
    b0 = len(organ.blocks)
    r = s.deepen("m")
    assert "refusal" not in r, r
    o2 = s.lc._load_working("m")[0]
    m = H                                   # default m -> H
    assert o2.n_params() == p0 + (H * m + m + H * m)
    assert len(o2.blocks) == b0 + 1
    assert r["params"] == o2.n_params()
    assert "deepen" in _events(s, "m")
    # explicit m: hand delta again
    r2 = s.deepen("m", m=4)
    assert "refusal" not in r2, r2
    o3 = s.lc._load_working("m")[0]
    assert o3.n_params() == o2.n_params() + (H * 4 + 4 + H * 4)


# ---------------- F-2 deepen / hosts ----------------

@pytest.mark.parametrize("sub,pol", [
    ("transformer", None),
    ("growable_attention", {"substrate_params": ATT_SP}),
])
def test_f2_deepen_hosts_global_and_position(tmp_path, sub, pol):
    s = _sys(tmp_path)
    organ = _mk(s, "h", substrate=sub, policy=pol)
    L0 = organ.L
    probe = ROWS[0]["input"]
    pre = s.infer("h", probe, working=True)["output"]
    r = s.deepen("h")                       # global = END
    assert "refusal" not in r, r
    o2 = s.lc._load_working("h")[0]
    assert o2.L == L0 + 1
    post = s.infer("h", probe, working=True)["output"]
    assert np.asarray(pre).tolist() == np.asarray(post).tolist()
    # positional insert works
    r2 = s.deepen("h", position=1)
    assert "refusal" not in r2, r2
    assert s.lc._load_working("h")[0].L == L0 + 2
    # scope on a host: loud refusal naming the host
    r3 = s.deepen("h", scope=[0, 1])
    assert "refusal" in r3
    assert type(organ).__name__ in r3["refusal"]


# ---------------- F-3 scoped deepen / mlp ----------------

def test_f3_scoped_deepen_warns_and_executes(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    p0 = organ.n_params()
    probe = ROWS[0]["input"]
    pre = s.infer("m", probe, working=True)["output"]
    with pytest.warns(UserWarning, match="functionally CLOSED"):
        r = s.deepen("m", scope=[0, 1])
    assert "refusal" not in r, r
    assert any("functionally CLOSED" in t
               for t in r.get("warning", [])), r
    o2 = s.lc._load_working("m")[0]
    assert o2.n_params() > p0                # executed
    post = s.infer("m", probe, working=True)["output"]
    assert float(pre) == float(post)         # zero-birth exact


# ---------------- F-4 budget + force semantics ----------------

def test_f4_budget_refuses_force_gates_only(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m", policy={"max_params_mult": 1})
    pre = _hash(organ)
    r = s.deepen("m")
    assert r.get("refusal") == "params budget"
    o2 = s.lc._load_working("m")[0]
    assert _hash(o2) == pre                  # untouched
    assert "deepen_refused" in _events(s, "m")
    # force does NOT bypass the budget
    r2 = s.deepen("m", force=True)
    assert r2.get("refusal") == "params budget"
    # force DOES bypass a refusing gate
    s2 = _sys(tmp_path / "b")
    _mk(s2, "g")
    out = s2.set_policy("g", growth_params={
        "gate_scope_min_width": 99, "gate_scope_mode": "refuse"})
    assert "refusal" not in out, out
    r3 = s2.deepen("g", scope=[0, 1])
    assert "refusal" in r3 and "G-SCOPE" in r3["refusal"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r4 = s2.deepen("g", scope=[0, 1], force=True)
    assert "refusal" not in r4, r4
    o3 = s2.lc._load_working("g")[0]
    assert o3.gain_ledger[-1].get("forced") is True


# ---------------- F-5 removals ----------------

def test_f5_removals_hand_deltas_hosts_refuse(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    H = organ.H
    s.deepen("m", m=4)
    o1 = s.lc._load_working("m")[0]
    p1 = o1.n_params()
    r = s.remove_block("m", 0)
    assert "refusal" not in r, r
    o2 = s.lc._load_working("m")[0]
    assert o2.n_params() == p1 - (4 * H + 4 + H * 4)
    assert r["params_delta"] == -(4 * H + 4 + H * 4)
    # grown body removal: -(body + assembly) per the slot
    g = s.grow("m", k_nodes=1, hidden=4)
    assert g.get("grown"), g
    o3 = s.lc._load_working("m")[0]
    key = o3.gain_ledger[-1]["site"]
    slot = next(sl for sl in o3._port_site.bodies
                if sl.get("key") == key)
    dn = int(slot.n_params())
    p3 = o3.n_params()
    r2 = s.remove_grown("m", key)
    assert "refusal" not in r2, r2
    o4 = s.lc._load_working("m")[0]
    assert o4.n_params() == p3 - dn
    assert r2["params_delta"] == -dn
    # bad targets refuse loudly, nothing changes
    for bad in (s.remove_block("m", 99),
                s.remove_grown("m", "nope")):
        assert "refusal" in bad
    # hosts refuse loudly (shrink = version rollback)
    st = _sys(tmp_path / "t")
    _mk(st, "h", substrate="transformer")
    for r3 in (st.remove_block("h", 0), st.remove_grown("h", 0)):
        assert "refusal" in r3 and "rollback" in r3["refusal"]


# ---------------- F-6 propose: 7 moves, verdict==reality ------

def test_f6_propose_all_moves_no_mutation(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    out = s.set_policy("m", growth_params={"loop_enabled": True})
    assert "refusal" not in out, out
    g = s.grow("m", k_nodes=1, hidden=4)     # one grown site
    assert g.get("grown"), g
    o0 = s.lc._load_working("m")[0]
    key = o0.gain_ledger[-1]["site"]
    pre = _hash(o0)

    # deepen: success arm
    rep = s.propose("m", "deepen", {"m": 4})
    assert rep["would_refuse"] is False
    assert rep["cost_params"] == 4 * o0.H + 4 + o0.H * 4
    # grow: refusal arm (site already composite) + success arm
    j = int(str(key).split("/")[0]) if "/" in str(key) else key
    assert s.propose("m", "grow", {"j": j,
                                   "hidden": 4})["would_refuse"] \
        is True
    # remove_block: no blocks yet -> refuse; after deepen ->
    # success (reality judged below)
    assert s.propose("m", "remove_block",
                     {"k": 0})["would_refuse"] is True
    # remove_grown: success + refusal arms
    assert s.propose("m", "remove_grown",
                     {"key": key})["would_refuse"] is False
    assert s.propose("m", "remove_grown",
                     {"key": "nope"})["would_refuse"] is True
    # remove_loop: refusal (no loop yet)
    assert s.propose("m", "remove_loop", {})["would_refuse"] \
        is True
    # insert_layer on mlp: loud refusal dict (no such operator;
    # matches reality — the organ has no insert_layer)
    r = s.propose("m", "insert_layer", {"position": 0})
    assert "refusal" in r and "insert_layer" in r["refusal"]
    # grow_site dry-run on mlp: would_refuse verdict (the 3-D
    # site grammar does not parse here)
    assert s.propose("m", "grow_site",
                     {"site_path": "zz",
                      "hidden": 4})["would_refuse"] is True
    # unknown move: loud refusal
    r = s.propose("m", "teleport", {})
    assert "refusal" in r and "teleport" in r["refusal"]
    # READ-ONLY: state untouched through all of the above
    assert _hash(s.lc._load_working("m")[0]) == pre

    # grow: refusal REALITY (the composite j propose refused
    # above) + success arm on a fresh site, reality on a twin
    import copy
    o0 = s.lc._load_working("m")[0]
    with pytest.raises(ValueError, match="already composite"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            copy.deepcopy(o0).grow(j, hidden=4)
    import re
    jf = next(int(m.group(1)) for sp, _ in o0.growth_sites()
              for m in [re.match(r"root\[(\d+)\]$", str(sp))]
              if m and int(m.group(1)) != j)
    assert s.propose("m", "grow",
                     {"j": jf, "hidden": 4})["would_refuse"] \
        is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        copy.deepcopy(o0).grow(jf, hidden=4)   # reality: succeeds

    # deepen: refusal arm under a refusing gate — verdict must
    # equal reality WHICHEVER way the width estimator falls
    # (propose and preset_layer consult the same wd.met)
    out = s.set_policy("m", growth_params={
        "gate_deepen_mode": "refuse", "loop_enabled": True})
    assert "refusal" not in out, out
    repd = s.propose("m", "deepen", {"m": 2})
    o0 = s.lc._load_working("m")[0]
    rd = s.deepen("m", m=2, position=len(o0.blocks))  # gated
    if repd["would_refuse"]:
        assert "refusal" in rd and "G-DEEPEN" in rd["refusal"]
    else:
        assert "refusal" not in rd, rd
    out = s.set_policy("m", growth_params={
        "gate_deepen_mode": "advise", "loop_enabled": True})
    assert "refusal" not in out, out

    # verdict == reality (each arm above, executed for real)
    assert "refusal" not in s.deepen("m", m=4)          # succ
    assert "refusal" not in s.remove_block("m", 0)      # succ
    assert "refusal" not in s.remove_grown("m", key)    # succ
    assert "refusal" in s.remove_grown("m", key)        # gone
    r = s.loop("m")                                     # loop on
    assert "refusal" not in r, r
    assert s.propose("m", "remove_loop", {})["would_refuse"] \
        is False
    assert "refusal" not in s.remove_loop("m")
    # hosts: insert_layer/grow_site succeed via propose there
    st = _sys(tmp_path / "t")
    ot = _mk(st, "h", substrate="transformer")
    rep = st.propose("h", "insert_layer", {"position": 0})
    assert rep["would_refuse"] is False
    assert rep["cost_params"] > 0
    assert "refusal" not in st.deepen("h", position=0)  # reality
    sp = "layer0/ffn[0]"                    # the host grammar
    rep2 = st.propose("h", "grow_site", {"site_path": sp,
                                         "hidden": 4})
    assert rep2["would_refuse"] is False
    st.lc._load_working("h")[0].grow_site(sp, hidden=4)
    #                     ^ reality on an unsaved working copy


# ---------------- F-7 plans ----------------

def _plan():
    # MIXED vocabulary per the plan text: deepen + remove
    # (remove_block k=0 removes the block the first step made)
    return {"steps": [{"move": "deepen", "args": {"m": 4}},
                      {"move": "remove_block", "args": {"k": 0}},
                      {"move": "deepen", "args": {"m": 2}}],
            "limits": {"max_events": 2}}


def test_f7_plan_validate_and_run(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    H, p0 = organ.H, organ.n_params()
    plan = _plan()
    v = s.plan_validate("m", plan)
    assert "refusal" not in v, v
    # static walk on the CURRENT state: remove_block k=0 has
    # no block yet -> would_refuse with cost 0 (dry-run truth)
    assert v["cumulative_cost_params"] == \
        (H * 4 + 4 + H * 4) + 0 + (H * 2 + 2 + H * 2)
    assert v["steps"][1]["proposal"]["would_refuse"] is True
    # file-path dialect == inline dialect
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    assert s.plan_validate("m", str(pf)) == v
    # run: deepen(+d4) then remove_block(-d4), HALTS at
    # max_events=2; hand params walk nets to p0
    W1_pre = np.asarray(s.lc._load_working("m")[0].W1).copy()
    r = s.plan_run("m", str(pf), examples=ROWS,
                   steps_between=5)
    assert "refusal" not in r, r
    assert r["halted"] == "limit:max_events"
    assert [e["move"] for e in r["events"]] == \
        ["deepen", "remove_block"]
    o2 = s.lc._load_working("m")[0]
    assert o2.n_params() == p0                # +d4 then -d4
    assert len(o2.blocks) == 0
    # steps_between trained (weights moved)
    assert not np.array_equal(
        np.asarray(o2.W1), W1_pre)
    # every plan event trigger="policy"; adoption+halt records
    led = o2.gain_ledger
    evs = [x["event"] for x in led]
    assert "plan_adopted" in evs and "plan_halted" in evs
    dp = [x for x in led if x["event"] == "deepen"][-1]
    assert dp["trigger"] == "policy"
    rb = [x for x in led if x["event"] == "prune_block"][-1]
    assert rb["trigger"] == "policy"
    assert rb["params_added"] == -(4 * H + 4 + H * 4)
    # examples=None -> pure structural mode (no training)
    s2 = _sys(tmp_path / "b")
    _mk(s2, "n")
    W_pre = np.asarray(s2.lc._load_working("n")[0].W1).copy()
    r2 = s2.plan_run("n", {"steps": [{"move": "deepen",
                                      "args": {"m": 2}}]})
    assert "refusal" not in r2, r2
    o3 = s2.lc._load_working("n")[0]
    assert np.array_equal(np.asarray(o3.W1), W_pre)
    # a bad plan refuses loudly
    bad = s.plan_run("m", {"steps": [{"move": "teleport"}]})
    assert "refusal" in bad


# ---------------- F-8 trial ----------------

def test_f8_trial_report_and_bit_restore(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    pre = _hash(organ)
    r = s.trial("m", "deepen", {"m": 4}, budget_steps=3,
                examples=ROWS)
    assert "refusal" not in r, r
    assert len(r["losses"]) == 3
    assert r["realized_gain"] == r["loss_before"] - r["losses"][-1]
    assert r["wall_ms"] > 0
    assert _hash(s.lc._load_working("m")[0]) == pre  # restored
    # lib dialect: no examples -> loud refusal naming it
    r2 = s.trial("m", "deepen", {"m": 4})
    assert "refusal" in r2 and "examples" in r2["refusal"]


# ---------------- F-9 describe / assess ----------------

def test_f9_describe_assess_readonly_jsonsafe(tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    pre = _hash(organ)
    d = s.describe("m")
    a = s.assess("m")
    json.dumps(d), json.dumps(a)             # JSON-safe
    assert d["params_total"] == organ.n_params()
    assert a["census"]["params"] == organ.n_params()
    assert _hash(s.lc._load_working("m")[0]) == pre


# ---------------- F-10 warning transport ----------------

def test_f10_warning_transport(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m", policy={"max_params_mult": 50})
    r = s.deepen("m")                        # global: NO key
    assert "warning" not in r
    with pytest.warns(UserWarning):
        r2 = s.deepen("m", scope=[0, 1])
    assert isinstance(r2.get("warning"), list) and r2["warning"]


# ---------------- F-11 contract ----------------

NEW_VERBS = ("deepen", "remove_block", "remove_grown",
             "propose", "plan_validate", "plan_run", "trial",
             "describe", "assess")


def test_f11_contract_tuple_and_surface(tmp_path):
    sms = REPO.parent / "SoftModelSystem"
    sys.path.insert(0, str(sms))
    try:
        from sms.common import contract
    finally:
        sys.path.remove(str(sms))
    assert len(NEW_VERBS) == 9
    for v in NEW_VERBS:
        assert v in contract.SYSTEM_VERBS, v
        assert hasattr(System, v), v
    # 33 legacy + 9 (59C) + innovation_report (owner
    # ruling 2026-07-24, 59C review)
    assert len(contract.SYSTEM_VERBS) == 33 + 9 + 1
    src = (sms / "sms" / "common" / "contract.py").read_text()
    assert "not a System verb" not in src    # stale comment gone
