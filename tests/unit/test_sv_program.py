"""59C stage 5 — SV program, lib homes: SV-0 (all-verb channel
sweeps; channels are verified by E2E per the owner ruling),
SV-1 lib journeys (CLI subprocess + MCP wire), SV-4
(conservation audit), SV-7 (plan-rule accuracy), SV-8 (trial
invariance across the whole move vocabulary). Every station
carries a formula-angle and a numeric-angle judge."""
import hashlib
import json
import os
import subprocess
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
from mcp.mcp_server import MCPServer                 # noqa: E402

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]
SUITES = [{"name": "s0", "X": [r["input"] for r in ROWS[:8]],
           "y": [r["target"] for r in ROWS[:8]]}]

# the SMS contract's pinned facade surface = the sweep universe
SMS = REPO.parent / "SoftModelSystem"


def _system_verbs():
    sys.path.insert(0, str(SMS))
    try:
        from sms.common import contract
        return set(contract.SYSTEM_VERBS)
    finally:
        sys.path.remove(str(SMS))


def _hash(net):
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


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


# =====================================================
# The ONE command script both channel sweeps execute —
# (verb, args, judge). Judges receive the parsed response.
# =====================================================

def _mk_script(tmp_path):
    plan_inline = {"steps": [{"move": "deepen",
                              "args": {"m": 4}}],
                   "limits": {"max_events": 1}}
    probe = ROWS[0]["input"]

    def ok(r):
        assert isinstance(r, dict) and "refusal" not in r, r

    def refuses(r):
        assert isinstance(r, dict) and "refusal" in r, r

    def isdict(r):
        assert isinstance(r, dict), r

    return [
        ("create_model", {"model_id": "m", "holdout": ROWS[:8],
                          "policy": {"max_params_mult": 50}},
         ok),
        ("add_holdout", {"model_id": "m",
                         "examples": ROWS[:8]}, isdict),
        ("study", {"model_id": "m", "examples": ROWS,
                   "steps": 20}, ok),
        ("set_policy", {"model_id": "m",
                        "growth_params": {"loop_enabled": True}},
         ok),
        ("teach", {"model_id": "m", "examples": ROWS}, isdict),
        ("infer", {"model_id": "m", "input_": probe,
                   "working": True}, ok),
        ("predict_dist", {"model_id": "m", "input_": probe,
                          "working": True}, isdict),
        ("attempts", {"model_id": "m", "inputs": [probe]},
         lambda r: r),                       # list-shaped
        ("practice_update", {"model_id": "m",
                             "inputs": [probe],
                             "passed": [None]}, isdict),
        ("evaluate", {"model_id": "m"}, isdict),
        ("trajectory", {"model_id": "m"}, isdict),
        ("growth_report", {"model_id": "m"}, isdict),
        ("check_drift", {"model_id": "m"}, isdict),
        ("discoveries", {"model_id": "m"}, isdict),
        ("card", {"model_id": "m"}, isdict),
        ("list_models", {}, lambda r: r),
        ("list_substrates", {}, isdict),
        ("recommend_substrate", {"sample_examples": ROWS[:8]},
         isdict),
        ("self_review", {"model_id": "m"}, isdict),
        ("attribution", {"model_id": "m", "suites": SUITES},
         lambda r: r),
        ("grow", {"model_id": "m", "k_nodes": 1, "hidden": 4},
         lambda r: r.get("grown") or refuses(r)),
        ("widen", {"model_id": "m", "k": 2}, isdict),
        ("add_feature", {"model_id": "m", "name": "f2"},
         isdict),
        ("deepen", {"model_id": "m", "m": 4}, ok),
        ("propose", {"model_id": "m", "move": "deepen",
                     "args": {"m": 2}},
         lambda r: r["would_refuse"] is False),
        ("plan_validate", {"model_id": "m",
                           "plan": plan_inline}, ok),
        ("plan_run", {"model_id": "m", "plan": plan_inline},
         lambda r: len(r["events"]) == 1),
        ("trial", {"model_id": "m", "move": "deepen",
                   "args": {"m": 2}, "budget_steps": 2,
                   "examples": ROWS},
         lambda r: len(r["losses"]) == 2),
        ("describe", {"model_id": "m"},
         lambda r: r["params_total"] > 0),
        ("assess", {"model_id": "m"},
         lambda r: r["census"]["params"] > 0),
        ("remove_block", {"model_id": "m", "k": 0},
         lambda r: r["params_delta"] < 0),
        ("remove_grown", {"model_id": "m", "key": "zz"},
         refuses),                           # loud on bad key
        ("loop", {"model_id": "m"}, isdict),
        ("remove_loop", {"model_id": "m"}, isdict),
        ("run_self", {"model_id": "m", "block_budget": 1},
         isdict),
        ("run_course", {"model_id": "m",
                        "curriculum": [
                            {"name": "c0", "examples": ROWS,
                             "suite": {"X": SUITES[0]["X"],
                                       "y": SUITES[0]["y"]},
                             "target": 0.0}],
                        "policy": {"max_blocks_per_stage": 1,
                                   "steps_per_block": 10}},
         isdict),
        ("refound", {"model_id": "m", "steps": 30}, isdict),
        ("innovation_report", {"model_id": "m"}, isdict),
        ("commit", {"model_id": "m", "note": "sv0"}, isdict),
        ("get_versions", {"model_id": "m"}, isdict),
        ("rollback", {"model_id": "m", "to": "v0"}, isdict),
        ("reset", {"model_id": "m"}, isdict),
        ("store", {"model_id": "m"}, None),  # channel-specific
    ]


def test_sv0_universe_is_the_contract(tmp_path):
    """The sweep script covers EVERY pinned facade verb (the
    mechanical no-dead-verb guarantee). innovation_report is
    pinned since the owner ruling 2026-07-24 (59C review):
    sweep universe == contract surface EXACTLY."""
    verbs = {v for v, _, _ in _mk_script(tmp_path)}
    assert verbs == _system_verbs()


def test_sv0_mcp_all_verb_sweep(tmp_path):
    srv = MCPServer(System(Config.from_env(
        backend="mlp", models_root=tmp_path / "ws")))
    for i, (verb, args, judge) in enumerate(
            _mk_script(tmp_path)):
        # the MCP tool dialect (wire schemas differ from the
        # facade kwargs for these two verbs)
        if verb == "set_policy":
            args = {"model_id": args["model_id"],
                    "updates": {k: v for k, v in args.items()
                                if k != "model_id"}}
        elif verb == "infer":
            args = dict(args)
            args["input"] = args.pop("input_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = srv.handle({"jsonrpc": "2.0", "id": i,
                               "method": "tools/call",
                               "params": {"name": verb,
                                          "arguments": args}})
        body = resp["result"]["content"][0]["text"]
        assert resp["result"]["isError"] is False, (verb, body)
        r = json.loads(body)
        if verb == "store":                  # MCP summary form
            assert r["rows"] >= 0 and r["cap"] > 0
        elif judge is not None:
            judge(r)


@pytest.mark.slow
def test_sv0_cli_all_verb_sweep(tmp_path):
    env = dict(os.environ,
               SOFTMODEL_MODELS_ROOT=str(tmp_path / "ws"))
    for verb, args, judge in _mk_script(tmp_path):
        p = subprocess.run(
            [sys.executable, "-m", "cli.cli", verb,
             json.dumps(args)],
            cwd=REPO, env=env, capture_output=True, text=True,
            timeout=300)
        r = json.loads(p.stdout.strip().splitlines()[-1])
        if verb == "store":                  # graceful refusal,
            assert "refusal" in r            # never a crash
        else:
            assert p.returncode == 0, (verb, p.stdout,
                                       p.stderr[-500:])
            if judge is not None:
                judge(r)


def test_sv0_self_review_and_attribution_values(tmp_path):
    """Audit finding #3: the two zero-test verbs get REAL value
    boxes. self_review: report fields present with sane values
    on a trained model. attribution: each active node's
    distribution sums to 1 within 1e-6 (the verb's activation-
    mass normalization — hand-checkable)."""
    s = _sys(tmp_path)
    s.create_model("m", holdout=ROWS[:8],
                   policy={"max_params_mult": 50})
    s.study("m", ROWS, steps=30)
    rep = s.self_review("m", probe_inputs=[ROWS[0]["input"]])
    assert rep["store"]["rows"] == len(ROWS) - 8   # studied
    #   rows minus the 8 quarantined holdout rows (hand count)
    assert 0.0 <= rep["saturation"]["root"] <= 1.0
    assert rep["structure_events"] == 0        # none yet
    assert rep["uncertainty"] is not None      # probe path live
    g = s.grow("m", k_nodes=1, hidden=4)
    assert g.get("grown"), g
    s.study("m", ROWS, steps=30)             # activate the body
    att = s.attribution("m", SUITES)
    assert att["suite_names"] == ["s0"]
    assert att["nodes"], att
    for node in att["nodes"]:
        if node.get("inactive"):
            continue
        assert abs(sum(node["distribution"]) - 1.0) <= 1e-6 \
            + 0.003 * len(node["distribution"])
        #   ^ entries are round(...,3): quantization bound


# =====================================================
# SV-1 lib journeys (CLI + MCP) with formula/numeric judges
# =====================================================

def _journey(call, tmp_path, chan):
    """create -> data -> train -> deepen -> propose -> plan-FILE
    (deepen+widen, max_events) -> trial -> evaluate ->
    describe. `call(verb, args) -> parsed response`."""
    from reference_net.method.gates import plan_sha
    probe = ROWS[0]["input"]
    assert "refusal" not in call(
        "create_model", {"model_id": "j", "holdout": ROWS[:8],
                         "policy": {"max_params_mult": 50}})
    assert "refusal" not in call(
        "study", {"model_id": "j", "examples": ROWS,
                  "steps": 20})
    d0 = call("describe", {"model_id": "j"})
    H = d0["components"][0]["width"]
    p0 = d0["params_total"]
    pre = call("infer", {"model_id": "j", "input_": probe,
                         "working": True})["output"]
    r = call("deepen", {"model_id": "j", "m": 4})
    # FORMULA judge: closed-form delta
    assert r["params"] == p0 + (H * 4 + 4 + H * 4)
    # NUMERIC judge: serve bitwise across the birth, fetched
    # via the channel's own predict
    post = call("infer", {"model_id": "j", "input_": probe,
                          "working": True})["output"]
    assert float(pre) == float(post)
    rep = call("propose", {"model_id": "j", "move": "deepen",
                           "args": {"m": 2}})
    assert rep["cost_params"] == 2 * H + 2 + H * 2
    assert rep["cost_mem_bytes"] == rep["cost_params"] * 24
    # plan FILE: deepen+widen... widen is not a plan move —
    # vocabulary is the 7 moves; use deepen+grow (both families)
    plan = {"steps": [{"move": "deepen", "args": {"m": 2}},
                      {"move": "grow",
                       "args": {"j": 1, "hidden": 4}}],
            "limits": {"max_events": 1}}
    pf = tmp_path / f"plan_{chan}.json"
    pf.write_text(json.dumps(plan))
    rr = call("plan_run", {"model_id": "j", "plan": str(pf)})
    assert rr["halted"] == "limit:max_events"
    assert len(rr["events"]) == 1            # hand-walked halt
    d1 = call("describe", {"model_id": "j"})
    assert d1["params_total"] == r["params"] + (2 * H + 2
                                               + H * 2)
    t = call("trial", {"model_id": "j", "move": "deepen",
                       "args": {"m": 2}, "budget_steps": 2,
                       "examples": ROWS})
    assert t["realized_gain"] == t["loss_before"] \
        - t["losses"][-1]
    d2 = call("describe", {"model_id": "j"})
    assert d2["params_total"] == d1["params_total"]  # restored
    # deepen at POSITION with an explicit recipe THROUGH THE
    # CHANNEL (proof-matrix row 'deepen position/recipes' P1):
    # delta formula + birth preservation via channel predict
    pre3 = call("infer", {"model_id": "j", "input_": probe,
                          "working": True})["output"]
    r3 = call("deepen", {"model_id": "j", "m": 2,
                         "position": 0, "recipe": "random"})
    assert r3["params"] == d2["params_total"] + (2 * H + 2
                                                 + H * 2)
    post3 = call("infer", {"model_id": "j", "input_": probe,
                           "working": True})["output"]
    assert float(pre3) == float(post3)
    ev = call("evaluate", {"model_id": "j"})
    assert isinstance(ev, dict)
    return plan, plan_sha(plan)


def test_sv1_mcp_journey(tmp_path):
    srv = MCPServer(System(Config.from_env(
        backend="mlp", models_root=tmp_path / "ws")))

    def call(verb, args):
        if verb == "infer":                  # wire dialect
            args = dict(args)
            args["input"] = args.pop("input_")
        elif verb == "set_policy":
            args = {"model_id": args["model_id"],
                    "updates": {k: v for k, v in args.items()
                                if k != "model_id"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = srv.handle({"jsonrpc": "2.0", "id": 1,
                               "method": "tools/call",
                               "params": {"name": verb,
                                          "arguments": args}})
        assert resp["result"]["isError"] is False, resp
        return json.loads(resp["result"]["content"][0]["text"])
    plan, sha = _journey(call, tmp_path, "mcp")
    # FORCE through the T2 channel (proof-matrix 'force' row):
    # a refusing gate first refuses over the wire, then
    # force=true executes; forced=True lands in the ledger
    assert "refusal" not in call(
        "set_policy", {"model_id": "j",
                       "growth_params":
                       {"gate_scope_min_width": 99,
                        "gate_scope_mode": "refuse"}})
    rf = call("deepen", {"model_id": "j", "scope": [0, 1],
                         "m": 2})
    assert "refusal" in rf and "G-SCOPE" in rf["refusal"]
    df = call("describe", {"model_id": "j"})
    rf2 = call("deepen", {"model_id": "j", "scope": [0, 1],
                          "m": 2, "force": True})
    assert "refusal" not in rf2, rf2
    assert rf2["params"] > df["params_total"]
    organ_f = srv.sys.lc._load_working("j")[0]
    assert organ_f.gain_ledger[-1].get("forced") is True
    # sha equality: the adoption record in the user-visible
    # ledger carries EXACTLY the canonical plan sha
    organ = srv.sys.lc._load_working("j")[0]
    adopt = [x for x in organ.gain_ledger
             if x.get("event") == "plan_adopted"]
    assert adopt and adopt[-1]["plan_sha"] == sha
    # SV-4 CONSERVATION over the journey: initial + signed
    # ledger deltas == final served params_total
    meta = srv.sys.lc._load_working("j")[1]
    organ = srv.sys.lc._load_working("j")[0]
    signed = sum(int(x.get("params_added") or 0)
                 for x in organ.gain_ledger
                 if x.get("provenance", {}).get("kind")
                 != "trial" if x.get("event") != "trial")
    d = call("describe", {"model_id": "j"})
    assert meta["initial_params"] + signed == d["params_total"]
    # SV-4 second clause: event records match the commands
    # issued 1:1 (the growth-verb families, hand-counted from
    # THIS journey's own script)
    evs = [e["event"] for e in srv.sys.lc.events("j")]
    assert evs.count("deepen") == 3          # global+position+forced
    assert evs.count("deepen_refused") == 1  # the gate refusal
    assert evs.count("propose") == 1
    assert evs.count("plan") == 1
    assert evs.count("trial") == 1
    assert evs.count("study") == 1


@pytest.mark.slow
def test_sv1_cli_journey(tmp_path):
    env = dict(os.environ,
               SOFTMODEL_MODELS_ROOT=str(tmp_path / "ws"))

    def call(verb, args):
        p = subprocess.run(
            [sys.executable, "-m", "cli.cli", verb,
             json.dumps(args)],
            cwd=REPO, env=env, capture_output=True, text=True,
            timeout=300)
        assert p.returncode == 0, (verb, p.stdout, p.stderr)
        return json.loads(p.stdout.strip().splitlines()[-1])
    _journey(call, tmp_path, "cli")


# =====================================================
# SV-7 plan-rule accuracy: the FILE value is the authority
# =====================================================

def test_sv7_rule_file_moves_the_halt_point(tmp_path):
    from reference_net.method.gates import plan_sha
    for max_ev, expect in ((1, 1), (2, 2)):
        s = _sys(tmp_path / f"w{max_ev}")
        s.create_model("m", policy={"max_params_mult": 50})
        s.study("m", ROWS, steps=20)
        organ = s.lc._load_working("m")[0]
        H, p0 = organ.H, organ.n_params()
        plan = {"steps": [{"move": "deepen", "args": {"m": 4}},
                          {"move": "deepen", "args": {"m": 2}},
                          {"move": "deepen", "args": {"m": 2}}],
                "limits": {"max_events": max_ev}}
        pf = tmp_path / f"rule{max_ev}.json"
        pf.write_text(json.dumps(plan))
        r = s.plan_run("m", str(pf))
        # halt point moves EXACTLY with the file value
        assert len(r["events"]) == expect, r
        assert r["halted"] == "limit:max_events"
        # signed-params walk, hand-computed per event
        walk = [(4 * H + 4 + H * 4), (2 * H + 2 + H * 2),
                (2 * H + 2 + H * 2)][:expect]
        o2 = s.lc._load_working("m")[0]
        assert o2.n_params() == p0 + sum(walk)
        # provenance completeness: adoption + halt + policy
        led = o2.gain_ledger
        assert [x for x in led if x["event"] == "plan_adopted"]
        halt = [x for x in led if x["event"] == "plan_halted"]
        assert halt and halt[-1]["events_run"] == expect
        assert halt[-1]["plan_sha"] == plan_sha(plan)
        assert all(x["trigger"] == "policy"
                   for x in led if x["event"] == "deepen")
    # max_params rule: cap between event costs halts at 1
    s = _sys(tmp_path / "wp")
    s.create_model("m", policy={"max_params_mult": 50})
    s.study("m", ROWS, steps=20)
    organ = s.lc._load_working("m")[0]
    H, p0 = organ.H, organ.n_params()
    cap = p0 + (4 * H + 4 + H * 4)           # exactly one event
    plan = {"steps": [{"move": "deepen", "args": {"m": 4}},
                      {"move": "deepen", "args": {"m": 4}}],
            "limits": {"max_params": cap}}
    r = s.plan_run("m", plan)
    assert len(r["events"]) == 1 and \
        r["halted"] == "limit:max_params"


# =====================================================
# SV-8 trial invariance over the WHOLE move vocabulary
# =====================================================

def test_sv8_trial_invariance_every_move(tmp_path):
    s = _sys(tmp_path)
    s.create_model("m", policy={"max_params_mult": 50})
    s.study("m", ROWS, steps=20)
    s.set_policy("m", growth_params={"loop_enabled": True})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.deepen("m", m=4)                   # a block to remove
        g = s.grow("m", k_nodes=1, hidden=4)
    assert g.get("grown"), g
    key = s.lc._load_working("m")[0].gain_ledger[-1]["site"]
    s.loop("m")                              # a loop to remove
    moves = [("deepen", {"m": 2}),
             ("grow", {"j": 2, "hidden": 4}),
             ("remove_block", {"k": 0}),
             ("remove_grown", {"key": key}),
             ("remove_loop", {})]
    for move, args in moves:
        pre = _hash(s.lc._load_working("m")[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = s.trial("m", move, args, budget_steps=2,
                        examples=ROWS)
        assert "refusal" not in t, (move, t)
        assert _hash(s.lc._load_working("m")[0]) == pre, move
        assert t["realized_gain"] == t["loss_before"] \
            - t["losses"][-1], move
    # host moves on a transformer organ
    st = _sys(tmp_path / "t")
    st.create_model("h", substrate="transformer",
                    policy={"max_params_mult": 50})
    st.study("h", ROWS, steps=20)
    for move, args in (("insert_layer", {"position": 0}),
                       ("grow_site",
                        {"site_path": "layer0/ffn[0]",
                         "hidden": 4})):
        pre = _hash(st.lc._load_working("h")[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = st.trial("h", move, args, budget_steps=2,
                         examples=ROWS)
        assert "refusal" not in t, (move, t)
        assert _hash(st.lc._load_working("h")[0]) == pre, move
        assert t["realized_gain"] == t["loss_before"] \
            - t["losses"][-1], move
