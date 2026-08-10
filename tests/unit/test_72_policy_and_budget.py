"""Doc 72B D-3 boxes (owner order: tests FIRST — this file is
committed RED on the unfixed branch, then flipped GREEN by
the D-1/D-2 fixes; the RED transcript is the tests-first
evidence).

T-72-1 merge preserves unrelated keys (DEFECT-1 carrier)
T-72-2 removal sentinel; {} is a no-op
T-72-3 E-6 window cycle exactness (three-clause form, RV F2)
T-72-4 seam-facade budget parity (DEFECT-2 carrier)
T-72-5 whole-act rider inclusion at the seam, no partial
       mutation on refusal
T-72-6 G-BUDGET shift-left: propose/plan_validate == run
       reality; expert-tier graceful absence (RV F7)
Boxes assert VALUES (key sets, caps, params, hashes), never
just flags."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.facade import System                    # noqa: E402
from core.wiring import Config                    # noqa: E402

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0},
         "target": float(i) / 24.0} for i in range(25)]
ASPECT4 = {"gate_aspect_min": 8.0,
           "gate_aspect_mode": "refuse",
           "aspect_auto": "widen_first",
           "aspect_auto_max_widen": 16}


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


def _mk(s, mid="m", mult=50):
    out = s.create_model(mid, policy={"max_params_mult": mult})
    assert not (isinstance(out, dict)
                and out.get("refusal")), out
    r = s.study(mid, ROWS, steps=20)
    assert "refusal" not in r, r
    return s.lc._load_working(mid)[0]


def _gp(s, mid="m"):
    return dict(s.lc.policy(mid).get("growth_params") or {})


def _hash(net):
    """State fingerprint (F-8/b3 precedent): every array and
    scalar except the audit ledger AND the growth policy
    (policy presence is asserted separately by these boxes)."""
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


# ------------- T-72-1 merge preserves unrelated keys --------

def test_t72_1_window_write_preserves_aspect_stack(tmp_path):
    """DEFECT-1 carrier: installing an E-6-style protection
    window must NOT discard the previously installed aspect
    stack — neither in policy.json nor on the reloaded
    organ."""
    s = _sys(tmp_path)
    _mk(s)
    s.set_policy("m", growth_params=dict(ASPECT4))
    s.set_policy("m", growth_params={
        "train_lr_scales": {"encoder": 0.15}})   # open window
    pol = _gp(s)
    for k, v in ASPECT4.items():
        assert pol.get(k) == v, (k, pol)          # kept
    assert pol["train_lr_scales"] == {"encoder": 0.15}
    organ = s.lc._load_working("m")[0]
    ip = getattr(organ, "_growth_policy", {})
    assert ip.get("gate_aspect_min") == 8.0       # installed
    s.set_policy("m", growth_params={
        "train_lr_scales": {}})                   # close ({})
    pol = _gp(s)
    for k, v in ASPECT4.items():
        assert pol.get(k) == v, (k, pol)          # still kept


# ------------- T-72-2 removal sentinel + {} no-op -----------

def test_t72_2_none_sentinel_removes_and_empty_is_noop(
        tmp_path):
    s = _sys(tmp_path)
    _mk(s)
    s.set_policy("m", growth_params=dict(ASPECT4))
    s.set_policy("m", growth_params={"gate_aspect_min": None})
    pol = _gp(s)
    assert "gate_aspect_min" not in pol           # removed
    assert pol.get("gate_aspect_mode") == "refuse"  # intact
    assert pol.get("aspect_auto") == "widen_first"
    before = json.dumps(_gp(s), sort_keys=True)
    s.set_policy("m", growth_params={})           # no-op
    assert json.dumps(_gp(s), sort_keys=True) == before


# ------------- T-72-3 E-6 cycle exactness (3 clauses) -------

def test_t72_3_window_cycle_exactness(tmp_path):
    s = _sys(tmp_path)
    _mk(s)
    s.set_policy("m", growth_params=dict(ASPECT4))
    pre = json.dumps(_gp(s), sort_keys=True)
    # (a)+(b): open, then close via the None SENTINEL ->
    # growth_params byte-identical to pre-window
    s.set_policy("m", growth_params={
        "train_lr_scales": {"encoder": 0.15}})
    s.set_policy("m", growth_params={
        "train_lr_scales": None})
    assert json.dumps(_gp(s), sort_keys=True) == pre
    # (c): close via {} leaves the key inert-empty and the
    # aspect keys intact
    s.set_policy("m", growth_params={
        "train_lr_scales": {"encoder": 0.15}})
    s.set_policy("m", growth_params={
        "train_lr_scales": {}})
    pol = _gp(s)
    assert pol["train_lr_scales"] == {}
    for k, v in ASPECT4.items():
        assert pol.get(k) == v


# ------------- T-72-4 seam-facade budget parity -------------

def test_t72_4_seam_refuses_at_cap_like_facade(tmp_path):
    """DEFECT-2 carrier: at a state where the next deepen
    crosses the cap, facade / plan_run / trial must ALL
    refuse — with NO mutation through any door. Values: the
    cap number and the params count are asserted."""
    s = _sys(tmp_path)
    organ = _mk(s, mult=2)                 # cap = 2x initial
    meta = json.loads((s.lc._mdir("m") / "working" /
                       "meta.json").read_text())
    cap = meta["initial_params"] * 2
    p0 = organ.n_params()
    assert p0 + 228 > cap                  # next deepen crosses
    # facade
    r = s.deepen("m", m=4, position=0)
    assert r.get("refusal"), r
    assert "budget" in str(r["refusal"])
    # plan seam
    r = s.plan_run("m", {"steps": [{"move": "deepen",
                                    "args": {"m": 4,
                                             "position": 0}}]},
                   examples=ROWS, steps_between=0)
    assert (r.get("refusal") or r.get("halted")), r
    # trial seam
    r = s.trial("m", "deepen", {"m": 4, "position": 0},
                budget_steps=2, examples=ROWS)
    assert r.get("refusal"), r
    organ2 = s.lc._load_working("m")[0]
    assert organ2.n_params() == p0         # nothing mutated


# ------------- T-72-5 whole-act rider at the seam -----------

def test_t72_5_seam_whole_act_rider_inclusive(tmp_path):
    """With a floor that forces a widen rider, the SEAM must
    judge deepen+widen as ONE act against the cap and refuse
    crossings with zero partial mutation (60D F-4 semantics,
    now at the seam). Cap chosen so the PRIMARY move alone
    fits but primary+rider crosses."""
    s = _sys(tmp_path)
    organ = _mk(s, mult=50)
    # walk to the aspect edge exactly as the facade F-4 box
    # does: three deepens are aspect-clean at H=28
    for _ in range(2):
        r = s.plan_run("m", {"steps": [{"move": "deepen",
                                        "args": {"m": 4,
                                                 "position": 0}}]},
                       examples=ROWS, steps_between=0)
        assert not r.get("refusal") and r.get("halted") is None, r
    organ = s.lc._load_working("m")[0]
    p_now = organ.n_params()
    meta = json.loads((s.lc._mdir("m") / "working" /
                       "meta.json").read_text())
    # floor 8.0 at H=16 (d_in=2 birth): the NEXT deepen
    # (depth 3->4, 16/4=4) forces a widen rider k=16
    # (H 16->32; rider added ~320). Headroom 450 is chosen
    # so the RIDER ALONE fits (320 <= 450 — the existing
    # rider-only check must ADMIT) while rider+deepen
    # (~320+260=580) CROSSES — carrying exactly the
    # missing-whole-act defect. (First cut used 300, which
    # the rider alone crossed — the old rider-only check
    # refused and the box was a false GREEN; disclosed.)
    mult = (p_now + 450) / meta["initial_params"]
    s.set_policy("m", max_params_mult=mult)
    s.set_policy("m", growth_params=dict(ASPECT4))
    h_before = _hash(s.lc._load_working("m")[0])
    r = s.plan_run("m", {"steps": [{"move": "deepen",
                                    "args": {"m": 4,
                                             "position": 0}}]},
                   examples=ROWS, steps_between=0)
    assert (r.get("refusal") or r.get("halted")), r
    organ2 = s.lc._load_working("m")[0]
    assert organ2.n_params() == p_now      # no partial widen
    assert _hash(organ2) == h_before       # bitwise untouched


# ------------- T-72-6 G-BUDGET shift-left + tier law --------

def test_t72_6_gbudget_shift_left_and_expert_absence(
        tmp_path):
    s = _sys(tmp_path)
    organ = _mk(s, mult=2)                 # next deepen crosses
    meta = json.loads((s.lc._mdir("m") / "working" /
                       "meta.json").read_text())
    cap = meta["initial_params"] * 2
    # policy tier: propose carries G-BUDGET with values
    q = s.propose("m", "deepen", {"m": 4, "position": 0})
    gb = q.get("gates", {}).get("G-BUDGET")
    assert gb is not None, q
    assert gb["met"] is False
    assert gb["cap"] == cap
    assert gb["post"] == organ.n_params() + q["cost_params"]
    assert q["would_refuse"] is True
    # plan_validate row agrees with run reality
    v = s.plan_validate("m", {"steps": [
        {"move": "deepen", "args": {"m": 4, "position": 0}}]})
    row = v["steps"][0]["proposal"]
    assert row["gates"]["G-BUDGET"]["met"] is False
    assert row["would_refuse"] is True
    # expert tier (RV F7): direct gates.propose WITHOUT the
    # injected cap emits NO G-BUDGET row, no budget refusal
    from reference_net.method.gates import propose as _prop
    gpp = dict(getattr(organ, "_growth_policy", {}) or {})
    gpp.pop("params_budget_cap", None)
    rep = _prop(organ, "deepen", gpp, m=4, position=0)
    assert "G-BUDGET" not in rep["gates"]


# ------------- T-72-7 (review fix): unknown-move trial -------

def test_t72_7_unknown_move_trial_still_refuses_loudly(
        tmp_path):
    """Review finding: the budget pre-check must not turn the
    legacy loud refusal for unknown moves into a crash."""
    s = _sys(tmp_path)
    _mk(s)
    r = s.trial("m", "widen", {"k": 2}, budget_steps=2,
                examples=ROWS)
    assert r.get("refusal"), r
    assert "widen" in str(r["refusal"])


# ---- T-72-8 (full-review additions): walk index + None ----

def test_t72_8_cumulative_walk_marks_crossing_step(tmp_path):
    """[RV F5] obligation: in a multi-step plan the budget
    walk marks the CROSSING step — step 1 met, step 2 not."""
    s = _sys(tmp_path)
    organ = _mk(s, mult=4)          # room for one deepen only
    meta = json.loads((s.lc._mdir("m") / "working" /
                       "meta.json").read_text())
    cap = meta["initial_params"] * 4
    v = s.plan_validate("m", {"steps": [
        {"move": "deepen", "args": {"m": 4, "position": 0}},
        {"move": "deepen", "args": {"m": 4, "position": 0}}]})
    g1 = v["steps"][0]["proposal"]["gates"]["G-BUDGET"]
    g2 = v["steps"][1]["proposal"]["gates"]["G-BUDGET"]
    assert g1["met"] is True and g2["met"] is False, (g1, g2)
    assert g2["cap"] == cap
    # top-level None is a no-op ([RV F6])
    before = json.dumps(_gp(s), sort_keys=True)
    s.set_policy("m", growth_params=None)
    assert json.dumps(_gp(s), sort_keys=True) == before


# ---- T-72-9 (doc 74): rider-pricing FUNCTION coverage ------

def test_t72_9_propose_rider_pricing_executes_and_prices(
        tmp_path):
    """Doc 74 D-2: the 72-added rider-inclusive pricing in
    propose must EXECUTE (it shipped with no test running it
    — the E-7 crash) and price the whole act correctly
    (hand-built expected value from the organ's own state)."""
    s = _sys(tmp_path)
    _mk(s)
    # E-7-shaped organ: mixed growth history (deepen + grow)
    for step in ({"move": "deepen", "args": {"m": 4,
                                             "position": 0}},
                 {"move": "grow", "args": {"j": 0,
                                           "hidden": 16}},
                 {"move": "deepen", "args": {"m": 4,
                                             "position": 0}}):
        r = s.plan_run("m", {"steps": [step]},
                       examples=ROWS, steps_between=0)
        assert not r.get("refusal") and \
            r.get("halted") is None, (step, r)
    s.set_policy("m", growth_params=dict(ASPECT4))
    organ = s.lc._load_working("m")[0]
    import numpy as _np
    q = s.propose("m", "deepen", {"m": 4, "position": 0})
    assert not q.get("refusal"), q          # no crash path
    ga = q["gates"]["G-ASPECT"]
    assert ga["met"] is False               # rider fires
    k_r = int(_np.ceil(ga["floor"] * ga["depth_after"])
              - ga["width"])
    H_new = int(ga["width"]) + k_r
    exp_act = k_r * (organ.d_in + 2)
    for b in organ.blocks:
        exp_act += 2 * int(_np.asarray(b["bb"]).size) * k_r
    exp_act += 4 * H_new + 4 + H_new * 4    # deepen at H_new
    gb = q["gates"]["G-BUDGET"]
    assert gb["post"] == organ.n_params() + exp_act, \
        (gb, exp_act)


# ---- T-72-10 (74 S-3 gate finds): the two uncovered paths --

def test_t72_10_trial_precheck_exception_passthrough(
        tmp_path):
    """The pre-check's except-branch: a GROWTH move whose
    propose raises (grow_site on the Network family) must
    fall through to the legacy loud refusal — never crash."""
    s = _sys(tmp_path)
    _mk(s)
    r = s.trial("m", "insert_layer", {"position": 0},
                budget_steps=2, examples=ROWS)
    assert r.get("refusal"), r


def test_t72_10b_rider_pricing_covers_loop_block(tmp_path):
    """The loop-block term of the rider formula executes and
    prices (organ WITH a loop block; doc 74 S-3 gate)."""
    s = _sys(tmp_path)
    _mk(s)
    for step in ({"move": "deepen", "args": {"m": 4,
                                             "position": 0}},
                 {"move": "deepen", "args": {"m": 4,
                                             "position": 0}}):
        r = s.plan_run("m", {"steps": [step]},
                       examples=ROWS, steps_between=0)
        assert not r.get("refusal"), r
    rs = s.set_policy("m", growth_params={
        "loop_enabled": True})       # opt-in (loud contract)
    assert not (isinstance(rs, dict)
                and rs.get("refusal")), rs
    lr = s.loop("m")
    assert not (isinstance(lr, dict) and lr.get("refusal")), lr
    s.set_policy("m", growth_params=dict(ASPECT4))
    q = s.propose("m", "deepen", {"m": 4, "position": 0})
    assert not q.get("refusal"), q
    assert q["gates"]["G-ASPECT"]["met"] is False
    assert q["gates"]["G-BUDGET"]["post"] > 0
