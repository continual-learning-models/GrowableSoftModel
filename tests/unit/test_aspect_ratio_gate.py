"""Aspect-ratio guardrail: the G-ASPECT gate, the
headroom observability arm, and the AUTO-COMPLIANCE rule
(default = the industry-mature width-first practice), plus the
omega-on-deepened-scope zero-extension it requires.

Boxes written FIRST (strict TDD). RED honesty: the refuse/warn
arms and T-1 are the RED carriers (no gate exists yet); the
advise arms and T-2 are BEHAVIOR PINS, green before and after
by design. Every asserted number is hand-computed from the
fixture's own dimensions (hand-value totality).

T-7's SMS model_policy arm lives in the SMS repo with the
product E2E (T-21); this file is the lib tier. T-21/T-22 are
not in this file (SMS venv / sim-tests station)."""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet",
           REPO / "tests" / "unit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.facade import System                       # noqa: E402
from core.wiring import Config                       # noqa: E402

# the T-2 fixture's own recipe — VERBATIM (capture 2026-07-25,
# pre-implementation; sha ded82fd73f85fe83)
ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]
TR_SP = {"d_model": 8, "n_layers": 1, "n_heads": 1, "seed": 3}
ATT_SP = {"d_model": 8, "n_layers": 1, "heads_spec": [[1]],
          "seed": 3}
KEYS4 = {"gate_aspect_min": 2.0, "gate_aspect_mode": "warn",
         "aspect_auto": "off", "aspect_auto_max_widen": 8}


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


def _mk(s, mid="m", substrate=None, sp=None, extra_pol=None):
    pol = {"max_params_mult": 50}
    if sp:
        pol["substrate_params"] = sp
    if extra_pol:
        pol.update(extra_pol)
    kw = {"substrate": substrate} if substrate else {}
    out = s.create_model(mid, policy=pol, **kw)
    assert not (isinstance(out, dict)
                and out.get("refusal")), out
    r = s.study(mid, ROWS, steps=20)
    assert "refusal" not in r, r
    return s.lc._load_working(mid)[0]


def _set(s, mid, **gp):
    """growth_params door with MERGE semantics (lc.set_policy
    replaces the sub-dict wholesale; partial updates here must
    not drop earlier keys)."""
    cur = dict(s.lc.policy(mid).get("growth_params") or {})
    cur.update(gp)
    out = s.set_policy(mid, growth_params=cur)
    assert not (isinstance(out, dict)
                and out.get("refusal")), out


def _xy():
    X = np.array([[r["input"]["a"], r["input"]["b"],
                   r["input"]["c"]] for r in ROWS])
    y = np.array([[r["target"]] for r in ROWS])
    return X, y


# ================= T-1 key registration =================

def test_t1_four_keys_registered(tmp_path):
    """All FOUR keys route through the growth_params door; a
    typo refuses loudly naming itself; the DEFAULT dict is
    UNTOUCHED (I-8 hash safety); registry TEN -> FOURTEEN (the
    stage-0 census consciously updated — the companion count
    in test_g8_policy_key_registration is updated with S-2)."""
    from reference_net.growthpolicy import (
        DEFAULT_GROWTH_POLICY, EXTENDED_GROWTH_KEYS)
    pre_keys = set(DEFAULT_GROWTH_POLICY)
    s = _sys(tmp_path)
    _mk(s, "m")
    for k, v in KEYS4.items():
        _set(s, "m", **{k: v})
        assert k in EXTENDED_GROWTH_KEYS, k
    out = s.set_policy("m",
                       growth_params={"gate_aspect_minn": 1})
    assert isinstance(out, dict) and out.get("refusal")
    assert "gate_aspect_minn" in str(out["refusal"])
    assert set(DEFAULT_GROWTH_POLICY) == pre_keys       # I-8
    for k in KEYS4:
        assert k not in DEFAULT_GROWTH_POLICY
    assert len(EXTENDED_GROWTH_KEYS) == 37   # TEN -> FOURTEEN
    # -> THIRTY-FIVE (84 D-4: +21 preference.* keys, the
    #    same registry-set mechanism; I-8 re-asserted above)
    # -> THIRTY-SEVEN (correction round C-1: +world_type,
    #    +explore_draw_min)


# ================= T-2 default-off bitwise =================

def test_t2_default_off_bitwise_vs_captured_fixture(tmp_path):
    """BEHAVIOR PIN. The fixture bytes were captured DURING
    S-1, BEFORE any implementation existed (the pre-60D
    truth). With the feature entirely off (no keys set), the
    same seeded construction + deepen must reproduce them
    byte-identically — the feature adds EXACTLY nothing until
    a user turns it on. ONE normalization, per the standing
    SR-22 declaration (net.py _cost_fields: "wall_ms is THIS
    operation's own duration (run-varying; the ledger sits
    outside all bit gates)"): wall_ms is zeroed on BOTH sides
    — verified the SOLE differing field by a full recursive
    state diff 2026-07-25. The judge is field-by-field exact
    equality (dtype + bit-exact array values + every scalar),
    not a pickle byte compare: pickle MEMO layout differs
    between a fresh state (aliased objects) and a reloaded
    fixture — representation, never values."""
    def _norm(st):
        for rec in st.get("gain_ledger", []):
            if "wall_ms" in rec:
                rec["wall_ms"] = 0.0
        return st

    def _same(a, b, path="$"):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert (isinstance(a, np.ndarray)
                    and isinstance(b, np.ndarray)), path
            assert a.dtype == b.dtype and a.shape == b.shape, path
            assert np.array_equal(a, b), path       # bit-exact
        elif isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), path
            for k in a:
                _same(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, (list, tuple)) or (
                hasattr(a, "__iter__") and hasattr(a, "maxlen")):
            aa, bb = list(a), list(b)               # deque too
            assert len(aa) == len(bb), path
            for i, (x, y) in enumerate(zip(aa, bb)):
                _same(x, y, f"{path}[{i}]")
        elif hasattr(a, "__dict__"):
            assert type(a) is type(b), path
            _same(vars(a), vars(b), path + "~")
        else:
            assert type(a) is type(b) and a == b, (path, a, b)

    s = _sys(tmp_path)
    s.create_model("m", policy={"max_params_mult": 50})
    s.study("m", ROWS, steps=20)
    s.deepen("m", m=4)
    organ = s.lc._load_working("m")[0]
    # pickle is safe here: the fixture is repo-tracked and was
    # captured by this project (the battery's own pin format)
    fx = pickle.loads(
        (REPO / "tests" / "fixtures"
         / "aspect_default_off_organ.pkl").read_bytes())
    _same(_norm(organ.__getstate__()), _norm(fx))


# ============ T-3 mlp arithmetic + modes (hand values) ======

def test_t3_mlp_modes_and_boundary(tmp_path):
    """EXECUTION REVISION (disclosed in 60D s3): the library
    default mlp trunk is H=16 (birth-derived, self-shaping
    doctrine — never overridable), not the doc draft's 8; all
    mlp hand values recomputed at the fixture's OWN width.
    H=16, blocks=1 -> deepen makes depth_after=3,
    aspect_after = 16/3 ~= 5.33 (message formats %.2f);
    floor 6.0: advise silent / warn both numbers / refuse
    loud naming G-ASPECT; force executes with forced=True in
    the ledger. BOUNDARY on the FIRST deepen: blocks=0 ->
    depth_after=2, aspect exactly 16/2 = 8.0; floor 8.0 ->
    ADMITTED (>= semantics; exactly-representable ratio)."""
    # boundary twin FIRST (its own workspace)
    sb = _sys(tmp_path / "b")
    _mk(sb, "x")
    _set(sb, "x", gate_aspect_min=8.0,
         gate_aspect_mode="refuse", aspect_auto="off")
    rb = sb.deepen("x", m=4)               # 16/2 = 8.0 == floor
    assert "refusal" not in rb, rb          # >= ADMITS
    # main organ: H=16, one block in place
    s = _sys(tmp_path)
    organ = _mk(s, "m")
    assert organ.H == 16
    s.deepen("m", m=4)                      # blocks=1, stages=2
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="advise",
         aspect_auto="off")
    r = s.deepen("m", m=4)                  # advise = silent
    assert "refusal" not in r and "warning" not in r, r
    s.remove_block("m", 1)                  # back to blocks=1
    _set(s, "m", gate_aspect_mode="warn")
    with pytest.warns(UserWarning, match="G-ASPECT"):
        r = s.deepen("m", m=4)
    assert "refusal" not in r, r
    assert any("5.33" in t and "6.0" in t
               for t in r.get("warning", [])), r
    s.remove_block("m", 1)
    _set(s, "m", gate_aspect_mode="refuse")
    r = s.deepen("m", m=4)
    assert "refusal" in r and "G-ASPECT" in r["refusal"], r
    assert "5.33" in r["refusal"] and "6.0" in r["refusal"]
    r = s.deepen("m", m=4, force=True)      # C-5: force wins
    assert "refusal" not in r, r
    o2 = s.lc._load_working("m")[0]
    assert o2.gain_ledger[-1].get("forced") is True


# ============ T-4 hosts x2 (never a representative) =========

@pytest.mark.parametrize("sub,sp", [
    ("transformer", TR_SP), ("growable_attention", ATT_SP)])
def test_t4_host_modes(tmp_path, sub, sp):
    """d_model=8, L=1 -> insert_layer makes depth_after=2,
    aspect_after = 8/2 = 4.00; floor 5 -> refuse fires; same
    mode sweep on each host family."""
    s = _sys(tmp_path)
    organ = _mk(s, "h", substrate=sub, sp=sp)
    assert organ.d == 8 and organ.L == 1
    _set(s, "h", gate_aspect_min=5.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    r = s.deepen("h")
    assert "refusal" in r and "G-ASPECT" in r["refusal"], r
    assert "4.00" in r["refusal"] and "5.0" in r["refusal"]
    assert s.lc._load_working("h")[0].L == 1    # untouched
    _set(s, "h", gate_aspect_mode="warn")
    with pytest.warns(UserWarning, match="G-ASPECT"):
        r = s.deepen("h")
    assert "refusal" not in r, r
    assert s.lc._load_working("h")[0].L == 2
    _set(s, "h", gate_aspect_mode="advise")
    r = s.deepen("h")                       # 8/3 < 5, silent
    assert "refusal" not in r and "warning" not in r, r


# ============ T-5 propose mirror (dry-run == reality) =======

def test_t5_propose_mirror_verdict_equals_reality(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages=2
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    rep = s.propose("m", "deepen", {"m": 4})
    g = rep["gates"]["G-ASPECT"]
    assert abs(g["aspect_after"] - 16 / 3) < 1e-12
    assert g["floor"] == 6.0 and g["met"] is False
    assert rep["would_refuse"] is True
    assert "refusal" in s.deepen("m", m=4)      # reality
    # host mirror (insert_layer branch)
    st_ = _sys(tmp_path / "t")
    _mk(st_, "h", substrate="transformer", sp=TR_SP)
    _set(st_, "h", gate_aspect_min=5.0,
         gate_aspect_mode="refuse", aspect_auto="off")
    rep2 = st_.propose("h", "insert_layer", {"position": 1})
    g2 = rep2["gates"]["G-ASPECT"]
    assert g2["aspect_after"] == 4.0 and g2["met"] is False
    assert rep2["would_refuse"] is True
    assert "refusal" in st_.deepen("h")         # reality


# ============ T-6 plan/trial: same seam, no bypass ==========

def test_t6_plan_and_trial_same_seam(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages=2
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    r = s.plan_run("m", {"steps": [{"move": "deepen",
                                    "args": {"m": 4}}]})
    assert "refusal" in r and "G-ASPECT" in r["refusal"], r
    t = s.trial("m", "deepen", {"m": 4}, budget_steps=2,
                examples=ROWS)
    assert "refusal" in t and "G-ASPECT" in t["refusal"], t


# ============ T-7 served doors + warning transport ==========

def test_t7_served_doors_persistence_warn_transport(tmp_path):
    """Facade set_policy routes all four keys; persistence
    across reload (policy.json -> organ install); warn mode
    set through the facade -> the deepen RESPONSE carries the
    G-ASPECT text in response["warning"] (59B 2.4: served
    users cannot see Python warnings). SMS model_policy arm:
    SMS repo (with T-21)."""
    s = _sys(tmp_path)
    _mk(s, "m")
    _set(s, "m", **KEYS4)
    pol = s.lc.policy("m")["growth_params"]
    for k, v in KEYS4.items():
        assert pol[k] == v, k                   # persisted
    o = s.lc._load_working("m")[0]              # reload installs
    for k, v in KEYS4.items():
        assert o._growth_policy[k] == v, k
    s.deepen("m", m=4)                         # 16/2 >= 2.0: ok
    _set(s, "m", gate_aspect_min=6.0)          # now 16/3 < 6.0
    with pytest.warns(UserWarning):
        r = s.deepen("m", m=4)
    assert any("G-ASPECT" in t for t in r.get("warning", [])), r


# ============ T-8 doc guards (key rows) =====================

def test_t8_parameter_reference_rows():
    doc = (REPO / "docs" / "PARAMETER_REFERENCE.md").read_text()
    for k in KEYS4:
        assert k in doc, k


# ============ T-9 scoped deepen judges the GLOBAL shape =====

def test_t9_scoped_deepen_global_shape(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages=2
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    r = s.deepen("m", m=2, scope=[0, 1])    # global 16/3 < 6
    assert "refusal" in r and "G-ASPECT" in r["refusal"], r
    _set(s, "m", gate_aspect_min=2.0)       # global 16/3 >= 2
    with warnings.catch_warnings():         # scope advisory
        warnings.simplefilter("ignore")
        r = s.deepen("m", m=2, scope=[0, 1])
    assert "refusal" not in r, r


# ============ T-10 observability (assess census) ============

def test_t10_assess_aspect_field(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages 2
    a = s.assess("m")
    assert a["census"]["aspect"] == 8.0    # 16/2 EXACT, pre-op
    assert a["census"]["depth_headroom"] is None    # gate off
    json.dumps(a)                           # JSON-safe
    s.deepen("m", m=4)                      # stages 3
    a2 = s.assess("m")
    assert a2["census"]["aspect"] == round(16 / 3, 4)
    st_ = _sys(tmp_path / "t")
    _mk(st_, "h", substrate="transformer", sp=TR_SP)
    st_.deepen("h")                         # L=2
    ah = st_.assess("h")
    assert ah["census"]["aspect"] == 4.0    # 8/2
    json.dumps(ah)


# ============ T-11 headroom invariant (both families) =======

@pytest.mark.parametrize("sub,sp,floor", [
    (None, None, 4.0), ("transformer", TR_SP, 2.0)])
def test_t11_headroom_gate_equivalence(tmp_path, sub, sp,
                                       floor):
    """mlp: floor=4.0, width 16, stages 2 -> headroom =
    16//4 - 2 = 2; host: floor=2.0, width 8 -> 8//2 - 2 = 2.
    Walks 2 -> 1 -> 0 across two admitted deepens (the mlp
    third instant is the 16/4 = 4.0 boundary — ADMITTED); at
    0 the NEXT deepen refuses — the metric<->gate equivalence
    asserted BOTH ways; gate off -> None."""
    s = _sys(tmp_path)
    _mk(s, "m", substrate=sub, sp=sp)
    r0 = s.deepen("m", m=4) if sub is None else s.deepen("m")
    assert "refusal" not in r0, r0              # stages -> 2
    _set(s, "m", gate_aspect_min=floor,
         gate_aspect_mode="refuse", aspect_auto="off")
    walk = []
    for _ in range(2):
        h = s.assess("m")["census"]["depth_headroom"]
        walk.append(h)
        assert h > 0
        r = (s.deepen("m", m=4) if sub is None
             else s.deepen("m"))
        assert "refusal" not in r, r    # headroom>0 -> admitted
    walk.append(s.assess("m")["census"]["depth_headroom"])
    assert walk == [2, 1, 0]
    r = s.deepen("m", m=4) if sub is None else s.deepen("m")
    assert "refusal" in r               # ==0 -> next refuses
    _set(s, "m", gate_aspect_min=0.0)   # off again
    assert s.assess("m")["census"]["depth_headroom"] is None


# ============ T-12 shift-left (plan judged whole) ===========

def test_t12_plan_validate_shift_left(tmp_path):
    """Fresh organ (stages 1), floor 5.0, three deepen steps:
    the CUMULATIVE walk crosses at step index 2 (16/2=8.0 ok,
    16/3~=5.33 ok, 16/4=4.0 < 5.0) -> plan_validate marks
    exactly that step would_refuse BEFORE any run; plan_run
    under refuse mode then halts loudly at the same step
    (validation verdict == run reality)."""
    s = _sys(tmp_path)
    _mk(s, "m")
    _set(s, "m", gate_aspect_min=5.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    plan = {"steps": [{"move": "deepen", "args": {"m": 4}}
                      for _ in range(3)]}
    v = s.plan_validate("m", plan)
    gA = [row["proposal"]["gates"]["G-ASPECT"]
          for row in v["steps"]]
    assert [g["met"] for g in gA] == [True, True, False]
    assert abs(gA[0]["aspect_after"] - 8.0) < 1e-12
    assert abs(gA[1]["aspect_after"] - 16 / 3) < 1e-12
    assert abs(gA[2]["aspect_after"] - 4.0) < 1e-12
    assert [row["proposal"]["would_refuse"]
            for row in v["steps"]] == [False, False, True]
    r = s.plan_run("m", plan)
    assert "refusal" in r and "G-ASPECT" in r["refusal"], r


# ============ T-13 production profile doc guard =============

def test_t13_production_profile_row():
    doc = (REPO / "docs" / "PARAMETER_REFERENCE.md").read_text()
    assert "RECOMMENDED PRODUCTION PROFILE" in doc


# ============ T-14 AUTO widen_first, mlp (hand values) ======

def test_t14_auto_widen_first_mlp(tmp_path):
    """H=16, stages=2, floor=6 -> deepen request -> the
    system AUTO-widens by k = ceil(6*3) - 16 = 2 (width 18),
    THEN deepens; final aspect 18/3 = 6.0 >= floor (boundary
    ADMITS), NO collision;
    both events ledgered (widen with provenance aspect_auto);
    serve BITWISE unchanged across the whole act (omega + the
    D-8 zero-extension + zero-birth delta are all exact)."""
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages 2, H=8
    probe = ROWS[0]["input"]
    pre = s.infer("m", probe, working=True)["output"]
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="widen_first", aspect_auto_max_widen=16)
    r = s.deepen("m", m=4)
    assert "refusal" not in r, r            # NO gate collision
    assert r.get("auto_widened") == 2       # k = ceil(6*3)-16
    o2 = s.lc._load_working("m")[0]
    assert o2.H == 18 and len(o2.blocks) == 2   # 18/3 = 6.0
    post = s.infer("m", probe, working=True)["output"]
    # VALUE-EXACT serve; ADJUDICATED line (60C precedent,
    # recorded in 60D s4): the zero-extension adds only exact
    # +0.0 terms, but a width-changing matmul lets BLAS
    # re-associate the sum — observed exactly ONE ulp at
    # H 16->18 (5.6e-17 at |y|~0.117), ZERO ulp at 16->20
    # (T-19). Line = 4 ulp of |pre|; never quietly tuned.
    assert abs(float(pre) - float(post)) <= \
        4 * np.spacing(abs(float(pre)))
    b0 = o2.blocks[0]                       # D-8 on the OLD block
    assert np.asarray(b0["Bin"]).shape == (4, 18)
    assert np.asarray(b0["Bout"]).shape == (18, 4)
    assert np.all(np.asarray(b0["Bin"])[:, 16:] == 0.0)
    assert np.all(np.asarray(b0["Bout"])[16:, :] == 0.0)
    evs = s.lc.events("m")
    w = [e for e in evs if e["event"] == "widen"
         and e.get("aspect_auto")]
    assert len(w) == 1 and w[0]["k"] == 2   # audited provenance


# ============ T-15 closed-loop plan (P-b pre-widen) =========

def test_t15_plan_pre_widen_closed_loop(tmp_path):
    """floor=4, H=16, stages=2, plan of three deepen steps ->
    final stages 5 need width >= ceil(4x5) = 20 -> ONE
    up-front auto-widen (+4 -> width 20), then the plan runs
    to completion with ZERO refusals; per-step aspects 20/3,
    20/4, 20/5 = 6.67/5.00/4.00 all >= 4 (every instant
    compliant, a fortiori; the last is the boundary —
    ADMITTED)."""
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages 2, H=16
    _set(s, "m", gate_aspect_min=4.0, gate_aspect_mode="refuse",
         aspect_auto="widen_first", aspect_auto_max_widen=16)
    plan = {"steps": [{"move": "deepen", "args": {"m": 4}}
                      for _ in range(3)]}
    r = s.plan_run("m", plan)
    assert "refusal" not in r, r
    assert r.get("auto_widened") == 4       # ceil(4*5) - 16
    assert len(r["events"]) == 3            # ZERO refusals
    o2 = s.lc._load_working("m")[0]
    assert o2.H == 20 and len(o2.blocks) == 4   # 1 + 3
    for stages in (3, 4, 5):                # hand walk
        assert 20 / stages >= 4.0
    evs = s.lc.events("m")
    assert len([e for e in evs if e["event"] == "widen"
                and e.get("aspect_auto")]) == 1     # ONE record


# ============ T-16 hosts defer (named boundary) =============

def test_t16_hosts_defer_named_boundary(tmp_path):
    """No d_model-widen operator exists -> on hosts,
    widen_first behaves as defer with the boundary NAMED: the
    direct verb answers deferred_aspect (not a refusal, no
    mutation); a plan excludes the crossing steps and reports
    deferred_aspect_steps; no crash anywhere."""
    s = _sys(tmp_path)
    _mk(s, "h", substrate="transformer", sp=TR_SP)
    _set(s, "h", gate_aspect_min=5.0, gate_aspect_mode="refuse",
         aspect_auto="widen_first")
    r = s.deepen("h")                       # 8/2 = 4.0 < 5
    assert "refusal" not in r, r
    assert r.get("deferred_aspect") is True
    assert "d_model" in r.get("note", "")
    assert s.lc._load_working("h")[0].L == 1    # untouched
    rp = s.plan_run("h", {"steps": [{"move": "insert_layer",
                                     "args": {"position": 1}}]})
    assert "refusal" not in rp, rp
    assert rp.get("deferred_aspect_steps") == [0]
    assert "d_model" in rp.get("note", "")
    assert len(rp["events"]) == 0
    assert s.lc._load_working("h")[0].L == 1


# ============ T-17 off switch + budget supremacy ============

def test_t17_off_switch_and_budget_supremacy(tmp_path):
    # off: gate-only behavior — collision possible, the
    # user's explicit choice
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    assert "refusal" in s.deepen("m", m=4)  # 16/3 < 6, collide
    # budget supremacy: params budget cap 1x -> the auto-widen
    # cannot fit -> loud refusal, deepen NOT attempted, organ
    # bitwise untouched (budgets are NEVER bypassed)
    s2 = _sys(tmp_path / "b")
    _mk(s2, "b", extra_pol={"max_params_mult": 1})
    _set(s2, "b", gate_aspect_min=9.0,
         gate_aspect_mode="refuse", aspect_auto="widen_first")
    o_pre = pickle.dumps(
        s2.lc._load_working("b")[0].__getstate__())
    r2 = s2.deepen("b", m=4)               # 16/2=8<9 -> widen
    assert "refusal" in r2 and "budget" in r2["refusal"], r2
    o_post = pickle.dumps(
        s2.lc._load_working("b")[0].__getstate__())
    assert o_pre == o_post                  # untouched


# ============ T-18 C-5 audit + manual precedence ============

def test_t18_manual_precedence_and_audit(tmp_path):
    s = _sys(tmp_path)
    _mk(s, "m")
    # floor unset: manual verbs exactly as without the feature
    # (T-2 is the byte judge of "exactly"; here: no refusal,
    # no automation fields)
    r = s.widen("m", k=2)
    assert isinstance(r, dict) and "refusal" not in r
    r = s.deepen("m", m=4)
    assert "refusal" not in r and "auto_widened" not in r
    # automation on: a DIRECT widen stays the user's own act —
    # never blocked, never re-attributed
    _set(s, "m", gate_aspect_min=2.0, aspect_auto="widen_first")
    r = s.widen("m", k=2)
    assert "refusal" not in r
    evs = s.lc.events("m")
    manual_w = [e for e in evs if e["event"] == "widen"
                and not e.get("aspect_auto")]
    assert len(manual_w) == 2               # both stay manual


# ============ T-19 D-8 exactness (blocks + loop) ============

def test_t19_d8_widen_with_blocks_and_loop_bitwise(tmp_path):
    """Organ with 2 TRAINED blocks + a loop, widen k=4 ->
    serve BITWISE unchanged; shapes hand-asserted: Bin m x 20
    / Bout 20 x m with ZEROS exactly in the new stripes; loop
    L_in/L_out likewise. (RED today: widen_net refuses looped
    scopes — the lifted boundary.)"""
    s = _sys(tmp_path)
    _mk(s, "m")
    _set(s, "m", loop_enabled=True)
    s.deepen("m", m=4)
    s.deepen("m", m=2)
    s.study("m", ROWS, steps=20)            # train the blocks
    r = s.loop("m")
    assert "refusal" not in r, r
    probe = ROWS[0]["input"]
    pre = s.infer("m", probe, working=True)["output"]
    r = s.widen("m", k=4)                   # the D-8 arm
    assert isinstance(r, dict) and "refusal" not in r, r
    o = s.lc._load_working("m")[0]
    assert o.H == 20
    b0, b1 = o.blocks[0], o.blocks[1]
    assert np.asarray(b0["Bin"]).shape == (4, 20)
    assert np.asarray(b0["Bout"]).shape == (20, 4)
    assert np.asarray(b1["Bin"]).shape == (2, 20)
    assert np.asarray(b1["Bout"]).shape == (20, 2)
    assert np.all(np.asarray(b0["Bin"])[:, 16:] == 0.0)
    assert np.all(np.asarray(b0["Bout"])[16:, :] == 0.0)
    assert np.all(np.asarray(b1["Bin"])[:, 16:] == 0.0)
    assert np.all(np.asarray(b1["Bout"])[16:, :] == 0.0)
    lb = o.loop_block
    assert np.asarray(lb["L_in"]).shape[1] == 20
    assert np.asarray(lb["L_out"]).shape[0] == 20
    assert np.all(np.asarray(lb["L_in"])[:, 16:] == 0.0)
    assert np.all(np.asarray(lb["L_out"])[16:, :] == 0.0)
    post = s.infer("m", probe, working=True)["output"]
    # same adjudicated 4-ulp line as T-14 (portability across
    # BLAS builds); observed ZERO ulp here (16 -> 20)
    assert abs(float(pre) - float(post)) <= \
        4 * np.spacing(abs(float(pre)))


# ============ T-20 D-8 trains after extension ===============

def test_t20_d8_trains_after_extension(tmp_path):
    """Finite-difference gradient truth on the EXTENDED block
    (the deepen-backward FD judge verbatim: grads vs 0.5*FD,
    rel < 1e-5) and a training run moves the new stripes off
    zero (total plasticity — nothing frozen)."""
    import test_deepen_backward as tdb
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)
    s.study("m", ROWS, steps=20)
    r = s.widen("m", k=4)                   # extend to H=20
    assert isinstance(r, dict) and "refusal" not in r, r
    o = s.lc._load_working("m")[0]
    assert np.all(
        np.asarray(o.blocks[0]["Bin"])[:, 16:] == 0.0)
    X, y = _xy()
    Xs = o._std_x(X)
    ys = (y - o._y_mu) / o._y_sd
    g, _ = tdb._grad_map(o, Xs, ys)
    blk = o.blocks[0]
    assert tdb._rel(g["Bin0"],
                    0.5 * tdb._fd(o, Xs, ys, blk["Bin"])) < 1e-5
    assert tdb._rel(g["Bout0"],
                    0.5 * tdb._fd(o, Xs, ys, blk["Bout"])) < 1e-5
    s.study("m", ROWS, steps=40)            # train
    o2 = s.lc._load_working("m")[0]
    stripe = np.asarray(o2.blocks[0]["Bin"])[:, 16:]
    assert not np.all(stripe == 0.0)        # moved off zero


# ============ T-23..T-27: review-pass-3 coverage fill ========
# (60D execution record C-1..C-5: obligations of already-
# specified behaviors that lacked an owning box)

def test_t23_defer_mode_on_network_family(tmp_path):
    """C-1: aspect_auto="defer" on the NETWORK family (T-16
    covered hosts only). Verb: clean deferred_aspect answer,
    organ untouched. Plan: BOTH crossing steps excluded."""
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                      # stages 2, H=16
    _set(s, "m", gate_aspect_min=6.0, gate_aspect_mode="refuse",
         aspect_auto="defer")
    o_pre = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    r = s.deepen("m", m=4)                  # 16/3 = 5.33 < 6
    assert "refusal" not in r, r
    assert r.get("deferred_aspect") is True
    o_post = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    assert o_pre == o_post                  # untouched
    rp = s.plan_run("m", {"steps": [
        {"move": "deepen", "args": {"m": 4}},
        {"move": "deepen", "args": {"m": 4}}]})
    assert "refusal" not in rp, rp
    assert rp.get("deferred_aspect_steps") == [0, 1]
    assert len(rp["events"]) == 0
    assert len(s.lc._load_working("m")[0].blocks) == 1


def test_t24_auto_widen_cap_refusal(tmp_path):
    """C-2: a deficit ABOVE aspect_auto_max_widen refuses
    loudly NAMING the key, before any mutation (distinct from
    T-17's params-budget arm). floor 20, stages 2 -> k =
    ceil(20*3) - 16 = 44 > cap 8."""
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)
    _set(s, "m", gate_aspect_min=20.0,
         gate_aspect_mode="refuse", aspect_auto="widen_first",
         aspect_auto_max_widen=8)
    o_pre = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    r = s.deepen("m", m=4)
    assert "refusal" in r, r
    assert "aspect_auto_max_widen" in r["refusal"]
    assert "44" in r["refusal"]              # the hand k
    o_post = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    assert o_pre == o_post                   # untouched


def test_t25_refusal_leaves_no_junk_snapshot():
    """C-3: the pre-snapshot placement's own promise — a
    REFUSED deepen takes no auto-snapshot (D-4: 'a refusal
    must not leave a junk auto-snapshot'). Net-level: the
    in-memory ring length is unchanged across the refusal."""
    from reference_net.net import Network
    from reference_net.growthpolicy import \
        DEFAULT_GROWTH_POLICY
    net = Network(3, 8, lr=1e-2, seed=7)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              gate_aspect_min=99.0,
                              gate_aspect_mode="refuse")
    n0 = len(getattr(net, "_snapshots", []))
    with pytest.raises(ValueError, match="G-ASPECT"):
        net.deepen(m=4)
    assert len(getattr(net, "_snapshots", [])) == n0
    assert len(net.blocks) == 0              # no mutation
    # and an ADMITTED deepen still snapshots (the placement
    # moved the gate, not the snapshot)
    net._growth_policy["gate_aspect_min"] = 1.0
    net.deepen(m=4)
    assert len(net._snapshots) == n0 + 1


def test_t26_loop_excluded_from_depth(tmp_path):
    """C-4: loop blocks are iteration, not a static stage —
    EXCLUDED from depth (D-2). With 2 blocks + a loop
    (H=16): census aspect == 16/3 (stages 3, loop ignored);
    a refusing deepen names depth 4 (3+1), not 5."""
    s = _sys(tmp_path)
    _mk(s, "m")
    _set(s, "m", loop_enabled=True)
    s.deepen("m", m=4)
    s.deepen("m", m=2)
    s.study("m", ROWS, steps=20)
    r = s.loop("m")
    assert "refusal" not in r, r
    a = s.assess("m")["census"]["aspect"]
    assert a == round(16 / 3, 4)             # loop ignored
    _set(s, "m", gate_aspect_min=5.0, gate_aspect_mode="refuse",
         aspect_auto="off")
    r = s.deepen("m", m=2)                   # 16/4 = 4.0 < 5
    assert "refusal" in r and "depth 4" in r["refusal"], r


def test_t27_headroom_clamps_at_zero(tmp_path):
    """C-5: floor above width//stages never serves a negative
    headroom. width 16, stages 2, floor 9 -> raw 16//9 - 2 =
    -1 -> served 0."""
    s = _sys(tmp_path)
    _mk(s, "m")
    s.deepen("m", m=4)                       # stages 2
    _set(s, "m", gate_aspect_min=9.0, aspect_auto="off")
    assert s.assess("m")["census"]["depth_headroom"] == 0


def test_t28_act_atomicity_widen_fits_deepen_does_not(tmp_path):
    """F-4 (review pass 5): the widen+deepen ACT is atomic.
    cap 162 (mult=2 on 81); floor 8.5, H=16, stages 1 ->
    deficit k=1; the widen ALONE fits (86 <= 162) but the
    act does not (deepen cost at H=17 is 4*17+4+17*4 = 140;
    86+140 = 226 > 162) -> ONE loud refusal BEFORE any
    mutation: no widen event in the log, refusal reports the
    TRUE persisted params (81), organ bitwise untouched;
    the only new log row is the deepen refusal."""
    s = _sys(tmp_path)
    _mk(s, "m", extra_pol={"max_params_mult": 2})
    _set(s, "m", gate_aspect_min=8.5, gate_aspect_mode="refuse",
         aspect_auto="widen_first")
    o_pre = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    r = s.deepen("m", m=4)
    assert "refusal" in r and "budget" in r["refusal"], r
    assert r.get("params") == 81             # persisted truth
    o_post = pickle.dumps(
        s.lc._load_working("m")[0].__getstate__())
    assert o_pre == o_post                   # untouched
    evs = s.lc.events("m")
    assert [e for e in evs if e["event"] == "widen"] == []
    assert evs[-1]["event"] == "deepen_refused"
