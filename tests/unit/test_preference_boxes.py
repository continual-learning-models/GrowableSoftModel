"""TB-01..TB-16 — GrowthPreference unit referee boxes
(doc 83 v1.16 §8.1 + §4.8; plan 84 v2.13 D-2, RED before D-3).

Interface under test (doc 83 §4.2 shapes; importables fixed here):
  reference_net.growthpolicy.preference:
    GrowthPreference(policy: dict)      # merged over defaults
      .score(ctx) -> float              # single ctx => 1.0 edge
      .score_set(ctxs) -> list[float]   # the decision-set call
      .credit(ev) -> None               # PreferenceAuditError on
                                        #   governance-derived ev
      .explore_offer(ctx, raw_eff) -> bool
      .snapshot() -> blob / .restore(blob) -> {"ok"|"refusal"}
      .rebuild(credit_events) -> None
      .inspect() -> dict
      .on_changepoint(bucket, evidence) -> None
      .audit_events: list[dict]         # drained by caller
    PreferenceAuditError, PREFERENCE_DEFAULTS,
    validate_preference_policy(policy) -> None | refusal dict,
    bucket_of(ctx, policy) -> str,
    migrate_snapshot(blob) -> blob | refusal dict

Every expected number is HAND-DERIVED from doc 83 M1/M4/M5
normative equations (worksheets in docstrings). T-AX3 second
routes: rebuild identity (TB-13), two-process replay (TP-01),
byte-identity at system scale (TI-02).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules" / "ReferenceNet"))

DECAY = 0.98


def _mod():
    from reference_net.growthpolicy import preference
    return preference


def _part(**over):
    m = _mod()
    pol = {"seed": 0}
    pol.update(over)
    return m.GrowthPreference(pol)


def _ev(bucket, advantage, **over):
    ev = {"event_id": f"e-{bucket}-{advantage}", "bucket": bucket,
          "move": bucket.split("|")[0], "batch": 0,
          "quoted_gain": 0.0, "window_gains": [advantage],
          "credited_gain": advantage, "advantage": advantage}
    ev.update(over)
    return ev


# ---------------------------------------------------------- TB-01
def test_tb01_ema_fold_hand_computed_two_buckets():
    """TB-01: normative fold (doc 83 M1) at decay g=0.98,
    6 events, 2 buckets.
    WORKSHEET bucket A, advantages [0.40, 0.10, -0.20]:
      e1: w=1, m=0.4, v=0
      e2: w=1.98; m=(0.98*0.4+0.1)/1.98=0.492/1.98=0.2484848485
          M2=(0.98*0.16+0.01)/1.98=0.1668/1.98=0.0842424242
          v=0.0842424242-0.2484848485^2=0.0224977043
      e3: w=2.9404
          m=(1.9404*0.2484848485-0.2)/2.9404=0.0959597333696096
          M2=(1.9404*0.0842424242+0.04)/2.9404
          v=M2-m^2=0.0599877573229598
    WORKSHEET bucket B, advantages [0.30, 0.30, 0.00]:
      end state: w=2.9404, m=0.1979730648891307,
                 v=0.0201985850451433
    """
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.decay": DECAY})
    for a in (0.40, 0.10, -0.20):
        p.credit(_ev("grow|s1", a))
    for a in (0.30, 0.30, 0.00):
        p.credit(_ev("deepen|s1", a))
    st = p.inspect()["stats"]
    A, B = st["grow|s1"], st["deepen|s1"]
    assert abs(A["w"] - 2.9404) < 1e-12
    assert abs(A["m"] - 0.0959597333696096) < 1e-12
    assert abs(A["v"] - 0.0599877573229598) < 1e-12
    assert A["n_raw"] == 3
    assert abs(B["w"] - 2.9404) < 1e-12
    assert abs(B["m"] - 0.1979730648891307) < 1e-12
    assert abs(B["v"] - 0.0201985850451433) < 1e-12


# ---------------------------------------------------------- TB-02
def test_tb02_credit_window_fold_and_advantage():
    """TB-02: RewardCrediting arithmetic (doc 83 M3 v1.17 —
    CUMULATIVE convention, dimensional law: every window gain
    measured against the FIXED pre-adoption base, so each is
    total-gain-scale and commensurable with the quote).
    WORKSHEET: E_before=1.00, window energies [0.90,0.85,0.83]
      => gains g_i = E_before - E_i = [0.10, 0.15, 0.17]
      weights (1, .5, .25):
      credited = (0.10 + 0.075 + 0.0425)/1.75 = 0.2175/1.75
               = 0.1242857142857143
      quoted_gain = 0.06
      advantage = 0.0642857142857143
    """
    m = _mod()
    ev = m.assemble_credit_event(
        event_id="ev-1", bucket="grow|s1", move="grow", batch=7,
        quoted_gain=0.06, window_gains=[0.10, 0.15, 0.17],
        weights=[1, 0.5, 0.25])
    assert abs(ev["credited_gain"] - 0.1242857142857143) < 1e-12
    assert abs(ev["advantage"] - 0.0642857142857143) < 1e-12


# ---------------------------------------------------------- TB-03
def test_tb03_mean_clip_normalization_and_saturation():
    """TB-03 (v1.17): mean_clip through the REFERENCE-
    DISTRIBUTION envelope. Credits 0.1 then 0.7 build the event
    stats by the stable fold at decay 0.98:
      e1: n=1, w=1, m=0.1, v=0
      e2: w=1.98; d=0.6; m=0.1+0.6/1.98=0.4030303030
          v=(0.6*(0.7-0.4030303))/1.98=0.0899908...
          sd=0.2999846950
    WORKSHEET multipliers (z=(m_b-mu_ev)/sd_ev):
      grow   m=0.1: z=-1.0101525 -> exp=0.3641634 -> clip 0.5
      deepen m=0.7: z=+0.9899495 -> exp=2.6910986 -> clip 2.0
    GRADED referee (the magnitude-awareness the old set-z could
    not express): a bucket at m=0.45 -> z=+0.1565736 ->
    mult = 1.1694968861 (NOT saturated).
    """
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.min_count": 1})
    p.credit(_ev("grow|s0", 0.1))
    p.credit(_ev("deepen|s0", 0.7))
    got = p.score_set([{"move": "grow", "slope": -0.5},
                       {"move": "deepen", "slope": -0.5}])
    assert np.allclose(got, [0.5, 2.0], atol=1e-12)
    p._force_stats("refine|s0", w=3.0, m=0.45, v=0.0, n_raw=3)
    graded = p.score({"move": "refine", "slope": -0.5})
    assert abs(graded - 1.1694968861) < 1e-9


def test_tb03b_degenerate_normalization_edge_exact_one():
    """TB-03 edge (v1.17 tolerance law): a zero-dispersion
    advantage history (all credits equal => sd_ev=0 within
    tolerance) carries no reference scale => every multiplier
    is EXACTLY 1.0; likewise with fewer than 2 events."""
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.min_count": 1})
    p.credit(_ev("grow|s0", 0.3))
    p.credit(_ev("deepen|s0", 0.3))
    assert p.score({"move": "grow", "slope": -0.5}) == 1.0
    got = p.score_set([{"move": "grow", "slope": -0.5},
                       {"move": "deepen", "slope": -0.5}])
    assert got == [1.0, 1.0]
    q = _part(**{"preference.rule": "mean_clip",
                 "preference.min_count": 1})
    q.credit(_ev("grow|s0", 0.9))      # single event: n_ev=1
    assert q.score({"move": "grow", "slope": -0.5}) == 1.0


# ---------------------------------------------------------- TB-04
def test_tb04_thompson_seeded_draw_sequence():
    """TB-04: discounted TS draws (doc 83 M4 normative forms).
    WORKSHEET (v1.19 production fixed-scale form): credits
    (0.4,0.3,-0.1,0.2) at decay 0.98 give reference variance
    v_ev = 0.0348896359715334 (mpmath-50 referee:
    0.0348896359715333863...). Bucket forced to w=4:
      se = sqrt(v_ev/(w+1)) = sqrt(v_ev/5)
         = 0.0835339882581137
    (mpmath-50: 0.08353398825811369859...) — the per-bucket
    empirical variance v NEVER enters the scale (production
    form: TF-Agents / SMPyBandits / BanditPyLib; measured
    101.3 vs MABWiser 117.1 on the 20-arm benchmark), so the
    v=0.09 and v=0 cases draw with the IDENTICAL se — the
    lock-out trap (a lucky-coinciding arm drawing a dust
    scale) is structurally impossible. draws draw_k = m +
    se * z_k, z_k the generator's standard normals (stream
    seed 0+10000) — replayed independently. raw = m only
    when the REFERENCE distribution is degenerate (v_ev=0).
    AUDIT-FIELD LAW (v1.19 follow-up, tests-first): the
    logged draw event's "se" field IS the actual sampling
    scale used by the rule — here sqrt(v_ev/5) =
    0.0835339882581137 — NOT the bucket-empirical sqrt(v/w)
    (= 0.15 for the forced v=0.09 case; a checker computing
    mean + se*z from the log must reproduce the draw).
    """
    p = _part(**{"preference.rule": "thompson",
                 "preference.min_count": 1})
    for a in (0.4, 0.3, -0.1, 0.2):     # builds v_ev
        p.credit(_ev("grow|s1", a))
    p._force_stats("grow|s1", w=4.0, m=0.2, v=0.09, n_raw=4)
    se = 0.0835339882581137
    ref = np.random.default_rng(0 + 10000)
    want = [0.2 + se * ref.standard_normal() for _ in range(3)]
    got = []
    for _ in range(3):
        p.score({"move": "grow", "slope": 0.0})
        last = [e for e in p.audit_events
                if e["kind"] == "preference_draw"][-1]
        got.append(last["draw"])
        assert abs(last["se"] - se) < 1e-12   # audit-field law
    assert np.allclose(got, want, atol=1e-12)
    p._force_stats("grow|s1", w=4.0, m=0.25, v=0.0, n_raw=4)
    p.score({"move": "grow", "slope": 0.0})
    last = [e for e in p.audit_events
            if e["kind"] == "preference_draw"][-1]
    want2 = 0.25 + se * ref.standard_normal()   # SAME se: v-free
    assert abs(last["draw"] - want2) < 1e-12   # NOT locked to m
    q = _part(**{"preference.rule": "thompson",
                 "preference.min_count": 1})
    q._force_stats("grow|s1", w=2.0, m=0.3, v=0.0, n_raw=2)
    q.score({"move": "grow", "slope": 0.0})    # v_ev=0 (no credits)
    e0 = [e for e in q.audit_events
          if e["kind"] == "preference_draw"][-1]
    assert e0["draw"] == 0.3        # degenerate reference => raw=m


# ---------------------------------------------------------- TB-05
def test_tb05_ucb_index_including_small_w():
    """TB-05: ucb raw = m + c*sqrt(v/w) (doc 83 M4).
    WORKSHEET: m=0.2, v=0.09, w=4, c=1 => 0.2+0.15 = 0.35
               m=0.2, v=0.04, w=0.5, c=1 => 0.2+sqrt(0.08)
                 = 0.4828427124746190
    Referee reads the logged raw from the draw event.
    """
    p = _part(**{"preference.rule": "ucb",
                 "preference.min_count": 1,
                 "preference.ucb_c": 1.0})
    p.credit(_ev("grow|s1", 0.1))
    p._force_stats("grow|s1", w=4.0, m=0.2, v=0.09, n_raw=4)
    p.score({"move": "grow", "slope": 0.0})
    e1 = [e for e in p.audit_events
          if e["kind"] == "preference_draw"][-1]
    assert abs(e1["draw"] - 0.35) < 1e-12
    p._force_stats("grow|s1", w=0.5, m=0.2, v=0.04, n_raw=4)
    p.score({"move": "grow", "slope": 0.0})
    e2 = [e for e in p.audit_events
          if e["kind"] == "preference_draw"][-1]
    assert abs(e2["draw"] - 0.4828427124746190) < 1e-12


# ---------------------------------------------------------- TB-06
def test_tb06_eps_greedy_hit_and_miss_branches():
    """TB-06 (v1.17): eps rule in GAIN UNITS — u < eps =>
    raw ~ Normal(mu_ev, sd_ev) (unit-correct: an unscaled
    N(0,1) among gain-unit raws would be a unit error), logged
    rule=eps_random; degenerate event stats => raw = m.
    WORKSHEET: credits 0.2 then 0.4 (stable fold, decay .98):
      mu_ev = 0.3010101010, sd_ev = 0.0999948983.
    eps=1.0 hit: raw = mu_ev + sd_ev * z1 with z1 the stream's
    next standard normal after the uniform; replayed
    independently. eps=0.0 miss: raw = m.
    """
    p = _part(**{"preference.rule": "eps_greedy",
                 "preference.min_count": 1,
                 "preference.eps": 1.0})
    p.credit(_ev("grow|s1", 0.2))
    p.credit(_ev("grow|s1", 0.4))
    mu_ev, sd_ev = p._ref_stats()
    assert abs(mu_ev - 0.3010101010101010) < 1e-12
    assert abs(sd_ev - 0.0999948983496128) < 1e-12
    ref = np.random.default_rng(10000)
    ref.uniform()
    want_raw = mu_ev + sd_ev * ref.standard_normal()
    p.score({"move": "grow", "slope": 0.0})
    e = [e for e in p.audit_events
         if e["kind"] == "preference_draw"][-1]
    assert e["rule"] == "eps_random"
    assert abs(e["draw"] - want_raw) < 1e-12
    q = _part(**{"preference.rule": "eps_greedy",
                 "preference.min_count": 1,
                 "preference.eps": 0.0})
    q.credit(_ev("grow|s1", 0.2))
    q.credit(_ev("grow|s1", 0.4))
    q.score({"move": "grow", "slope": 0.0})
    e2 = [e for e in q.audit_events
          if e["kind"] == "preference_draw"][-1]
    assert e2["rule"] == "eps_greedy"
    assert abs(e2["draw"] - 0.3010101010101010) < 1e-12


# ---------------------------------------------------------- TB-07
def test_tb07_rule_mix_single_blend_point():
    """TB-07: rule_mix blends RAW scores BEFORE the shared
    normalization (doc 83 M4 composition contract).
    WORKSHEET: mix {mean_clip: 0.5, ucb: 0.5} on bucket m=0.2,
    v=0.09, w=4, c=1: raw = 0.5*0.2 + 0.5*0.35 = 0.275.
    """
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.rule_mix": {"mean_clip": 0.5,
                                         "ucb": 0.5},
                 "preference.min_count": 1,
                 "preference.ucb_c": 1.0})
    p.credit(_ev("grow|s1", 0.1))
    p._force_stats("grow|s1", w=4.0, m=0.2, v=0.09, n_raw=4)
    p.score({"move": "grow", "slope": 0.0})
    e = [e for e in p.audit_events
         if e["kind"] == "preference_draw"][-1]
    assert abs(e["draw"] - 0.275) < 1e-12


# ---------------------------------------------------------- TB-08
def test_tb08_bucketer_cut_boundaries_and_fallback():
    """TB-08: bucketer (doc 83 M2). b0: bucket = move. b1 bands
    via bisect_right over preference.slope_cuts [-0.01, 0.01]:
      slope -0.02 -> grow|s0 ; -0.01 -> grow|s1 (boundary right-
      closed into the upper band); 0.005 -> grow|s1; 0.01 ->
      grow|s2; 0.02 -> grow|s2.
    slope=None under b1 => b0-form bucket + AUDITED fallback.
    b2 is RESERVED (doc 83 M2): validate refuses it.
    """
    m = _mod()
    b0pol = {"preference.bucket_spec": "b0"}
    assert m.bucket_of({"move": "grow", "slope": 0.5},
                       b0pol) == "grow"
    b1pol = {"preference.bucket_spec": "b1",
             "preference.slope_cuts": [-0.01, 0.01]}
    for slope, want in ((-0.02, "grow|s0"), (-0.01, "grow|s1"),
                        (0.005, "grow|s1"), (0.01, "grow|s2"),
                        (0.02, "grow|s2")):
        assert m.bucket_of({"move": "grow", "slope": slope},
                           b1pol) == want, slope
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.bucket_spec": "b1",
                 "preference.slope_cuts": [-0.01, 0.01]})
    p.score({"move": "grow", "slope": None})
    kinds = [e["kind"] for e in p.audit_events]
    assert "b1_fallback" in kinds
    r = m.validate_preference_policy(
        {"preference.bucket_spec": "b2"})
    assert r and "refusal" in r and "reserved" in r["refusal"]


# ---------------------------------------------------------- TB-09
def test_tb09_min_count_floor_and_clip_adversarial():
    """TB-09 (v1.17): floor law + INDEPENDENCE — a floored
    bucket returns exactly 1.0 and (reference normalization)
    has ZERO effect on any other candidate's multiplier.
    WORKSHEET: event stats forced to mu=0.1, sd=0.1
    (n=5, w=4, v=0.01):
      floored grow (n_raw=2 < 3), m=99  -> 1.0 exactly
      deepen m=0.1 -> z=0 -> 1.0
      deepen unchanged whether grow is floored or not
      (independence — the old set-z coupled them).
    Then grow matured (n_raw=3): z=(99-0.1)/0.1 huge -> 2.0;
    a bucket at m=-99 -> 0.5; graded m=0.13 -> z=0.3 ->
    exp = 1.3498588076.
    """
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.min_count": 3})
    p._force_event_stats(5, 4.0, 0.1, 0.01)
    p._force_stats("grow|s1", w=2.0, m=99.0, v=0.0, n_raw=2)
    p._force_stats("deepen|s1", w=5.0, m=0.1, v=0.0, n_raw=5)
    got = p.score_set([{"move": "grow", "slope": 0.0},
                       {"move": "deepen", "slope": 0.0}])
    assert got[0] == 1.0                 # floored exactly
    assert got[1] == 1.0                 # z=0 -> 1.0, INDEPENDENT
    p._force_stats("grow|s1", w=3.0, m=99.0, v=0.0, n_raw=3)
    got2 = p.score_set([{"move": "grow", "slope": 0.0},
                        {"move": "deepen", "slope": 0.0}])
    assert got2 == [2.0, 1.0]            # saturated hi; other
    p._force_stats("refine|s1", w=3.0,   # candidate UNTOUCHED
                   m=-99.0, v=0.0, n_raw=3)
    assert p.score({"move": "refine", "slope": 0.0}) == 0.5
    p._force_stats("refine|s1", w=3.0, m=0.13, v=0.0, n_raw=3)
    assert abs(p.score({"move": "refine", "slope": 0.0})
               - 1.3498588075760032) < 1e-12


# ---------------------------------------------------------- TB-10
def test_tb10_explore_quota_accounting_and_exhaustion():
    """TB-10 (v1.17; draw form v1.19): quota with the NUMERIC
    favorability law — favorable iff the bucket draw's
    normalized multiplier > explore_draw_min (default 1.2).
    WORKSHEET: event stats mu=0, sd=0.1, v_ev=0.01; bucket
    m=0.5, w=5 => draw = 0.5 + sqrt(0.01/6)*z_k
    (= 0.5 + 0.0408248290463863*z_k, seed-0 stream:
    z_1=0.1257302, z_2=-0.1321049) => z_norm = 5.051/4.946
    => mult saturates 2.0 > 1.2 favorable BOTH draws (any
    |z_k| < 11.8 is favorable here). First offer True, second
    False (quota=1 exhausted); every call audited with
    quota_left."""
    p = _part(**{"preference.rule": "thompson",
                 "preference.min_count": 1,
                 "preference.explore_quota": 1})
    p._force_event_stats(5, 4.0, 0.0, 0.01)
    p._force_stats("grow|s1", w=5.0, m=0.5, v=0.0, n_raw=5)
    ctx = {"move": "grow", "slope": 0.0}
    assert p.explore_offer(ctx, raw_eff=-0.2) is True
    assert p.explore_offer(ctx, raw_eff=-0.2) is False
    evs = [e for e in p.audit_events
           if e["kind"] == "explore_activation"]
    assert len(evs) == 2
    assert evs[0]["quota_left"] == 0
    assert evs[1]["quota_left"] == 0
    assert evs[0]["granted"] is True      # audit-field pinning
    assert evs[1]["granted"] is False     # (quota exhausted)
    assert evs[0]["mult"] == 2.0          # saturated per worksheet
    assert evs[1]["mult"] == 2.0


# ---------------------------------------------------------- TB-11
def test_tb11_audit_only_law_governance_refused():
    """TB-11: the audit-only law (doc 83 §2.4): a CreditEvent
    derived from a governance ledger entry raises
    PreferenceAuditError and does NOT touch statistics."""
    m = _mod()
    p = _part(**{"preference.rule": "mean_clip"})
    bad = _ev("grow|s1", 0.5, source_kind="governance")
    with pytest.raises(m.PreferenceAuditError):
        p.credit(bad)
    assert p.inspect()["stats"] == {}
    bad2 = _ev("grow|s1", 0.5)
    bad2["credited_gain"] = None          # gain=None convention
    with pytest.raises(m.PreferenceAuditError):
        p.credit(bad2)
    assert p.inspect()["stats"] == {}


# ---------------------------------------------------------- TB-12
def test_tb12_snapshot_restore_roundtrip_all_fields():
    """TB-12: snapshot blob (doc 83 §4.1 PreferenceState) round-
    trips exactly: schema, rule, bucket_spec, stats, quota_used,
    rng_state (full generator state), fold_cursor, pending."""
    p = _part(**{"preference.rule": "thompson",
                 "preference.min_count": 1})
    for a in (0.4, -0.1):
        p.credit(_ev("grow|s1", a))
    p.score({"move": "grow", "slope": 0.0})    # advance rng
    blob = p.snapshot()
    assert blob["schema"] == "pref-v1"
    q = _part(**{"preference.rule": "thompson",
                 "preference.min_count": 1})
    r = q.restore(blob)
    assert r.get("ok") is True
    assert q.snapshot() == blob
    a = p.score({"move": "grow", "slope": 0.0})
    b = q.score({"move": "grow", "slope": 0.0})
    assert a == b                               # identical continuation
    da = [e for e in p.audit_events
          if e["kind"] == "preference_draw"][-1]["draw"]
    db = [e for e in q.audit_events
          if e["kind"] == "preference_draw"][-1]["draw"]
    assert da == db


# ---------------------------------------------------------- TB-13
def test_tb13_rebuild_equals_incremental_fold_property():
    """TB-13: replay invariant (doc 83 M1/§4.2):
    snapshot() == fold(credit_events) — rebuild from the event
    stream equals the incrementally folded table, over seeded
    randomized sequences (the ema_fold second method, T-AX3)."""
    rng = np.random.default_rng(42)
    for trial in range(5):
        p = _part(**{"preference.rule": "mean_clip"})
        events = []
        for i in range(int(rng.integers(3, 30))):
            b = ["grow|s0", "grow|s1", "deepen|s1"][
                int(rng.integers(0, 3))]
            ev = _ev(b, float(rng.normal(0, 0.3)))
            ev["event_id"] = f"t{trial}-e{i}"
            events.append(ev)
            p.credit(ev)
        q = _part(**{"preference.rule": "mean_clip"})
        q.rebuild(events)
        sp, sq = p.inspect()["stats"], q.inspect()["stats"]
        assert sp.keys() == sq.keys()
        for k in sp:
            for f in ("w", "m", "v"):
                assert abs(sp[k][f] - sq[k][f]) < 1e-12, (k, f)
            assert sp[k]["n_raw"] == sq[k]["n_raw"]
        assert q.snapshot()["fold_cursor"] == len(events)
        assert p.snapshot()["event_stats"] == \
            q.snapshot()["event_stats"]


# ---------------------------------------------------------- TB-14
def test_tb14_refusal_semantics():
    """TB-14: doc 83 §4.5 — unknown rule/bucket_spec refused at
    validate time (never silent fallback); corrupt or unknown-
    schema blob refused at restore (never silently zero)."""
    m = _mod()
    r1 = m.validate_preference_policy(
        {"preference.rule": "quantum_leap"})
    assert r1 and "refusal" in r1
    r2 = m.validate_preference_policy(
        {"preference.bucket_spec": "b7"})
    assert r2 and "refusal" in r2
    assert m.validate_preference_policy(
        {"preference.rule": "thompson"}) is None
    p = _part()
    r3 = p.restore({"schema": "pref-v9", "stats": {}})
    assert "refusal" in r3
    r4 = p.restore({"schema": "pref-v1", "stats": "garbage"})
    assert "refusal" in r4
    assert p.inspect()["stats"] == {}     # untouched, not zeroed-in


# ---------------------------------------------------------- TB-15
def test_tb15_bocpd_deepening_rescale_referee():
    """TB-15: M5 deepening action (doc 83 v1.14 normative):
    on a detected break, w <- rho*w with rho =
    preference.bocpd_depth; m and v UNCHANGED; audited.
    WORKSHEET: w=2.5, rho=0.5 => w'=1.25 exactly; m=0.3, v=0.02
    unchanged to 1e-12. Coupling off => no-op, no event."""
    p = _part(**{"preference.rule": "mean_clip",
                 "preference.bocpd_coupling": True,
                 "preference.bocpd_depth": 0.5})
    p._force_stats("grow|s1", w=2.5, m=0.3, v=0.02, n_raw=4)
    p.on_changepoint("grow|s1", {"run_length_drop": 0.9})
    st = p.inspect()["stats"]["grow|s1"]
    assert abs(st["w"] - 1.25) < 1e-12
    assert abs(st["m"] - 0.3) < 1e-12
    assert abs(st["v"] - 0.02) < 1e-12
    assert [e for e in p.audit_events
            if e["kind"] == "bocpd_deepen"]
    q = _part(**{"preference.rule": "mean_clip",
                 "preference.bocpd_coupling": False})
    q._force_stats("grow|s1", w=2.5, m=0.3, v=0.02, n_raw=4)
    q.on_changepoint("grow|s1", {"run_length_drop": 0.9})
    assert q.inspect()["stats"]["grow|s1"]["w"] == 2.5
    assert not [e for e in q.audit_events
                if e["kind"] == "bocpd_deepen"]


# ---------------------------------------------------------- TB-16
def test_tb16_schema_migration_and_sampled_audit_replay():
    """TB-16: P-1 migration referee + P-2 sampled-audit replay.
    (a) hand-built pref-v1 blob loads unchanged (same-version);
    (b) unknown pref-v9 => refusal, then the rebuild path
        reconstructs state == fold(events);
    (c) audit_draw_mode=sampled:2 halves preference_draw volume
        (every 2nd + boundary cases), each sampled event carries
        an rng checkpoint, and the FINAL SNAPSHOT equals the
        full-audit twin's (replay invariant preserved)."""
    m = _mod()
    blob = {"schema": "pref-v1", "rule": "mean_clip",
            "bucket_spec": "b1",
            "stats": {"grow|s1": {"w": 1.0, "m": 0.2, "v": 0.0,
                                  "n_raw": 1}},
            "prior_fingerprint": None, "quota_used": 0,
            "rng_state": None, "fold_cursor": 1, "pending": []}
    out = m.migrate_snapshot(dict(blob))
    assert out["schema"] == "pref-v1" and out["stats"] == blob[
        "stats"]
    bad = m.migrate_snapshot({"schema": "pref-v9"})
    assert "refusal" in bad
    p = _part()
    events = [_ev("grow|s1", 0.2)]
    assert "refusal" in p.restore({"schema": "pref-v9"})
    p.rebuild(events)
    assert abs(p.inspect()["stats"]["grow|s1"]["m"] - 0.2) < 1e-12
    full = _part(**{"preference.rule": "thompson",
                    "preference.min_count": 1,
                    "preference.audit_draw_mode": "full"})
    samp = _part(**{"preference.rule": "thompson",
                    "preference.min_count": 1,
                    "preference.audit_draw_mode": "sampled:2"})
    for p2 in (full, samp):
        p2._force_stats("grow|s1", w=4.0, m=0.2, v=0.09, n_raw=4)
        for _ in range(4):
            p2.score({"move": "grow", "slope": 0.0})
    df = [e for e in full.audit_events
          if e["kind"] == "preference_draw"]
    ds = [e for e in samp.audit_events
          if e["kind"] == "preference_draw"]
    assert len(df) == 4 and len(ds) == 2
    assert all("rng_checkpoint" in e for e in ds)
    sf, ss = full.snapshot(), samp.snapshot()
    assert sf["rng_state"] == ss["rng_state"]
    assert sf["stats"] == ss["stats"]
