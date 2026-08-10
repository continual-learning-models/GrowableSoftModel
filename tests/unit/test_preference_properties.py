"""TP-01..TP-05 — GrowthPreference property/invariant boxes
(doc 83 v1.16 §8.2; plan 84 v2.13 D-2, RED before D-3).

T-AX3 note: TP-01 is the REPLAY second method for every seeded
draw (two independently constructed processes); TP-02/03/05 are
range/identity invariants over seeded sweeps; TP-04 is the
monotonicity law on deterministic rules and in fixed-seed
expectation for stochastic ones (eps random branch exempt by
definition, doc 83 v1.3 scope).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules" / "ReferenceNet"))


def _mod():
    from reference_net.growthpolicy import preference
    return preference


def _part(**over):
    m = _mod()
    pol = {"seed": 0}
    pol.update(over)
    return m.GrowthPreference(pol)


def _ev(bucket, advantage, i=0):
    return {"event_id": f"p-{bucket}-{i}", "bucket": bucket,
            "move": bucket.split("|")[0], "batch": i,
            "quoted_gain": 0.0, "window_gains": [advantage],
            "credited_gain": advantage, "advantage": advantage}


def _seeded_life(part, rng):
    """One deterministic mini-life: credits + score_set calls."""
    outs = []
    for i in range(12):
        b = ["grow|s0", "grow|s1", "deepen|s1"][
            int(rng.integers(0, 3))]
        part.credit(_ev(b, float(rng.normal(0, 0.3)), i))
        outs.append(part.score_set(
            [{"move": "grow", "slope": -0.02},
             {"move": "grow", "slope": 0.0},
             {"move": "deepen", "slope": 0.0}]))
    return outs


def test_tp01_determinism_two_processes():
    """TP-01: identical (events, keys, seed) => identical draws,
    tables, multipliers — two independently built parts."""
    for rule in ("mean_clip", "thompson", "ucb", "eps_greedy"):
        pol = {"preference.rule": rule, "preference.min_count": 1}
        a = _part(**pol)
        b = _part(**pol)
        oa = _seeded_life(a, np.random.default_rng(9))
        ob = _seeded_life(b, np.random.default_rng(9))
        assert oa == ob, rule
        assert a.snapshot() == b.snapshot(), rule
        assert json.dumps(a.audit_events, sort_keys=True) == \
            json.dumps(b.audit_events, sort_keys=True), rule


def test_tp02_boundedness_all_rules():
    """TP-02: for every rule and any stats, multiplier stays in
    [clip_lo, clip_hi] (and is finite)."""
    for rule in ("fixed", "mean_clip", "thompson", "ucb",
                 "eps_greedy"):
        p = _part(**{"preference.rule": rule,
                     "preference.min_count": 1})
        rng = np.random.default_rng(3)
        for i in range(8):
            p._force_stats("grow|s1", w=float(rng.uniform(0, 9)),
                           m=float(rng.normal(0, 50)),
                           v=float(rng.uniform(0, 100)),
                           n_raw=int(rng.integers(1, 9)))
            p._force_stats("deepen|s1",
                           w=float(rng.uniform(0, 9)),
                           m=float(rng.normal(0, 50)),
                           v=float(rng.uniform(0, 100)),
                           n_raw=int(rng.integers(1, 9)))
            for mult in p.score_set(
                    [{"move": "grow", "slope": 0.0},
                     {"move": "deepen", "slope": 0.0}]):
                assert 0.5 <= mult <= 2.0, (rule, mult)


def test_tp03_inertness_fixed_rule_exact_one_no_side_effects():
    """TP-03: rule=fixed => multiplier ≡ 1.0 for every ctx, with
    ZERO side effects (no draws, no audit events, no rng use —
    the M2 inertness precedence at unit scale)."""
    p = _part()                      # shipping default rule=fixed
    for slope in (None, -0.5, 0.0, 0.5):
        assert p.score({"move": "grow", "slope": slope}) == 1.0
    assert p.score_set([{"move": "grow", "slope": 0.0},
                        {"move": "deepen", "slope": 0.0}]) \
        == [1.0, 1.0]
    assert p.audit_events == []
    s = p.snapshot()
    assert s["stats"] == {} and s["quota_used"] == 0


def test_tp04_monotonicity_deterministic_and_expected():
    """TP-04: raising ONE bucket's advantage mean never lowers its
    multiplier. Deterministic rules exactly; thompson in fixed-
    seed expectation (same seed, same draw noise z: raw = m +
    se*z is increasing in m for fixed se)."""
    for rule in ("mean_clip", "ucb"):
        lo = _part(**{"preference.rule": rule,
                      "preference.min_count": 1})
        hi = _part(**{"preference.rule": rule,
                      "preference.min_count": 1})
        for p2, m_ in ((lo, 0.1), (hi, 0.4)):
            p2._force_stats("grow|s1", w=4.0, m=m_, v=0.01,
                            n_raw=4)
            p2._force_stats("deepen|s1", w=4.0, m=0.2, v=0.01,
                            n_raw=4)
        a = lo.score_set([{"move": "grow", "slope": 0.0},
                          {"move": "deepen", "slope": 0.0}])[0]
        b = hi.score_set([{"move": "grow", "slope": 0.0},
                          {"move": "deepen", "slope": 0.0}])[0]
        assert b >= a - 1e-12, rule
    lo = _part(**{"preference.rule": "thompson",
                  "preference.min_count": 1})
    hi = _part(**{"preference.rule": "thompson",
                  "preference.min_count": 1})
    for p2, m_ in ((lo, 0.1), (hi, 0.4)):
        p2._force_stats("grow|s1", w=4.0, m=m_, v=0.01, n_raw=4)
        p2._force_stats("deepen|s1", w=4.0, m=0.2, v=0.01,
                        n_raw=4)
    a = lo.score_set([{"move": "grow", "slope": 0.0},
                      {"move": "deepen", "slope": 0.0}])[0]
    b = hi.score_set([{"move": "grow", "slope": 0.0},
                      {"move": "deepen", "slope": 0.0}])[0]
    assert b >= a - 1e-12       # same seed => same z, raw ↑ in m


def test_tp05_prior_cap_post_load_weights(tmp_path):
    """TP-05: fleet-prior load (doc 83 M8): post-load bucket
    weights <= preference.prior_weight_cap; prior_load audited
    with the artifact fingerprint.
    WORKSHEET: artifact bucket w=50 with cap 12 => loaded w=12;
    m, v carried verbatim."""
    m = _mod()
    art = {"schema": "prior-v2",
           "strata": {"supervised": {"mu": 0.02, "sd": 0.08,
                                     "n": 6}},
           "buckets": {"supervised":
                       {"grow|s1": {"w": 50.0, "m": 0.15,
                                    "v": 0.04, "n": 6}}},
           "source_sha": ["deadbeef"], "created": "2026-07-28"}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(art))
    p = m.GrowthPreference({"seed": 0,
                            "preference.rule": "mean_clip",
                            "preference.prior_path": str(path),
                            "preference.prior_weight_cap": 12.0})
    st = p.inspect()["stats"]["grow|s1"]
    assert st["w"] == 12.0
    assert st["m"] == 0.15 and st["v"] == 0.04
    assert st["n_raw"] == 6        # v1.17: fleet evidence
    #   satisfies the min_count floor — prior is ACTIONABLE
    #   at cold start, not floored inert
    mu_ev, sd_ev = p._ref_stats()  # event stats seeded from
    assert abs(mu_ev - 0.02) < 1e-15   # the stratum record
    assert abs(sd_ev - 0.08) < 1e-15
    loads = [e for e in p.audit_events if e["kind"] ==
             "prior_load"]
    assert loads and loads[0]["weight_cap_applied"] is True
    # prior-v1 (z-unit) artifacts are UNKNOWN: refusal + inert
    old_art = {"schema": "prior-v1", "buckets": {}}
    path2 = tmp_path / "old.json"
    path2.write_text(json.dumps(old_art))
    q = m.GrowthPreference({"seed": 0,
                            "preference.rule": "mean_clip",
                            "preference.prior_path": str(path2)})
    assert q.inspect()["stats"] == {}
    assert any(e["kind"] == "prior_refusal"
               for e in q.audit_events)
