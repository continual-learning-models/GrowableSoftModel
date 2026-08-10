"""TI-01..TI-07 — GrowthPreference integration boxes (live organ)
(doc 83 v1.16 §8.3; plan 84 v2.13 D-2 RED specs, GREEN at D-3).

SCRIPTED LIFE (shared with the TI-02 baseline capture,
scripts/capture_preference_ti02_baseline.py): additive-law mini
world, Network(d_in=1, hidden=3, lr=1e-2, seed=5), 600 warmup
steps, then 2 x [decide/grow_with_policy + 300 steps], policy
routes through real tier-2 probes (probe_steps=80). Everything
seeded; the decision trace and the served function are hashed.

D-3 wiring contract fixed by these boxes:
  - decide()/grow_with_policy consult the preference part ONLY
    when the policy carries preference.* keys with rule != fixed
    (M2 inertness precedence); the state blob rides the net as
    net._preference_blob (attach/read helpers in preference.py).
  - decision dict gains {"pref_mult": {arm: mult}} when enabled.
  - preference.on_window(net, window_energy, policy) drives the
    K-window credit pipeline (caller-invoked batch-end hook).
  - preference.apply_rollback(net, policy, mode) implements
    rollback_mode keep|revert.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

FIXTURE = ROOT / "tests" / "fixtures" / \
    "preference_ti02_baseline.json"

ADDI = lambda x: np.sin(2 * x) + 0.6 * np.cos(5 * x)   # noqa: E731

BASE_POL = {"min_energy_points": 10 ** 9, "min_window_rows": 64,
            "probe_steps": 80, "seed": 0}


def _life(policy):
    """The scripted life (MUST stay byte-identical to the capture
    script's copy — any edit here requires recapturing)."""
    import reference_net.growthpolicy as gp
    from reference_net.net import Network
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = ADDI(X[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=5)
    for _ in range(600):
        net.train_step(X, y)
    decisions = []
    for _k in range(2):
        d = gp.grow_with_policy(net, dict(policy))
        decisions.append(d)
        for _ in range(300):
            net.train_step(X, y)
    grid = np.linspace(-2, 2, 257).reshape(-1, 1)
    pred = net.predict(grid)
    fn_sha = hashlib.sha256(
        np.ascontiguousarray(pred).tobytes()).hexdigest()
    trace = [{k: v for k, v in d.items()
              if k != "policy_snapshot"} for d in decisions]
    dec_sha = hashlib.sha256(json.dumps(
        _jsonable(trace), sort_keys=True).encode()).hexdigest()
    mse = float(np.mean((net.predict(X) - y) ** 2))
    return net, decisions, fn_sha, dec_sha, mse


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _pref():
    from reference_net.growthpolicy import preference
    return preference


# ---------------------------------------------------------- TI-01
def test_ti01_seam_records_pref_mult_and_ranking_algebra():
    """TI-01: enabled decision cycle (rule=mean_clip, min_count=1)
    on the mini world: the decision record carries pref_mult, and
    the recorded arm equals the argmax of the ADJUSTED scores
    recomputed HERE from the decision's own recorded
    intermediates (score_arm = E_ref - asymptote_arm; adjusted =
    score * mult if score > 0 else score; CI-overlap tie keeps
    the additive default) — the ranking algebra referee."""
    pref = _pref()
    pol = dict(BASE_POL)
    pol.update({"preference.rule": "mean_clip",
                "preference.min_count": 1,
                "preference.bucket_spec": "b0"})
    import reference_net.growthpolicy as gp
    from reference_net.net import Network
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = ADDI(X[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=5)
    for _ in range(600):
        net.train_step(X, y)
    blob = pref.GrowthPreference(
        {"seed": 0, "preference.rule": "mean_clip",
         "preference.min_count": 1,
         "preference.bucket_spec": "b0"})
    blob._force_stats("widen", w=4.0, m=0.3, v=0.0, n_raw=4)
    blob._force_stats("deepen", w=4.0, m=-0.3, v=0.0, n_raw=4)
    blob._force_event_stats(6, 5.0, 0.0, 0.09)   # mu=0, sd=0.3
    #   -> widen z=+1 -> mult e^1 clip 2.0; deepen z=-1 -> 0.5
    pref.attach_blob(net, blob.snapshot())
    d = gp.decide(net, pol)
    assert "pref_mult" in d, d.get("reasons")
    mult = d["pref_mult"]
    assert set(mult) == {"widen", "deepen"}
    prices = d["prices"]
    e_ref = d["pref_e_ref"]
    raw = {arm: e_ref - prices[arm]["asymptote"]
           for arm in ("widen", "deepen")}
    adj = {a: (raw[a] * mult[a] if raw[a] > 0 else raw[a])
           for a in raw}
    fw, fd = prices["widen"], prices["deepen"]
    overlap = not (fw.get("ci_high", np.inf) < fd.get(
        "ci_low", -np.inf)
        or fd.get("ci_high", np.inf) < fw.get("ci_low", -np.inf))
    want = "widen" if (overlap or adj["widen"] >= adj["deepen"]) \
        else "deepen"
    assert d["arm"] == want, (d["arm"], adj, overlap)


# ---------------------------------------------------------- TI-02
def test_ti02_inertness_byte_identity_vs_prebranch_baseline():
    """TI-02: INERTNESS AT SYSTEM SCALE. The DEFAULT-POLICY life
    (no preference keys — and equally rule=fixed) is
    BYTE-IDENTICAL to the pre-branch baseline captured at D-1/D-2
    on the unmodified tree (decision-trace sha + served-function
    sha + mse). Also the machine proof that NO slope probe runs
    under rule=fixed (extra probes would shift the decision
    trace). Runs THREE configs against one baseline: default,
    explicit rule=fixed, rule=fixed+b1+slope_probes-on-paper."""
    base = json.loads(FIXTURE.read_text())
    _, _, fn_sha, dec_sha, mse = _life(BASE_POL)
    assert fn_sha == base["fn_sha"]
    assert dec_sha == base["dec_sha"]
    assert abs(mse - base["mse"]) < 1e-15
    pol2 = dict(BASE_POL)
    pol2.update({"preference.rule": "fixed"})
    _, _, fn2, dec2, _ = _life(pol2)
    assert fn2 == base["fn_sha"] and dec2 == base["dec_sha"]
    pol3 = dict(BASE_POL)
    pol3.update({"preference.rule": "fixed",
                 "preference.bucket_spec": "b1",
                 "preference.slope_probes": True})
    _, _, fn3, dec3, _ = _life(pol3)
    assert fn3 == base["fn_sha"] and dec3 == base["dec_sha"]


# ---------------------------------------------------------- TI-03
def test_ti03_credit_pipeline_end_to_end():
    """TI-03: adopted event -> K windows -> credit lands with the
    correct advantage. The pending event is created at adoption
    with quoted_gain from the decision's own price read; window
    energies fed via preference.on_window; after K=3 windows the
    CreditEvent folds with weights (1,.5,.25).
    REFEREE: advantage recomputed HERE from the recorded window
    energies and the recorded quoted_gain, to 1e-12."""
    pref = _pref()
    pol = dict(BASE_POL)
    pol.update({"preference.rule": "mean_clip",
                "preference.min_count": 1,
                "preference.bucket_spec": "b0",
                "preference.credit_weights": [1, 0.5, 0.25]})
    import reference_net.growthpolicy as gp
    from reference_net.net import Network
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = ADDI(X[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=5)
    for _ in range(600):
        net.train_step(X, y)
    pref.attach_blob(net, None)          # fresh enabled state
    d = gp.grow_with_policy(net, pol)
    assert d.get("applied"), d["reasons"]
    energies = []
    for _w in range(3):
        for _ in range(120):
            net.train_step(X, y)
        e = float(np.mean((net.predict(X) - y) ** 2))
        energies.append(e)
        pref.on_window(net, e, pol)
    blob = pref.read_blob(net)
    evs = blob["credited_events_tail"]
    assert len(evs) == 1
    ev = evs[0]
    e0 = ev["e_before"]
    gains = [e0 - energies[0],           # CUMULATIVE vs the
             e0 - energies[1],           # fixed pre-adoption
             e0 - energies[2]]           # base (M3 v1.17)
    want_credit = (gains[0] + 0.5 * gains[1] + 0.25 * gains[2]) \
        / 1.75
    assert abs(ev["credited_gain"] - want_credit) < 1e-12
    assert abs(ev["advantage"]
               - (want_credit - ev["quoted_gain"])) < 1e-12
    stats = blob["stats"]
    assert any(s["n_raw"] == 1 for s in stats.values())


# ---------------------------------------------------------- TI-04
def test_ti04_persistence_identical_continuation():
    """TI-04: mid-life snapshot -> fresh part restore -> identical
    continuation vs the uninterrupted twin (multipliers, draws,
    stats) — the persistence contract at life scale."""
    pref = _pref()
    polkw = {"seed": 0, "preference.rule": "thompson",
             "preference.min_count": 1}
    a = pref.GrowthPreference(dict(polkw))
    rng = np.random.default_rng(11)
    ctxs = [{"move": "grow", "slope": 0.0},
            {"move": "deepen", "slope": 0.0}]
    for i in range(6):
        a.credit({"event_id": f"x{i}", "bucket":
                  ["grow|s1", "deepen|s1"][i % 2],
                  "move": "grow", "batch": i, "quoted_gain": 0.0,
                  "window_gains": [0.1],
                  "credited_gain": float(rng.normal(0, 0.2)),
                  "advantage": float(rng.normal(0, 0.2))})
        a.score_set(ctxs)
    blob = a.snapshot()
    b = pref.GrowthPreference(dict(polkw))
    assert b.restore(copy.deepcopy(blob)).get("ok") is True
    for _ in range(4):
        assert a.score_set(ctxs) == b.score_set(ctxs)
    assert a.snapshot() == b.snapshot()


# ---------------------------------------------------------- TI-05
def test_ti05_rollback_mode_keep_and_revert():
    """TI-05: rollback semantics (doc 83 M1): keep (default) =
    the table SURVIVES a rollback (lessons retained); revert =
    table restored to the with-model snapshot. Referee: ledger-
    replay equality — revert state == fold(events up to the
    snapshot), keep state == fold(all events)."""
    pref = _pref()
    polkw = {"seed": 0, "preference.rule": "mean_clip"}
    p = pref.GrowthPreference(dict(polkw))
    evs = []
    for i, a_ in enumerate((0.4, 0.1, -0.2)):
        ev = {"event_id": f"r{i}", "bucket": "grow|s1",
              "move": "grow", "batch": i, "quoted_gain": 0.0,
              "window_gains": [a_], "credited_gain": a_,
              "advantage": a_}
        evs.append(ev)
        p.credit(ev)
    model_snap = p.snapshot()             # persisted with model
    ev3 = {"event_id": "r3", "bucket": "grow|s1", "move": "grow",
           "batch": 3, "quoted_gain": 0.0, "window_gains": [0.3],
           "credited_gain": 0.3, "advantage": 0.3}
    evs.append(ev3)
    p.credit(ev3)
    keep = pref.apply_rollback(p.snapshot(), model_snap,
                               mode="keep")
    revert = pref.apply_rollback(p.snapshot(), model_snap,
                                 mode="revert")
    q_all = pref.GrowthPreference(dict(polkw))
    q_all.rebuild(evs)
    q_pre = pref.GrowthPreference(dict(polkw))
    q_pre.rebuild(evs[:3])
    assert keep["stats"] == q_all.snapshot()["stats"]
    assert revert["stats"] == q_pre.snapshot()["stats"]


# ---------------------------------------------------------- TI-06
def test_ti06_bocpd_coupling_on_synthetic_break():
    """TI-06: the EXISTING changepoint registry part watches the
    advantage residual stream; a planted break deepens the
    bucket's discount (w <- rho*w) exactly once.
    Stream: 40 residuals N(0, .05) then 12 at +1.0 (seeded)."""
    pref = _pref()
    p = pref.GrowthPreference({"seed": 0,
                               "preference.rule": "mean_clip",
                               "preference.bocpd_coupling": True,
                               "preference.bocpd_depth": 0.5})
    p._force_stats("grow|s1", w=2.0, m=0.2, v=0.01, n_raw=5)
    rng = np.random.default_rng(21)
    stream = np.concatenate([rng.normal(0, 0.05, 40),
                             rng.normal(1.0, 0.05, 12)])
    fired = pref.watch_changepoint(p, "grow|s1", stream,
                                   {"seed": 0})
    assert fired is True
    assert abs(p.inspect()["stats"]["grow|s1"]["w"] - 1.0) < 1e-12
    assert [e for e in p.audit_events
            if e["kind"] == "bocpd_deepen"]
    q = pref.GrowthPreference({"seed": 0,
                               "preference.rule": "mean_clip",
                               "preference.bocpd_coupling": True,
                               "preference.bocpd_depth": 0.5})
    q._force_stats("grow|s1", w=2.0, m=0.2, v=0.01, n_raw=5)
    calm = rng.normal(0, 0.05, 52)
    assert pref.watch_changepoint(q, "grow|s1", calm,
                                  {"seed": 0}) is False
    assert q.inspect()["stats"]["grow|s1"]["w"] == 2.0


# ---------------------------------------------------------- TI-07
def test_ti07_explore_offer_through_normal_gate_path():
    """TI-07: the explore path (doc 83 §4.4): consulted only when
    no candidate has a positive adjusted score; on True the
    offered candidate enters ranking with the minimal positive
    epsilon (trial-eligible), everything downstream the NORMAL
    flow. Referee'd at the seam-helper level
    (rank_with_preference — the exact function the combiner
    calls), plus the bitwise-restoration law re-asserted on this
    world's real probe path (pricer fingerprint equality)."""
    pref = _pref()
    part = pref.GrowthPreference(
        {"seed": 0, "preference.rule": "thompson",
         "preference.min_count": 1,
         "preference.bucket_spec": "b0",
         "preference.explore_quota": 1})
    part._force_stats("widen", w=6.0, m=0.5, v=0.0, n_raw=6)
    part._force_stats("deepen", w=6.0, m=-0.5, v=0.0, n_raw=6)
    part._force_event_stats(6, 5.0, 0.0, 0.01)  # sd=0.1: widen
    #   draw 0.5 -> z=5 -> mult 2.0 > 1.2 => favorable
    ctxs = [{"move": "widen", "slope": 0.0},
            {"move": "deepen", "slope": 0.0}]
    out = pref.rank_with_preference(
        part, raw_scores=[-0.02, -0.05], ctxs=ctxs)
    assert out["explore_offer"] is True
    assert out["offered_index"] == 0          # favorable bucket
    assert out["scores_adj"][0] > 0.0         # epsilon-positive
    assert out["scores_adj"][1] <= 0.0        # untouched negative
    assert part.snapshot()["quota_used"] == 1
    out2 = pref.rank_with_preference(
        part, raw_scores=[-0.02, -0.05], ctxs=ctxs)
    assert out2["explore_offer"] is False     # quota exhausted
    out3 = pref.rank_with_preference(
        part, raw_scores=[0.30, -0.05], ctxs=ctxs)
    assert out3["explore_offer"] is False     # positive exists =>
    assert out3["scores_adj"][0] > 0.0        # not consulted
    # bitwise-restoration law on this world's REAL probe path:
    import reference_net.growthpolicy as gp
    from reference_net.net import Network
    from reference_net.growthpolicy.interfaces import get
    from reference_net.growthpolicy.pricer_zero_attach import \
        _fingerprint
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = ADDI(X[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=5)
    for _ in range(600):
        net.train_step(X, y)
    pol = dict(gp.DEFAULT_GROWTH_POLICY)
    pol.update(BASE_POL)
    fp_before = _fingerprint(net)
    get("pricer", pol["pricer"]).price(net, pol)
    assert _fingerprint(net) == fp_before
