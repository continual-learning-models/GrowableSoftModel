"""run_self — the granted self-study session (S9-2 review; S9-5 adds
proactive growth behind an explicit flag).

CONTROLLABILITY (C1-C6): this function runs ONLY when called — the
grant IS the call, with an explicit block budget (C1, C3). It reads
material exclusively from the model's experience store (C2). Every
session and every action lands in the event log (C4). Nothing here
commits — the gate keeps sole promotion authority (C5). The loop is
per-block and stops at budget (C6).
"""
from __future__ import annotations
from reference_net.growthpolicy import OP_OMEGA

import numpy as np

from core.plasticity.self_review import self_review, RETENTION_SAG
from core.plasticity import policy as siting

STEPS_PER_BLOCK = 200
LP_FLAT = 0.02          # |window LP| below this = flat (frozen S9-1)
SAT_DEMAND = 0.35       # saturation indicating unmet demand (frozen)
QUIZ_N = 96             # self-quiz sample size per block
VAR_EPS = 0.05          # variation self-check: +-5% input transform
VAR_FLAG = 3.0          # inconsistency = local response > 3x typical


def _ss(pol, key, default):
    """selfstudy/growth policy read (param-interface batch S5,
    docs/system/22 items 13-27): loud validation, no silent
    adjustment (all keys in this family are positive numbers)."""
    v = pol.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or v <= 0:
        raise ValueError(f"{key} must be a number > 0; got {v!r}")
    return v


def run_self(system, model_id, block_budget: int,
             suites=None, allow_growth: bool = False) -> dict:
    """One granted self-study session.
    suites: optional evaluate suites [{name, X, y}] so the session can
    track its own retention/LP honestly (recommended).
    allow_growth: S9-5 — the session may PROPOSE structural growth
    (policy-sited) inside the working state; commit stays the gate's.
    """
    lc = system.lc
    pol = lc.policy(model_id)
    _steps = int(_ss(pol, 'selfstudy_steps', STEPS_PER_BLOCK))
    _satd = _ss(pol, 'selfstudy_sat_demand', SAT_DEMAND)
    _lpf = _ss(pol, 'selfstudy_lp_flat', LP_FLAT)
    _qn = int(_ss(pol, 'selfstudy_quiz_n', QUIZ_N))
    if block_budget < 1:
        return {"refusal": "zero budget (C3)"}
    store = system.store(model_id)
    if len(store) == 0:
        return {"refusal": "empty experience store (C2: nothing to "
                           "replay; the model never fetches data)"}
    organ, _m = lc._load_working(model_id)
    if organ is None:
        return {"refusal": "no working state"}
    lc._log(model_id, "self_study_start", budget=block_budget,
            allow_growth=allow_growth)
    actions = []
    for block in range(block_budget):
        review = self_review(system, model_id)
        did = None
        quiz = self_quiz(system, model_id, n=_qn)
        # self-quiz is DIAGNOSTIC ONLY (SZ-1 verdict): quiz_acc and
        # weak rows are REPORTED; digestion stays uniform mixed replay
        # 1) proactive growth (S9-5, explicit flag): LP flat + demand
        if allow_growth and _lp_flat(system, model_id, suites, lp_flat=_lpf) \
                and max(review["saturation"].values() or [0]) >= _satd:
            organ, _m = lc._load_working(model_id)
            d = siting.decide(
                organ,
                widen_sat=_ss(pol, 'widen_sat',
                              siting.WIDEN_SAT),
                uniform_factor=_ss(pol, 'uniform_factor',
                                   siting.UNIFORM_FACTOR),
                escalate_disposed=int(_ss(
                    pol, 'escalate_disposed',
                    siting.ESCALATE_DISPOSED)))
            if d["action"] == OP_OMEGA:
                out = system.widen(model_id, container=d["container"])
            else:
                out = system.grow(model_id, k_nodes=1)
            rows = store.sample(min(128, len(store)))
            system.lc.study(model_id, rows, steps=_steps)
            did = {"block": block, "action": f"grow:{d['action']}",
                   "detail": {k: v for k, v in d.items() if k != "action"},
                   "result": "refused" if isinstance(out, dict)
                   and out.get("refusal") else "applied"}
        # 2) default: consolidation replay (deep scales mature here)
        else:
            rows = store.sample(min(128, len(store)))
            system.lc.study(model_id, rows, steps=_steps)
            did = {"block": block, "action": "consolidate",
                   "rows": len(rows)}
        did["quiz_acc"] = quiz["quiz_acc"]
        did["weak_count"] = len(quiz["weak_rows"])
        if suites is not None:
            did["accs"] = system.evaluate(model_id, suites)["stage_accs"]
        actions.append(did)
    variation = variation_check(system, model_id)
    lc._log(model_id, "self_study_end", budget=block_budget,
            actions=[a["action"] for a in actions],
            variation_flags=len(variation["flags"]))
    return {"blocks": len(actions), "actions": actions,
            "variation": variation,
            "note": "working-state only; commit() gates promotion"}


def self_quiz(system, model_id, n=QUIZ_N, tol_frac=0.25) -> dict:
    """SELF-QUESTION, SELF-ANSWER, SELF-GRADE — against the store's
    REAL labels (the model's own past experience; nothing fabricated).
    Returns the failed rows so digestion can target them."""
    import numpy as np
    store = system.store(model_id)
    rows = store.sample(min(n, len(store)))
    if not rows:
        return {"quiz_acc": None, "weak_rows": []}
    lc = system.lc
    organ, meta = lc._load_working(model_id)
    X = lc._build_X(meta, [r["input"] for r in rows])
    pred = organ.predict(X)[:, 0]
    y = np.array([float(r["target"]) for r in rows])
    tol = max(tol_frac * (float(y.std()) + 1e-9), 1e-6)
    wrong = np.abs(pred - y) > tol
    return {"quiz_acc": float(1.0 - wrong.mean()),
            "weak_rows": [rows[i] for i in np.where(wrong)[0]]}


def variation_check(system, model_id, n=32,
                    var_eps=VAR_EPS, var_flag=VAR_FLAG) -> dict:
    """TRANSFER SELF-CHECK (label-free): transform stored cases by
    +-VAR_EPS per feature and test the model's own CONSISTENCY — a
    response wildly larger than typical marks 'memorized, not
    understood'. Flags join the question list for the brain; the
    model NEVER trains on self-made labels."""
    import numpy as np
    store = system.store(model_id)
    rows = store.sample(min(n, len(store)))
    if not rows:
        return {"flags": [], "typical_response": None}
    lc = system.lc
    organ, meta = lc._load_working(model_id)
    X = lc._build_X(meta, [r["input"] for r in rows])
    base = organ.predict(X)[:, 0]
    resp = np.zeros(len(rows))
    for j in range(X.shape[1]):
        Xp = X.copy()
        step = var_eps * (np.abs(Xp[:, j]) + 1.0)
        Xp[:, j] = Xp[:, j] + step
        resp = np.maximum(resp,
                          np.abs(organ.predict(Xp)[:, 0] - base))
    typical = float(np.median(resp)) + 1e-12
    flags = [{"input": rows[i]["input"],
              "reason": "variation_inconsistent",
              "score": round(float(resp[i] / typical), 3)}
             for i in np.where(resp > var_flag * typical)[0]]
    return {"flags": flags, "typical_response": typical}


def _lp_flat(system, model_id, suites, lp_flat=LP_FLAT) -> bool:
    if suites is None:
        return True          # no suites given: demand check carries it
    evs = [e for e in system.lc.events(model_id)
           if e.get("event") == "evaluate" and "stage_accs" in e]
    if len(evs) < 2:
        return True
    last, prev = evs[-1]["stage_accs"], evs[-2]["stage_accs"]
    return all(abs(a - b) < lp_flat for a, b in zip(last, prev))
