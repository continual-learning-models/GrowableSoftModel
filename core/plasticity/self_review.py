"""S9-1 — self_review: the extension reads its own record (read-only).

The inward half of the reprocessing loop: retention trends, learning
progress, uncertainty summary, memory stats, saturation — everything
the brain (or a future granted self-study session) needs to decide
what to do next. NO behavior change anywhere: pure derivation over
event logs and working state.

CONSENT RULE: this module only REPORTS. Nothing here trains, grows,
or schedules anything.
"""
from __future__ import annotations

import numpy as np

from core.plasticity.metrics import (
    lp_curve, retention_trend, saturation_report)

# Threshold derived from the existing comparison-round evaluate logs
# (E0 corpus: healthy mastered suites drift within +-0.02 per
# evaluation; genuine sagging shows sustained <-0.03). FROZEN here,
# before SR-1 runs (E0 lesson: derive-then-freeze).
RETENTION_SAG = -0.03


def self_review(system, model_id, probe_inputs=None,
                retention_sag=None) -> dict:
    """Read-only self-report for one model. probe_inputs (optional):
    rows the caller supplies for uncertainty probing."""
    lc = system.lc
    events = lc.events(model_id)
    evals = [e for e in events if e.get("event") == "evaluate"
             and "stage_accs" in e]
    n_suites = max((len(e["stage_accs"]) for e in evals), default=0)
    suites = []
    for k in range(n_suites):
        lp = lp_curve(evals, k)
        trend = retention_trend(evals, k)
        suites.append({"suite": k,
                       "last_acc": evals[-1]["stage_accs"][k],
                       "lp_tail": [round(v, 4) for v in lp[-3:]],
                       "retention_trend": round(trend, 4),
                       "sagging": bool(trend < (
                           retention_sag if retention_sag
                           is not None else
                           system.lc.policy(model_id).get(
                               "retention_sag",
                               RETENTION_SAG)))})
    organ, _meta = lc._load_working(model_id)
    sat = saturation_report(organ) if organ is not None else {}
    uncertainty = None
    if probe_inputs and organ is not None:
        uncertainty = _uncertainty_map(system, model_id, probe_inputs)
    store = system.store(model_id)
    grows = [e for e in events if e.get("event") in
             ("grow", "widen", "add_feature", "refound")]
    return {"model_id": model_id,
            "suites": suites,
            "sagging_suites": [s["suite"] for s in suites if s["sagging"]],
            "saturation": {k: round(v, 4) for k, v in sat.items()},
            "structure_events": len(grows),
            "store": {"rows": len(store), "seen": store.n_seen},
            "uncertainty": uncertainty,
            "consent_note": "report only; self-study runs only when "
                            "explicitly granted"}


def _uncertainty_map(system, model_id, inputs, sigma=0.01) -> dict:
    """Rank the caller-supplied inputs by the extension's own doubt:
    perturbation sensitivity + working-vs-committed disagreement.
    Emits the QUESTION LIST (the extension never answers itself)."""
    lc = system.lc
    organ, meta = lc._load_working(model_id)
    X = lc._build_X(meta, inputs)
    base = organ.predict(X)[:, 0]
    rng = np.random.default_rng(system.f.config.seed)
    pert = organ.perturb(rng, sigma) if hasattr(organ, "perturb") else None
    sens = (np.abs(pert.predict(X)[:, 0] - base)
            if pert is not None else np.zeros(len(X)))
    try:
        committed = np.array([system.infer(model_id, i)["output"]
                              for i in inputs], dtype=float)
        disagree = np.abs(committed - base)
    except Exception:
        disagree = np.zeros(len(X))
    spread = float(base.std()) + 1e-9
    score = (sens + disagree) / spread
    order = np.argsort(-score)
    questions = [{"input": inputs[i],
                  "reason": ("version_disagreement"
                             if disagree[i] > sens[i]
                             else "perturb_sensitive"),
                  "score": round(float(score[i]), 4)}
                 for i in order[:10]]
    return {"questions": questions,
            "mean_score": round(float(score.mean()), 4)}
