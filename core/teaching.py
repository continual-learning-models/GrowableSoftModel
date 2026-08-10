"""Teaching instrumentation (IWP3): trajectory verdicts + attribution.

Suites/trajectory are TEACHER-SIDE pedagogy instruments and NEVER gate
promotion (VII.2 — the commit gate anchors on the model's own holdout).

Trajectory verdict math: PORTED from SelfGrow@7fe0a0f/
reference_net/instrument.py (trajectory method) — the frozen original is
welded to its Instrument storage; ~30 pure lines re-hosted over the
system's event-derived score matrix. PORT_MANIFEST entry #1.
"""
from __future__ import annotations

import numpy as np

from core.lifecycle import Lifecycle

# thresholds (identical to the frozen Phase-2 values)
V_MAX = 0.05
R_MIN = 0.90
SPIKE = 0.15
STUCK_GAIN = 0.01
STUCK_WINDOW = 4


def trajectory(lc: Lifecycle, mid: str, current_stage: int | None = None,
               v_max=V_MAX, r_min=R_MIN, spike=SPIKE,
               stuck_gain=STUCK_GAIN, stuck_window=STUCK_WINDOW):
    # threshold kwargs: param-interface batch S4 (docs/system/22
    # items 8-12). MIRROR of reference_net/instrument.py's
    # trajectory (documented port, PORT_MANIFEST #1) — same names,
    # same defaults; equality asserted by test_param_verdict.
    rows = lc.score_matrix(mid)
    if len(rows) < 2:
        return {"verdict": "INSUFFICIENT", "n_evals": len(rows)}
    M = np.array([r["stage_accs"] for r in rows])
    K = M.shape[1]
    k = K - 1 if current_stage is None else current_stage
    cur = M[:, k]
    best_hist = np.maximum.accumulate(cur)
    progress = float(best_hist[-1])
    gain_recent = float(best_hist[-1] - best_hist[-stuck_window]
                        if len(cur) > stuck_window else best_hist[-1])
    w = cur[-stuck_window:]
    vol = float(np.mean(np.maximum(0.0, w[:-1] - w[1:]))) if len(w) > 1 else 0.0
    if k > 0:
        peaks = M[:, :k].max(axis=0) + 1e-9
        retention = float(np.min(M[-1, :k] / peaks))
    else:
        retention = 1.0
    spiked = bool(len(cur) >= 3 and cur[-2] - cur[-3] > spike
                  and cur[-1] < cur[-2] - spike / 2)
    if spiked or (vol > v_max and gain_recent > 0):
        verdict = "FALSE_SPIKE"
    elif retention < r_min:
        verdict = "FALSE_SWAP"
    elif gain_recent < stuck_gain:
        verdict = "STUCK"
    else:
        verdict = "REAL"
    return {"verdict": verdict, "progress": round(progress, 4),
            "recent_gain": round(gain_recent, 4),
            "volatility": round(vol, 4),
            "retention": round(retention, 4), "n_evals": len(rows)}


def attribution(lc: Lifecycle, mid: str, suites):
    """Which suite each grown node serves (activation-mass distribution)."""
    from generator.data import featurize
    organ, meta = lc._load_working(mid)
    out = []

    def walk(n, path=""):
        for j, inner in n.inner.items():
            mags = []
            for s in suites:
                X = np.array([featurize(i, meta["features"]) for i in s["X"]])
                mags.append(float(np.mean(np.abs(
                    inner.predict(n._std_x(X))[:, 0]))))
            tot = sum(mags)
            if tot < 1e-9:
                # freshly grown, untrained inner net: zero output by
                # function preservation — honestly inactive, no distribution
                out.append({"node": f"{path or 'root'}[{j}]",
                            "inactive": True})
            else:
                out.append({"node": f"{path or 'root'}[{j}]",
                            "distribution": [round(m / tot, 3) for m in mags],
                            "majority_suite": int(np.argmax(mags))})
            walk(inner, f"{path}/{j}" if path else str(j))
        port = getattr(n, "_port_site", None)
        if port is None:
            return
        # fullwidth port bodies (Growth Interface Reform): the
        # SERVED contribution is u @ A — a zero-born assembly
        # keeps the honest "inactive by function preservation"
        for g, slot in enumerate(port.bodies):
            j = slot.get("key", g)
            body, A = slot["body"], np.asarray(slot["A"])
            mags = []
            for s in suites:
                X = np.array([featurize(i, meta["features"])
                              for i in s["X"]])
                u = np.asarray(body.predict(n._std_x(X)))
                mags.append(float(np.mean(np.abs(u @ A))))
            tot = sum(mags)
            if tot < 1e-9:
                out.append({"node": f"{path or 'root'}[{j}]",
                            "inactive": True})
            else:
                out.append({"node": f"{path or 'root'}[{j}]",
                            "distribution": [round(m / tot, 3)
                                             for m in mags],
                            "majority_suite": int(np.argmax(mags))})
            walk(body, f"{path}/{j}" if path else str(j))
    walk(organ)
    return {"nodes": out, "suite_names": [s["name"] for s in suites]}
