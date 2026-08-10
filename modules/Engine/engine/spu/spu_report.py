"""SPU readiness-field report and JSONL artifacts
(DESIGN_SPU v1.3 §6; DEV_PLAN S4).

Aggregates the event stream of an SPUNetwork into the per-scope
readiness field: which newborn inner networks are being
self-processed, how hard the loop works, and where skips happen.
Observation only — nothing here feeds back into computation.
"""
import json
from pathlib import Path


def build_report(net):
    """Aggregate net.spu_events into the readiness field."""
    spus = {}
    skips = {}
    steps_summaries = []
    events = (getattr(net, "spu_events", None)
              or getattr(net, "_spu_events", []))
    for e in events:
        if e.get("path") == "__step__":
            steps_summaries.append(e)
            continue
        if e.get("skip") is not None:
            skips[e["skip"]] = skips.get(e["skip"], 0) + 1
            continue
        rec = spus.setdefault(e["path"], {
            "events": 0, "steps_total": 0, "steps_max": 0,
            "converged": 0, "disc_hits": 0, "clips": 0,
            "j_inv_first": e["j_inv_before"],
            "j_inv_last": e["j_inv_after"]})
        rec["events"] += 1
        rec["steps_total"] += e["steps"]
        rec["steps_max"] = max(rec["steps_max"], e["steps"])
        rec["converged"] += int(e["converged"])
        rec["disc_hits"] += e["disc_hits"]
        rec["clips"] += e["clips"]
        rec["j_inv_last"] = e["j_inv_after"]
    for rec in spus.values():
        rec["steps_mean"] = (rec["steps_total"] / rec["events"]
                             if rec["events"] else 0.0)
    interference = None
    if steps_summaries:
        deltas = [s["task_mse_after"] - s["task_mse_before"]
                  for s in steps_summaries]
        interference = {"steps": len(steps_summaries),
                        "mse_delta_mean": sum(deltas) / len(deltas),
                        "mse_delta_max": max(deltas)}
    n_counted_skips = 0
    for reason, n in getattr(net, "_spu_skip_counts", {}).items():
        skips[reason] = skips.get(reason, 0) + n
        n_counted_skips += n
    return {"spus": spus, "skips": skips,
            "processed_steps": len(steps_summaries),
            "interference": interference,
            "total_events": len(events) + n_counted_skips}


def write_events_jsonl(path, events):
    """Commit-grade artifact: one JSON object per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return p


def read_events_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
