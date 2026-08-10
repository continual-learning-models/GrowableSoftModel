"""TF-GROWTH-AUDIT — log-verified multi-scale SELF-growth on the
transformer host (repeatable; log SAVED for inspection).

The automatic loop (run_course) drives a transformer student through an
ESCALATING simulated curriculum; growth decisions are the RUNNER's own
(STUCK / persistent-FALSE escalation) — no manual grow calls. The
model's event log (events.jsonl) is then SAVED to tests/logs/ and
AUDITED to confirm, from the records:

  G1  grow events exist, each logging {nodes, params, depth}
      -> the network self-grew, and the growth is on the record
  G2  depth increased (multi-scale structure actually formed)
  G3  params increased across grow events (structure, not relabeling)
  G4  growth occurred only AFTER the early stages were mastered
      (complexity-forced, non-uniform in time — the owner's principle)
  G5  gated commits surround the run (versions = v0 + commits; growth
      that stays was paid for through the gate path)

Run: python3 tests/integration/test_transformer_growth_audit.py
Log: tests/logs/transformer_growth_events.jsonl (overwritten per run)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.facade import System

LOG_OUT = ROOT / "tests" / "logs" / "transformer_growth_events.jsonl"

LAW = lambda x: float(x[0] * x[1] + max(x[1] - x[2], 0.0) * x[0] + 2 * x[2])
BOUNDS = [(1, 1, 2), (3, 3, 3), (6, 6, 4), (12, 12, 8), (20, 20, 12)]
TARGETS = [0.85, 0.85, 0.85, 0.75, 0.65]


def stage(k, seed=0):
    rng = np.random.default_rng(seed + k)
    b = np.array(BOUNDS[k])
    mk = lambda X: [{"input": {"a": float(x[0]), "b": float(x[1]),
                               "c": float(x[2])},
                     "target": str(round(LAW(x), 6))} for x in X]
    Xs = rng.uniform(0, 1, (200, 3)) * b
    Xh = rng.uniform(0, 1, (30, 3)) * b
    Xe = rng.uniform(0, 1, (50, 3)) * b
    return {"name": f"stage{k + 1}", "examples": mk(Xs), "holdout": mk(Xh),
            "suite": {"X": Xe.tolist(),
                      "y": [str(round(LAW(x), 6)) for x in Xe]},
            "target": TARGETS[k]}


def main():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        curriculum = [stage(k) for k in range(len(BOUNDS))]
        s.create_model("tfg", holdout=curriculum[0]["examples"][:40],
                       substrate="transformer")
        rep = s.run_course("tfg", curriculum,
                           policy={"max_blocks_per_stage": 10,
                                   "steps_per_block": 100,
                                   "post_grow_blocks": 3})
        # ---- SAVE the log (owner requirement) ----
        events = s.lc.events("tfg")
        LOG_OUT.parent.mkdir(exist_ok=True)
        LOG_OUT.write_text("\n".join(json.dumps(e) for e in events))
        print(f"log saved: {LOG_OUT.relative_to(ROOT)} "
              f"({len(events)} events)")

        # ---- print the trail ----
        trail = []
        for e in events:
            t = e["event"]
            if t == "grow":
                t = f"GROW({','.join(e['nodes'])} -> depth {e['depth']}, " \
                    f"{e['params']}p)"
            if t == "commit":
                t = f"commit({e['version']} {e['score']:.2f})"
            trail.append(t)
        print("EVENT TRAIL:", " -> ".join(trail))

        # ---- AUDIT from the records ----
        grows = [e for e in events if e["event"] == "grow"]
        commits = [e for e in events if e["event"] == "commit"]
        stage_grows = {st["name"]: st["grows"] for st in rep["stages"]}
        print("per-stage grows:", stage_grows,
              "| final:", [round(st["final_accs"][i], 2)
                           for i, st in enumerate(rep["stages"])])

        assert grows, "G1: no grow events — self-growth did not occur"
        assert all({"nodes", "params", "depth"} <= set(e) for e in grows), \
            "G1: grow events lack structural records"
        print(f"  G1 self-growth on record: {len(grows)} grow event(s), "
              f"nodes {sum((e['nodes'] for e in grows), [])}")

        max_depth = max(e["depth"] for e in grows)
        assert max_depth >= 2, "G2: no multi-scale structure"
        print(f"  G2 multi-scale: depth 1 -> {max_depth}")

        params = [e["params"] for e in grows]
        assert params[-1] > 10817, "G3: params did not increase"
        print(f"  G3 structural growth: params -> {params}")

        early = sum(stage_grows.get(f"stage{i+1}", 0) for i in range(2))
        late = sum(stage_grows.get(f"stage{i+1}", 0)
                   for i in range(2, len(BOUNDS)))
        assert late >= 1 and early == 0, \
            f"G4: growth not complexity-forced (early={early}, late={late})"
        print(f"  G4 complexity-forced: early stages 0 grows, "
              f"hard stages {late}")

        vs = s.get_versions("tfg")
        assert len(vs["versions"]) == 1 + len(commits), "G5: lineage broken"
        print(f"  G5 gated lineage: v0 + {len(commits)} commits")
        print("TRANSFORMER MULTI-SCALE SELF-GROWTH AUDIT: PASS (5/5)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
