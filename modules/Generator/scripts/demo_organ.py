"""Organ (mlp backend) acceptance demo — real learning with GENERALIZATION.

Proves the architecture:
  - the SoftModel model is a small net trained from scratch (no base LLM, no GPU),
  - inputs are LLM-extracted features, outputs are labels + confidence,
  - teach -> held-out metric improves on inputs NEVER seen in training,
  - a contradictory (garbage) batch is rejected by the gate,
  - versions + rollback work.

Run:  python scripts/demo_organ.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec
from generator.data import read_jsonl

MODEL = "demo_risk"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "risk"


def line(c="-"):
    print(c * 64)


def main() -> int:
    cfg = Config.from_env(backend="mlp")
    f = SoftModelFactory(cfg)

    # deterministic re-runs
    if f.registry.model_dir(MODEL).exists():
        shutil.rmtree(f.registry.model_dir(MODEL))

    line("=")
    print(f"SoftModel organ demo — model '{MODEL}' — backend={cfg.backend}")
    print("Task: transaction risk (features extracted by the calling LLM)")
    line("=")

    holdout = read_jsonl(EXAMPLES_DIR / "holdout.jsonl")
    train = read_jsonl(EXAMPLES_DIR / "train.jsonl")
    f.create(ModelSpec(
        model_id=MODEL,
        description="risk triage organ: LOW/HIGH from LLM-extracted features",
        holdout=holdout,
    ))

    probe = {"amount": 70, "night": 1, "foreign": 1}   # HIGH, and NOT in train set

    # 1) untrained organ (v0)
    print("\n[1] Untrained organ (v0):")
    r = f.infer(MODEL, probe)
    print(f"   infer({probe}) -> {r['output']} (conf {r['confidence']})")
    e0 = f.evaluate(MODEL)
    print(f"   held-out metric: {e0['metric']:.2f} ({e0['correct']}/{e0['n']})")

    # 2) teach with the training set (disjoint from holdout!)
    print("\n[2] teach() with 12 labeled examples (all DIFFERENT from held-out):")
    r1 = f.teach(MODEL, train)
    _report(r1)
    r = f.infer(MODEL, probe)
    print(f"   infer({probe}) -> {r['output']} (conf {r['confidence']})   <- generalization")

    # 3) teach a contradictory (label-flipped) batch -> gate must reject
    print("\n[3] teach() with contradictory labels -> gate rejects:")
    garbage = [{"input": ex["input"],
                "target": ("HIGH" if ex["target"] == "LOW" else "LOW")}
               for ex in train[:6]]
    r2 = f.teach(MODEL, garbage)
    _report(r2)

    # 4) versions + rollback
    print("\n[4] versions:")
    for v in f.versions(MODEL)["versions"]:
        s = "n/a" if v["score"] is None else f"{v['score']:.2f}"
        print(f"   {v['version']:<4} parent={str(v['parent']):<4} score={s:<5} {v['note']}")
    print(f"   active = {f.versions(MODEL)['active']}")

    print("\n[5] rollback to v0, then re-promote v1:")
    f.rollback(MODEL, "v0")
    print(f"   active = {f.versions(MODEL)['active']}")
    f.rollback(MODEL, r1["candidate_version"])
    print(f"   active = {f.versions(MODEL)['active']}")

    line("=")
    ok = (r1["promoted"] and r1["candidate_metric"] >= 0.9
          and not r2["promoted"]
          and f.infer(MODEL, probe)["output"] == "HIGH")
    print("ORGAN ACCEPTANCE:", "PASS" if ok else "FAIL")
    line("=")
    return 0 if ok else 1


def _report(r):
    print(f"   candidate {r['candidate_version']}: metric "
          f"{r['live_metric_before']:.2f} -> {r['candidate_metric']:.2f} | "
          f"promoted={r['promoted']} | active={r['active_version']}")


if __name__ == "__main__":
    raise SystemExit(main())
