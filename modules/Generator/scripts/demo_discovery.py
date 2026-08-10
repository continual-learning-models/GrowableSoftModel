"""M3 Type-D demo — the model DISCOVERS the regularities governing a domain.

Narrative:
  [1] feed raw transaction observations (no rules given anywhere)
  [2] the model discovers the regularities, validated on held-out reality —
      readable statements the brain (LLM) can incorporate into its reasoning
  [3] inference is explainable: each answer cites the rule that fired
  [4] a contradictory batch is rejected by the gate
  [5] reality changes (M2 composes): drift detected -> windowed re-teach ->
      the discovered regularities THEMSELVES evolve

Run:  python scripts/demo_discovery.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec
from test_drift import (FEATURES, TRAIN_OLD, HOLDOUT_OLD, TRAIN_NEW,
                        HOLDOUT_NEW)

MODEL = "demo_risk_rules"


def line(c="-"):
    print(c * 64)


def show(discoveries: dict):
    for ln in discoveries["regularities"]:
        print(f"     {ln}")
    metric = discoveries["metric"]
    if metric is not None:
        print(f"     [held-out metric of this rule set: {metric:.2f}]")


def main() -> int:
    cfg = Config.from_env(backend="mlp", gate_recent_n=8, drift_tolerance=0.15)
    f = SoftModelFactory(cfg)
    if f.registry.model_dir(MODEL).exists():
        shutil.rmtree(f.registry.model_dir(MODEL))

    line("=")
    print(f"SoftModel discovery demo — Type-D model '{MODEL}'")
    print("Task: DISCOVER what governs transaction risk from raw observations")
    line("=")

    f.create(ModelSpec(MODEL, holdout=HOLDOUT_OLD))

    print("\n[1] teach() 12 raw observations (no rules given anywhere):")
    r1 = f.teach(MODEL, TRAIN_OLD)
    print(f"   candidate {r1['candidate_version']}: metric "
          f"{r1['candidate_metric']:.2f} | promoted={r1['promoted']}")

    print("\n[2] What the model DISCOVERED (readable by the brain/LLM):")
    d1 = f.discoveries(MODEL)
    show(d1)

    print("\n[3] Explainable inference (the fired rule is cited):")
    for probe in ({"amount": 800, "night": 0, "foreign": 0},
                  {"amount": 75, "night": 0, "foreign": 1}):
        out = f.infer(MODEL, probe)
        print(f"   {probe} -> {out['output']}  because: {out['rule']}")

    print("\n[4] teach() with contradictory labels -> gate rejects:")
    garbage = [{"input": ex["input"],
                "target": ("HIGH" if ex["target"] == "LOW" else "LOW")}
               for ex in TRAIN_OLD[:6]]
    r2 = f.teach(MODEL, garbage)
    print(f"   candidate {r2['candidate_version']}: metric "
          f"{r2['candidate_metric']:.2f} | promoted={r2['promoted']} "
          f"| live regularities untouched")

    print("\n[5] REALITY CHANGES (fraud moves to daytime-foreign):")
    f.add_holdout(MODEL, HOLDOUT_NEW)
    d = f.check_drift(MODEL)
    print(f"   drift: recent={d['recent_metric']:.2f} vs "
          f"baseline={d['baseline_metric']:.2f} -> needs_reteach={d['needs_reteach']}")
    r3 = f.teach(MODEL, TRAIN_NEW, window=len(TRAIN_NEW))
    print(f"   re-teach: candidate {r3['candidate_version']} metric "
          f"{r3['candidate_metric']:.2f} | promoted={r3['promoted']}")
    print("\n   The discovered regularities THEMSELVES evolved:")
    d3 = f.discoveries(MODEL)
    show(d3)

    probe = {"amount": 75, "night": 0, "foreign": 1}
    out = f.infer(MODEL, probe)
    print(f"\n   {probe} -> {out['output']}  because: {out['rule']}")

    line("=")
    interaction = any(("foreign == 1" in ln and "night == 0" in ln)
                      for ln in d3["regularities"])
    ok = (r1["promoted"] and d1["n_rules"] >= 1
          and not r2["promoted"]
          and d["needs_reteach"] and r3["promoted"]
          and interaction and out["output"] == "HIGH")
    print("DISCOVERY ACCEPTANCE:", "PASS" if ok else "FAIL")
    line("=")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
