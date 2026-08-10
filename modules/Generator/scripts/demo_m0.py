"""M0 acceptance demo (T7) — runs end-to-end with the mock backend (no GPU).

Proves the SoftModel control loop:
    call it (weak) -> teach it -> it measurably improves -> gate promotes only
    improvements -> versioned -> rollback works.

Run:  python scripts/demo_m0.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.registry import ModelRegistry
from generator.model_manager import ModelManager
from generator.trainer import Trainer
from generator.evaluator import Evaluator
from generator.evolve import Evolve
from generator.data import write_jsonl

MODEL = "demo_status"

HOLDOUT = [
    {"input": "shipped out",           "target": "SHIPPED"},
    {"input": "on its way",            "target": "SHIPPED"},
    {"input": "delivered!!!",          "target": "DELIVERED"},
    {"input": "arrived at destination","target": "DELIVERED"},
    {"input": "cancelled by user",     "target": "CANCELLED"},
    {"input": "order canceled",        "target": "CANCELLED"},
    {"input": "awaiting payment",      "target": "PENDING"},
    {"input": "not yet processed",     "target": "PENDING"},
]
ROUND1 = [HOLDOUT[i] for i in (0, 2, 4, 6)]      # half the mappings
ROUND2 = [HOLDOUT[i] for i in (1, 3, 5, 7)]      # the other half
GARBAGE = [{"input": "blah blah", "target": "NOISE"},
           {"input": "random text", "target": "NOISE"}]   # no held-out coverage


def line(c="-"):
    print(c * 64)


def main():
    cfg = Config.from_env(backend="mock")
    reg = ModelRegistry(cfg)

    # deterministic re-runs
    if reg.model_dir(MODEL).exists():
        shutil.rmtree(reg.model_dir(MODEL))

    mm = ModelManager(cfg, reg)
    trainer = Trainer(cfg, reg, mm)
    evaluator = Evaluator(cfg, mm, reg)
    evolve = Evolve(cfg, reg, mm, trainer, evaluator)

    line("=")
    print(f"SoftModel M0 demo — model '{MODEL}' — backend={cfg.backend}")
    line("=")

    reg.create_model(MODEL)
    write_jsonl(reg.holdout_path(MODEL), HOLDOUT)

    # 1) baseline (v0 = untrained)
    print("\n[1] Baseline (v0, untrained):")
    print("   infer('shipped out') ->", repr(mm.infer(MODEL, "v0", "shipped out")["output"]))
    v0 = evaluator.eval_version(MODEL, "v0")
    print(f"   held-out metric: {v0['metric']:.2f} ({v0['correct']}/{v0['n']})")

    # 2) teach round 1
    print("\n[2] teach() round 1 — 4 examples:")
    r1 = evolve.teach(MODEL, ROUND1)
    _report(r1)

    # 3) teach round 2
    print("\n[3] teach() round 2 — 4 more examples:")
    r2 = evolve.teach(MODEL, ROUND2)
    _report(r2)
    print("   infer('shipped out') ->",
          repr(mm.infer(MODEL, reg.active(MODEL), "shipped out")["output"]))

    # 4) teach a non-improving (garbage) batch -> gate should REJECT
    print("\n[4] teach() round 3 — garbage (no held-out coverage) -> gate rejects:")
    r3 = evolve.teach(MODEL, GARBAGE)
    _report(r3)

    # 5) versions + rollback
    print("\n[5] versions:")
    for v in reg.versions(MODEL):
        s = "n/a" if v["score"] is None else f"{v['score']:.2f}"
        print(f"   {v['version']:<4} parent={str(v['parent']):<4} score={s:<5} {v['note']}")
    print(f"   active = {reg.active(MODEL)}")

    print("\n[6] rollback to v1, then re-infer:")
    reg.set_active(MODEL, "v1")
    print("   active =", reg.active(MODEL),
          "| metric =", f"{evaluator.eval_version(MODEL, 'v1')['metric']:.2f}")

    line("=")
    ok = (r1["promoted"] and r2["promoted"] and not r3["promoted"]
          and r2["candidate_metric"] == 1.0)
    print("M0 ACCEPTANCE:", "PASS ✅" if ok else "CHECK ❌")
    line("=")
    return 0 if ok else 1


def _report(r):
    print(f"   candidate {r['candidate_version']}: metric "
          f"{r['live_metric_before']:.2f} -> {r['candidate_metric']:.2f} | "
          f"promoted={r['promoted']} | active={r['active_version']}")


if __name__ == "__main__":
    raise SystemExit(main())
