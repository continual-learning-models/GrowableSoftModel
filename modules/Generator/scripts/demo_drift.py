"""M2 drift-awareness demo — reality changes, the system notices and adapts.

Narrative:
  [1] organ learns OLD reality (fraud = big amounts, or foreign at night)
  [2] reality CHANGES (fraudsters move to daytime-foreign, small amounts);
      fresh labeled examples are appended to the held-out stream
  [3] check_drift: the live organ's recent-slice metric collapses -> DRIFTED
  [4] re-teach with new-reality cases (windowed): the gate re-evaluates the
      live version FRESH on the recent slice, so the adapted candidate wins —
      under frozen-holdout (M1) gating it would have been rejected
  [5] drift cleared; lineage shows the whole story

Run:  python scripts/demo_drift.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_drift import TRAIN_OLD, HOLDOUT_OLD, TRAIN_NEW, HOLDOUT_NEW, FEATURES

MODEL = "demo_drift_risk"


def line(c="-"):
    print(c * 64)


def main() -> int:
    cfg = Config.from_env(backend="mlp", gate_recent_n=8, drift_tolerance=0.15)
    f = SoftModelFactory(cfg)
    if f.registry.model_dir(MODEL).exists():
        shutil.rmtree(f.registry.model_dir(MODEL))

    line("=")
    print(f"SoftModel drift demo — model '{MODEL}' (gate_recent_n=8)")
    line("=")

    f.create(ModelSpec(MODEL, holdout=HOLDOUT_OLD))

    print("\n[1] Learn OLD reality (fraud = big amounts, or foreign AT NIGHT):")
    r1 = f.teach(MODEL, TRAIN_OLD)
    print(f"   v1: metric {r1['live_metric_before']:.2f} -> "
          f"{r1['candidate_metric']:.2f} | promoted={r1['promoted']}")
    d0 = f.check_drift(MODEL)
    print(f"   drift check: recent={d0['recent_metric']:.2f} "
          f"baseline={d0['baseline_metric']:.2f} -> drifted={d0['drifted']}")

    print("\n[2] REALITY CHANGES (fraud moves to DAYTIME-foreign, small amounts).")
    print("    Fresh labeled examples appended to the held-out stream:")
    info = f.add_holdout(MODEL, HOLDOUT_NEW)
    print(f"   holdout: {info['holdout_size']} examples "
          f"(+{info['added']} fresh)")

    print("\n[3] Drift check against the recent slice:")
    d1 = f.check_drift(MODEL)
    print(f"   recent={d1['recent_metric']:.2f} vs baseline="
          f"{d1['baseline_metric']:.2f} (tol {d1['drift_tolerance']}) "
          f"-> drifted={d1['drifted']} needs_reteach={d1['needs_reteach']}")

    print("\n[4] Re-teach with new-reality cases (window sheds contradicted labels):")
    r2 = f.teach(MODEL, TRAIN_NEW, window=len(TRAIN_NEW))
    print(f"   live re-scored FRESH on recent slice: {r2['live_metric_before']:.2f}"
          f"  (stale experience exposed)")
    print(f"   candidate {r2['candidate_version']}: {r2['candidate_metric']:.2f} "
          f"| promoted={r2['promoted']}")
    print("   (under a frozen held-out set this adaptation would have been rejected)")

    print("\n[5] Drift check after adaptation:")
    d2 = f.check_drift(MODEL)
    print(f"   recent={d2['recent_metric']:.2f} baseline="
          f"{d2['baseline_metric']:.2f} -> drifted={d2['drifted']}")

    print("\n[6] Lineage:")
    for v in f.versions(MODEL)["versions"]:
        s = "n/a" if v["score"] is None else f"{v['score']:.2f}"
        print(f"   {v['version']:<4} parent={str(v['parent']):<4} score={s:<5} {v['note']}")
    print(f"   active = {f.versions(MODEL)['active']}")

    line("=")
    ok = (r1["promoted"] and d1["drifted"] and r2["promoted"]
          and not d2["drifted"] and r2["candidate_metric"] >= 0.85)
    print("DRIFT ACCEPTANCE:", "PASS" if ok else "FAIL")
    line("=")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
