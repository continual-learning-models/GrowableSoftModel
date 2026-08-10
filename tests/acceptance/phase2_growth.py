"""Acceptance B — growth: recursive deepening (rho), widening (omega),
exact function preservation, auditable lineage, budget refusals.
(Course-FORCED growth under difficulty is additionally covered by the
in-repo integration audit: tests/integration/test_transformer_growth_audit.py.)"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(cid, ok, evidence):
    RESULTS.append((cid, bool(ok), evidence))
    print(f"  {cid}: {'PASS' if ok else 'FAIL'} - {evidence}")


def rows(seed, n):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-2, 2, size=(n, 3))
    return [{"input": {"a": float(a), "b": float(b), "c": float(c)},
             "target": float(a * b + c * c + a - b)}
            for a, b, c in A]


def main():
    with tempfile.TemporaryDirectory() as td:
        os.environ["SOFTMODEL_MODELS_ROOT"] = td
        from core.facade import System
        s = System()
        s.create_model("acc_grow", holdout=rows(2, 40))
        s.study("acc_grow", rows(1, 300), steps=600)
        q = {"a": 1.2, "b": -0.7, "c": 0.4}
        p0 = s.infer("acc_grow", q, working=True)["output"]

        # B1 rho deepen preserves function exactly and is logged
        g1 = s.grow("acc_grow", k_nodes=2)
        p1 = s.infer("acc_grow", q, working=True)["output"]
        check("B1 deepen (rho) exact-preserving",
              "refusal" not in g1 and abs(p1 - p0) < 1e-6,
              f"|delta|={abs(p1 - p0):.2e}")

        # B2 recursion: growth INSIDE a grown site (multi-scale)
        s.study("acc_grow", rows(3, 300), steps=400)
        rep = s.growth_report("acc_grow")
        deep = [c for c in rep["candidates"] if "/" in c["site"]
                or "[" in c["site"]]
        g2 = s.grow("acc_grow", k_nodes=1)
        check("B2 recursive deepening available",
              "refusal" not in g2 and rep["depth"] >= 2 and len(deep) > 0,
              f"depth {rep['depth']}, sites incl. inner paths")

        # B3 omega widen preserves exactly
        p2 = s.infer("acc_grow", q, working=True)["output"]
        w = s.widen("acc_grow", k=2)
        p3 = s.infer("acc_grow", q, working=True)["output"]
        check("B3 widen (omega) exact-preserving",
              "refusal" not in w and abs(p3 - p2) < 1e-6,
              f"|delta|={abs(p3 - p2):.2e}")

        # B4 lineage: every structural act is an event
        ev = [e["event"] for e in s.lc.events("acc_grow")]
        check("B4 auditable lineage",
              ev.count("grow") >= 2 and ev.count("widen") == 1,
              f"events: grow x{ev.count('grow')}, widen x1")

        # B5 budgets refuse (params cap)
        s.set_policy("acc_grow", max_params_mult=1.0)
        g5 = s.grow("acc_grow", k_nodes=2)
        w5 = s.widen("acc_grow", k=2)
        check("B5 growth budgets enforced",
              "refusal" in g5 and "refusal" in w5,
              "params-cap refusals on both operators")

    failed = [r for r in RESULTS if not r[1]]
    print(f"GROWTH ACCEPTANCE: {'PASS' if not failed else 'FAIL'} "
          f"({len(RESULTS) - len(failed)}/{len(RESULTS)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
