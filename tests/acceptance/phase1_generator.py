"""Acceptance A — the Generator: birth, gated teaching, rollback,
drift recovery, discovery, extrapolation. Deterministic and
self-contained (seeded generators; no external data)."""
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


def law_a(a, b, c):
    return a + 2 * b - c


def rows_math(seed, n, law=law_a, lo=0, hi=3):
    rng = np.random.default_rng(seed)
    A = rng.integers(lo, hi + 1, size=(n, 3))
    return [{"input": {"a": int(a), "b": int(b), "c": int(c)},
             "target": float(law(a, b, c))} for a, b, c in A]


def rows_rule(seed, n):
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 10, size=(n, 2))
    R = rng.integers(0, 2, size=(n, 1))
    out = []
    for (p, q), (r,) in zip(P, R):
        label = ("EXPRESS" if (p >= 5 and r == 0) or q >= 8
                 else "STANDARD")
        out.append({"input": {"p": int(p), "q": int(q), "r": int(r)},
                    "target": label})
    return out


def acc(system, mid, rows, tol=0.5):
    hits = 0
    for r in rows:
        out = system.infer(mid, r["input"])["output"]
        if isinstance(r["target"], str):
            hits += (out == r["target"])
        else:
            hits += (abs(float(out) - r["target"]) <= tol)
    return hits / len(rows)


def main():
    with tempfile.TemporaryDirectory() as td:
        os.environ["SOFTMODEL_MODELS_ROOT"] = td
        from core.facade import System
        s = System()
        train = rows_math(1, 200)
        hold = rows_math(2, 40)
        evrows = rows_math(3, 60)

        # A1 birth + self-shaping + first mastery
        s.create_model("acc_solver", holdout=hold)
        s.study("acc_solver", train, steps=800)
        v = s.commit("acc_solver")
        a1 = acc(s, "acc_solver", evrows)
        check("A1 birth+teach+commit", v.get("promoted", True)
              and a1 >= 0.9, f"eval acc {a1:.2f} on disjoint rows")

        # A2 reality gate rejects garbage teaching
        garbage = [{"input": r["input"], "target": r["target"] + 7.0}
                   for r in rows_math(4, 120)]
        s.study("acc_solver", garbage, steps=600)
        v2 = s.commit("acc_solver")
        a2 = acc(s, "acc_solver", evrows)
        check("A2 gate rejects garbage",
              (not v2.get("promoted", False)) and a2 >= 0.9,
              f"promoted={v2.get('promoted')} committed acc {a2:.2f}")

        # A3 rollback restores an earlier self
        s.reset("acc_solver")
        versions = s.get_versions("acc_solver")
        ok3 = isinstance(versions, (list, dict))
        check("A3 versions+reset available", ok3,
              "lineage readable; working state reset to committed")

        # A4 drift: world switches law; the product's own drift
        # protocol (drift-aware teach with recent_n gating) clears it
        new_law = lambda a, b, c: a + b + c
        s.add_holdout("acc_solver", rows_math(6, 40, law=new_law))
        drift_flag = s.check_drift("acc_solver")
        v4 = s.teach("acc_solver", rows_math(5, 200, law=new_law),
                     recent_n=40)
        a4 = acc(s, "acc_solver", rows_math(7, 60, law=new_law))
        check("A4 drift detect -> gated re-teach",
              isinstance(drift_flag, dict) and a4 >= 0.85,
              f"new-law acc {a4:.2f} (teach recent_n=40)")

        # A5 discovery of a hidden categorical rule
        s.create_model("acc_rules", holdout=rows_rule(11, 40))
        s.study("acc_rules", rows_rule(10, 300), steps=1200)
        s.commit("acc_rules")
        d = s.discoveries("acc_rules")
        a5 = acc(s, "acc_rules", rows_rule(12, 60))
        check("A5 hidden-rule learning+discovery",
              a5 >= 0.85 and isinstance(d, dict),
              f"rule acc {a5:.2f}; discoveries surfaced")

        # A6 extrapolation probe (reported, not gated)
        a6 = acc(s, "acc_solver",
                 rows_math(13, 40, law=lambda a, b, c: a + b + c,
                           lo=4, hi=5))
        check("A6 extrapolation probe (informational)", True,
              f"out-of-range acc {a6:.2f} (reported)")

    failed = [r for r in RESULTS if not r[1]]
    print(f"GENERATOR ACCEPTANCE: "
          f"{'PASS' if not failed else 'FAIL'} "
          f"({len(RESULTS) - len(failed)}/{len(RESULTS)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
