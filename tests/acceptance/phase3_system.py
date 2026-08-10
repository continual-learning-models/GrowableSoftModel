"""Acceptance C — the plastic system: schema growth (sigma),
re-founding (Phi) under the gate, and a granted self-study session
under the consent rules C1-C6."""
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


def rows2(seed, n):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-2, 2, size=(n, 2))
    return [{"input": {"a": float(a), "b": float(b)},
             "target": float(a * b + a - b)} for a, b in A]


def main():
    with tempfile.TemporaryDirectory() as td:
        os.environ["SOFTMODEL_MODELS_ROOT"] = td
        from core.facade import System
        from core.plasticity.store import QuarantineViolation
        s = System()
        s.create_model("acc_sys", holdout=rows2(2, 40))
        s.study("acc_sys", rows2(1, 200), steps=600)
        q = {"a": 1.0, "b": -1.0}
        p0 = s.infer("acc_sys", q, working=True)["output"]

        # C1 sigma: mid-stream new feature, exact entry, learnable
        out = s.add_feature("acc_sys", "c", default=0.0)
        p1 = s.infer("acc_sys", {**q, "c": 5.0}, working=True)["output"]
        check("C1 add_feature exact entry",
              out["features"][-1] == "c" and abs(p1 - p0) < 1e-6,
              f"|delta|={abs(p1 - p0):.2e}; schema={out['features']}")

        # C2 memory + quarantine by default
        st = s.store("acc_sys")
        try:
            st.add(rows2(2, 40)[:1], source="study")
            quarantined = False
        except QuarantineViolation:
            quarantined = True
        check("C2 experience memory + holdout quarantine",
              len(st) > 0 and quarantined,
              f"{len(st)} rows retained; holdout content refused loudly")

        # C3 granted self-study session (consent C1-C6)
        r = s.run_self("acc_sys", 2)
        ev = [e["event"] for e in s.lc.events("acc_sys")]
        check("C3 self-study: granted, budgeted, logged",
              r["blocks"] == 2 and "self_study_start" in ev
              and "self_study_end" in ev,
              "2 blocks; session events on the record")
        r0 = s.run_self("acc_sys", 0)
        check("C4 zero-budget refused (controllability)",
              "refusal" in r0, r0.get("refusal", ""))

        # C5 Phi re-found from own memory; the gate stays in charge
        out = s.refound("acc_sys", steps=600)
        verdict = s.commit("acc_sys")
        check("C5 refound gated by commit",
              "candidate_params" in out and "promoted" in verdict,
              f"candidate {out['candidate_params']} params; "
              f"gate promoted={verdict['promoted']}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"SYSTEM ACCEPTANCE: {'PASS' if not failed else 'FAIL'} "
          f"({len(RESULTS) - len(failed)}/{len(RESULTS)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
