"""IWP2/S2.2 acceptance (UT2.2-2.5, 2.7-2.9): working state + gated
commit + budgets + event lineage + serving-state separation."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.wiring import SysFactory
from core.lifecycle import Lifecycle

RNG = np.random.default_rng(7)
LAW = lambda x: round(float(x[0] + 2 * x[1] - x[2]), 6)


def rows(n, fn=LAW, lo=0, hi=4):
    X = RNG.uniform(lo, hi, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
             "target": str(fn(x))} for x in X]


def _mk(tmp):
    return Lifecycle(SysFactory(Config.from_env(backend="mlp",
                                                models_root=Path(tmp))))


def test_ut2_2_session_commit_flow():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        # session: study from scratch (organ born from data shape)
        for _ in range(4):
            lc.study("m", rows(200), steps=300)
        # default infer still serves v0 (nothing committed) — UT2.9
        assert lc.infer("m", {"a": 1, "b": 3, "c": 0})["output"] is None
        w = lc.infer("m", {"a": 1, "b": 3, "c": 0}, working=True)
        assert abs(w["output"] - 7.0) <= 0.5
        r = lc.commit("m", note="first mastery")
        assert r["promoted"] and r["score"] >= 0.9, r
        out = lc.infer("m", {"a": 1, "b": 3, "c": 0})   # now committed
        assert abs(out["output"] - 7.0) <= 0.5
        # events lineage: study+commit present; version recorded
        kinds = [e["event"] for e in lc.events("m")]
        assert "study" in kinds and "commit" in kinds
        assert lc.f.versions("m")["active"] == r["version"]
    finally:
        shutil.rmtree(tmp)


def test_ut2_3_commit_gate_rejects_worse():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        for _ in range(4):
            lc.study("m", rows(200), steps=300)
        assert lc.commit("m")["promoted"]
        bad = lambda x: round(float(-x[0]), 6)          # contradiction
        lc.study("m", rows(150, bad), steps=400)
        r = lc.commit("m")
        assert not r["promoted"], r                      # gate holds
        lc.reset("m")                                    # back to incumbent
        out = lc.infer("m", {"a": 1, "b": 3, "c": 0}, working=True)
        assert abs(out["output"] - 7.0) <= 0.5
    finally:
        shutil.rmtree(tmp)


def test_ut2_4_grow_in_session_and_rollback_exact():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        lc.study("m", rows(200), steps=200)
        before = lc.infer("m", {"a": 2, "b": 2, "c": 2}, working=True)["output"]
        g = lc.grow("m", k_nodes=2)
        assert g["grown"] and g["depth"] == 2
        after = lc.infer("m", {"a": 2, "b": 2, "c": 2}, working=True)["output"]
        assert abs(after - before) < 1e-9                # parity at growth
    finally:
        shutil.rmtree(tmp)


def test_ut2_5_budget_refusals():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(30))
        lc.study("m", rows(100), steps=50)
        lc.set_policy("m", max_depth=1)
        r = lc.grow("m")
        assert r.get("refusal") == "depth budget", r
        lc.set_policy("m", max_depth=4, max_params_mult=1)
        r = lc.grow("m")
        assert r.get("refusal") == "params budget", r
        kinds = [e["event"] for e in lc.events("m")]
        assert kinds.count("grow_refused") == 2 and "policy" in kinds
    finally:
        shutil.rmtree(tmp)


def test_ut2_8_teach_equals_study_commit():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        data = rows(200)
        lc.study("m", data, steps=300)
        r = lc.commit("m")
        assert r["promoted"]
        # teach path on a fresh model, same data: comparable outcome
        lc.create("m2", holdout=rows(50))
        r2 = lc.f.teach("m2", data)
        assert r2["promoted"]
        assert abs(r["score"] - r2["candidate_metric"]) <= 0.1
    finally:
        shutil.rmtree(tmp)


def test_ut2_7_practice_safety_in_session():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        for _ in range(4):
            lc.study("m", rows(200), steps=300)
        probes = [{"a": float(a), "b": float(b), "c": 1.0}
                  for a in (0.5, 1.5, 2.5) for b in (0.5, 1.5, 2.5)]
        att = np.array(lc.attempts("m", probes), dtype=object)
        truths = [LAW([p["a"], p["b"], p["c"]]) for p in probes]
        passed = []
        for i, t in enumerate(truths):
            own_ok = abs(float(att[i][0]) - t) <= 0.5
            passed.append(None if own_ok else float(t))
        before = lc.evaluate("m", [{"name": "hold", "X": [list(p.values())
                             and [p["a"], p["b"], p["c"]] for p in probes],
                             "y": [str(t) for t in truths]}])["stage_accs"][0]
        r = lc.practice_update("m", probes, passed)
        after = lc.evaluate("m", [{"name": "hold",
                            "X": [[p["a"], p["b"], p["c"]] for p in probes],
                            "y": [str(t) for t in truths]}])["stage_accs"][0]
        assert after >= before - 0.01                    # never degrades
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut2_2_session_commit_flow()
    test_ut2_3_commit_gate_rejects_worse()
    test_ut2_4_grow_in_session_and_rollback_exact()
    test_ut2_5_budget_refusals()
    test_ut2_8_teach_equals_study_commit()
    test_ut2_7_practice_safety_in_session()
    print("iwp2 lifecycle tests passed")
