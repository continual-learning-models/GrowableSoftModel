"""IWP3 acceptance (UT3.1, 3.6, 3.7): verdict patterns; drift x
trajectory coexistence; gate ignores suites."""
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
from core.teaching import trajectory, attribution

RNG = np.random.default_rng(3)
LAW = lambda x: round(float(x[0] + 2 * x[1] - x[2]), 6)


def rows(n, fn=LAW):
    X = RNG.uniform(0, 4, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
             "target": str(fn(x))} for x in X]


def _mk(tmp):
    return Lifecycle(SysFactory(Config.from_env(backend="mlp",
                                                models_root=Path(tmp))))


def test_ut3_1_verdict_patterns():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("v", holdout=rows(10))

        def feed(rows_):
            (lc._mdir("v") / "events.jsonl").write_text("\n".join(
                __import__("json").dumps({"ts": 0, "event": "evaluate",
                                          "names": ["a", "b"],
                                          "stage_accs": r}) for r in rows_))
        feed([[.9, .2], [.9, .25], [.9, .3], [.9, .55], [.9, .3]])
        assert trajectory(lc, "v")["verdict"] == "FALSE_SPIKE"
        feed([[.9, .2], [.9, .4], [.8, .55], [.6, .7], [.5, .8]])
        assert trajectory(lc, "v")["verdict"] == "FALSE_SWAP"
        feed([[.9, .3], [.9, .31], [.9, .3], [.9, .305], [.9, .3], [.9, .3]])
        assert trajectory(lc, "v")["verdict"] == "STUCK"
        feed([[.9, .2], [.92, .3], [.93, .4], [.95, .5], [.95, .6]])
        assert trajectory(lc, "v")["verdict"] == "REAL"
    finally:
        shutil.rmtree(tmp)


def test_ut3_7_gate_ignores_suites_and_attribution():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        for _ in range(4):
            lc.study("m", rows(200), steps=300)
        # log deliberately terrible suite evaluations — gate must not care
        junk = [{"name": "junk",
                 "X": [[9.0, 9.0, 9.0]] * 5, "y": ["0"] * 5}]
        lc.evaluate("m", junk)
        r = lc.commit("m")
        assert r["promoted"], r        # holdout decided, suites ignored
        # attribution over grown structure: fresh nodes are honestly
        # INACTIVE (zero output by function preservation); after training
        # they acquire a distribution
        lc.grow("m", k_nodes=2)
        suites = [{"name": "s1", "X": [[1, 1, 1], [2, 2, 2]],
                   "y": ["2", "4"]}]
        att = attribution(lc, "m", suites)
        assert len(att["nodes"]) == 2
        assert all(n.get("inactive") for n in att["nodes"])
        lc.study("m", rows(200), steps=200)      # inner nets now move
        att = attribution(lc, "m", suites)
        active = [n for n in att["nodes"] if not n.get("inactive")]
        assert active and all(abs(sum(n["distribution"]) - 1.0) < 0.01
                              for n in active)
    finally:
        shutil.rmtree(tmp)


def test_ut3_6_drift_and_trajectory_coexist():
    tmp = tempfile.mkdtemp()
    try:
        lc = _mk(tmp)
        lc.create("m", holdout=rows(50))
        for _ in range(3):
            lc.study("m", rows(200), steps=300)
            lc.evaluate("m", [{"name": "hold",
                               "X": [[1, 3, 0], [2, 2, 2], [0, 1, 1]],
                               "y": ["7", "4", "1"]}])
        assert lc.commit("m")["promoted"]
        # reality drifts (frozen M2 machinery through the factory)
        law_b = lambda x: round(float(x[0] + x[1] + x[2]), 6)
        lc.f.add_holdout("m", rows(40, law_b))
        d = lc.f.check_drift("m", recent_n=40)
        assert d["drifted"]                      # drift: reality moved
        t = trajectory(lc, "m")
        assert t["verdict"] in ("REAL", "STUCK")  # learning quality separate
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut3_1_verdict_patterns()
    test_ut3_7_gate_ignores_suites_and_attribution()
    test_ut3_6_drift_and_trajectory_coexist()
    print("iwp3 tests passed")
