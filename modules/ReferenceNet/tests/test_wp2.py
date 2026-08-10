"""WP2 acceptance (A-WP2): a scripted (non-LLM) driver runs a full course
END-TO-END through the Instrument API only; all metrics populated; events
logged; verdicts behave; attribution works; practice protocol works."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference_net.curriculum import make_splits, law, TOL
from reference_net.instrument import Instrument


def suites(stages):
    return [{"name": s["name"], "X": s["X"]["eval"].tolist(),
             "y": s["y"]["eval"].tolist()} for s in stages]


def test_full_course_via_api_only():
    tmp = tempfile.mkdtemp()
    try:
        ins = Instrument(root=tmp, seed=5)
        stages = make_splits()
        ev_suites = suites(stages)
        ins.create_student("s1", hidden=16)

        for k, st in enumerate(stages):
            # cumulative study through the API (driver-side pedagogy)
            X = np.vstack([stages[i]["X"]["study"] for i in range(k + 1)])
            y = np.vstack([stages[i]["y"]["study"] for i in range(k + 1)])
            for _ in range(14):
                ins.study("s1", X.tolist(), y.tolist(), steps=200)
                accs = ins.evaluate("s1", ev_suites)["stage_accs"]
                traj = ins.trajectory("s1", current_stage=k)
                if accs[k] >= 0.9:
                    break
                if traj["verdict"] == "STUCK":
                    rep = ins.growth_report("s1")
                    assert rep["candidates"]
                    before = accs[k]
                    g = ins.grow("s1", k_nodes=2)
                    for _ in range(4):
                        ins.study("s1", X.tolist(), y.tolist(), steps=200)
                        accs = ins.evaluate("s1", ev_suites)["stage_accs"]
                    if accs[k] < before + 0.01:
                        ins.rollback("s1", g["ckpt"])

            # practice cycle (verifier outside; attempt-0 protocol: only
            # fix problems the student\'s own answer fails, within reach)
            Xp = st["X"]["practice"][:40]
            yp = law(Xp)
            att = np.array(ins.attempts("s1", Xp.tolist()))
            passed = []
            for i in range(len(Xp)):
                if abs(att[i, 0] - yp[i]) <= TOL:
                    passed.append(None)
                    continue
                ok = np.where(np.abs(att[i] - yp[i]) <= TOL)[0]
                passed.append(float(att[i][ok[np.argmin(
                    np.abs(att[i, ok] - att[i, 0]))]]) if len(ok) else None)
            pr = ins.practice_update("s1", Xp.tolist(), passed)
            assert pr["passed"] + pr["beyond_reach"] == len(Xp)

        # WP2 tests API completeness, not mastery (mastery = WP1/WP3):
        # early stages mastered; later (hardest) stages progressing with
        # growth having occurred THROUGH the API.
        final = ins.evaluate("s1", ev_suites)["stage_accs"]
        assert min(final[:4]) >= 0.85, final
        assert min(final) >= 0.55, final
        traj = ins.trajectory("s1")
        assert traj["retention"] >= 0.9, traj

        # attribution present for every composite node
        card = ins.card("s1")
        if card["depth"] >= 2:
            att = ins.attribution("s1", ev_suites)
            assert att["nodes"]
            assert all(abs(sum(n["distribution"]) - 1.0) < 0.01
                       for n in att["nodes"])

        # storage populated
        d = Path(tmp) / "s1"
        assert (d / "events.jsonl").exists() and (d / "score_matrix.jsonl").exists()
        kinds = [json.loads(l)["event"]
                 for l in (d / "events.jsonl").read_text().splitlines()]
        assert "study" in kinds and "practice" in kinds
    finally:
        shutil.rmtree(tmp)


def test_verdict_classifier_patterns():
    tmp = tempfile.mkdtemp()
    try:
        ins = Instrument(root=tmp)
        ins.create_student("v")
        mp = ins._matrix_path("v")

        def write(rows):
            mp.write_text("\n".join(json.dumps(
                {"t": i, "names": ["a", "b"], "stage_accs": r})
                for i, r in enumerate(rows)))

        # FALSE_SPIKE: jump then drop
        write([[.9, .2], [.9, .25], [.9, .3], [.9, .55], [.9, .3]])
        assert ins.trajectory("v")["verdict"] == "FALSE_SPIKE"
        # FALSE_SWAP: new rises, old collapses
        write([[.9, .2], [.9, .4], [.8, .55], [.6, .7], [.5, .8]])
        assert ins.trajectory("v")["verdict"] == "FALSE_SWAP"
        # STUCK: flat below target
        write([[.9, .3], [.9, .31], [.9, .3], [.9, .305], [.9, .3], [.9, .3]])
        assert ins.trajectory("v")["verdict"] == "STUCK"
        # REAL: steady rise, retention high
        write([[.9, .2], [.92, .3], [.93, .4], [.95, .5], [.95, .6]])
        assert ins.trajectory("v")["verdict"] == "REAL"
    finally:
        shutil.rmtree(tmp)


def test_rollback_via_api():
    tmp = tempfile.mkdtemp()
    try:
        ins = Instrument(root=tmp, seed=1)
        st = make_splits()[0]
        ins.create_student("r", hidden=8)
        ins.study("r", st["X"]["study"].tolist(), st["y"]["study"].tolist(), 100)
        before = ins._students["r"].predict(st["X"]["eval"]).copy()
        g = ins.grow("r")
        ins.study("r", st["X"]["study"].tolist(), st["y"]["study"].tolist(), 100)
        ins.rollback("r", g["ckpt"])
        after = ins._students["r"].predict(st["X"]["eval"])
        assert np.allclose(before, after)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_full_course_via_api_only()
    test_verdict_classifier_patterns()
    test_rollback_via_api()
    print("wp2 tests passed")
