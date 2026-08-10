"""WP3 experiments (PLAN Part II; A-WP3). One command, seeded, CPU.

E1    grown vs never-grown (same curriculum, same student config)
E2    vs equal-final-params flat net trained end-to-end on the same data
E-S3  study-only vs study+practice (value of the practice phase)
E-S4  planted pathologies -> verdict classifier must flag them
E-S5  deeper curriculum forces recursion (depth >= 2 with NO code change)
E-S6  growth-location attribution to the triggering stage

Growth is forced by PROBLEM COMPLEXITY (the curriculum's later stages),
never by an artificially small student (owner's principle).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference_net.curriculum import make_splits, law, TOL, accuracy
from reference_net.net import Network
from reference_net.trainer import Course

RESULTS = []


def check(eid, ok, evidence):
    RESULTS.append((eid, ok, evidence))
    print(f"  {eid}: {'PASS' if ok else 'FAIL'} — {evidence}")


def note(eid, evidence):
    print(f"  {eid}: INFO — {evidence}")


def run_course(H=16, grow_budget=10, targets=0.9, stages=None, seed=5):
    stages = stages or make_splits()
    net = Network(3, H, seed=seed)
    c = Course(net, stages, eval_every=200, plateau_patience=4, seed=seed)
    m, e = c.run_scripted([targets] * len(stages), max_blocks=40,
                          grow_budget=grow_budget, t_post_blocks=6)
    return c, m, e


def main() -> int:
    t0 = time.time()
    stages = make_splits()

    # ---- E1: grown vs never-grown --------------------------------------
    print("[E1] grown vs never-grown (fixed student, escalating curriculum)")
    c_g, m_g, e_g = run_course(grow_budget=10)
    c_0, m_0, e_0 = run_course(grow_budget=0)
    f_g, f_0 = m_g[-1]["stage_accs"], m_0[-1]["stage_accs"]
    grows = sum(1 for x in e_g if x["event"] == "grow") - \
        sum(1 for x in e_g if x["event"] == "rollback")
    check("E1", grows >= 1 and min(f_g) >= 0.9 > min(f_0),
          f"grown kept={grows} final={[round(a,2) for a in f_g]} vs "
          f"never-grown final={[round(a,2) for a in f_0]}")

    # ---- E2: equal-final-params flat baseline --------------------------
    print("[E2] equal-final-params flat net, same data, same steps")
    P = c_g.net.n_params()
    H_flat = max(4, round((P - 1) / 5))           # params = 5H+1
    accs_flat = []
    for seed in (5, 6, 7):
        flat = Network(3, H_flat, seed=seed)
        X = np.vstack([s["X"]["study"] for s in stages])
        y = np.vstack([s["y"]["study"] for s in stages])
        for _ in range(c_g.t):
            flat.train_step(X, y)
        accs_flat.append(min(accuracy(flat.predict(s["X"]["eval"]),
                                      s["y"]["eval"]) for s in stages))
    check("E2", min(m_g[-1]["stage_accs"]) >= np.mean(accs_flat) - 0.05,
          f"grown min-stage {min(m_g[-1]['stage_accs']):.2f} vs flat "
          f"(H={H_flat}, {P} params) min-stage {np.mean(accs_flat):.2f}"
          f"±{np.std(accs_flat):.2f} (3 seeds)")

    # ---- E-S5: recursion under deeper pressure -------------------------
    print("[E-S5] depth from complexity (no code change)")
    depth = c_g.net.depth()
    hist = {}
    for row in c_g.net.structure():
        hist[row["path"].count("/") + (row["path"] != "root")] = \
            hist.get(row["path"].count("/") + (row["path"] != "root"), 0) + 1
    check("E-S5", depth >= 2,
          f"final depth={depth}, structure levels={hist} "
          f"params={c_g.net.n_params()}")

    # ---- E-S6: attribution of growth to triggering stages --------------
    print("[E-S6] growth-location attribution")
    from reference_net.instrument import Instrument
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    ins = Instrument(root=tmp, seed=5)
    ins._students["g"] = c_g.net
    ins._ckpts["g"] = {}
    suites = [{"name": s["name"], "X": s["X"]["eval"].tolist(),
               "y": s["y"]["eval"].tolist()} for s in stages]
    att = ins.attribution("g", suites)
    grow_stages = [x["stage"] for x in e_g if x["event"] == "grow"]
    stage_idx = {s["name"]: i for i, s in enumerate(stages)}
    trig = min(stage_idx[st] for st in grow_stages) if grow_stages else None
    ok_nodes = sum(1 for n in att["nodes"] if n["majority_suite"] >= (trig or 0))
    frac = ok_nodes / max(1, len(att["nodes"]))
    check("E-S6", frac >= 0.7,
          f"{ok_nodes}/{len(att['nodes'])} grown nodes attribute to the "
          f"triggering stage or later (first trigger: {grow_stages[:1]})")
    shutil.rmtree(tmp)

    # ---- E-S4: planted pathologies vs verdicts -------------------------
    print("[E-S4] verdict classifier on planted pathologies")
    tmp = tempfile.mkdtemp()
    ins = Instrument(root=tmp, seed=5)
    ins.create_student("p", hidden=16)
    suites6 = suites
    # pathology A: skip straight to the hardest stage (level-skipping)
    Xh = stages[-1]["X"]["study"]
    yh = stages[-1]["y"]["study"]
    verdicts = []
    for _ in range(6):
        ins.study("p", Xh.tolist(), yh.tolist(), steps=200)
        ins.evaluate("p", suites6)
        verdicts.append(ins.trajectory("p", current_stage=len(stages) - 1)
                        ["verdict"])
    flagged = any(v in ("STUCK", "FALSE_SPIKE", "FALSE_SWAP")
                  for v in verdicts[2:])
    # pathology B: memorization (tiny data, hard stage)
    ins.create_student("q", hidden=16)
    Xt, yt = Xh[:8], yh[:8]
    verdicts_b = []
    for _ in range(6):
        ins.study("q", Xt.tolist(), yt.tolist(), steps=200)
        ins.evaluate("q", suites6)
        verdicts_b.append(ins.trajectory("q")["verdict"])
    flagged_b = any(v.startswith("FALSE") or v == "STUCK"
                    for v in verdicts_b[2:])
    check("E-S4", flagged and flagged_b,
          f"skip-ahead verdicts={verdicts} | memorization verdicts={verdicts_b}")
    shutil.rmtree(tmp)

    # ---- E-S3: value of practice ----------------------------------------
    print("[E-S3] study-only vs study+practice (recorded finding)")
    tmp = tempfile.mkdtemp()
    finals = {}
    for mode in ("study_only", "with_practice"):
        ins = Instrument(root=tmp, seed=5)
        ins.create_student(mode, hidden=16)
        for k, st in enumerate(stages):
            X = np.vstack([stages[i]["X"]["study"] for i in range(k + 1)])
            y = np.vstack([stages[i]["y"]["study"] for i in range(k + 1)])
            for _ in range(6):
                ins.study(mode, X.tolist(), y.tolist(), steps=200)
                accs = ins.evaluate(mode, suites)["stage_accs"]
                if accs[k] >= 0.9:
                    break
            if mode == "with_practice":
                Xp = st["X"]["practice"][:40]
                yp = law(Xp)
                a = np.array(ins.attempts(mode, Xp.tolist()))
                passed = []
                for i in range(len(Xp)):
                    if abs(a[i, 0] - yp[i]) <= TOL:
                        passed.append(None)
                        continue
                    ok = np.where(np.abs(a[i] - yp[i]) <= TOL)[0]
                    passed.append(float(a[i][ok[np.argmin(
                        np.abs(a[i, ok] - a[i, 0]))]]) if len(ok) else None)
                ins.practice_update(mode, Xp.tolist(), passed)
        finals[mode] = ins.evaluate(mode, suites)["stage_accs"]
    note("E-S3", f"study-only={[round(a,2) for a in finals['study_only']]} "
                 f"with-practice={[round(a,2) for a in finals['with_practice']]} "
                 f"(honest recorded finding; practice is safe by protocol)")
    shutil.rmtree(tmp)

    failed = [e for e, ok, _ in RESULTS if not ok]
    print()
    print(f"EXPERIMENTS: {'PASS' if not failed else 'FAIL(' + ','.join(failed) + ')'}"
          f"  wall={time.time()-t0:.0f}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
