"""run_course acceptance: the automatic loop masters a graded curriculum,
growing when stuck, committing through the gate, stopping on judgment
calls."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.facade import System

LAW = lambda x: float(x[0] * x[1] + max(x[1] - x[2], 0.0) * x[0] + 2 * x[2])
BOUNDS = [(1, 1, 2), (3, 3, 3), (6, 6, 4)]


def stage(k, n_study=250, n_eval=60, seed=0):
    rng = np.random.default_rng(seed + k)
    b = np.array(BOUNDS[k])
    Xs = rng.uniform(0, 1, (n_study, 3)) * b
    Xh = rng.uniform(0, 1, (30, 3)) * b
    Xe = rng.uniform(0, 1, (n_eval, 3)) * b
    mk = lambda X: [{"input": {"a": float(x[0]), "b": float(x[1]),
                               "c": float(x[2])},
                     "target": str(round(LAW(x), 6))} for x in X]
    return {"name": f"stage{k+1}",
            "examples": mk(Xs),
            "holdout": mk(Xh),
            "suite": {"X": Xe.tolist(),
                      "y": [str(round(LAW(x), 6)) for x in Xe]},
            "target": 0.85}


def test_run_course_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        curriculum = [stage(k) for k in range(3)]
        s.create_model("auto", holdout=curriculum[0]["examples"][:40])
        rep = s.run_course("auto", curriculum)
        assert rep.get("completed"), rep.get("stopped")
        assert all(st["mastered"] for st in rep["stages"])
        # gate-driven commits happened; events recorded
        kinds = [e["event"] for e in s.lc.events("auto")]
        assert "commit" in kinds and "study" in kinds
        # final model serves the hardest stage
        out = s.infer("auto", {"a": 5.0, "b": 5.0, "c": 2.0})
        want = LAW([5.0, 5.0, 2.0])
        assert abs(out["output"] - want) <= 1.0, (out, want)
        print("course report:", [(st["name"], st["blocks"], st["grows"],
                                  round(st["final_accs"][i], 2))
                                 for i, st in enumerate(rep["stages"])])
    finally:
        shutil.rmtree(tmp)


def test_run_course_stops_for_judgment():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        cur = [stage(0)]
        s.create_model("j", holdout=cur[0]["examples"][:40])
        s.set_policy("j", max_depth=1, max_params_mult=1)   # growth impossible
        # force STUCK by demanding an impossible target
        cur[0]["target"] = 1.01
        rep = s.run_course("j", cur, policy={"max_blocks_per_stage": 6})
        assert rep["stopped"], rep      # returned control to the teacher
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_run_course_end_to_end()
    test_run_course_stops_for_judgment()
    print("course tests passed")
