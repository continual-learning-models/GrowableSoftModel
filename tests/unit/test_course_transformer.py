"""Gap-closure test (owner's question): the AUTOMATIC LOOP (run_course)
on the TRANSFORMER host — the system-level analog of Phase-2's full
curriculum runs, simulated data."""
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


def stage(k, seed=0):
    rng = np.random.default_rng(seed + k)
    b = np.array(BOUNDS[k])
    mk = lambda X: [{"input": {"a": float(x[0]), "b": float(x[1]),
                               "c": float(x[2])},
                     "target": str(round(LAW(x), 6))} for x in X]
    Xs = rng.uniform(0, 1, (250, 3)) * b
    Xh = rng.uniform(0, 1, (30, 3)) * b
    Xe = rng.uniform(0, 1, (60, 3)) * b
    return {"name": f"stage{k+1}", "examples": mk(Xs), "holdout": mk(Xh),
            "suite": {"X": Xe.tolist(),
                      "y": [str(round(LAW(x), 6)) for x in Xe]},
            "target": 0.85}


def test_run_course_on_transformer_host():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        curriculum = [stage(k) for k in range(3)]
        out = s.create_model("tfauto", holdout=curriculum[0]["examples"][:40],
                             substrate="transformer")
        assert out["substrate"] == "transformer"
        rep = s.run_course("tfauto", curriculum,
                           policy={"max_blocks_per_stage": 10,
                                   "steps_per_block": 120})
        assert rep.get("completed"), rep.get("stopped")
        assert all(st["mastered"] for st in rep["stages"])
        # committed serving from the transformer host
        probe = {"a": 5.0, "b": 5.0, "c": 2.0}
        o = s.infer("tfauto", probe)
        want = LAW([5.0, 5.0, 2.0])
        assert abs(o["output"] - want) / want <= 0.15, (o, want)
        card = s.card("tfauto")
        assert card["learned_shape"]["substrate"] == "transformer"
        grows = sum(st["grows"] for st in rep["stages"])
        print(f"   transformer run_course: stages "
              f"{[round(st['final_accs'][i], 2) for i, st in enumerate(rep['stages'])]}, "
              f"grows={grows}, depth={card['learned_shape']['depth']}, "
              f"params={card['learned_shape']['params']}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_run_course_on_transformer_host()
    print("transformer-host automatic loop: PASS")
