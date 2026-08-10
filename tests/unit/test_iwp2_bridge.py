"""IWP2/S2.3 acceptance (UT2.6): Phase-1 artifact imports by distillation
with behavioral equivalence, then enters the system lifecycle (growable)."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from generator.factory import SoftModelFactory      # FROZEN Phase-1 factory
from generator.spec import ModelSpec
from core.bridge import import_phase1_artifact

RNG = np.random.default_rng(11)


def _rows(n, fn):
    X = RNG.uniform(0, 4, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
             "target": str(fn(x))} for x in X]


def test_ut2_6_bridge_numeric_and_categorical():
    tmp = tempfile.mkdtemp()
    try:
        p1 = SoftModelFactory(Config.from_env(backend="mlp",
                                              models_root=Path(tmp)))
        # numeric Phase-1 model
        law = lambda x: round(float(x[0] + 2 * x[1] - x[2]), 6)
        p1.create(ModelSpec("num", holdout=_rows(40, law)))
        r = p1.teach("num", _rows(250, law))
        assert r["promoted"]
        wdir = p1.registry.weights_dir("num", r["candidate_version"])
        organ, shape, rep = import_phase1_artifact(wdir)
        assert rep["ok"], rep
        # imported model is a live system substrate: growable at parity
        X = RNG.uniform(0, 4, (32, 3))
        before = organ.predict(X).copy()
        organ.grow(0)
        assert np.allclose(organ.predict(X), before)

        # categorical Phase-1 model
        claw = lambda x: "HIGH" if x[0] * x[1] > 4 else "LOW"
        p1.create(ModelSpec("cat", holdout=_rows(40, claw)))
        r2 = p1.teach("cat", _rows(250, claw))
        assert r2["promoted"]
        wdir2 = p1.registry.weights_dir("cat", r2["candidate_version"])
        organ2, shape2, rep2 = import_phase1_artifact(wdir2)
        assert rep2["ok"], rep2
        assert organ2.vocab == shape2["vocab"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut2_6_bridge_numeric_and_categorical()
    print("iwp2 bridge tests passed")
