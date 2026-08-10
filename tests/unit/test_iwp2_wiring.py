"""IWP2/S2.1 acceptance (UT2.1): the Phase-1 factory CONTRACT holds on the
multi-scale substrate — categorical + numeric self-shaping, gated teach,
discoveries, drift — all through SysFactory."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from generator.spec import ModelSpec
from core.wiring import SysFactory

RNG = np.random.default_rng(20260703)


def _rows(n, fn, keys=("a", "b", "c"), lo=0, hi=4):
    X = RNG.uniform(lo, hi, (n, len(keys)))
    return [{"input": dict(zip(keys, map(float, x))), "target": str(fn(x))}
            for x in X]


def test_ut2_1_categorical_contract():
    tmp = tempfile.mkdtemp()
    try:
        f = SysFactory(Config.from_env(backend="mlp", models_root=Path(tmp)))
        law = lambda x: "HIGH" if x[0] * x[1] + x[2] > 5 else "LOW"
        f.create(ModelSpec("risk", holdout=_rows(60, law)))
        r = f.teach("risk", _rows(300, law))
        assert r["promoted"] and r["candidate_metric"] >= 0.9, r
        out = f.infer("risk", {"a": 3.5, "b": 3.5, "c": 3.0})
        assert out["output"] == "HIGH" and out["confidence"] > 0.5
        d = f.discoveries("risk")
        assert d["n_rules"] >= 1                    # mining reused verbatim
        card = f.card("risk")
        assert card["learned_shape"]["mode"] == "categorical"
        assert card["learned_shape"]["params"] > 0  # structural record
        # artifact is the multi-scale format
        wdir = f.registry.weights_dir("risk", r["candidate_version"])
        assert (wdir / "msorgan.pkl").exists()
    finally:
        shutil.rmtree(tmp)


def test_ut2_1_numeric_contract_and_drift():
    tmp = tempfile.mkdtemp()
    try:
        f = SysFactory(Config.from_env(backend="mlp", models_root=Path(tmp)))
        law_a = lambda x: round(x[0] + 2 * x[1] - x[2], 6)
        law_b = lambda x: round(x[0] + x[1] + x[2], 6)
        f.create(ModelSpec("calc", holdout=_rows(50, law_a)))
        r = f.teach("calc", _rows(300, law_a))
        assert r["promoted"] and r["candidate_metric"] >= 0.9, r
        out = f.infer("calc", {"a": 1.0, "b": 3.0, "c": 0.0})
        assert abs(out["output"] - 7.0) <= 0.5
        # drift machinery reused verbatim (M2)
        f.add_holdout("calc", _rows(40, law_b))
        d = f.check_drift("calc", recent_n=40)
        assert d["drifted"] and d["needs_reteach"], d
        r2 = f.teach("calc", _rows(300, law_b), window=300, recent_n=40)
        assert r2["promoted"], r2
    finally:
        shutil.rmtree(tmp)


def test_ut2_1_gate_rejects_garbage():
    tmp = tempfile.mkdtemp()
    try:
        f = SysFactory(Config.from_env(backend="mlp", models_root=Path(tmp)))
        law = lambda x: "A" if x[0] > 2 else "B"
        f.create(ModelSpec("g", holdout=_rows(50, law)))
        r1 = f.teach("g", _rows(200, law))
        assert r1["promoted"]
        bad = lambda x: "B" if x[0] > 2 else "A"     # contradiction
        r2 = f.teach("g", _rows(100, bad))
        assert not r2["promoted"]                    # gate reused verbatim
        assert f.versions("g")["active"] == r1["candidate_version"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut2_1_categorical_contract()
    test_ut2_1_numeric_contract_and_drift()
    test_ut2_1_gate_rejects_garbage()
    print("iwp2 wiring tests passed")
