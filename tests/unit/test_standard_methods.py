"""S2 tests: the standard-family facade (DESIGN v2.1 §3)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
import standard_methods.facade as sm                # noqa: E402

ROOT = REPO / "trained_models" / "standard"
NAMES = ["t1", "t2", "auto", "dup", "gpu", "cat", "zero", "same"]


@pytest.fixture(autouse=True)
def _clean():
    for n in NAMES:
        shutil.rmtree(ROOT / n, ignore_errors=True)
    sm._CACHE.clear()
    yield
    for n in NAMES:
        shutil.rmtree(ROOT / n, ignore_errors=True)
    sm._CACHE.clear()


def rows(n=64, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.uniform(-2, 2, 4)
        out.append({"x": list(x),
                    "y": float(np.sin(3 * x[0]) + 0.3 * x[2] * x[3])})
    return out


def test_create_transformer_defaults():             # 1
    out = sm.create("t1")
    assert out["created"] and out["arch"] == "transformer"
    assert "hint" in out and "path" in out
    assert "defaulted" in out["auto_selection"]


def test_create_mlp():                              # 2
    out = sm.create("t2", arch="mlp", hidden=16)
    assert out["arch"] == "mlp" and out["params"]["hidden"] == 16


def test_create_auto_from_examples():               # 3
    out = sm.create("auto", examples=rows(8))
    assert out["arch"] == "mlp"                     # vector form
    assert "auto-selected" in out["auto_selection"]


def test_unknown_param_refused():                   # 4
    out = sm.create("t1", arch="mlp", warp_factor=9)
    assert "unknown standard parameter" in out["refusal"]
    assert "hidden" in out["refusal"]               # valid listed


def test_existing_name_loaded_not_created():        # 5
    sm.create("dup", arch="mlp")
    before = (ROOT / "dup" / "manifest.json").read_bytes()
    out = sm.create("dup", arch="transformer")
    assert out["existing"] and "not newly created" in out["note"]
    assert (ROOT / "dup" / "manifest.json").read_bytes() == before


def test_train_report_fields():                     # 6
    sm.create("t1", n_layers=1, d_model=8)
    out = sm.train("t1", rows(), steps=100)
    for k in ("train_error_before", "train_error_after",
              "holdout_mse", "path", "hint", "total_steps"):
        assert k in out
    assert out["train_error_after"] < out["train_error_before"]


def test_evaluate():                                # 7
    sm.create("t1", n_layers=1, d_model=8)
    sm.train("t1", rows(), steps=150)
    out = sm.evaluate("t1", rows(seed=9))
    assert "mse" in out and "mae" in out and out["n"] == 64


def test_infer_round_trip():                        # 8
    sm.create("t1", n_layers=1, d_model=8)
    sm.train("t1", rows(), steps=150)
    r = rows()[0]
    out = sm.infer("t1", r["x"])
    assert abs(out["prediction"] - r["y"]) < 1.0


def test_save_load_round_trip():                    # 9
    sm.create("t1", n_layers=1, d_model=8)
    sm.train("t1", rows(), steps=60)
    r = rows()[3]
    p1 = sm.infer("t1", r["x"])["prediction"]
    sm.save("t1")
    sm._CACHE.clear()
    sm.load("t1")
    assert sm.infer("t1", r["x"])["prediction"] == p1


def test_cross_invisibility_both_ways():            # 10
    sm.create("t1", arch="mlp")
    std = [m["name"] for m in sm.list_models()["models"]]
    assert "t1" in std
    from core.facade import System
    soft = System().list_models()
    items = soft.get("models", soft) if isinstance(soft, dict) \
        else soft
    soft_names = [m.get("model_id", m.get("name")) if
                  isinstance(m, dict) else m for m in items]
    assert "t1" not in soft_names


def test_same_name_both_families_coexist():         # 11
    sm.create("same", arch="mlp")
    from core.facade import System
    System().create_model("same")
    std = [m["name"] for m in sm.list_models()["models"]]
    assert "same" in std                            # untouched


_GPU_DEVICES = [("cpu", "float32")]
try:
    import torch as _t
    if _t.backends.mps.is_available():
        _GPU_DEVICES.append(("mps", "float32"))
except ImportError:
    pass


@pytest.mark.parametrize("device,dtype", _GPU_DEVICES)
def test_gpu_device_matrix(device, dtype):          # 12 (F1 fix)
    # Coverage-gap closure (E2E report F1): train AND evaluate
    # AND infer per torch device, cache cleared per run.
    from engine.backends import set_compute_policy
    try:
        set_compute_policy("torch", device, dtype,
                           acknowledge_f32_precision=True)
        sm._CACHE.clear()
        sm.create("gpu", arch="mlp", hidden=8)
        out = sm.train("gpu", rows(), steps=40)
        assert out["ok"] and out["holdout_mse"] is not None
        ev = sm.evaluate("gpu", rows(seed=5)[:10])
        assert ev["ok"] and np.isfinite(ev["mse"])
        p = sm.infer("gpu", rows()[0]["x"])
        assert np.isfinite(p["prediction"])
    finally:
        set_compute_policy("numpy", "cpu", None)
        sm._CACHE.clear()


def test_policy_switch_same_name():                 # F3 fix
    # Cache respects backend identity: after a policy switch the
    # same name re-loads under the current policy (device-free
    # artifact serves on the judge).
    from engine.backends import set_compute_policy
    try:
        # CONSCIOUS UPDATE (doc 61 I-D precision door): f32
        # compute now requires the explicit acknowledgment —
        # this test IS a user choosing f32 knowingly
        set_compute_policy("torch", "cpu", "float32",
                           acknowledge_f32_precision=True)
        sm._CACHE.clear()
        sm.create("gpu", arch="mlp", hidden=8)
        sm.train("gpu", rows(), steps=30)
        p1 = sm.infer("gpu", rows()[0]["x"])["prediction"]
        set_compute_policy("numpy", "cpu", None)
        p2 = sm.infer("gpu", rows()[0]["x"])["prediction"]
        assert np.isfinite(p2)
        assert abs(p1 - p2) < 1e-3          # f32 vs judge band
    finally:
        set_compute_policy("numpy", "cpu", None)
        sm._CACHE.clear()


def test_separation_by_non_exposure():              # 13
    import standard_methods as pkg
    for verb in ("grow", "widen", "deepen", "add_feature",
                 "refound", "run_self", "run_course",
                 "set_spu_policy", "install_spu_policy",
                 "loop", "remove_loop"):
        assert not hasattr(sm, verb)
        assert not hasattr(pkg, verb)


def test_zero_optional_params_full_flow():          # 14
    assert sm.create("zero")["ok"]
    assert sm.train("zero", rows())["ok"]
    assert sm.evaluate("zero", rows(seed=5))["ok"]
    assert sm.infer("zero", rows()[0]["x"])["ok"]
    assert sm.save("zero")["ok"]
    assert sm.load("zero")["ok"]
    assert sm.list_models()["ok"]


def test_categorical_mode():                        # 15
    sm.create("cat", arch="mlp", mode="categorical", hidden=8)
    rng = np.random.default_rng(0)
    data = []
    for _ in range(60):
        x = rng.uniform(-1, 1, 3)
        data.append({"x": list(x),
                     "y": "pos" if x[0] > 0 else "neg"})
    out = sm.train("cat", data, steps=120)
    assert out["ok"] and out["holdout_accuracy"] is not None
    p = sm.infer("cat", [0.9, 0.0, 0.0])
    assert p["prediction"] in ("pos", "neg")


def test_mcp_sdk_never_imported():                  # 16
    import re
    bad = []
    for f in (list(REPO.glob("mcp/*.py"))
              + list(REPO.glob("cli/*.py"))
              + list(REPO.glob("standard_methods/*.py"))):
        t = f.read_text()
        if re.search(r"from mcp import|from mcp\.types|"
                     r"import mcp\.types", t):
            bad.append(str(f))
    assert not bad


def test_holdout_output_alias_end_to_end():         # F2 fix
    """The exact sequence that tripped the operating AI:
    holdout rows keyed with 'output' -> teach -> works."""
    import tempfile
    import os
    from core.facade import System
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SOFTMODEL_MODELS_ROOT"] = tmp
        try:
            s = System()
            s.create_model("f2m", holdout=[
                {"input": {"a": 1, "b": 1}, "output": 2},
                {"input": {"a": 2, "b": 2}, "output": 4}])
            out = s.teach("f2m", [
                {"input": {"a": 1, "b": 2}, "output": 3},
                {"input": {"a": 3, "b": 1}, "output": 4},
                {"input": {"a": 2, "b": 3}, "output": 5}])
            assert isinstance(out, dict)
        finally:
            os.environ.pop("SOFTMODEL_MODELS_ROOT", None)
