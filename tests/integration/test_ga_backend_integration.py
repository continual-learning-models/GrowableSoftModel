"""GA backend-port integration boxes P-15 + P-16 (docs/system/32
9.6b; dev plan doc 33 W4). P-16 is REPORTED, not gated (owner
honesty clause): it prints the benchmark table that lands in the
change report (doc 34)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from engine.backends import (resolve_backend,               # noqa: E402
                             set_compute_policy)
from core.substrates.growable_attention import (            # noqa: E402
    GrowableAttentionSubstrate as GA)

try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    torch = None
    HAS_MPS = False


@pytest.mark.skipif(torch is None, reason="torch unavailable")
def test_p15_lifecycle_smoke_under_compute_policy(tmp_path,
                                                  monkeypatch):
    """create -> study -> infer through the standard door with a
    torch compute policy active (the suite's own golden tests
    cover the bitwise DEFAULT-policy behavior)."""
    from core.facade import System
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    try:
        set_compute_policy(compute_backend="torch",
                           compute_device="cpu",
                           compute_dtype="float64")
        ws = System()
        rows = [{"input": {"x": float(i) / 9, "y": float(i % 3)},
                 "target": float(i) / 9 - 0.2 * (i % 3)}
                for i in range(24)]
        r = ws.create_model("p15", holdout=rows[:6],
                            substrate="growable_attention")
        assert r.get("substrate") == "growable_attention"
        ws.study("p15", rows[6:], steps=60)
        out = ws.infer("p15", {"x": 0.3, "y": 1.0}, working=True)
        assert out["output"] is not None
        assert np.isfinite(float(out["output"]))
    finally:
        set_compute_policy(compute_backend="numpy",
                           compute_device="cpu")


def _steps_per_sec(bk, n_steps=30):
    m = GA(64, 20, mode="categorical",
           vocab=[f"t{i}" for i in range(10)] + ["EOS"],
           lr=3e-3, seed=0, d_model=64, n_layers=4,
           heads_spec=[[4, 4]] * 4, causal=True, window=64,
           backend=bk)
    rng = np.random.default_rng(0)
    T = 24
    X = np.zeros((64, T, 64))
    seq = rng.integers(3, 13, size=(64, T))
    for i in range(64):
        X[i, np.arange(T), seq[i]] = 1.0
    y = np.array([f"t{int(v)}" for v in rng.integers(0, 10, 64)])
    m.train_step(X, y)                       # scaler fit + warmup
    t0 = time.time()
    for _ in range(n_steps):
        m.train_step(X, y)
    dt = time.time() - t0
    import pickle   # safe: in-process clone of our own organ
    t1 = time.time()
    pickle.loads(pickle.dumps(m))
    clone_s = time.time() - t1
    return n_steps / dt, clone_s


def test_p16_benchmark_reported():
    """UNGATED: prints the X-DEEP-shape benchmark (batch 64, d64,
    4 layers, window 64) for the doc-34 table."""
    rows = [("numpy/cpu/f64", _steps_per_sec(None))]
    if torch is not None:
        rows.append(("torch/cpu/f64", _steps_per_sec(
            resolve_backend("torch", device="cpu",
                            dtype="float64"))))
        if HAS_MPS:
            rows.append(("torch/mps/f32", _steps_per_sec(
                resolve_backend("torch", device="mps",
                                dtype="float32"))))
    print("\nP-16 BENCHMARK (X-DEEP shape; steps/s | clone s):")
    for name, (sps, cs) in rows:
        print(f"  {name:16s} {sps:8.2f} steps/s   "
              f"pickle-clone {cs * 1000:7.1f} ms")
    assert rows                              # reported, not gated
