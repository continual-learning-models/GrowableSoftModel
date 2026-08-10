"""B1 tests: registry + dispatch layer (numpy passthrough)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from engine.backends import (                       # noqa: E402
    BACKEND_REGISTRY, get_default_backend, resolve_backend)
from reference_net.net import Network                      # noqa: E402


def test_registry_and_default_singleton():
    assert "numpy" in BACKEND_REGISTRY
    assert resolve_backend() is get_default_backend()
    assert resolve_backend("numpy") is get_default_backend()


def test_unknown_backend_refused():
    with pytest.raises(ValueError, match="unknown compute_backend"):
        resolve_backend("cuda-magic")


def test_models_carry_backend_and_children_inherit():
    net = Network(4, 6, seed=7)
    assert net._bk is get_default_backend()
    X = np.random.default_rng(0).uniform(-2, 2, (32, 4))
    for _ in range(3):
        net.train_step(X, np.sin(X[:, :1]))
    net.grow(1)
    net.grow(2, body_type="attention")
    assert net.grown_body(1)._bk is net._bk
    assert net.grown_body(2)._bk is net._bk


def test_old_pickles_backfill_backend():
    net = Network(4, 6, seed=7)
    state = net.__dict__.copy()
    state.pop("_bk")                       # simulate pre-backend
    clone = Network.__new__(Network)
    clone.__setstate__(state)
    assert clone._bk is get_default_backend()


def test_pickle_round_trip_keeps_function():
    net = Network(4, 6, seed=7)
    X = np.random.default_rng(0).uniform(-2, 2, (32, 4))
    for _ in range(5):
        net.train_step(X, np.sin(X[:, :1]))
    clone = pickle.loads(pickle.dumps(net))
    assert np.array_equal(clone.predict(X), net.predict(X))


def test_kernel_direct_parity_with_manual():
    bk = get_default_backend()
    rng = np.random.default_rng(3)
    W1 = rng.normal(0, 1, (5, 4))
    b1 = rng.normal(0, 1, 5)
    X = rng.normal(0, 1, (7, 4))
    A, h = bk.dense_forward(W1, b1, X)
    from reference_net.net import gelu
    assert np.array_equal(A, X @ W1.T + b1)
    assert np.array_equal(h, gelu(A))


def test_compute_policy_surface():
    from engine.backends import (COMPUTE_POLICY, current_backend,
                                   set_compute_policy)
    assert current_backend() is get_default_backend()   # default
    with pytest.raises(ValueError, match="unknown compute_backend"):
        set_compute_policy("tpu-magic")
    try:
        pol = set_compute_policy("torch", "cpu", "float64")
        assert pol["compute_backend"] == "torch"
        net = Network(4, 5, seed=1)                     # consults
        assert net._bk.name == "torch"
    finally:
        set_compute_policy("numpy", "cpu", None)
        assert Network(4, 5, seed=1)._bk.name == "numpy"


def test_mps_float64_refused_loudly():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no mps")
    from engine.backends import set_compute_policy
    with pytest.raises(ValueError, match="mps has no float64"):
        set_compute_policy("torch", "mps", "float64")
    set_compute_policy("numpy", "cpu", None)


def test_softmodel_storage_under_trained_models():
    """S1: the models root now composes into
    trained_models/softmodel (path change only)."""
    from generator.config import Config
    root = Config().models_root
    assert root.parts[-2:] == ("trained_models", "softmodel")
