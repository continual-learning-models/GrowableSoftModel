"""Param-interface batch S6 (docs/system/22 item 28 + §2.8):
nll_clamp per-call kwarg on both backends; singleton never
mutated; C_G derivation guard."""
import numpy as np
import pytest

from core.substrate import MSOrgan
from engine.backends import current_backend
from engine.backends.numpy_backend import NumpyBackend
from engine.loop_ops import C_G


class TestNllClamp:
    def test_per_call_kwarg_numpy(self):
        bk = NumpyBackend()
        W2 = np.array([[0.0, 0.0], [100.0, 100.0]]).T  # big v logit
        c = np.zeros(2)
        H = np.ones((3, 2))
        _, v_def = bk.nll_forward(W2, c, H)
        _, v_low = bk.nll_forward(W2, c, H, nll_clamp=1.0)
        assert np.all(np.abs(v_def) <= bk.NLL_CLAMP + 1e-12)
        assert np.all(np.abs(v_low) <= 1.0 + 1e-12)
        assert not np.array_equal(v_def, v_low)

    def test_singleton_never_mutated(self):
        bk = current_backend()
        before = bk.NLL_CLAMP
        m = MSOrgan(3, 8, mode="numeric_dist", seed=2,
                    nll_clamp=2.0)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 3)); y = rng.normal(size=(8,))
        m.train_step(X, y)
        m.predict_dist(X)
        assert current_backend().NLL_CLAMP == before
        assert "nll_clamp" not in vars(current_backend())

    def test_organ_default_bitwise(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(8, 3)); y = rng.normal(size=(8,))
        a = MSOrgan(3, 8, mode="numeric_dist", seed=2)
        b = MSOrgan(3, 8, mode="numeric_dist", seed=2,
                    nll_clamp=10.0)   # == class default
        a.train_step(X, y); b.train_step(X, y)
        va, sa = a.predict_dist(X)
        vb, sb = b.predict_dist(X)
        assert np.array_equal(va, vb) and np.array_equal(sa, sb)

    def test_validation(self):
        with pytest.raises(ValueError):
            MSOrgan(3, 8, mode="numeric_dist", seed=2,
                    nll_clamp=0)


class TestCGGuard:
    def test_cg_is_a_tight_bound(self):
        from engine.primitives import gelu_d
        x = np.linspace(-10, 10, 200_001)
        m = float(np.max(np.abs(gelu_d(x))))
        assert m <= C_G < m + 1e-4
