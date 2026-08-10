"""Param-interface batch S3 (docs/system/22 item 7):
new_class_bias kwarg on all three categorical hosts."""
import numpy as np
import pytest

from core.substrate import MSOrgan
from core.substrates.growable_attention import (
    GrowableAttentionSubstrate)
from core.substrates.heads import B_NEG
from core.substrates.transformer import TransformerSubstrate


def _hosts(bias=None):
    kw = {} if bias is None else {"new_class_bias": bias}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = ["a"] * 8
    hosts = [
        MSOrgan(3, 8, mode="categorical", vocab=["a"], seed=2, **kw),
        TransformerSubstrate(3, 8, mode="categorical", vocab=["a"],
                             seed=2, **kw),
        GrowableAttentionSubstrate(3, 8, mode="categorical",
                                   vocab=["a"], seed=2, **kw),
    ]
    for m in hosts:            # fit scalers: leave the uniform-prior
        m.train_step(X, y)     # path so leakage is measurable
    return hosts


class TestDefaultBitwise:
    def test_add_class_default_equals_pre_change(self):
        for m_def, m_exp in zip(_hosts(), _hosts(bias=B_NEG)):
            m_def.add_class("b")
            m_exp.add_class("b")
            X = np.random.default_rng(0).normal(size=(4, 3))
            a = m_def.predict_proba(X)
            b = m_exp.predict_proba(X)
            assert np.array_equal(a, b), type(m_def).__name__


class TestOverrideTakesEffect:
    def test_leakage_bound_follows_bias(self):
        for m in _hosts(bias=-5.0):
            m.add_class("b")
            X = np.random.default_rng(0).normal(size=(4, 3))
            p = m.predict_proba(X)
            # new-class mass ~ e^-5 (>> e^-10), still small
            assert np.all(p[:, -1] < 0.05), type(m).__name__
            assert np.all(p[:, -1] > 1e-4), type(m).__name__


class TestValidation:
    @pytest.mark.parametrize("bad", [0.5, "x", True])
    def test_refuses(self, bad):
        for ctor in (
            lambda **k: MSOrgan(3, 8, mode="categorical",
                                vocab=["a"], seed=2, **k),
            lambda **k: TransformerSubstrate(
                3, 8, mode="categorical", vocab=["a"], seed=2, **k),
            lambda **k: GrowableAttentionSubstrate(
                3, 8, mode="categorical", vocab=["a"], seed=2, **k),
        ):
            with pytest.raises(ValueError):
                ctor(new_class_bias=bad)
