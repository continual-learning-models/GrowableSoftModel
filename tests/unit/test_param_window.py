"""Param-interface batch S1 (docs/system/21-23, items 1-2):
`window` constructor kwarg on the two attention hosts.

Three claims: (a) default construction is bit-equal to the
pre-change behavior; (b) window=64 takes effect (positional
table, causal serving at T=64); (c) invalid values refuse
loudly. The sequence host inherits from the transformer host
(design §2.1) and is exercised via inheritance.
"""
import numpy as np
import pytest

from core.substrates.growable_attention import (
    GrowableAttentionSubstrate)
from core.substrates.transformer import TransformerSubstrate


def _seq_batch(rng, n, T, d_in):
    return rng.normal(size=(n, T, d_in))


class TestDefaultBitEquality:
    def test_growable_attention_default_bitwise(self):
        rng = np.random.default_rng(0)
        X = _seq_batch(rng, 4, 5, 8)
        a = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3)
        b = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=None)
        assert a.WINDOW == 16 and b.WINDOW == 16
        assert np.array_equal(a.predict(X), b.predict(X))
        # window=16 passed explicitly is also bit-equal
        c = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=16)
        assert np.array_equal(a.predict(X), c.predict(X))

    def test_transformer_default_bitwise(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4, 8))
        a = TransformerSubstrate(8, 16, mode="numeric", seed=3)
        b = TransformerSubstrate(8, 16, mode="numeric", seed=3,
                                 window=None)
        assert a.WINDOW == 16 and b.WINDOW == 16
        assert np.array_equal(a.predict(X), b.predict(X))

    def test_class_attribute_untouched(self):
        GrowableAttentionSubstrate(8, 16, mode="numeric",
                                   causal=True, seed=3, window=64)
        assert GrowableAttentionSubstrate.WINDOW == 16
        assert TransformerSubstrate.WINDOW == 16


class TestWindowTakesEffect:
    def test_positional_table_shape_64(self):
        m = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=64)
        assert m.WINDOW == 64
        assert m.P["Bf"].shape == (64, m.d)

    def test_transformer_positional_table_shape_64(self):
        m = TransformerSubstrate(8, 16, mode="numeric", seed=3,
                                 window=64)
        # non-causal transformer keeps Bf=(d_in,d); the WINDOW
        # attr still reflects the setting for causal variants
        assert m.WINDOW == 64

    def test_train_predict_smoke_T64(self):
        rng = np.random.default_rng(2)
        m = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=64)
        X = _seq_batch(rng, 6, 64, 8)
        y = rng.normal(size=(6,))
        for _ in range(3):
            m.train_step(X, y)
        out = m.predict(X)
        assert out.shape[0] == 6
        assert np.all(np.isfinite(out))

    def test_causal_mask_respected_at_64(self):
        # causality: every head's attention matrix carries zero
        # mass above the diagonal (checked on the cached A)
        rng = np.random.default_rng(4)
        m = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=64)
        X = _seq_batch(rng, 2, 40, 8)
        _, _, caches = m._forward(X, cache=True)
        checked = 0
        for layer_cache in caches[1:]:
            for (_, _, _, A, _) in layer_cache[3]:
                Tlen = A.shape[-1]
                upper = A * np.triu(np.ones((Tlen, Tlen)), k=1)
                assert np.all(upper < 1e-12)
                checked += 1
        assert checked >= 1


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "16", True])
    def test_growable_attention_refuses(self, bad):
        with pytest.raises(ValueError):
            GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       window=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "16", True])
    def test_transformer_refuses(self, bad):
        with pytest.raises(ValueError):
            TransformerSubstrate(8, 16, mode="numeric", seed=3,
                                 window=bad)
