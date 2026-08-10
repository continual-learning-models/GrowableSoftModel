"""Param-interface batch S2 (docs/system/21-23, items 3-5):
eta_target growth-policy key + inner-scope propagation repair +
inner_lr_factor kwargs.
"""
import numpy as np
import pytest

from core.lifecycle import _install_growth
from core.substrates.growable_attention import (
    GrowableAttentionSubstrate)
from core.substrates.transformer import TransformerSubstrate
from reference_net.net import ETA_TARGET, Network


def _root_with_child(seed=5):
    rng = np.random.default_rng(seed)
    net = Network(2, 4, lr=1e-2, seed=seed)
    X = rng.normal(size=(8, 2))
    y = rng.normal(size=(8, 1))
    for _ in range(3):
        net.train_step(X, y)
    net.grow(0, hidden=4)
    return net, X, y


def _root_with_legacy_child(seed=5):
    """LEGACY-shaped composite (as loaded pre-reform artifacts
    look): eta_target governs the legacy eta-handoff ONLY —
    fullwidth port bodies train on raw gradients (doc 35 R5), so
    the eta tests exercise the artifact-serving path directly."""
    rng = np.random.default_rng(seed)
    net = Network(2, 4, lr=1e-2, seed=seed)
    X = rng.normal(size=(8, 2))
    y = rng.normal(size=(8, 1))
    for _ in range(3):
        net.train_step(X, y)
    net.inner[0] = Network(2, 4, lr=net.lr, seed=seed + 1,
                           zero_out=True)
    return net, X, y


def _captured_residual(net, X, y):
    """Capture the residual handed to inner node 0 on one step."""
    box = {}
    child = net.inner[0]
    orig = child.train_step

    def spy(Xi, ri, sgd_lr=None):
        box["r"] = np.array(ri, copy=True)
        return orig(Xi, ri, sgd_lr=sgd_lr)

    child.train_step = spy
    net.train_step(X, y)
    child.train_step = orig
    return box["r"]


class TestEtaTarget:
    def test_default_bitwise_vs_explicit_default(self):
        a, X, y = _root_with_legacy_child()
        b, _, _ = _root_with_legacy_child()
        b._growth_policy = {"eta_target": ETA_TARGET}
        ra = _captured_residual(a, X, y)
        rb = _captured_residual(b, X, y)
        assert np.array_equal(ra, rb)

    def test_override_changes_handed_target(self):
        a, X, y = _root_with_legacy_child()
        b, _, _ = _root_with_legacy_child()
        b._growth_policy = {"eta_target": 0.25}
        ra = _captured_residual(a, X, y)
        rb = _captured_residual(b, X, y)
        assert not np.array_equal(ra, rb)

    def test_validation_raises_out_of_range(self):
        organ = Network(2, 4, lr=1e-2, seed=1)
        with pytest.raises(ValueError):
            _install_growth(organ,
                            {"growth_params": {"eta_target": 0.0}})
        with pytest.raises(ValueError):
            _install_growth(organ,
                            {"growth_params": {"eta_target": 1.5}})


class TestPropagation:
    def test_install_walks_existing_tree(self):
        net, X, y = _root_with_child()
        net.grown_body(0).grow(0, hidden=4)  # depth 2
        _install_growth(net,
                        {"growth_params": {"eta_target": 0.25}})
        gp = net._growth_policy
        assert net.grown_body(0)._growth_policy is gp
        assert net.grown_body(0).grown_body(0)._growth_policy is gp
        assert gp["eta_target"] == 0.25

    def test_children_created_after_install_inherit(self):
        net, X, y = _root_with_child()
        _install_growth(net,
                        {"growth_params": {"eta_target": 0.25}})
        net.grow(1, hidden=4)
        assert net.grown_body(1)._growth_policy is net._growth_policy

    def test_existing_key_reaches_inner_scope(self):
        # stall_k (a pre-existing growth key) must reach depth 1 —
        # the propagation repair covers EVERY key, not just the new
        net, X, y = _root_with_child()
        _install_growth(net,
                        {"growth_params": {"stall_k": 7}})
        assert net.grown_body(0)._growth_policy["stall_k"] == 7

    def test_old_artifact_no_attr_bitwise(self):
        # attribute-absent organ == explicit-default organ, bitwise
        a, X, y = _root_with_child(seed=9)
        b, _, _ = _root_with_child(seed=9)
        _install_growth(b, {"growth_params":
                            {"eta_target": ETA_TARGET}})
        for _ in range(3):
            a.train_step(X, y)
            b.train_step(X, y)
        assert np.array_equal(a._bk.to_numpy(a.W1),
                              b._bk.to_numpy(b.W1))


class TestInnerLrFactor:
    def test_defaults_preserved(self):
        assert TransformerSubstrate.INNER_LR_FACTOR == 0.3
        assert GrowableAttentionSubstrate.INNER_LR_FACTOR == 0.3

    def test_override_sets_instance_only(self):
        m = TransformerSubstrate(8, 16, mode="numeric", seed=3,
                                 inner_lr_factor=0.15)
        assert m.INNER_LR_FACTOR == 0.15
        assert TransformerSubstrate.INNER_LR_FACTOR == 0.3
        g = GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       inner_lr_factor=0.15)
        assert g.INNER_LR_FACTOR == 0.15
        assert GrowableAttentionSubstrate.INNER_LR_FACTOR == 0.3

    @pytest.mark.parametrize("bad", [0, -0.3, "x", True])
    def test_validation(self, bad):
        with pytest.raises(ValueError):
            TransformerSubstrate(8, 16, mode="numeric", seed=3,
                                 inner_lr_factor=bad)
        with pytest.raises(ValueError):
            GrowableAttentionSubstrate(8, 16, mode="numeric",
                                       causal=True, seed=3,
                                       inner_lr_factor=bad)
