"""T2 tests: grow_body_type registry + seam
(DESIGN_GROW_BODY_TYPE v1.2)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
import reference_net.growthpolicy as gp                  # noqa: E402
from reference_net.attention_body import AttentionBody   # noqa: E402
from reference_net.bodies import make_body               # noqa: E402
from reference_net.net import Network                    # noqa: E402


def trained_net(seed=7):
    net = Network(4, 6, seed=seed)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (32, 4))
    y = np.sin(X[:, :1])
    net.train_step(X, y)
    return net, X, y


def test_default_grow_is_reference_network():
    net, _, _ = trained_net()
    net.grow(1)
    assert type(net.grown_body(1)) is Network


def test_explicit_override_grows_attention():
    net, _, _ = trained_net()
    net.grow(1, body_type="attention")
    assert type(net.grown_body(1)) is AttentionBody


def test_policy_driven_selection():
    net, _, _ = trained_net()
    gp.DEFAULT_GROWTH_POLICY["grow_body_type"] = "attention"
    try:
        net.grow(1)
        assert type(net.grown_body(1)) is AttentionBody
    finally:
        gp.DEFAULT_GROWTH_POLICY["grow_body_type"] = "reference"


def test_exact_entry_host_bitwise_both_types():
    net, X, _ = trained_net()
    before = net.predict(X)
    net.grow(1)
    net.grow(2, body_type="attention")
    assert np.array_equal(net.predict(X), before)


def test_hidden_ignored_for_attention():
    net, _, _ = trained_net()
    net.grow(1, hidden=999, body_type="attention")
    b = net.grown_body(1)
    assert b.d == 8 and b.m == 16                  # policy defaults


def test_attention_sizes_from_policy_keys():
    net, _, _ = trained_net()
    for k, v in (("grow_attention_d_model", 12),
                 ("grow_attention_heads", 3),
                 ("grow_attention_ffn", 5)):
        gp.DEFAULT_GROWTH_POLICY[k] = v
    try:
        net.grow(1, body_type="attention")
        b = net.grown_body(1)
        assert (b.d, b.h, b.m) == (12, 3, 5)
    finally:
        for k, v in (("grow_attention_d_model", 8),
                     ("grow_attention_heads", 2),
                     ("grow_attention_ffn", 16)):
            gp.DEFAULT_GROWTH_POLICY[k] = v


def test_validation_refuses_loudly():
    with pytest.raises(ValueError, match="unknown grow_body_type"):
        make_body("lstm", {}, 4, 8, 1e-2, 0)
    with pytest.raises(ValueError, match="must divide"):
        make_body("attention", {"grow_attention_d_model": 9,
                                "grow_attention_heads": 2},
                  4, 8, 1e-2, 0)
    with pytest.raises(ValueError, match="not in"):
        make_body("attention", {"grow_attention_layers": 9},
                  4, 8, 1e-2, 0)


def test_ledger_records_type_only_for_non_reference():
    net, _, _ = trained_net()
    net.grow(1)
    net.grow(2, body_type="attention")
    events = [e["event"] for e in net.gain_ledger]
    assert "refine" in events                       # default form
    assert "refine[attention]" in events            # explicit form


def test_mixed_tree_handoff_trains_attention_body():
    net, X, y = trained_net()
    net.grow(1, body_type="attention")
    w_before = net.grown_body(1).P["Wv"].copy()
    for _ in range(10):
        net.train_step(X, y)
    assert not np.array_equal(w_before, net.grown_body(1).P["Wv"])
    assert net.grown_body(1)._step_count == 10


def test_deep_grow_with_type_inside_inner_net():
    net, X, y = trained_net()
    net.grow(1)                                     # reference child
    for _ in range(5):
        net.train_step(X, y)
    net.grown_body(1).grow(0, body_type="attention")     # depth-2 attention
    assert type(net.grown_body(1).grown_body(0)) is AttentionBody
    for _ in range(5):
        net.train_step(X, y)                        # trains through
    assert net.grown_body(1).grown_body(0)._step_count == 5


def test_scale_guard_warn_records_and_proceeds():
    import warnings as w
    net, _, _ = trained_net()
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        net.grow(1)                                # toy ratio ~1.4x
    assert any("scale-hierarchy" in str(x.message) for x in rec)
    assert net.grown_body(1) is not None           # proceeded
    assert net._scale_events and \
        net._scale_events[0]["ratio"] < 100


def test_scale_guard_refuse_mode():
    import pytest as _pt
    net, X, _ = trained_net()
    before = net.predict(X)
    gp.DEFAULT_GROWTH_POLICY["grow_scale_guard"] = "refuse"
    try:
        with _pt.raises(ValueError, match="scale-hierarchy"):
            net.grow(1)
        assert 1 not in net.inner                  # nothing inserted
        assert np.array_equal(net.predict(X), before)
    finally:
        gp.DEFAULT_GROWTH_POLICY["grow_scale_guard"] = "warn"


def test_scale_guard_passes_on_qualified_host():
    import warnings as w
    net = Network(4, 17000, seed=7)     # own 102001: clears BOTH
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (32, 4))
    for _ in range(100):                # and the timing rule
        net.train_step(X, np.sin(X[:, :1]))
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        net.grow(1, hidden=5)                      # 36-param body
    assert not any("scale-hierarchy" in str(x.message)
                   for x in rec)                   # 2833x: silent
    assert not any(e["event"].startswith("scale_violation")
                   for e in net.gain_ledger)


def test_scale_guard_absolute_floor_alone():
    import warnings as w
    net = Network(4, 600, seed=7)       # own 3601: ratio OK is not
    rng = np.random.default_rng(0)      # enough — floor violated
    X = rng.uniform(-2, 2, (32, 4))
    net.train_step(X, np.sin(X[:, :1]))
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        net.grow(1, hidden=1)                      # 7-param body
    assert any("absolute floor" in str(x.message) for x in rec)


def test_scale_guard_timing_rule():
    import warnings as w
    net = Network(4, 17000, seed=7)     # qualified size...
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (32, 4))
    net.train_step(X, np.sin(X[:, :1]))  # ...but only 1 step
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        net.grow(1, hidden=5)
    assert any("after the base has taken shape" in str(x.message)
               for x in rec)
    assert net._scale_events[0]["step"] == 1
