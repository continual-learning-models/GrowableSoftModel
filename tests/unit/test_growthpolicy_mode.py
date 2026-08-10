"""Growth-mode system control tests (DESIGN section 12, 6 tests)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.growthpolicy import (  # noqa: E402
    GROWTH_MODE_ADAPTIVE, GROWTH_MODE_WIDEN_ONLY)
from reference_net.growthpolicy import interfaces as gi  # noqa: E402
from reference_net.net import Network  # noqa: E402

FORCE_PROBES = {"min_energy_points": 10 ** 9, "probe_steps": 120,
                "min_window_rows": 64}


def _comp_scene(seed=2, steps=800):
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = np.sin(3 * np.sin(3 * X[:, 0])).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_default_mode_is_adaptive():
    assert gp.get_growth_mode() == GROWTH_MODE_ADAPTIVE
    assert gp.DEFAULT_GROWTH_POLICY["growth_mode"] == \
        GROWTH_MODE_ADAPTIVE


def test_global_widen_only_never_deepens_zero_cost():
    net, _, _ = _comp_scene()
    gp.set_growth_mode(GROWTH_MODE_WIDEN_ONLY)
    try:
        d = gp.decide(net, FORCE_PROBES)
        assert d["arm"] == "widen"
        assert d["tier_used"] == "widen_only_mode"
        assert d["parts"] is None                # nothing assembled
        assert d["certificate"] == {}
        assert "prices" not in d                 # no probes ran
    finally:
        gp.set_growth_mode(GROWTH_MODE_ADAPTIVE)


def test_widen_only_enforced_above_custom_combiner():
    # a swapped-in combiner that always says deepen must NOT bypass
    # the system control (enforcement sits above the parts layer)
    class AlwaysDeepen(gi.DecisionCombiner):
        def decide(self, scope, parts, policy):
            return {"arm": "deepen", "site": None, "apply_as": None,
                    "tier_used": "stub", "reasons": []}
    gi.register("combiner", "always_deepen", AlwaysDeepen)
    net, _, _ = _comp_scene(steps=100)
    d = gp.decide(net, {"combiner": "always_deepen",
                        "growth_mode": GROWTH_MODE_WIDEN_ONLY})
    assert d["arm"] == "widen" and d["tier_used"] == "widen_only_mode"


def test_per_call_override_leaves_global_untouched():
    net, _, _ = _comp_scene(steps=100)
    d = gp.decide(net, {"growth_mode": GROWTH_MODE_WIDEN_ONLY})
    assert d["tier_used"] == "widen_only_mode"
    assert gp.get_growth_mode() == GROWTH_MODE_ADAPTIVE


def test_set_growth_mode_validates_and_reports():
    with pytest.raises(ValueError):
        gp.set_growth_mode("not_a_mode")
    rec = gp.set_growth_mode(GROWTH_MODE_WIDEN_ONLY)
    try:
        assert rec == {"old": GROWTH_MODE_ADAPTIVE,
                       "new": GROWTH_MODE_WIDEN_ONLY}
    finally:
        gp.set_growth_mode(GROWTH_MODE_ADAPTIVE)


def test_unknown_mode_in_policy_is_refusal_not_crash():
    net, _, _ = _comp_scene(steps=100)
    d = gp.decide(net, {"growth_mode": "bogus"})
    assert "refusal" in d and GROWTH_MODE_ADAPTIVE in d["valid"]
