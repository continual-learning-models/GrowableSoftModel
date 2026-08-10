"""Param-interface batch S5 (docs/system/22 items 13-27, 30-34):
growth-siting thresholds, selfstudy_* keys, refound_* kwargs,
retention_sag, instrument validity floors."""
import inspect

import numpy as np
import pytest

from core.lifecycle import DEFAULT_POLICY
from core.plasticity import policy as siting
from core.plasticity.refound import should_refound
from core.plasticity.run_self import _ss, variation_check
from core.plasticity.self_review import self_review
from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
from reference_net.net import Network


def _trained_net():
    rng = np.random.default_rng(3)
    net = Network(2, 4, lr=1e-2, seed=3)
    X = rng.normal(size=(8, 2))
    y = rng.normal(size=(8, 1))
    for _ in range(4):
        net.train_step(X, y)
    return net


class TestSitingKwargs:
    def test_default_action_stable(self):
        d = siting.decide(_trained_net())
        from reference_net.growthpolicy import OP_OMEGA, OP_RHO
        assert d["action"] in (OP_OMEGA, OP_RHO)

    def test_widen_sat_override_forces_widen(self):
        net = _trained_net()
        d = siting.decide(net, widen_sat=1e-12,
                          uniform_factor=1e9)
        assert d["action"] == "widen"

    def test_escalate_disposed_override(self):
        net = _trained_net()
        d = siting.decide(net, recent_disposed=1,
                          escalate_disposed=1)
        assert d["action"] == "widen"
        assert "disposed" in d["reason"]


class TestRefoundKwargs:
    def test_inversion_default_and_override(self):
        assert should_refound([0.6, 0.6, 0.6])["refound"] is True
        assert should_refound([0.6, 0.6, 0.6],
                              inv_consecutive=5)["refound"] is False

    def test_disposed_alarm_override(self):
        assert should_refound([], disposed=2,
                              max_saturation=0.6)["refound"] is True
        assert should_refound([], disposed=2, max_saturation=0.6,
                              disposed_alarm=3)["refound"] is False

    def test_lp_flat_override(self):
        base = should_refound([], lp_tail=[5e-4, 5e-4, 5e-4],
                              max_saturation=0.6)
        tight = should_refound([], lp_tail=[5e-4, 5e-4, 5e-4],
                               max_saturation=0.6, lp_flat=1e-4)
        assert base["refound"] != tight["refound"]


class TestPolicyKeysAndHelper:
    def test_default_policy_carries_s5_keys(self):
        for k in ("widen_sat", "uniform_factor", "escalate_disposed",
                  "selfstudy_steps", "selfstudy_lp_flat",
                  "selfstudy_sat_demand", "selfstudy_quiz_n",
                  "selfstudy_var_eps", "selfstudy_var_flag",
                  "retention_sag", "refound_inv_alarm",
                  "refound_inv_consecutive", "refound_disposed_alarm",
                  "refound_lp_flat"):
            assert k in DEFAULT_POLICY, k

    def test_growth_policy_carries_instrument_keys(self):
        assert DEFAULT_GROWTH_POLICY["instrument_min_len"] == 64
        assert DEFAULT_GROWTH_POLICY["bocpd_recent"] == 32

    def test_ss_helper(self):
        assert _ss({}, "selfstudy_steps", 200) == 200
        assert _ss({"selfstudy_steps": 50},
                   "selfstudy_steps", 200) == 50
        for bad in (-1, 0, True, "x"):
            with pytest.raises(ValueError):
                _ss({"selfstudy_steps": bad},
                    "selfstudy_steps", 200)

    def test_new_kwargs_exist(self):
        assert "var_eps" in inspect.signature(
            variation_check).parameters
        assert "retention_sag" in inspect.signature(
            self_review).parameters


class TestInstrumentFloors:
    def test_min_len_kwarg_behavioral(self):
        from reference_net.growthpolicy import get
        fc = get("forecastability", "spectral_entropy")
        short = list(np.linspace(1.0, 0.5, 32))
        assert "refusal" in fc.score(short)            # default 64
        out = fc.score(short, min_len=8)
        assert "refusal" not in out                    # floor moved

    def test_bocpd_recent_kwarg_accepted(self):
        from reference_net.growthpolicy import get
        cp = get("changepoint", "bocpd")
        y = list(np.linspace(1.0, 0.5, 96))
        out = cp.detect(y, min_len=8, recent=16)
        assert isinstance(out, dict)
