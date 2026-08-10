"""Param-interface batch S6b (REQ-1b): end-to-end settability
through the facade for each channel."""
import pickle

import numpy as np
import pytest


def _rows(seed, n):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-2, 2, size=(n, 3))
    return [{"input": {"a": float(a), "b": float(b), "c": float(c)},
             "target": float(a * b + c * c + a - b)}
            for a, b, c in A]


@pytest.fixture()
def system(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    from core.facade import System
    return System()


class TestEndToEnd:
    def test_all_three_channels_effective(self, system):
        s = system
        out = s.create_model(
            "e2e", holdout=_rows(2, 30),
            policy={"substrate": "transformer",
                    "substrate_params": {"inner_lr_factor": 0.15,
                                         "window": 64,
                                         "seed": 7},
                    "growth_params": {"eta_target": 0.25},
                    "gate_tol": 0.1,
                    "selfstudy_quiz_n": 8})
        assert "refusal" not in out, out
        s.study("e2e", _rows(1, 60), steps=30)
        organ, _ = s.lc._load_working("e2e")
        # C1 constructor channel
        assert organ.INNER_LR_FACTOR == 0.15
        assert organ.WINDOW == 64
        # C2 growth-policy channel (tree-walk installed)
        assert organ._growth_policy["eta_target"] == 0.25
        # C2 per-model policy channel
        pol = s.lc.policy("e2e")
        assert pol["gate_tol"] == 0.1
        assert pol["selfstudy_quiz_n"] == 8

    def test_unknown_substrate_param_refused_loudly(self, system):
        out = system.create_model(
            "bad", holdout=_rows(2, 30),
            policy={"substrate": "transformer",
                    "substrate_params": {"no_such_knob": 1}})
        assert "refusal" in out

    def test_roundtrip_without_new_params_bitwise(self, system):
        s = system
        s.create_model("plain", holdout=_rows(2, 30))
        s.study("plain", _rows(1, 60), steps=30)
        organ, _ = s.lc._load_working("plain")
        X = np.random.default_rng(0).normal(size=(4, 3))
        before = organ.predict(X)
        blob = pickle.dumps(organ)
        again = pickle.loads(blob).predict(X)
        assert np.array_equal(before, again)
