"""Param-interface batch S3 (docs/system/22 item 6): gate_tol on
both gate surfaces (lifecycle _match; Generator factory
evaluator) + the curriculum accuracy mirror.
"""
import numpy as np
import pytest

from core.lifecycle import TOL, _gate_tol, _match


class TestLifecycleMatch:
    def test_default_matches_pre_change_rule(self):
        assert _match(1.4, 1.0) == (abs(1.4 - 1.0) <= TOL)
        assert _match(1.6, 1.0) == (abs(1.6 - 1.0) <= TOL)

    def test_tol_changes_outcome(self):
        assert _match(1.4, 1.0) is True          # default 0.5
        assert _match(1.4, 1.0, tol=0.1) is False
        assert _match(1.05, 1.0, tol=0.1) is True

    def test_gate_tol_read_and_validation(self):
        assert _gate_tol({}) == TOL
        assert _gate_tol({"gate_tol": 0.1}) == 0.1
        for bad in (-1, "x", True):
            with pytest.raises(ValueError):
                _gate_tol({"gate_tol": bad})


class TestFactoryEvaluator:
    def _evaluator(self, **cfg):
        from generator.evaluator import Evaluator

        class _Cfg:
            min_gain = 0.0
        for k, v in cfg.items():
            setattr(_Cfg, k, v)
        return Evaluator(_Cfg(), None, None)

    def test_default_tol(self):
        ev = self._evaluator()
        assert ev.tol == 0.5
        assert ev._match(1.4, 1.0, tol=ev.tol) is True

    def test_config_override_one_rule_both_surfaces(self):
        ev = self._evaluator(gate_tol=0.1)
        assert ev.tol == 0.1
        assert ev._match(1.4, 1.0, tol=ev.tol) is False
        # same rule as the lifecycle surface at the same tol
        assert _match(1.4, 1.0, tol=0.1) is False

    def test_validation(self):
        with pytest.raises(ValueError):
            self._evaluator(gate_tol=-0.5)


class TestCurriculumMirror:
    def test_accuracy_default_and_override(self):
        from reference_net.curriculum import TOL as CTOL, accuracy
        assert CTOL == TOL  # mirror equality (drift -> failure)
        pred = np.array([1.4, 2.0])
        truth = np.array([1.0, 2.0])
        assert accuracy(pred, truth) == 1.0
        assert accuracy(pred, truth, tol=0.1) == 0.5
