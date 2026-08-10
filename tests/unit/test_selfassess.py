"""Selfassess W1 boxes (docs 27 7 / 29 6 / 30 W1):
QA0 modularity, QA1 dual implementation, QA4 metamorphic,
QA6 verdict machine, QA7 ladder/backoff, QA8 accumulation,
QA9 config completeness + read-only."""
import math
import pickle   # safe: in-process round-trips only
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.selfassess import (                          # noqa: E402
    InnovationAssessor, InnovationCriterion, MDLCriterion,
    SliceLedger, V27_DEFAULTS, _validate_innovation, get,
    install, list_available, register)


def _cfg(**over):
    base = {**V27_DEFAULTS, "innovation_slice_mode":
            "level_tag"}
    base.update(over)
    return base


def _assessor(**over):
    return InnovationAssessor(_validate_innovation(
        {k: v for k, v in _cfg(**over).items()}))


def _feed_cycles(a, series, n=300):
    """series: {slice: [l per cycle]} — feed + close cycles."""
    cycles = max(len(v) for v in series.values())
    for k in range(cycles):
        for s, vals in series.items():
            if k < len(vals):
                a.observe(s, vals[k], n)
        a.close_cycle()


class TestQA0Modularity:
    def test_mdl_registered_default(self):
        assert "mdl" in list_available()
        assert get("mdl") is MDLCriterion

    def test_unknown_method_refused_with_listing(self):
        with pytest.raises(ValueError) as e:
            _validate_innovation(
                {"innovation_method": "nope"})
        assert "available" in str(e.value)

    def test_stub_second_criterion_selectable(self):
        @register
        class StubCriterion(InnovationCriterion):
            NAME = "stub_test"

            def demand_stalled(self, l_now, l_then, cfg):
                return False

            def score_trial(self, *a, **k):
                return {"slice_ok": False, "gain_nats": 0.0,
                        "cost_nats": 0.0, "units": "stub"}
        a = _assessor(innovation_method="stub_test")
        assert a.criterion.NAME == "stub_test"

    def test_engine_never_imports_method_internals(self):
        src = (REPO / "core" / "selfassess.py").read_text()
        eng = src[src.index("class InnovationAssessor"):
                  src.index("# ---------------- public "
                            "installer")]
        # the engine consults only the part's role methods
        assert "MDLCriterion" not in eng
        assert "half_log_n" not in eng


class TestQA1DualImplementation:
    def test_ledger_vs_naive(self):
        rng = np.random.default_rng(0)
        led = SliceLedger(hist_len=4)
        naive = {}
        for _ in range(200):
            s = f"L{rng.integers(3)}"
            v = float(rng.uniform(0.5, 3.0))
            n = int(rng.integers(1, 9))
            led.record(s, v, n)
            su, ct = naive.get(s, (0.0, 0))
            naive[s] = (su + v * n, ct + n)
        closed = led.close_cycle()
        for s, (su, ct) in naive.items():
            assert abs(closed[s] - su / ct) < 1e-12

    def test_score_trial_vs_naive(self):
        c = MDLCriterion()
        cfg = _cfg(innovation_cost_form="half_log_n")
        inc, cand = {"a": 2.0}, {"a": 1.8}
        r = c.score_trial(inc, cand, "a", added_params=100,
                          n_positions=5000, lifetime_n=8000,
                          cfg=cfg)
        naive_gain = (2.0 - 1.8) * 5000
        naive_cost = 0.5 * math.log(8000) * 100
        assert abs(r["gain_nats"] - naive_gain) < 1e-9
        assert abs(r["cost_nats"] - naive_cost) < 1e-9
        assert r["slice_ok"] == (naive_gain > naive_cost)


class TestQA4Metamorphic:
    def test_gain_antisymmetry(self):
        c = MDLCriterion()
        cfg = _cfg()
        a = c.score_trial({"s": 2.0}, {"s": 1.5}, "s", 10,
                          1000, 1000, cfg)
        b = c.score_trial({"s": 1.5}, {"s": 2.0}, "s", 10,
                          1000, 1000, cfg)
        assert abs(a["gain_nats"] + b["gain_nats"]) < 1e-12

    def test_cost_monotone_in_params(self):
        c = MDLCriterion()
        cfg = _cfg()
        costs = [c.score_trial({"s": 2.0}, {"s": 1.9}, "s", p,
                               1000, 1000, cfg)["cost_nats"]
                 for p in (10, 100, 1000)]
        assert costs[0] < costs[1] < costs[2]

    def test_amortize_h1_strictest(self):
        c = MDLCriterion()
        # gain sits between cost and cost/2
        r1 = c.score_trial({"s": 2.0}, {"s": 1.9}, "s", 100,
                           1000, 1000,
                           _cfg(innovation_cost_per_param=1.5,
                                innovation_amortize_h=1.0))
        r2 = c.score_trial({"s": 2.0}, {"s": 1.9}, "s", 100,
                           1000, 1000,
                           _cfg(innovation_cost_per_param=1.5,
                                innovation_amortize_h=2.0))
        assert not r1["slice_ok"] and r2["slice_ok"]

    def test_positions_scale_linearly(self):
        c = MDLCriterion()
        cfg = _cfg()
        g1 = c.score_trial({"s": 2.0}, {"s": 1.9}, "s", 10,
                           1000, 1000, cfg)["gain_nats"]
        g2 = c.score_trial({"s": 2.0}, {"s": 1.9}, "s", 10,
                           3000, 1000, cfg)["gain_nats"]
        assert abs(g2 - 3 * g1) < 1e-9

    def test_slice_relabel_invariance(self):
        c = MDLCriterion()
        cfg = _cfg()
        a = c.score_trial({"x": 2.0}, {"x": 1.7}, "x", 10,
                          1000, 500, cfg)
        b = c.score_trial({"y": 2.0}, {"y": 1.7}, "y", 10,
                          1000, 500, cfg)
        assert a["gain_nats"] == b["gain_nats"]
        assert a["cost_nats"] == b["cost_nats"]


class TestQA6VerdictMachine:
    def test_learning_then_stalled_then_demand_resolved(self):
        a = _assessor(innovation_slice_min_obs=100)
        _feed_cycles(a, {"s": [3.0, 2.5, 2.0, 2.0, 2.0]})
        assert a.cycle_verdicts()["s"] == "STALLED"
        rec = a.next_probe()
        assert rec is not None and rec["slice"] == "s"
        a.register_spawn("s", rec["ladder_class"])
        assert a.cycle_verdicts()["s"] == "UNDER_TRIAL"
        a.register_outcome("s", True, rec["ladder_class"],
                           gain_nats=100.0)
        assert a.cycle_verdicts()["s"] == "LEARNING"
        assert a.innovation_degree("s")[
            "cumulative_gain"] == 100.0

    def test_ladder_exhaustion_to_moi(self):
        a = _assessor(innovation_slice_min_obs=100,
                      innovation_class_fails=1)
        _feed_cycles(a, {"s": [2.0, 2.0, 2.0, 2.0]})
        for cls in a.cfg["innovation_ladder_order"]:
            a.register_spawn("s", cls)
            a.register_outcome("s", False, cls)
        assert a.cycle_verdicts()["s"] == \
            "MASTERED_OR_INEXPRESSIBLE"
        assert "s" in a.innovation_report()["mastered"]

    def test_arrival_reset_full_and_scoped(self):
        a = _assessor(innovation_slice_min_obs=100,
                      innovation_class_fails=1)
        _feed_cycles(a, {"s": [2.0, 2.0, 2.0, 2.0],
                         "t": [1.0, 1.0, 1.0, 1.0]})
        for cls in a.cfg["innovation_ladder_order"]:
            a.register_spawn("s", cls)
            a.register_outcome("s", False, cls)
        a.on_arrival(slices=["t"])          # scoped: s keeps MOI
        assert a.cycle_verdicts()["s"] == \
            "MASTERED_OR_INEXPRESSIBLE"
        assert a.cycle_verdicts()["t"] == "LEARNING"
        a.on_arrival()                       # full reset
        assert a.cycle_verdicts()["s"] == "LEARNING"

    def test_min_obs_gates_verdicts(self):
        a = _assessor(innovation_slice_min_obs=10_000)
        _feed_cycles(a, {"s": [2.0, 2.0, 2.0, 2.0]}, n=100)
        assert a.cycle_verdicts().get("s") is None

    def test_no_harm_veto(self):
        a = _assessor(innovation_slice_min_obs=100,
                      innovation_class_fails=1)
        _feed_cycles(a, {"m": [1.0] * 4, "s": [2.0] * 4})
        for cls in a.cfg["innovation_ladder_order"]:
            a.register_spawn("m", cls)
            a.register_outcome("m", False, cls)   # m -> mastered
        assert "m" in a.innovation_report()["mastered"]
        r = a.score_trial({"s": 2.0, "m": 1.0},
                          {"s": 1.5, "m": 1.2},   # harms m
                          "s", added_params=10,
                          n_positions=1000)
        assert r["slice_ok"] and not r["no_harm_ok"]
        assert not r["accept"]


class TestQA7LadderBackoff:
    def test_class_fails_advances_ladder(self):
        a = _assessor(innovation_slice_min_obs=100,
                      innovation_class_fails=2)
        _feed_cycles(a, {"s": [2.0] * 4})
        order = a.cfg["innovation_ladder_order"]
        assert a.next_probe()["ladder_class"] == order[0]
        for _ in range(2):                   # 2 fails -> class 2
            a.register_spawn("s", order[0])
            a.register_outcome("s", False, order[0])
        a._skip["s"] = 99                    # bypass cadence
        assert a.next_probe()["ladder_class"] == order[1]

    def test_backoff_cadence_and_reset(self):
        a = _assessor(innovation_slice_min_obs=100)
        _feed_cycles(a, {"s": [2.0] * 4})
        a._rejections["s"] = 2               # -> every 2nd
        a._skip["s"] = 0
        assert a.next_probe() is None        # skip 1
        assert a.next_probe() is not None    # skip 2 fires
        a._rejections["s"] = 4               # -> every 4th
        a._skip["s"] = 0
        for _ in range(3):
            assert a.next_probe() is None
        assert a.next_probe() is not None

    def test_priority_empty_band_and_ordering(self):
        a = _assessor(innovation_slice_min_obs=100)
        _feed_cycles(a, {"lo": [1.5] * 4, "hi": [3.0] * 4})
        rec = a.next_probe()
        assert rec["slice"] == "hi"          # raw-l ordering


class TestQA8Accumulation:
    def test_long_stream_error_bound(self):
        led = SliceLedger(hist_len=4)
        rng = np.random.default_rng(1)
        vals = rng.uniform(1.0, 2.0, 1_000_000)
        for i in range(0, 1_000_000, 1000):
            chunk = vals[i:i + 1000]
            led.record("s", float(np.mean(chunk)), 1000)
        closed = led.close_cycle()
        exact = float(np.mean(vals))
        assert abs(closed["s"] - exact) < 1e-9


class TestQA9ConfigAndReadOnly:
    def test_all_keys_reachable_and_validated(self):
        src = (REPO / "core" / "selfassess.py").read_text()
        for k in V27_DEFAULTS:
            assert f'"{k}"' in src
        for bad in ({"innovation_progress_eps": 2.0},
                    {"innovation_amortize_h": 0.5},
                    {"innovation_ladder_order": ("widen",)},
                    {"innovation_backoff_levels": ((0, 1),)},
                    {"innovation_slice_min_obs": True}):
            with pytest.raises(ValueError):
                _validate_innovation(bad)

    def test_reconfigure_preserves_state_and_audits(self):
        a = _assessor(innovation_slice_min_obs=100)
        _feed_cycles(a, {"s": [2.0] * 3})
        l_before = a.ledger.l("s")
        a.drain_events()
        a.reconfigure({"innovation_progress_eps": 0.05})
        assert a.ledger.l("s") == l_before   # state preserved
        assert a.cfg["innovation_progress_eps"] == 0.05
        ev = a.drain_events()
        assert ev[-1]["event"] == "selfassess_reconfig"
        assert ev[-1]["delta"][
            "innovation_progress_eps"] == [0.01, 0.05]

    def test_method_not_hot_swappable(self):
        a = _assessor()
        with pytest.raises(ValueError):
            a.reconfigure({"innovation_method": "stub_test"})

    def test_assessor_holds_no_organ_reference(self):
        class Organ:
            mode = "categorical"
        o = Organ()
        a = install(o, _cfg())
        assert a is o._selfassess
        import gc
        refs = gc.get_referents(a.__dict__)
        assert o not in refs                 # structural check

    def test_install_off_when_unconfigured(self):
        class Organ:
            mode = "categorical"
        o = Organ()
        assert install(o, {}) is None
        assert not hasattr(o, "_selfassess")

    def test_mode_check_mse_rules(self):
        class Organ:
            mode = "numeric"
        with pytest.raises(ValueError):
            install(Organ(), _cfg())
        with pytest.raises(ValueError):
            install(Organ(), _cfg(
                innovation_allow_mse=True,
                innovation_cost_form="half_log_n"))
        a = install(Organ(), _cfg(innovation_allow_mse=True))
        assert a is not None

    def test_picklable(self):
        a = _assessor(innovation_slice_min_obs=100)
        _feed_cycles(a, {"s": [2.0] * 3})
        b = pickle.loads(pickle.dumps(a))
        assert b.ledger.l("s") == a.ledger.l("s")
        assert b.config() == a.config()
