"""A7 registry tests (5)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.growthpolicy import interfaces as gi  # noqa: E402


def test_register_and_get():
    class Dummy(gi.Forecastability):
        def score(self, series, threshold=0.4):
            return {"score": 1.0, "passed": True}
    gi.register("forecastability", "dummy_fc", Dummy)
    part = gi.get("forecastability", "dummy_fc")
    assert part.score([1, 2, 3])["passed"]


def test_unknown_name_refusal_dict_no_raise():
    r = gi.get("pricer", "does_not_exist")
    assert "refusal" in r and "zero_attach_v1" in r["available"]


def test_list_available_sorted():
    names = gi.list_available("extrapolator")
    assert names == sorted(names) and "domhan2015" in names


def test_swap_by_policy_name_pipeline_runs():
    # T-SWAP: a stub part switched in by policy name only
    class ConstFc(gi.Forecastability):
        def score(self, series, threshold=0.4):
            return {"score": 0.99, "passed": True}
    gi.register("forecastability", "const_fc", ConstFc)
    import numpy as np
    from reference_net.net import Network
    net = Network(d_in=2, hidden=3, lr=1e-2, seed=1)
    X = np.random.default_rng(0).normal(size=(8, 2))
    y = X[:, :1]
    for _ in range(3):
        net.train_step(X, y)
    d = gp.decide(net, {"forecastability": "const_fc"})
    assert d["parts"]["forecastability"] == "const_fc"


def test_default_policy_complete_keys():
    need = {"extrapolator", "forecastability", "changepoint",
            "backtest", "pricer", "combiner", "stall_k", "stall_eps",
            "saturation_norm", "energy_floor", "min_energy_points",
            "min_ledger_events", "min_window_rows", "probe_steps",
            "probe_lr", "max_rel_ci_width", "forecastability_min",
            "changepoint_max", "backtest_max_err", "seed"}
    assert need <= set(gp.DEFAULT_GROWTH_POLICY)
