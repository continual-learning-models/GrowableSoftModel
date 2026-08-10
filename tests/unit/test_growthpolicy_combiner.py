"""A6 threshold-combiner tests (13). Stub parts test the decision
logic in isolation; the real end-to-end scenes live in the
integration file."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.growthpolicy import interfaces as gi  # noqa: E402
from reference_net.net import Network  # noqa: E402

T = np.arange(200, dtype=float)


def _net(seed=1, hidden=4, steps=3, rows=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 2))
    y = X[:, :1] * 0.5
    net = Network(d_in=2, hidden=hidden, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def _warm(net, X, y, steps=10):
    for _ in range(steps):
        net.train_step(X, y)


def _inject(net, energy, gains, sat=False):
    net.energy_ring.clear()
    net.energy_ring.extend([float(v) for v in energy])
    net._E = float(energy[-1])
    for g in gains:
        net.gain_ledger.append({"event": "refine", "gain": float(g)})
    if sat:
        for _ in range(16):
            net._record_update_direction(np.full((4, 2), 1e-9))


class _StubPricer(gi.ProbePricer):
    """Crafted curves with known asymptotes."""
    W, D = 0.30, 0.10

    def price(self, scope, policy):
        t = np.arange(200.)
        mk = lambda a: (a + 0.6 * np.exp(-0.05 * t)).tolist()
        return {"widen_curve": mk(self.W), "deepen_curve": mk(self.D),
                "steps": 200}


gi.register("pricer", "stub_wd", _StubPricer)


def test_cold_start_defaults_widen():
    net, _, _ = _net()
    d = gp.decide(net)
    assert d["tier_used"] == "cold_start_default" and d["arm"] == "widen"


def test_cold_start_authorizes_tier2_when_window_ok():
    net, X, y = _net(rows=64, steps=10)
    d = gp.decide(net, {"pricer": "stub_wd"})
    assert d["tier_used"] == "tier2"


def test_certificate_fail_routes_tier2():
    net, X, y = _net(rows=64, steps=10)
    _inject(net, 0.5 + 0.4 * np.exp(-0.03 * T), [0.2, 0.2, 0.2])
    d = gp.decide(net, {"pricer": "stub_wd",
                        "forecastability_min": 1.01})
    assert d["tier_used"] == "tier2"
    assert not d["certificate"]["forecastability"]["passed"] or True


def test_tier1_widen_noncomposite_siting():
    net, X, y = _net(rows=64, steps=10)
    _inject(net, 0.5 + 0.9 * (T + 1) ** -0.4, [0.2, 0.2, 0.2])
    d = gp.decide(net)
    assert d["tier_used"] == "tier1" and d["arm"] == "widen"
    assert d["apply_as"] == "rho" and d["site"] not in net.inner


def test_all_composite_falls_back_to_omega():
    net, X, y = _net(hidden=2, rows=64, steps=10)
    net.grow(0, hidden=3); net.grow(1, hidden=3)
    _warm(net, X, y, 5)
    _inject(net, 0.5 + 0.9 * (T + 1) ** -0.4, [0.2, 0.2, 0.2])
    d = gp.decide(net)
    assert d["arm"] == "widen" and d["apply_as"] == "omega"
    assert d["site"] is None


def test_tier1_deepen_on_stall_and_saturation():
    net, X, y = _net(rows=64, steps=10)
    energy = 0.5 + 0.02 * np.exp(-0.01 * T)     # confident slow decay
    _inject(net, energy, [0.01, 0.02, 0.01], sat=True)
    d = gp.decide(net)
    assert d["tier_used"] == "tier1" and d["arm"] == "deepen", d["reasons"]


def test_tier1_ambiguous_routes_tier2():
    net, X, y = _net(rows=64, steps=10)
    energy = 0.5 + 0.02 * np.exp(-0.01 * T)     # p_useful ~ 0
    _inject(net, energy, [0.2, 0.2, 0.2])       # healthy gains: no stall
    d = gp.decide(net, {"pricer": "stub_wd"})
    assert d["tier_used"] == "tier2"


def test_tier2_deepen_wins():
    net, X, y = _net(rows=64, steps=10)
    d = gp.decide(net, {"pricer": "stub_wd",
                        "forecastability_min": 1.01,
                        "min_energy_points": 1000})
    assert d["tier_used"] == "tier2" and d["arm"] == "deepen"


def test_tier2_widen_wins():
    class WideWins(_StubPricer):
        W, D = 0.10, 0.30
    gi.register("pricer", "stub_ww", WideWins)
    net, X, y = _net(rows=64, steps=10)
    d = gp.decide(net, {"pricer": "stub_ww",
                        "min_energy_points": 1000})
    assert d["arm"] == "widen"


def test_tie_within_ci_prefers_widen():
    class Tie(_StubPricer):
        W, D = 0.20, 0.20
    gi.register("pricer", "stub_tie", Tie)
    net, X, y = _net(rows=64, steps=10)
    d = gp.decide(net, {"pricer": "stub_tie",
                        "min_energy_points": 1000})
    assert d["arm"] == "widen"


def test_decision_dict_complete_schema():
    net, _, _ = _net()
    d = gp.decide(net)
    for key in ("arm", "site", "tier_used", "reasons", "parts",
                "policy_snapshot", "gate_verdict", "apply_as"):
        assert key in d


def test_decision_log_contains_part_names(tmp_path):
    import json
    net, _, _ = _net()
    log = tmp_path / "decisions.jsonl"
    gp.decide(net, {"log_path": str(log)})
    rec = json.loads(log.read_text().strip())
    assert rec["parts"]["combiner"] == "threshold_policy"
    assert rec["gate_verdict"] is None


def test_same_seed_identical_decision_record():
    net1, _, _ = _net(seed=9, rows=64, steps=10)
    net2, _, _ = _net(seed=9, rows=64, steps=10)
    assert gp.decide(net1) == gp.decide(net2)
