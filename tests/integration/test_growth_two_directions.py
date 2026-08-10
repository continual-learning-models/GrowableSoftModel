"""S5 system integration tests (DEV_PLAN inventory, 7 tests):
the whole selection pipeline on living networks, end to end."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.growthpolicy.pricer_zero_attach import (  # noqa: E402
    _fingerprint)
from reference_net.net import Network  # noqa: E402

FAST = {"probe_steps": 120, "min_window_rows": 64}


def _scene(seed=0, n=64, steps=30, hidden=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (np.sin(2 * X[:, 0]) + 0.4 * X[:, 1]).reshape(-1, 1)
    net = Network(d_in=2, hidden=hidden, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_e2e_additive_scene_full_lifecycle(tmp_path):
    net, X, y = _scene()
    net.gain_horizon = 25
    log = tmp_path / "d.jsonl"
    pol = dict(FAST, log_path=str(log))
    d1 = gp.grow_with_policy(net, pol)
    assert d1["applied"] is not None
    for _ in range(30):                     # past the gain horizon
        net.train_step(X, y)
    growth_events = [r for r in net.gain_ledger
                     if r["event"] in ("refine", "deepen")]
    assert growth_events and growth_events[-1]["gain"] is not None
    d2 = gp.decide(net, pol)
    assert d2["arm"] in ("widen", "deepen")
    assert len(log.read_text().strip().splitlines()) == 2


def test_e2e_decide_never_mutates_scope():
    net, X, y = _scene(seed=3)
    before = _fingerprint(net)
    led = list(net.gain_ledger)
    gp.decide(net, FAST)                    # real pricer probes run
    assert _fingerprint(net) == before
    assert list(net.gain_ledger) == led


def test_e2e_save_load_mid_lifecycle_deterministic():
    def run(interrupt):
        net, X, y = _scene(seed=5)
        net.grow(1, hidden=3)
        for _ in range(10):
            net.train_step(X, y)
        net.deepen(m=2)
        if interrupt:
            net = pickle.loads(pickle.dumps(net))
        for _ in range(10):
            net.train_step(X, y)
        return net, gp.decide(net, FAST)
    na, da = run(False)
    nb, db = run(True)
    assert _fingerprint(na) == _fingerprint(nb)
    assert da == db


def test_e2e_budget_refusals():
    net, X, y = _scene(seed=7)
    pol = dict(FAST, max_blocks=0, min_energy_points=1000)

    class ForceDeepen(gp.interfaces.ProbePricer):
        def price(self, scope, policy):
            t = np.arange(200.)
            return {"widen_curve": (0.5 + 0.4 * np.exp(-0.05 * t)
                                    ).tolist(),
                    "deepen_curve": (0.05 + 0.4 * np.exp(-0.05 * t)
                                     ).tolist(), "steps": 200}
    gp.interfaces.register("pricer", "force_deepen", ForceDeepen)
    pol["pricer"] = "force_deepen"
    d = gp.grow_with_policy(net, pol)
    assert d["arm"] == "deepen" and d["applied"] is None
    assert any("budget refusal" in r for r in d["reasons"])
    assert net.blocks == []


def test_e2e_replay_determinism(tmp_path):
    # wall_ms (58 D-1, FR-16 OPTION B) is DECLARED run-varying
    # — the one wall-clock column on ledger records; replay
    # determinism covers structure, parameters and events, so
    # the comparison scrubs exactly that column (same regime
    # that keeps the ledger outside the bit gates, SR-22).
    def _scrub(ledger):
        return [{k: v for k, v in r.items() if k != "wall_ms"}
                for r in ledger]

    def run(tag):
        net, X, y = _scene(seed=11)
        log = tmp_path / f"{tag}.jsonl"
        gp.grow_with_policy(net, dict(FAST, log_path=str(log)))
        for _ in range(10):
            net.train_step(X, y)
        return (_fingerprint(net), _scrub(net.gain_ledger),
                log.read_text())
    assert run("a") == run("b")


def test_e2e_unknown_part_refusal_not_crash():
    net, _, _ = _scene(seed=13)
    d = gp.decide(net, {"extrapolator": "not_a_part"})
    assert "refusal" in d and "domhan2015" in d["available"]


def test_e2e_no_global_state_leak_golden_guard():
    # decide() ran in this session (tests above); the golden zero-block
    # paths must still be bit-identical
    from tests.fixtures.make_golden import build_reference
    net, X, y = build_reference()
    ref = np.load(ROOT / "tests/fixtures/golden_predict_gp1.npz")
    assert np.array_equal(net.predict(ref["Xq"]), ref["pred"])


def test_e2e_growth_mode_system_control(tmp_path):
    """SYSTEM CONTROL e2e: under global widen_only the full pipeline
    grows purely linearly (no blocks ever); switching back restores
    adaptive deepening on the same scene (seeded)."""
    rng = np.random.default_rng(100)
    Xc = rng.uniform(-2, 2, size=(64, 1))
    yc = np.sin(3 * np.sin(3 * Xc[:, 0])).reshape(-1, 1)

    def scene():
        net = Network(d_in=1, hidden=3, lr=1e-2, seed=2)
        for _ in range(2000):
            net.train_step(Xc, yc)
        return net

    pol = {"min_energy_points": 10 ** 9, "probe_steps": 300,
           "min_window_rows": 64,
           "log_path": str(tmp_path / "mode.jsonl")}
    gp.set_growth_mode(gp.GROWTH_MODE_WIDEN_ONLY)
    try:
        net = scene()
        d1 = gp.grow_with_policy(net, pol)
        assert d1["applied"].startswith("rho")
        assert net.blocks == [] and net.serial_depth() == 1
        rec = json.loads((tmp_path / "mode.jsonl")
                         .read_text().splitlines()[0])
        assert rec["policy_snapshot"]["growth_mode"] == "widen_only"
    finally:
        gp.set_growth_mode(gp.GROWTH_MODE_ADAPTIVE)
    net2 = scene()
    d2 = gp.grow_with_policy(net2, pol)      # same scene, adaptive
    assert d2["applied"] == "deepen" and net2.serial_depth() == 2
