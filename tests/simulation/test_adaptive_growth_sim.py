"""Simulation tests (owner mandate): confirm that (1) deepening
REALLY deepens — serial depth grows, the block trains, the deep
path is load-bearing; (2) widen-vs-deepen selection is ADAPTIVE —
the same machinery under the same policy chooses different arms on
different data; (3) the trained small model is EFFECTIVE — held-out
error improves substantially and reaches an absolute quality bar
at kilobyte scale. All scenes are seeded and deterministic;
outcomes are recorded to tests/logs/simulation_adaptive.jsonl.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.net import Network  # noqa: E402

LOG = ROOT / "tests" / "logs" / "simulation_adaptive.jsonl"
POL = {"min_energy_points": 10 ** 9,        # route through real probes
       "probe_steps": 300, "min_window_rows": 64}

COMP = lambda x: np.sin(3 * np.sin(3 * x))          # compositional law
ADDI = lambda x: np.sin(2 * x) + 0.6 * np.cos(5 * x)  # additive law


def _train_scene(target, seed, H=3, steps=2000):
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = target(X[:, 0]).reshape(-1, 1)
    Xq = np.random.default_rng(200).uniform(-2, 2, size=(256, 1))
    yq = target(Xq[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=H, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y, Xq, yq


def _mse(net, Xq, yq):
    return float(np.mean((net.predict(Xq) - yq) ** 2))


def _record(rec):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def test_sim_adaptivity_same_machinery_different_choices():
    """The SAME policy and parts choose deepen on the compositional
    scene and widen on the additive scene — adaptivity, not fiat."""
    net_c, *_ = _train_scene(COMP, seed=2)
    net_a, *_ = _train_scene(ADDI, seed=1)
    d_c = gp.decide(net_c, POL)
    d_a = gp.decide(net_a, POL)
    _record({"sim": "adaptivity",
             "compositional_arm": d_c["arm"],
             "additive_arm": d_a["arm"],
             "tiers": [d_c["tier_used"], d_a["tier_used"]]})
    assert d_c["arm"] == "deepen", d_c["reasons"]
    assert d_a["arm"] == "widen", d_a["reasons"]


def test_sim_deepen_really_deepens_and_is_load_bearing():
    net, X, y, Xq, yq = _train_scene(COMP, seed=2)
    pre_mse = _mse(net, Xq, yq)
    assert net.serial_depth() == 1
    d = gp.grow_with_policy(net, POL)
    assert d["applied"] == "deepen"
    assert net.serial_depth() == 2                 # depth grew
    for _ in range(3000):
        net.train_step(X, y)
    post_mse = _mse(net, Xq, yq)
    blk = net.blocks[0]
    assert np.any(blk["Bout"] != 0.0)              # the block trained
    twin = copy.deepcopy(net)
    twin.remove_block(0)
    ablated = np.max(np.abs(twin.predict(Xq) - net.predict(Xq)))
    assert ablated > 1e-3                          # deep path load-bearing
    _record({"sim": "deepen_pays", "pre_mse": pre_mse,
             "post_mse": post_mse, "ablation_delta": ablated,
             "serial_depth": net.serial_depth(),
             "params": net.n_params()})
    assert post_mse < 0.7 * pre_mse                # >= 30% improvement


def test_sim_widen_scene_effective():
    net, X, y, Xq, yq = _train_scene(ADDI, seed=1)
    pre_mse = _mse(net, Xq, yq)
    d = gp.grow_with_policy(net, POL)
    assert d["applied"] is not None and d["arm"] == "widen"
    for _ in range(3000):
        net.train_step(X, y)
    post_mse = _mse(net, Xq, yq)
    _record({"sim": "widen_pays", "pre_mse": pre_mse,
             "post_mse": post_mse, "applied": d["applied"],
             "params": net.n_params()})
    assert post_mse < 0.7 * pre_mse


def test_sim_trained_small_model_absolutely_effective():
    """Quality bar: the deepened compositional model explains >= 80%
    of target variance on held-out inputs, at well under a kilobyte
    of parameters."""
    net, X, y, Xq, yq = _train_scene(COMP, seed=2)
    gp.grow_with_policy(net, POL)
    for _ in range(6000):       # compositional gains are back-loaded:
        net.train_step(X, y)    # R2 0.775 @3k -> 0.822 @6k (plateau)
    r2 = 1.0 - _mse(net, Xq, yq) / float(np.var(yq))
    _record({"sim": "absolute_quality", "held_out_r2": r2,
             "params": net.n_params(),
             "bytes_approx": net.n_params() * 8})
    assert r2 >= 0.8, r2
    assert net.n_params() * 8 < 1024               # < 1 KB of weights
