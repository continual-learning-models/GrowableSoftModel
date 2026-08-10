"""SPU S3 tests: SPUNetwork integration (DEV_PLAN T3.x) —
net.py untouched; integration scoped to the substrate level
(factory exposure is post-v0)."""
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp                    # noqa: E402
from reference_net.net import Network                      # noqa: E402
from engine.spu.spu_loop import self_process        # noqa: E402
from reference_net.spu.spu_network import (                # noqa: E402
    SPUNetwork, spu_prepass)

POL = {"spu_enabled": True, "spu_warmup_steps": 0}


def make(seed=8, grown=True, cls=SPUNetwork, steps=60):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (32, 2))
    y = np.sin(3 * X[:, :1]) + 0.5 * X[:, 1:]
    net = cls(d_in=2, hidden=4, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    if grown:
        net.grow(1, hidden=5)
        for _ in range(5):
            net.train_step(X, y)
    return net, X, y


def test_facade_equals_plain_network_install():          # T3.1'
    """v2.0: the walk lives in Network.train_step; the facade and
    a PLAIN Network with install_spu_policy are the same machine
    (the capability v0 lacked: no subclass needed)."""
    from reference_net.spu.spu_network import install_spu_policy
    a, X, y = make(cls=SPUNetwork)
    b, _, _ = make(cls=Network)
    a.set_spu_policy(POL)
    install_spu_policy(b, POL)
    for _ in range(5):
        a.train_step(X, y)
        b.train_step(X, y)
    assert np.array_equal(a.grown_body(1).W1, b.grown_body(1).W1)
    assert np.array_equal(a.W1, b.W1)
    assert len(getattr(b, "_spu_events", [])) > 0


def test_no_double_processing_depth2():                   # V1 new
    """Children never hold the policy: a depth-2 grandchild is
    processed exactly once per holder step."""
    net, X, y = make()
    net.grown_body(1).grow(0, hidden=4)
    net.set_spu_policy({"spu_enabled": True, "spu_warmup_steps": 0})
    for _ in range(3):
        net.train_step(X, y)
    deep = [e for e in net.spu_events
            if e.get("path") == "root/port[0]/port[0]" and "steps" in e]
    # W3 migration note: port bodies are born with FIT identity
    # scalers, so there is no scalers_unfit first step — the
    # grandchild processes on all 3 steps, EXACTLY once each
    # (the guarded property: no double processing per step)
    assert len(deep) == 3
    assert getattr(net, "_spu_skip_counts", {}).get(
        "scalers_unfit") is None
    assert getattr(net.grown_body(1), "_spu_policy", None) is None


def test_unknown_holder_refused():                        # V1 new
    from reference_net.spu.spu_network import install_spu_policy
    import pytest as _pt
    with _pt.raises(TypeError):
        install_spu_policy(object(), {"spu_enabled": True})


def test_eligibility_end_to_end():                        # T3.2
    net, X, y = make()
    net.set_spu_policy(POL)
    net.train_step(X, y)
    paths = [e["path"] for e in net.spu_events
             if e.get("skip") is None and "steps" in e]
    assert paths and all(p.startswith("root/") for p in paths)
    assert "root" not in paths                            # root never


def test_rho_changes_population():                        # T3.3
    net, X, y = make()
    net.set_spu_policy(POL)
    net.train_step(X, y)
    first = {e["path"] for e in net.spu_events
             if e.get("skip") is None}
    net.grown_body(1).grow(0, hidden=4)                        # leaf refines
    for _ in range(3):
        net.train_step(X, y)
    later = {e["path"] for e in net.spu_events
             if e.get("skip") is None}
    assert "root/port[0]" in first
    assert "root/port[0]/port[0]" in later                # newborn active


def test_spu_off_bit_identical_to_network():              # T3.4
    a, Xa, ya = make(cls=SPUNetwork)
    b, Xb, yb = make(cls=Network)
    for _ in range(20):
        a.train_step(Xa, ya)
        b.train_step(Xb, yb)
    assert np.array_equal(a.W1, b.W1)
    assert np.array_equal(a.grown_body(1).W2, b.grown_body(1).W2)
    assert a.spu_events == []


def test_no_targets_in_spu_api():                         # T3.5
    assert "y" not in inspect.signature(self_process).parameters
    assert "y" not in inspect.signature(spu_prepass).parameters


def test_interference_record_present():                   # T3.6
    net, X, y = make()
    net.set_spu_policy(POL)
    net.train_step(X, y)
    summ = [e for e in net.spu_events if e["path"] == "__step__"]
    assert summ and "task_mse_before" in summ[0]
    assert summ[0]["processed"] >= 1


def test_spu_every_gating():                              # T3.7
    net, X, y = make()
    net.set_spu_policy({"spu_enabled": True, "spu_every": 5,
                        "spu_warmup_steps": 0})
    for _ in range(10):
        net.train_step(X, y)
    n_proc = sum(1 for e in net.spu_events
                 if e.get("skip") is None and "steps" in e)
    assert 1 <= n_proc <= 3                               # every 5th


def test_probes_never_self_process():                     # T3.8
    net, X, y = make()
    net.set_spu_policy(POL)
    before = len(net.spu_events)
    gp.decide(net, {"min_energy_points": 10 ** 9,
                    "probe_steps": 60, "min_window_rows": 16})
    assert len(net.spu_events) == before                  # decide adds none


def test_instrumentation_intact_with_spu_on():            # T3.9
    net, X, y = make()
    net.set_spu_policy(POL)
    for _ in range(5):
        net.train_step(X, y)
    assert net.residual_energy() is not None
    assert len(net.energy_ring) > 0


def test_pickle_roundtrip_and_backfill():                 # T3.10
    net, X, y = make()
    net.set_spu_policy(POL)
    net.train_step(X, y)
    clone = pickle.loads(pickle.dumps(net))
    assert isinstance(clone, SPUNetwork)
    assert np.array_equal(clone.W1, net.W1)
    state = pickle.dumps(net)
    legacy = pickle.loads(state)
    del legacy.__dict__["_spu_policy"]
    revived = pickle.loads(pickle.dumps(legacy))
    assert revived.get_spu_policy() is None               # back-fill


def test_long_run_determinism():                          # T3.11
    outs = []
    for _ in range(2):
        net, X, y = make()
        net.set_spu_policy(POL)
        for _ in range(20):
            net.train_step(X, y)
        outs.append(net.predict(X))
    assert np.array_equal(outs[0], outs[1])


def test_widen_only_mode_forces_spu_off():                # T3.12
    net, X, y = make()
    net.set_spu_policy(POL)
    gp.set_growth_mode(gp.GROWTH_MODE_WIDEN_ONLY)
    try:
        net.train_step(X, y)
        assert net.spu_events == []
    finally:
        gp.set_growth_mode(gp.GROWTH_MODE_ADAPTIVE)
    net.train_step(X, y)
    assert len(net.spu_events) > 0


def test_serving_purity_predict_emits_nothing():          # I3.2
    net, X, y = make()
    net.set_spu_policy(POL)
    w = net.grown_body(1).W1.copy()
    for _ in range(50):
        net.predict(X)
    assert np.array_equal(net.grown_body(1).W1, w)
    assert net.spu_events == []


def test_spu_and_growth_round_coexist():                  # I3.4
    net, X, y = make()
    net.set_spu_policy(POL)
    d = gp.grow_with_policy(net, {"min_energy_points": 10 ** 9,
                                  "probe_steps": 60,
                                  "min_window_rows": 16})
    for _ in range(10):
        net.train_step(X, y)
    assert d["applied"] is not None
    assert any(e.get("skip") is None for e in net.spu_events)
