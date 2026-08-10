"""V3 tests: SPU on the attention host (DEV_PLAN v2.1)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
import reference_net.growthpolicy as gp                      # noqa: E402
from core.substrates.transformer import (               # noqa: E402
    TransformerSubstrate)
from reference_net.spu.spu_network import install_spu_policy  # noqa: E402
from engine.spu.spu_objective import (                 # noqa: E402
    batch_std, forward_parts)

POL = {"spu_enabled": True, "spu_warmup_steps": 5,
       "spu_newborn_steps": 60}


def data(seed=0, n=48):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 4))
    return X, np.sin(3 * X[:, :1]) + 0.3 * X[:, 2:3] * X[:, 3:4]


def host(seed=7, steps=30, grown=True):
    X, y = data()
    h = TransformerSubstrate(d_in=4, hidden=6, mode="numeric",
                             seed=seed)
    for _ in range(steps):
        h.train_step(X, y)
    if grown:
        h.grow_site("layer0/ffn[2]", hidden=5)
    return h, X, y


def processed(h):
    return [e for e in getattr(h, "_spu_events", [])
            if e.get("skip") is None and "steps" in e]


def test_dispatch_accepts_host():
    h, _, _ = host()
    pol = install_spu_policy(h, POL)
    assert pol["spu_enabled"] is True


def test_bodies_process_in_window():
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(40):
        h.train_step(X, y)
    evs = processed(h)
    # 40 steps - 1 cache-miss - 4 remaining warmup = 35
    assert len(evs) == 35
    assert all(e["path"] == "L0/u2" for e in evs)
    counts = h._spu_skip_counts
    assert counts.get("no_cached_input") == 1
    assert counts.get("warming_up") == 4


def test_staleness_semantics_operational():
    """Event s_entry == batch_std of the body's z0 computed on the
    PREVIOUS training forward's cached input with pre-step
    weights."""
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(20):
        h.train_step(X, y)
    body = h.grown_body(0, 2)
    cache = {l: v.copy() for l, v in h._spu_input_cache.items()}
    W1, b1 = body.W1.copy(), body.b1.copy()
    W2, c = body.W2.copy(), body.c.copy()
    x_mu, x_sd = body._x_mu.copy(), body._x_sd.copy()
    n_before = len(processed(h))
    h.train_step(X, y)
    ev = processed(h)[n_before]
    Xs = (cache[0] - x_mu) / x_sd
    _, _, z0 = forward_parts(W1, b1, W2, c, Xs)
    assert ev["s_entry"] == batch_std(z0)


def test_cache_miss_first_step_disclosed():
    h, X, y = host()
    install_spu_policy(h, POL)
    h.train_step(X, y)
    assert h._spu_skip_counts.get("no_cached_input") == 1
    assert not processed(h)


def test_spu_off_no_state_no_cost():
    h, X, y = host()
    for _ in range(10):
        h.train_step(X, y)
    assert not hasattr(h, "_spu_input_cache")
    assert not hasattr(h, "_spu_events")


def test_predict_never_writes_cache_or_events():
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(3):
        h.train_step(X, y)
    cache_snapshot = {l: v.copy()
                      for l, v in h._spu_input_cache.items()}
    n_ev = len(getattr(h, "_spu_events", []))
    n_sk = dict(getattr(h, "_spu_skip_counts", {}))
    body_w = h.grown_body(0, 2).W1.copy()
    for _ in range(25):
        h.predict(X)
    assert len(getattr(h, "_spu_events", [])) == n_ev
    assert dict(getattr(h, "_spu_skip_counts", {})) == n_sk
    assert all(np.array_equal(cache_snapshot[l],
                              h._spu_input_cache[l])
               for l in cache_snapshot)
    assert np.array_equal(h.grown_body(0, 2).W1, body_w)


def test_deep_growth_population_change():
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(20):
        h.train_step(X, y)
    n_body_before = sum(1 for e in processed(h)
                        if e["path"] == "L0/u2")
    h.grow_site("layer0/ffn[2]::root[0]", hidden=4)   # body's child
    for _ in range(15):
        h.train_step(X, y)
    # the body retires the moment it hosts a composite node: the
    # scan recurses through it silently (reference-identical
    # behavior) and it gains NO further processed events
    n_body_after = sum(1 for e in processed(h)
                       if e["path"] == "L0/u2")
    assert n_body_after == n_body_before
    # the depth-2 grandchild takes over (first unfit, then warmup,
    # then processing)
    grandchild = [e for e in h._spu_events
                  if e.get("path") == "L0/u2/port[0]"]
    assert grandchild
    assert any("steps" in e and e.get("skip") is None
               for e in grandchild)


def test_every_gating_on_host():
    h, X, y = host()
    install_spu_policy(h, dict(POL, spu_every=5))
    for _ in range(20):
        h.train_step(X, y)
    assert 2 <= len(processed(h)) <= 5


def test_widen_only_mode_forces_off():
    h, X, y = host()
    install_spu_policy(h, POL)
    gp.set_growth_mode(gp.GROWTH_MODE_WIDEN_ONLY)
    try:
        for _ in range(5):
            h.train_step(X, y)
        assert not processed(h)
    finally:
        gp.set_growth_mode(gp.GROWTH_MODE_ADAPTIVE)


def test_save_load_roundtrip_with_policy(tmp_path):
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(10):
        h.train_step(X, y)
    pred = h.predict(X)
    h.save(tmp_path)
    from core.substrates import load_artifact
    h2 = load_artifact(tmp_path)
    assert np.allclose(h2.predict(X), pred)
    for _ in range(3):                                 # resumes cleanly
        h2.train_step(X, y)


def test_contract_surface_with_policy():
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(10):
        h.train_step(X, y)
    sites = h.growth_sites()
    assert sites and isinstance(sites[0][0], str)
    assert np.isfinite(h.predict(X)).all()


def test_build_report_on_host():
    """The readiness report works on a duck-typed host holder
    (the V3-plan test that was missed; caught by P2-S4)."""
    from engine.spu.spu_report import build_report
    h, X, y = host()
    install_spu_policy(h, POL)
    for _ in range(20):
        h.train_step(X, y)
    r = build_report(h)
    assert r["processed_steps"] > 0
    assert "L0/u2" in r["spus"]
    assert r["total_events"] > 0
