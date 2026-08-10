"""SPU S2 tests: the bounded local solver (DEV_PLAN T2.1-T2.12)."""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network                     # noqa: E402
from engine.spu.spu_loop import (                  # noqa: E402
    SKIP_DISABLED, SKIP_EVERY,
    SKIP_HOSTS_COMPOSITE, SKIP_MATURED, SKIP_ROOT,
    SKIP_SMALL_BATCH, SKIP_UNFIT, SKIP_WARMUP, self_process,
    skip_reason)
from engine.spu.spu_policy import validate_spu_policy  # noqa: E402


def trained_leaf(seed=3, steps=30):
    rng = np.random.default_rng(seed)
    leaf = Network(d_in=3, hidden=5, lr=1e-2, seed=seed)
    X = rng.normal(0, 1, (32, 3))
    y = np.sin(X.sum(1, keepdims=True))
    for _ in range(steps):
        leaf.train_step(X, y)
    return leaf, X


def pol(**kw):
    base = {"spu_enabled": True, "spu_warmup_steps": 0}
    base.update(kw)
    return validate_spu_policy(base)


def test_budget_never_exceeded():                       # T2.1
    leaf, X = trained_leaf()
    # p_mask pinned: budget mechanics asserted at the stated
    # perturbation strength (S-P1 changed only the DEFAULT)
    ev = self_process(leaf, X, pol(spu_tau_rel=0.0, spu_S_max=3,
                                   spu_p_mask=0.25), 0)
    assert ev["steps"] <= 3 and not ev["converged"]


def test_stop_on_threshold():                           # T2.2
    leaf, X = trained_leaf()
    ev = self_process(leaf, X, pol(spu_tau_rel=1e9), 0)
    assert ev["converged"] and ev["steps"] == 0


def test_adam_moments_untouched():                      # T2.3
    leaf, X = trained_leaf()
    m_before = copy.deepcopy(leaf.opt.m)
    v_before = copy.deepcopy(leaf.opt.v)
    self_process(leaf, X, pol(), 0)
    assert all(np.array_equal(a, b)
               for a, b in zip(leaf.opt.m, m_before))
    assert all(np.array_equal(a, b)
               for a, b in zip(leaf.opt.v, v_before))


def test_update_norm_cap():                             # T2.4
    leaf, X = trained_leaf()
    t0 = np.sqrt(sum(float((p ** 2).sum()) for p in
                     [leaf.W1, leaf.b1, leaf.W2, leaf.c]))
    ev = self_process(leaf, X, pol(spu_eta=100.0, spu_clip=0.01,
                                   spu_S_max=1, spu_tau_rel=0.0), 0)
    t1 = np.sqrt(sum(float((p ** 2).sum()) for p in
                     [leaf.W1, leaf.b1, leaf.W2, leaf.c]))
    assert ev["clips"] == 1
    assert abs(t1 - t0) <= 0.011 * t0 + 1e-12


def test_jinv_decreases_over_loop():                    # T2.5
    leaf, X = trained_leaf()
    ev = self_process(leaf, X, pol(spu_S_max=5, spu_tau_rel=0.0), 0)
    assert ev["j_inv_after"] <= ev["j_inv_before"] * 1.05


def test_disabled_leaf_untouched():                     # T2.6
    leaf, X = trained_leaf()
    r = skip_reason(leaf, len(X), pol(spu_enabled=False), 0)
    assert r == SKIP_DISABLED


def test_skip_conditions_single_authority():            # T2.7-T2.10
    leaf, X = trained_leaf()
    p = pol()
    assert skip_reason(leaf, len(X), p, 0, is_root=True) == SKIP_ROOT
    assert skip_reason(leaf, 3, p, 0) == SKIP_SMALL_BATCH
    assert skip_reason(leaf, len(X), pol(spu_every=7), 3) == SKIP_EVERY
    leaf.deepen(m=2)
    # P2-S2: the has_blocks gate is deleted — deepened newborns
    # are eligible
    assert skip_reason(leaf, len(X), p, 0) is None
    leaf.blocks = []
    leaf.grow(0, hidden=4)
    assert skip_reason(leaf, len(X), p, 0) == SKIP_HOSTS_COMPOSITE
    fresh = Network(d_in=3, hidden=4, seed=1)
    assert skip_reason(fresh, len(X), p, 0) == SKIP_UNFIT
    old, _ = trained_leaf(steps=250)
    assert skip_reason(old, len(X), pol(spu_newborn_steps=200),
                       0) == SKIP_MATURED
    assert skip_reason(old, len(X),
                       pol(spu_scope="all_inner_nets"), 0) is None
    ok, _ = trained_leaf()
    assert skip_reason(ok, len(X), p, 0) is None


def test_enrollment_window():
    """Owner ruling: rough shape first — the window is
    [warmup, newborn), clock-based (signals can be transiently
    false; the clock is deterministic and auditable)."""
    default = validate_spu_policy({"spu_enabled": True})
    young, X = trained_leaf(steps=30)      # age 30 < warmup 100
    assert skip_reason(young, len(X), default, 0) == SKIP_WARMUP
    mid, _ = trained_leaf(steps=150)       # inside [100, 300)
    assert skip_reason(mid, len(X), default, 0) is None
    old, _ = trained_leaf(steps=350)       # past retirement
    assert skip_reason(old, len(X), default, 0) == SKIP_MATURED
    anyage = validate_spu_policy({"spu_enabled": True,
                                  "spu_scope": "all_inner_nets"})
    assert skip_reason(young, len(X), anyage, 0) is None


def test_determinism_bit_replay():                      # T2.11
    a, X = trained_leaf(seed=9)
    b, _ = trained_leaf(seed=9)
    ea = self_process(a, X, pol(), 5)
    eb = self_process(b, X, pol(), 5)
    assert ea == eb
    assert np.array_equal(a.W1, b.W1) and np.array_equal(a.W2, b.W2)


def test_event_schema_complete():                       # T2.12
    leaf, X = trained_leaf()
    ev = self_process(leaf, X, pol(), 0)
    for k in ("steps", "converged", "j_inv_before", "j_inv_after",
              "s_entry", "s_after", "disc_hits", "clips",
              "policy_snapshot", "skip"):
        assert k in ev


def test_deepened_newborn_processes_through_chain():  # P2-S2
    """A deepened newborn self-processes: the chain-aware loop
    runs, block parameters move, and the event records the block
    count."""
    import numpy as _np
    leaf, X = trained_leaf()
    leaf.deepen(m=3)
    blk = leaf.blocks[0]
    blk["Bout"][:] = _np.random.default_rng(9).normal(
        0, 0.3, blk["Bout"].shape)          # matured block: carries
    before = {k: blk[k].copy() for k in blk}
    ev = self_process(leaf, X, pol(), 0)
    assert ev["skip"] is None and ev["steps"] > 0
    assert ev["blocks"] == 1
    assert any(not _np.array_equal(before[k], blk[k])
               for k in blk)                # block params adjusted


def test_deepened_unit_adam_moments_untouched():       # P2-S2
    """Plain local SGD only: Adam moments (incl. the block slots
    created by _rebuild_opt at deepen time) stay bitwise
    untouched."""
    import numpy as _np
    leaf, X = trained_leaf()
    leaf.deepen(m=3)
    m_snap = [a.copy() for a in leaf.opt.m]
    v_snap = [a.copy() for a in leaf.opt.v]
    t_snap = leaf.opt.t
    self_process(leaf, X, pol(), 0)
    assert leaf.opt.t == t_snap
    assert all(_np.array_equal(a, b)
               for a, b in zip(m_snap, leaf.opt.m))
    assert all(_np.array_equal(a, b)
               for a, b in zip(v_snap, leaf.opt.v))
