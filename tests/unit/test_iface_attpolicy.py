"""S9.5b per-model ATTENTION policy (docs/system/6 D-G5; plan doc
7 T26 a-d). Fixes the audit-verified process-global bug: the 12
att_* keys lived in one module dict — two models in one process
shared them. Zero theory change: the keys' meanings and every
computation are untouched; only WHERE they are read from.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

from core.facade import System                          # noqa: E402
from core.substrates.growable_attention import (        # noqa: E402
    POLICY, GrowableAttentionSubstrate)

ROWS = [{"input": {"x": float(i) / 24.0, "y": float(24 - i) / 24.0},
         "target": (float(i) / 24.0) * 0.7 + 0.1}
        for i in range(24)]

SP = {"d_model": 8, "n_layers": 1, "heads_spec": [[1]], "seed": 3}


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_t26a_two_models_isolated(ws):
    """T26a: two models in ONE process with different att_lambda —
    each organ's _att_policy carries its own value, the module
    POLICY is untouched, and the training path READS the
    per-model value (the process-global bug is dead)."""
    lam0 = POLICY["att_lambda"]
    ws.create_model("t26a-a", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": SP,
                            "att_lambda": 0.5})
    ws.create_model("t26a-b", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": SP})
    for mid in ("t26a-a", "t26a-b"):
        ws.study(mid, ROWS[8:], steps=10)
    a, _ = ws.lc._load_working("t26a-a")
    b, _ = ws.lc._load_working("t26a-b")
    assert a._att_policy["att_lambda"] == 0.5
    assert getattr(b, "_att_policy", None) is None
    assert a._pol("att_lambda") == 0.5          # the read the hot
    assert b._pol("att_lambda") == lam0         # path uses
    assert POLICY["att_lambda"] == lam0         # module untouched
    # merged, not bare: non-overridden keys present on A
    assert a._att_policy["att_warmup"] == POLICY["att_warmup"]


def test_t26a2_training_actually_uses_per_model_value():
    """T26a strengthening (re-review of the plan's 'training uses
    it'): two same-seed organs, selfproc ON with warmup/age
    cleared PER MODEL, differing ONLY in att_lambda — their
    training losses must DIVERGE (the J_att term is scaled by the
    per-model lambda on the hot path), while two organs with the
    SAME per-model lambda stay bitwise equal."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(12, 2))
    y = np.sin(X.sum(1))
    def mk(lam):
        m = GrowableAttentionSubstrate(
            2, 4, d_model=8, n_layers=1, heads_spec=[[2]],
            seed=3, selfproc=True)
        m._att_policy = {**POLICY, "att_lambda": lam,
                         "att_warmup": 0, "att_head_age_min": 0}
        return m
    a, b, c = mk(0.05), mk(0.9), mk(0.05)
    diverged = False
    for _ in range(15):
        la = a.train_step(X, y)
        lb = b.train_step(X, y)
        lc = c.train_step(X, y)
        assert la == lc                    # same lambda: bitwise
        if la != lb:
            diverged = True
    assert diverged                        # different lambda: used


def test_t26b_unknown_att_key_refused(ws):
    """T26b: unknown att_* key -> loud refusal naming it, at BOTH
    doors, nothing stored."""
    r = ws.create_model("t26b", holdout=ROWS[:8],
                        substrate="growable_attention",
                        policy={"att_lambd": 0.5})
    assert "refusal" in r and "att_lambd" in r["refusal"]
    ws.create_model("t26b2", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": SP})
    r2 = ws.set_policy("t26b2", att_windw=100)
    assert "refusal" in r2 and "att_windw" in r2["refusal"]
    assert not any(k.startswith("att_")
                   for k in ws.lc.policy("t26b2"))


def test_t26c_no_att_keys_module_default_bitwise(ws):
    """T26c: no att_* keys -> the attribute never appears and the
    training trajectory is BITWISE identical to a direct-
    construction organ of the same seed (the module-POLICY path
    is untouched by the helper substitution)."""
    ws.create_model("t26c", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": SP})
    ws.study("t26c", ROWS[8:], steps=1)
    organ, _ = ws.lc._load_working("t26c")
    assert getattr(organ, "_att_policy", None) is None
    # the read path falls back to the module table
    direct = GrowableAttentionSubstrate(2, organ.m, **SP)
    assert organ._pol("att_h_lo") == direct._pol("att_h_lo") \
        == POLICY["att_h_lo"]


def test_t26c2_helper_fallback_bitwise():
    """T26c (bitwise form): two same-seed DIRECT organs, one given
    the full module table as _att_policy, one bare — identical
    training losses step for step (the helper's two branches are
    numerically the same table)."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 2))
    y = np.sin(X.sum(1, keepdims=True))
    a = GrowableAttentionSubstrate(2, 4, **SP)
    b = GrowableAttentionSubstrate(2, 4, **SP)
    b._att_policy = dict(POLICY)               # merged-full table
    for _ in range(10):
        la = a.train_step(X, y.ravel())
        lb = b.train_step(X, y.ravel())
        assert la == lb                        # bitwise every step


def test_t26d_direct_construction_module_override_still_works():
    """T26d: the experiment-driver surface is unchanged — mutating
    the MODULE POLICY still steers a bare organ (no _att_policy),
    exactly as drivers do today."""
    m = GrowableAttentionSubstrate(2, 4, **SP)
    old = POLICY["att_warmup"]
    try:
        POLICY["att_warmup"] = 7
        assert m._pol("att_warmup") == 7       # bare -> module table
        m._att_policy = {**POLICY, "att_warmup": 99}
        assert m._pol("att_warmup") == 99      # instance wins
    finally:
        POLICY["att_warmup"] = old
