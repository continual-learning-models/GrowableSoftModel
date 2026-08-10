"""59C stage 0 (= doc 61's G8, formerly 60): product-gate registration of
the growth-control policy keys. Boxes written FIRST from the
spec text; RED today at the named refusals."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TEN_KEYS = {
    "gate_deepen_mode": "warn",
    "gate_seam_min_width": 2,
    "gate_seam_mode": "warn",
    "gate_nest_mode": "warn",
    "gate_scope_min_width": 2,
    "gate_scope_mode": "warn",
    "gate_widen_mode": "warn",
    "train_lr_scales": {"encoder": 0.5},
    "growth_auto_snapshot": True,
    "growth_snapshot_keep": 4,
}


def _sys(tmp_path):
    from core.facade import System
    from core.wiring import Config
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


def _rows(seed=101, n=16):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2]
    return [{"input": {"a": float(a), "b": float(b),
                       "c": float(c)},
             "target": str(float(t))}
            for (a, b, c), t in zip(X, y)]


def test_g8_ten_keys_accepted_and_typo_refused(tmp_path):
    """Each of the TEN 59C keys routes through set_policy's
    growth_params door; a typo key still refuses loudly;
    DEFAULT_GROWTH_POLICY the DICT is unchanged (I-8).
    CONSCIOUS UPDATE (60D): the registry now holds FOURTEEN
    keys — the four aspect keys' registration census lives in
    test_aspect_ratio_gate.py T-1; this box keeps auditing
    the original ten."""
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    pre_keys = set(DEFAULT_GROWTH_POLICY)
    s = _sys(tmp_path)
    s.create_model("m", description="g8")
    for k, v in TEN_KEYS.items():
        out = s.set_policy("m", growth_params={k: v})
        assert not (isinstance(out, dict)
                    and out.get("refusal")), (k, out)
    out = s.set_policy("m", growth_params={"gate_seam_modee": 1})
    assert isinstance(out, dict) and out.get("refusal")
    assert "gate_seam_modee" in str(out["refusal"])
    # I-8 guard: registry never enters the dict
    assert set(DEFAULT_GROWTH_POLICY) == pre_keys
    for k in TEN_KEYS:
        assert k not in DEFAULT_GROWTH_POLICY


def test_g8_key_reaches_organ_half_speed(tmp_path):
    """The routed train_lr_scales REACHES the organ: one
    training step vs a full-speed twin — encoder
    displacement EXACTLY half (the T-B1 identity through
    the product gate), and the setting SURVIVES a fresh
    load (persistence)."""
    import copy
    s = _sys(tmp_path)
    s.create_model("m", description="g8")
    rows = _rows()
    s.teach("m", rows)
    s.study("m", rows, steps=30)
    organ, _ = s.lc._load_working("m")
    twin = copy.deepcopy(organ)
    W1_old = np.asarray(twin._bk.to_numpy(twin.W1)).copy()
    X = np.array([[r["input"]["a"], r["input"]["b"],
                   r["input"]["c"]] for r in rows])
    y = np.array([[float(r["target"])] for r in rows])
    twin.train_step(X, y)                      # full speed
    out = s.set_policy("m", growth_params={
        "train_lr_scales": {"encoder": 0.5}})
    assert not (isinstance(out, dict) and out.get("refusal"))
    organ2, _ = s.lc._load_working("m")        # fresh load —
    #                                            persistence
    organ2.train_step(X, y)                    # scaled
    got = np.asarray(organ2._bk.to_numpy(organ2.W1))
    full = np.asarray(twin._bk.to_numpy(twin.W1))
    expect = W1_old + 0.5 * (full - W1_old)
    assert np.array_equal(got, expect)         # EXACT half
