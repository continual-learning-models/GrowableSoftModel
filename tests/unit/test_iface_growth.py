"""S9.5 per-model growth policy + body_type (docs/system/6 D-N1;
plan doc 7 T24 a-e). THE one family-file edit of the supplement:
net.py's seven module-default reads become instance-first —
absent attribute == module default bit-identically (the
equivalence battery pins that system-wide; T24d pins it here).
Zero algorithm change: only where parameters are READ FROM.
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
from reference_net.growthpolicy import (                # noqa: E402
    DEFAULT_GROWTH_POLICY)
from reference_net.net import Network                   # noqa: E402

ROWS = [{"input": {"x": float(i) / 24.0, "y": float(24 - i) / 24.0},
         "target": (float(i) / 24.0) * 0.7 + 0.1}
        for i in range(24)]


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_t24a_growth_params_reach_the_organ(ws):
    """T24a: set_policy(growth_params={...}) -> the next load
    installs organ._growth_policy carrying the override merged
    over the module defaults."""
    ws.create_model("t24a", holdout=ROWS[:8], substrate="mlp")
    ws.study("t24a", ROWS[8:], steps=10)
    organ0, _ = ws.lc._load_working("t24a")
    assert getattr(organ0, "_growth_policy", None) is None
    ws.set_policy("t24a", growth_params={"loop_K_max": 8})
    organ, _ = ws.lc._load_working("t24a")
    gp = getattr(organ, "_growth_policy", None)
    assert gp is not None and gp["loop_K_max"] == 8
    assert gp["grow_body_type"] == \
        DEFAULT_GROWTH_POLICY["grow_body_type"]   # merged, not bare


def test_t24b_unknown_growth_key_refused(ws):
    """T24b: unknown growth key -> loud refusal naming it, at the
    door (set_policy AND create_model), nothing stored."""
    ws.create_model("t24b", holdout=ROWS[:8], substrate="mlp")
    r = ws.set_policy("t24b", growth_params={"loop_k_max": 8})
    assert "refusal" in r and "loop_k_max" in r["refusal"]
    assert not ws.lc.policy("t24b").get("growth_params")
    r2 = ws.create_model("t24b2", holdout=ROWS[:8], substrate="mlp",
                         policy={"growth_params": {"nope": 1}})
    assert "refusal" in r2 and "nope" in r2["refusal"]


def test_t24c_body_type_reaches_network_grow(ws):
    """T24c: grow(body_type=...) threads facade -> lifecycle ->
    grow_site -> Network.grow — POSITIVE reaching proof: the
    registered non-default body type 'attention' actually grows
    an attention body (not the reference Network)."""
    ws.create_model("t24c", holdout=ROWS[:8], substrate="mlp")
    ws.study("t24c", ROWS[8:], steps=30)
    out = ws.grow("t24c", k_nodes=1, body_type="attention")
    assert "refusal" not in out
    organ, _ = ws.lc._load_working("t24c")
    assert len(organ._port_site.bodies) >= 1
    body = organ._port_site.bodies[0]["body"]
    assert type(body).__name__ != "Network", type(body).__name__


def test_t24d_absent_attribute_module_default_bitwise():
    """T24d: an organ WITHOUT _growth_policy reads the module
    default at all seven sites — training trajectory bitwise
    identical to an organ of the same seed (equivalence pin in
    miniature; the battery pins it system-wide)."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 3))
    y = np.sin(X.sum(1, keepdims=True))
    a = Network(d_in=3, hidden=5, lr=1e-2, seed=11)
    b = Network(d_in=3, hidden=5, lr=1e-2, seed=11)
    assert not hasattr(a, "_growth_policy")
    for _ in range(30):
        la = a.train_step(X, y)
        lb = b.train_step(X, y)
        assert la == lb                        # bitwise, every step
    assert np.array_equal(a.W1, b.W1)


def test_t24e_non_loop_sites_honored_per_model(ws):
    """T24e (review F1 coverage): a NON-loop site honors the
    per-model policy — grow_body_type steers Network.grow's
    default-body choice for ITS model only; a sibling model in
    the same process keeps the module default."""
    for mid in ("t24e-a", "t24e-b"):
        ws.create_model(mid, holdout=ROWS[:8], substrate="mlp")
        ws.study(mid, ROWS[8:], steps=30)
    ws.set_policy("t24e-a", growth_params={"grow_body_type":
                                           "attention"})
    # model A: per-model grow_body_type steers site :405 -> its
    # grown body is an ATTENTION body (read is per-model)
    out_a = ws.grow("t24e-a", k_nodes=1)
    assert "refusal" not in out_a
    organ_a, _ = ws.lc._load_working("t24e-a")
    body_a = organ_a._port_site.bodies[0]["body"]
    assert type(body_a).__name__ != "Network", type(body_a).__name__
    # model B (same process): module default untouched -> grows the
    # reference Network body
    out = ws.grow("t24e-b", k_nodes=1)
    assert "refusal" not in out
    organ_b, _ = ws.lc._load_working("t24e-b")
    body_b = organ_b._port_site.bodies[0]["body"]
    assert type(body_b).__name__ == "Network", type(body_b).__name__
