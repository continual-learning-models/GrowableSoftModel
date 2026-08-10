"""T3 integration tests: mixed trees across every audited walk
(DESIGN_GROW_BODY_TYPE v1.2)."""
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
import reference_net.growthpolicy as gp                     # noqa: E402
from reference_net.attention_body import AttentionBody      # noqa: E402
from reference_net.growthpolicy import decide, grow_with_policy  # noqa: E402
from reference_net.growthpolicy.pricer_zero_attach import (  # noqa: E402
    _fingerprint)
from reference_net.net import Network                       # noqa: E402
from reference_net.spu.spu_network import SPUNetwork        # noqa: E402
from engine.spu.spu_report import build_report       # noqa: E402
from reference_net.trainer import collect_instability       # noqa: E402


def mixed_net(seed=7, steps=20, cls=Network):
    net = cls(4, 6, seed=seed)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (48, 4))
    y = np.sin(X[:, :1]) + 0.3 * X[:, 1:2]
    for _ in range(steps):
        net.train_step(X, y)
    net.grow(1)                                # reference child
    net.grow(2, body_type="attention")         # attention child
    for _ in range(steps):
        net.train_step(X, y)
    return net, X, y


def test_decide_refuses_attention_scope():
    net, _, _ = mixed_net()
    d = decide(net.grown_body(2))
    assert "4b" in d["refusal"]
    assert d["body_type"] == "attention"
    d2 = grow_with_policy(net.grown_body(2))
    assert "4b" in d2["refusal"]


def test_pricing_parent_scope_with_attention_in_subtree():
    net, _, _ = mixed_net()
    fp1 = _fingerprint(net)                    # would crash pre-T3
    fp2 = _fingerprint(net)
    assert fp1 == fp2                          # deterministic
    net.grown_body(2).P["Wv"][0, 0] += 1.0
    assert _fingerprint(net) != fp1            # sensitive to body


def test_decide_on_parent_of_mixed_tree_returns_decision():
    net, _, _ = mixed_net(steps=40)
    d = decide(net)
    assert isinstance(d, dict)
    assert "refusal" not in d or "4b" not in str(d.get("refusal"))


def test_collect_instability_mixed_tree():
    net, _, _ = mixed_net()
    rows = collect_instability(net)
    owners = {id(r[3]) for r in rows}
    assert id(net.grown_body(2)) not in owners      # body yields no rows
    assert any(r[0] == "1" for r in rows)      # reference child does


def test_attribution_walk_pattern_mixed_tree():
    net, X, _ = mixed_net()
    out = []
    kids = [(sl.get("key", g), sl["body"]) for g, sl in
            enumerate(net._port_site.bodies)]  # instrument.py walk shape
    for j, inner in kids:
        mag = float(np.mean(np.abs(
            inner.predict(net._std_x(X))[:, 0])))
        out.append((j, mag))
    assert len(out) == 2 and all(np.isfinite(m) for _, m in out)


def test_pickle_and_deepcopy_mixed_tree_bitwise():
    net, X, _ = mixed_net()
    for clone in (pickle.loads(pickle.dumps(net)),
                  copy.deepcopy(net)):
        assert np.array_equal(clone.predict(X), net.predict(X))


def test_removal_restores_function_bitwise():
    net, X, y = mixed_net()
    before = net.predict(X)
    net.grow(3, body_type="attention")          # exact entry, untrained
    assert np.array_equal(net.predict(X), before)
    net.remove_grown(3)
    assert np.array_equal(net.predict(X), before)


def test_spu_processes_attention_and_reference():
    net = SPUNetwork(4, 6, seed=7)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (48, 4))
    y = np.sin(X[:, :1])
    net.set_spu_policy({"spu_enabled": True, "spu_warmup_steps": 0})
    net.train_step(X, y)
    net.grow(1)                                 # reference newborn
    net.grow(2, body_type="attention")          # attention newborn
    for _ in range(10):
        net.train_step(X, y)
    # T6 inversion: attention bodies are now PROCESSED, not
    # disclosed-skipped (unknown types still are — unit-tested)
    assert getattr(net, "_spu_skip_counts", {}).get(
        "body_type_unsupported", 0) == 0
    att_events = [e for e in net.spu_events
                  if e.get("path") == "root/port[1]"
                  and e.get("body_type") == "attention"]
    assert att_events                           # attention processes
    ref_events = [e for e in net.spu_events
                  if e.get("path") == "root/port[0]" and "steps" in e
                  and "body_type" not in e]
    assert ref_events                           # sibling processes
    rep = build_report(net)
    assert rep["processed_steps"] > 0


def test_selector_applier_routes_through_policy():
    net, _, _ = mixed_net()
    gp.DEFAULT_GROWTH_POLICY["grow_body_type"] = "attention"
    try:
        # the applier's exact call shape (growthpolicy line ~153)
        net.grow(3, hidden=gp.DEFAULT_GROWTH_POLICY["refine_hidden"])
        assert type(net.grown_body(3)) is AttentionBody
        assert any(e["event"] == "refine[attention]"
                   for e in net.gain_ledger)
    finally:
        gp.DEFAULT_GROWTH_POLICY["grow_body_type"] = "reference"


def test_structure_depth_params_mixed():
    net, _, _ = mixed_net()
    rows = net.structure()
    assert len(rows) == 3
    att = [r for r in rows if r.get("body_type") == "attention"]
    assert len(att) == 1
    assert net.depth() == 2
    assert net.n_params() > 471                # includes the body
