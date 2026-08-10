"""SPU S4 tests: readiness-field report + JSONL artifacts
(DEV_PLAN T4.1-T4.7)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.spu.spu_network import SPUNetwork       # noqa: E402
from engine.spu.spu_report import (                 # noqa: E402
    build_report, read_events_jsonl, write_events_jsonl)


def processed_net(steps_on=6):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (32, 2))
    y = np.sin(3 * X[:, :1]) + 0.5 * X[:, 1:]
    net = SPUNetwork(d_in=2, hidden=4, lr=1e-2, seed=8)
    for _ in range(60):
        net.train_step(X, y)
    net.grow(1, hidden=5)
    for _ in range(5):
        net.train_step(X, y)
    net.set_spu_policy({"spu_enabled": True,
                        "spu_warmup_steps": 0})
    for _ in range(steps_on):
        net.train_step(X, y)
    return net, X, y


def test_report_schema():                              # T4.1
    net, _, _ = processed_net()
    r = build_report(net)
    for k in ("spus", "skips", "processed_steps",
              "interference", "total_events"):
        assert k in r
    unit = r["spus"]["root/port[0]"]
    for k in ("events", "steps_mean", "steps_max", "converged",
              "disc_hits", "clips", "j_inv_first", "j_inv_last"):
        assert k in unit


def test_counts_add_up():                              # T4.2
    net, _, _ = processed_net(steps_on=6)
    r = build_report(net)
    leaf_events = sum(v["events"] for v in r["spus"].values())
    skip_events = sum(r["skips"].values())
    assert (leaf_events + skip_events + r["processed_steps"]
            == r["total_events"])
    assert r["spus"]["root/port[0]"]["events"] == 6


def test_summary_per_processed_step():                 # T4.3
    net, _, _ = processed_net(steps_on=4)
    r = build_report(net)
    assert r["processed_steps"] == 4
    assert r["interference"]["steps"] == 4


def test_skip_reasons_enumerated():                    # T4.4
    net, X, y = processed_net()
    net.grown_body(1).deepen(m=2)                      # leaf gains blocks
    net.train_step(X, y)
    r = build_report(net)
    # P2-S2 inversion: the has_blocks gate is deleted — the
    # deepened leaf keeps processing (chain-aware), never skips
    assert "has_blocks" not in r["skips"]
    deep = [e for e in net.spu_events
            if e.get("path") == "root/port[0]" and e.get("blocks") == 1]
    assert deep


def test_report_on_off_model_wellformed():             # T4.5
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (16, 2))
    y = X[:, :1]
    net = SPUNetwork(d_in=2, hidden=4, seed=1)
    for _ in range(3):
        net.train_step(X, y)
    r = build_report(net)
    assert r["total_events"] == 0 and r["spus"] == {}
    assert r["interference"] is None


def test_artifact_roundtrip(tmp_path):                 # T4.6
    net, _, _ = processed_net()
    p = write_events_jsonl(tmp_path / "spu_events.jsonl",
                           net.spu_events)
    back = read_events_jsonl(p)
    assert back == net.spu_events


def test_predict_adds_nothing_to_report():             # T4.7
    net, X, _ = processed_net()
    before = build_report(net)
    for _ in range(30):
        net.predict(X)
    assert build_report(net) == before
