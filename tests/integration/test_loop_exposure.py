"""L3 exposure tests (DEV_PLAN v1.2, ~8)."""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY as GP  # noqa: E402
from mcp.mcp_server import MCPServer, TOOLS                    # noqa: E402


@pytest.fixture
def sysdir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", tmp)
        yield tmp


def call(srv, name, args):
    r = srv.handle({"jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
    return json.loads(r["result"]["content"][0]["text"])


def _shaped_model(srv, mid="lm"):
    rng = np.random.default_rng(0)
    rows = [{"input": {"a": float(a), "b": float(b), "c": float(c)},
             "target": float(2 * a + b)}
            for a, b, c in rng.normal(size=(48, 3))]
    call(srv, "create_model", {"model_id": mid})
    for _ in range(4):
        call(srv, "study", {"model_id": mid, "examples": rows,
                            "steps": 40})
    return rows


def test_1_tool_table_38():
    names = [t["name"] for t in TOOLS]
    assert len(names) == 52      # 59C stage 3 adds the 9 growth-control tools + store (sanctioned; was 42)
    assert "loop" in names and "remove_loop" in names


def test_2_mcp_loop_round_trip(sysdir):
    srv = MCPServer()
    rows = _shaped_model(srv)
    GP["loop_enabled"] = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = call(srv, "loop", {"model_id": "lm"})
        assert out["ok"] and out["params_added"] > 0
        assert "ITERATES at serving" in out["note"]
        out2 = call(srv, "remove_loop", {"model_id": "lm"})
        assert out2["ok"]
    finally:
        GP["loop_enabled"] = False


def test_3_refusal_surfaces_via_mcp(sysdir):
    srv = MCPServer()
    _shaped_model(srv)
    out = call(srv, "loop", {"model_id": "lm"})   # switch OFF
    assert "opt-in option" in out["refusal"]


def test_4_polite_non_reference_refusal(sysdir):
    srv = MCPServer()
    call(srv, "create_model", {"model_id": "tm",
                               "substrate": "transformer"})
    rng = np.random.default_rng(0)
    rows = [{"input": {"a": float(a), "b": float(b)},
             "target": float(a + b)}
            for a, b in rng.normal(size=(32, 2))]
    call(srv, "study", {"model_id": "tm", "examples": rows,
                        "steps": 30})
    GP["loop_enabled"] = True
    try:
        out = call(srv, "loop", {"model_id": "tm"})
    finally:
        GP["loop_enabled"] = False
    assert "no composition chain" in out["refusal"]


def test_5_bad_container_refusal(sysdir):
    srv = MCPServer()
    _shaped_model(srv)
    GP["loop_enabled"] = True
    try:
        out = call(srv, "loop", {"model_id": "lm",
                                 "container": "7/9"})
    finally:
        GP["loop_enabled"] = False
    assert "no scope at" in out["refusal"]


def test_6_spu_disclosed_skip():
    from engine.spu.spu_loop import SKIP_HAS_LOOP, skip_reason
    from reference_net.net import Network
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 3))
    y = X[:, :1] * 2
    host = Network(3, 6, seed=1)
    for _ in range(120):
        host.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        host.grow(0, hidden=4)
        host.grow(1, hidden=4)
    unit = host.grown_body(0)
    for _ in range(10):
        host.train_step(X, y)
    GP["loop_enabled"] = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unit.loop(2)
    finally:
        GP["loop_enabled"] = False
    pol = {"spu_enabled": True, "spu_every": 1,
           "spu_scope": "newborn_inner_nets",
           "spu_warmup_steps": 0, "spu_newborn_steps": 999,
           "spu_n_min": 8}
    n = 48
    assert skip_reason(unit, n, pol, 1,
                       is_root=False) == SKIP_HAS_LOOP
    sib = host.grown_body(1)
    assert skip_reason(sib, n, pol, 1,
                       is_root=False) is None


def test_7_cli_help_lists_loop():
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "cli/cli.py"),
                        "help"], capture_output=True, text=True)
    d = json.loads(r.stdout)
    soft = d["softmodel_verbs (the tool's method)"]
    assert "loop" in soft and "remove_loop" in soft


def test_8_card_reports_survive_loop(sysdir):
    srv = MCPServer()
    _shaped_model(srv)
    GP["loop_enabled"] = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call(srv, "loop", {"model_id": "lm"})
        card = call(srv, "card", {"model_id": "lm"})
        assert isinstance(card, dict)
    finally:
        GP["loop_enabled"] = False
