"""M5 MCP adapter tests: protocol surface + the full brain-organ loop driven
purely through JSON-RPC messages."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.mcp_server import MCPServer, TOOLS

from test_drift import (FEATURES, TRAIN_OLD, HOLDOUT_OLD, TRAIN_NEW,
                        HOLDOUT_NEW)  # noqa: E402


def _server(tmp):
    return MCPServer(SoftModelFactory(Config.from_env(
        backend="mlp", models_root=Path(tmp),
        gate_recent_n=8, drift_tolerance=0.15)))


def _call(server, name, args, msg_id=1):
    resp = server.handle({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
    result = resp["result"]
    payload = json.loads(result["content"][0]["text"]) if not result["isError"] \
        else result["content"][0]["text"]
    return result["isError"], payload


def test_protocol_surface():
    tmp = tempfile.mkdtemp()
    try:
        s = _server(tmp)
        # initialize
        r = s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                      "params": {"protocolVersion": "2024-11-05"}})
        assert r["result"]["serverInfo"]["name"] == "generator"
        assert "tools" in r["result"]["capabilities"]
        # notifications produce no response
        assert s.handle({"jsonrpc": "2.0",
                         "method": "notifications/initialized"}) is None
        # ping
        assert s.handle({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
        # tools/list
        r = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in r["result"]["tools"]}
        assert names == {t["name"] for t in TOOLS} and "infer" in names
        # unknown method -> -32601
        r = s.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
        assert r["error"]["code"] == -32601
        # unknown tool -> isError
        err, payload = _call(s, "nonexistent", {})
        assert err and "unknown tool" in payload
    finally:
        shutil.rmtree(tmp)


def test_full_loop_over_mcp():
    tmp = tempfile.mkdtemp()
    try:
        s = _server(tmp)
        # create a Type-D model
        err, _ = _call(s, "create_model", {
            "model_id": "risk", "holdout": HOLDOUT_OLD})
        assert not err
        # fleet discovery
        err, fleet = _call(s, "list_models", {})
        assert not err and fleet[0]["model_id"] == "risk"
        # teach + infer + discoveries
        err, r = _call(s, "teach", {"model_id": "risk", "examples": TRAIN_OLD})
        assert not err and r["promoted"] and r["candidate_metric"] >= 0.9
        err, out = _call(s, "infer", {"model_id": "risk",
                                      "input": {"amount": 800, "night": 0, "foreign": 0}})
        assert not err and out["output"] == "HIGH"
        err, d = _call(s, "discoveries", {"model_id": "risk"})
        assert not err and d["n_rules"] >= 1
        # drift loop
        err, _ = _call(s, "add_holdout", {"model_id": "risk", "examples": HOLDOUT_NEW})
        assert not err
        err, drift = _call(s, "check_drift", {"model_id": "risk"})
        assert not err and drift["needs_reteach"]
        err, r2 = _call(s, "teach", {"model_id": "risk", "examples": TRAIN_NEW,
                                     "window": len(TRAIN_NEW)})
        assert not err and r2["promoted"]
        err, out = _call(s, "infer", {"model_id": "risk",
                                      "input": {"amount": 75, "night": 0, "foreign": 1}})
        assert not err and out["output"] == "HIGH"
        # lineage + rollback
        err, v = _call(s, "get_versions", {"model_id": "risk"})
        assert not err and v["active"] == r2["candidate_version"]
        err, rb = _call(s, "rollback", {"model_id": "risk", "to": "v1"})
        assert not err and rb["active"] == "v1"
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_protocol_surface()
    test_full_loop_over_mcp()
    print("mcp tests passed")
