"""M5 MCP demo — the full vision, live: a "brain" drives the whole loop
purely through MCP JSON-RPC messages (exactly what an LLM client sends).

Narrative:
  [1] initialize + discover tools
  [2] create a Type-D model, teach it raw observations
  [3] use it (infer) + read its DISCOVERED regularities
  [4] reality changes -> add_holdout -> check_drift says re-teach
  [5] teach (windowed) -> regularities evolved -> explainable infer

Run:  python scripts/demo_mcp.py
(For a real LLM: claude mcp add generator -- python3 -m generator.mcp_server)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.mcp_server import MCPServer
from test_drift import (FEATURES, TRAIN_OLD, HOLDOUT_OLD, TRAIN_NEW,
                        HOLDOUT_NEW)

MODEL = "demo_mcp_risk"


def line(c="-"):
    print(c * 64)


def main() -> int:
    factory = SoftModelFactory(Config.from_env(
        backend="mlp", gate_recent_n=8, drift_tolerance=0.15))
    if factory.registry.model_dir(MODEL).exists():
        shutil.rmtree(factory.registry.model_dir(MODEL))
    server = MCPServer(factory)
    counter = {"id": 0}

    def rpc(method, params=None):
        counter["id"] += 1
        return server.handle({"jsonrpc": "2.0", "id": counter["id"],
                              "method": method, "params": params or {}})

    def call(tool, args):
        resp = rpc("tools/call", {"name": tool, "arguments": args})
        result = resp["result"]
        assert not result["isError"], result["content"][0]["text"]
        return json.loads(result["content"][0]["text"])

    line("=")
    print("SoftModel MCP demo — a 'brain' drives the loop via JSON-RPC only")
    line("=")

    print("\n[1] initialize + tools/list:")
    init = rpc("initialize", {"protocolVersion": "2024-11-05"})
    info = init["result"]["serverInfo"]
    print(f"   connected to {info['name']} v{info['version']}")
    rpc("notifications/initialized")
    tools = rpc("tools/list")["result"]["tools"]
    print(f"   {len(tools)} tools: {', '.join(t['name'] for t in tools)}")

    print("\n[2] create_model + teach raw observations:")
    call("create_model", {"model_id": MODEL,
                          "description": "transaction risk regularities",
                          "holdout": HOLDOUT_OLD})
    r1 = call("teach", {"model_id": MODEL, "examples": TRAIN_OLD})
    print(f"   teach -> {r1['candidate_version']} metric "
          f"{r1['candidate_metric']:.2f} promoted={r1['promoted']}")

    print("\n[3] use it + read what it discovered:")
    out = call("infer", {"model_id": MODEL,
                         "input": {"amount": 800, "night": 0, "foreign": 0}})
    print(f"   infer -> {out['output']} (conf {out['confidence']})")
    d1 = call("discoveries", {"model_id": MODEL})
    for ln in d1["regularities"]:
        print(f"     {ln}")

    print("\n[4] reality changes -> add_holdout -> check_drift:")
    call("add_holdout", {"model_id": MODEL, "examples": HOLDOUT_NEW})
    drift = call("check_drift", {"model_id": MODEL})
    print(f"   drift: recent={drift['recent_metric']:.2f} vs baseline="
          f"{drift['baseline_metric']:.2f} -> needs_reteach={drift['needs_reteach']}")

    print("\n[5] re-teach (windowed) -> regularities evolve:")
    r2 = call("teach", {"model_id": MODEL, "examples": TRAIN_NEW,
                        "window": len(TRAIN_NEW)})
    print(f"   teach -> {r2['candidate_version']} metric "
          f"{r2['candidate_metric']:.2f} promoted={r2['promoted']}")
    d2 = call("discoveries", {"model_id": MODEL})
    for ln in d2["regularities"]:
        print(f"     {ln}")
    probe = {"amount": 75, "night": 0, "foreign": 1}
    out = call("infer", {"model_id": MODEL, "input": probe})
    print(f"   infer {probe} -> {out['output']}  because: {out.get('rule')}")

    fleet = call("list_models", {})
    entry = next(e for e in fleet if e["model_id"] == MODEL)
    print(f"\n[6] list_models: {entry['model_id']} "
          f"learned_shape={entry['learned_shape']['mode']} "
          f"active={entry['active_version']} score={entry['active_score']}")

    line("=")
    ok = (r1["promoted"] and drift["needs_reteach"] and r2["promoted"]
          and out["output"] == "HIGH"
          and any(("foreign == 1" in ln and "night == 0" in ln)
                  for ln in d2["regularities"]))
    print("MCP ACCEPTANCE:", "PASS" if ok else "FAIL")
    line("=")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
