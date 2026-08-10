"""MCP stdio server (R6): expose the Instrument to an LLM Teacher.

Dependency-free newline-delimited JSON-RPC 2.0 (Phase-1 pattern). The
Teacher drives the whole course through these tools; pedagogy lives in
the Teacher, verification lives with the caller's test harness (S0).

Run:      python3 -m reference_net.mcp_server
Register: claude mcp add reference_net -- python3 -m reference_net.mcp_server
Env:      SOFTSCALE2_ROOT (students dir), SOFTSCALE2_SEED
"""
from __future__ import annotations

import json
import os
import sys

from .instrument import Instrument

PROTOCOL_VERSION = "2024-11-05"
_ARR = {"type": "array"}

TOOLS = [
    {"name": "create_student",
     "description": ("Create a fresh student (recursive multi-scale network,"
                     " all-atomic start). Growth happens later, driven by"
                     " problem complexity."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "d_in": {"type": "integer"},
         "hidden": {"type": "integer"}}, "required": ["id"]}},
    {"name": "study",
     "description": ("Supervised STUDY block: labeled examples (X rows,"
                     " y values). The student learns; returns train MSE."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "X": _ARR, "y": _ARR,
         "steps": {"type": "integer"}}, "required": ["id", "X", "y"]}},
    {"name": "attempts",
     "description": ("PRACTICE phase 1: the student answers problems X with"
                     " varied attempts. Attempt 0 is its own answer. Verify"
                     " them yourself; then call practice_update."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "X": _ARR,
         "n_attempts": {"type": "integer"}}, "required": ["id", "X"]}},
    {"name": "practice_update",
     "description": ("PRACTICE phase 2: per problem give a verified-correct"
                     " answer ONLY where attempt 0 FAILED your verifier"
                     " (else null). Consolidation update; beyond-reach"
                     " count returned (STUCK evidence)."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "X": _ARR, "passed": _ARR},
         "required": ["id", "X", "passed"]}},
    {"name": "evaluate",
     "description": ("Score the student on suites [{name,X,y}] (held-out)."
                     " Appends to the score matrix used by trajectory."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "suites": _ARR},
         "required": ["id", "suites"]}},
    {"name": "trajectory",
     "description": ("S4 dashboard: progress, recent_gain, volatility"
                     " (backward drop), retention, and a verdict —"
                     " REAL / FALSE_SPIKE / FALSE_SWAP / STUCK. Never"
                     " advance the curriculum on a FALSE verdict; on STUCK"
                     " consider remediation or growth."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "current_stage": {"type": "integer"}},
         "required": ["id"]}},
    {"name": "growth_report",
     "description": ("Ranked most-unstable nodes across ALL depths (the"
                     " owner's oscillation signal) + params/depth."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}}, "required": ["id"]}},
    {"name": "grow",
     "description": ("Refine the k most unstable nodes into inner networks"
                     " (function-preserving; checkpointed). Verify the"
                     " investment pays via trajectory, else rollback."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "k_nodes": {"type": "integer"},
         "hidden": {"type": "integer"}}, "required": ["id"]}},
    {"name": "rollback",
     "description": "Restore the checkpoint taken by a grow call.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "ckpt": {"type": "string"}},
         "required": ["id", "ckpt"]}},
    {"name": "attribution",
     "description": ("Which suite each grown node serves (activation-mass"
                     " distribution) — verifies growth landed where the"
                     " difficulty was."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}, "suites": _ARR},
         "required": ["id", "suites"]}},
    {"name": "card",
     "description": "Student card: params, depth, recursive structure.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string"}}, "required": ["id"]}},
]


class MCPServer:
    def __init__(self, instrument: Instrument | None = None):
        self.ins = instrument or Instrument(
            root=os.environ.get("SOFTSCALE2_ROOT",
                                str(Instrument.__init__.__defaults__[0])),
            seed=int(os.environ.get("SOFTSCALE2_SEED", "7")))

    def _dispatch(self, name, a):
        i = self.ins
        if name == "create_student":
            return i.create_student(a["id"], a.get("d_in", 3),
                                    a.get("hidden", 16))
        if name == "study":
            return i.study(a["id"], a["X"], a["y"], a.get("steps", 200))
        if name == "attempts":
            return {"attempts": i.attempts(a["id"], a["X"],
                                           a.get("n_attempts", 8))}
        if name == "practice_update":
            return i.practice_update(a["id"], a["X"], a["passed"])
        if name == "evaluate":
            return i.evaluate(a["id"], a["suites"])
        if name == "trajectory":
            return i.trajectory(a["id"], a.get("current_stage"))
        if name == "growth_report":
            return i.growth_report(a["id"])
        if name == "grow":
            return i.grow(a["id"], a.get("k_nodes", 2), a.get("hidden", 16))
        if name == "rollback":
            return i.rollback(a["id"], a["ckpt"])
        if name == "attribution":
            return i.attribution(a["id"], a["suites"])
        if name == "card":
            return i.card(a["id"])
        raise ValueError(f"unknown tool: {name}")

    def handle(self, msg):
        method, mid = msg.get("method"), msg.get("id")
        if method is None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32600, "message": "no method"}}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "reference_net", "version": "0.1.0"}}}
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        if method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            try:
                out = self._dispatch(name, args)
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps(out)}],
                    "isError": False}}
            except Exception as exc:                 # noqa: BLE001
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"error: {exc}"}],
                    "isError": True}}
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"unknown: {method}"}}


def main():
    server = MCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = server.handle(json.loads(line))
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
