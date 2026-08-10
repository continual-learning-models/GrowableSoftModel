"""IT-LLM — system-run test: the CLOSED LOOP is usable BY AN LLM.

Simulates exactly how an LLM calls the system: a real MCP server in a
separate process, newline-delimited JSON-RPC over stdio, tool calls only
(no Python imports of system internals on the client side). Synthetic
(mock) data throughout — this verifies THE SYSTEM RUNS, not any real-data
case.

Checks:
  LLM-1  handshake + tool discovery (29 tools incl. plasticity + self-study verbs)
  LLM-2  ONE tool call runs the whole closed loop (run_course over a
         3-stage graded synthetic curriculum): stages mastered, commits
         gated, versions created, events logged — verified through MCP
         reads only
  LLM-3  the committed model answers correctly via infer (MCP)
  LLM-4  judgment-stop: with growth forbidden and an impossible target
         the loop STOPS and returns control (no thrashing) — via MCP
  LLM-5  a SECOND fresh server process serves the same model (the LLM
         can come back tomorrow)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------- synthetic curriculum (mock data; one law, nested regions) ----
LAW = lambda x: float(x[0] * x[1] + max(x[1] - x[2], 0.0) * x[0] + 2 * x[2])
BOUNDS = [(1, 1, 2), (3, 3, 3), (6, 6, 4)]


def make_stage(k, seed=0):
    rng = np.random.default_rng(seed + k)
    b = np.array(BOUNDS[k])
    mk = lambda X: [{"input": {"a": float(x[0]), "b": float(x[1]),
                               "c": float(x[2])},
                     "target": str(round(LAW(x), 6))} for x in X]
    Xs = rng.uniform(0, 1, (250, 3)) * b
    Xh = rng.uniform(0, 1, (30, 3)) * b
    Xe = rng.uniform(0, 1, (60, 3)) * b
    return {"name": f"stage{k + 1}", "examples": mk(Xs), "holdout": mk(Xh),
            "suite": {"X": Xe.tolist(),
                      "y": [str(round(LAW(x), 6)) for x in Xe]},
            "target": 0.85}


class MCPClient:
    """A minimal stand-in for an LLM MCP client: subprocess + JSON-RPC."""

    def __init__(self, models_root):
        env = dict(os.environ, SOFTMODEL_MODELS_ROOT=str(models_root),
                   SOFTMODEL_BACKEND="mlp")
        self.p = subprocess.Popen(
            [sys.executable, "-m", "mcp.mcp_server"],
            cwd=ROOT, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self._id = 0
        self.initialize = self._send({"method": "initialize", "params": {}})

    def _send(self, m):
        self._id += 1
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, **m}) + "\n")
        self.p.stdin.flush()
        while True:
            r = json.loads(self.p.stdout.readline())
            if r.get("id") == self._id:
                return r["result"]

    def tools(self):
        return self._send({"method": "tools/list"})["tools"]

    def call(self, tool, args):
        r = self._send({"method": "tools/call",
                        "params": {"name": tool, "arguments": args}})
        assert not r["isError"], f"{tool}: {r['content'][0]['text']}"
        return json.loads(r["content"][0]["text"])

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def main():
    tmp = tempfile.mkdtemp()
    try:
        c = MCPClient(tmp)

        # LLM-1: handshake + discovery
        assert c.initialize["serverInfo"]["name"] == "core"
        names = {t["name"] for t in c.tools()}
        assert len(names) == 29 and {"run_self", "self_review",
                                     "run_course", "set_policy",
                                     "widen", "add_feature", "refound",
                             "list_substrates",
                             "recommend_substrate"} <= names
        print("  LLM-1 handshake + 29 tools discovered: PASS")

        # LLM-2: one call runs the whole closed loop
        curriculum = [make_stage(k) for k in range(3)]
        c.call("create_model", {"model_id": "loop",
                                "holdout": curriculum[0]["examples"][:40]})
        rep = c.call("run_course", {"model_id": "loop",
                                    "curriculum": curriculum})
        assert rep.get("completed"), rep.get("stopped")
        assert all(st["mastered"] for st in rep["stages"])
        vs = c.call("get_versions", {"model_id": "loop"})
        assert vs["active"] != "v0"                      # commits happened
        card = c.call("card", {"model_id": "loop"})
        assert card["events"] > 0                        # lineage recorded
        finals = [round(st["final_accs"][i], 2)
                  for i, st in enumerate(rep["stages"])]
        print(f"  LLM-2 closed loop via ONE call: PASS "
              f"(stage accs {finals}, active {vs['active']})")

        # LLM-3: the committed model answers via MCP
        probe = {"a": 5.0, "b": 5.0, "c": 2.0}
        out = c.call("infer", {"model_id": "loop", "input": probe})
        want = LAW([5.0, 5.0, 2.0])
        assert abs(out["output"] - want) <= 1.0, (out, want)
        print(f"  LLM-3 committed model serves: PASS "
              f"(probe -> {out['output']:.2f}, truth {want:.2f})")

        # LLM-4: judgment-stop instead of thrashing
        c.call("create_model", {"model_id": "hard",
                                "holdout": curriculum[0]["examples"][:40]})
        c.call("set_policy", {"model_id": "hard",
                              "updates": {"max_depth": 1,
                                          "max_params_mult": 1}})
        cur = [dict(make_stage(0), target=1.01)]
        rep = c.call("run_course", {"model_id": "hard", "curriculum": cur,
                                    "policy": {"max_blocks_per_stage": 5}})
        assert rep.get("stopped"), rep
        print(f"  LLM-4 judgment-stop returned to teacher: PASS "
              f"({rep['stopped']['reason']})")
        c.close()

        # LLM-5: a fresh server process serves the same model
        c2 = MCPClient(tmp)
        out2 = c2.call("infer", {"model_id": "loop", "input": probe})
        assert abs(out2["output"] - out["output"]) < 1e-9
        c2.close()
        print("  LLM-5 fresh process, same model: PASS")

        print("SYSTEM-RUN TEST (LLM-callable closed loop): PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
