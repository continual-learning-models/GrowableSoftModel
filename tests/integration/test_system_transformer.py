"""IT-TF — whole-system run flow ON THE TRANSFORMER HOST (repeatable).

The transformer-substrate analog of the integration suite's core flows,
driven the way an LLM drives the system (real MCP server subprocess,
JSON-RPC only), simulated data:

  TF-1  create with substrate="transformer" -> session study -> gated
        commit -> committed serving (accuracy checked)
  TF-2  growth through the surface (growth_report -> grow) -> continued
        study -> second gated commit; lineage = v0 + commits exactly
  TF-3  trajectory + rollback across the growth boundary (exact pointer)
  TF-4  restart recovery: server killed; a FRESH process serves the SAME
        transformer model (registry-dispatched artifact loading)
  TF-5  drift on the transformer host: reality switches -> check_drift
        fires -> windowed session re-teach -> gated commit clears it

Run: python3 tests/integration/test_system_transformer.py
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

LAW_A = lambda x: round(float(x[0] * x[1] + 2 * x[2]), 6)
LAW_B = lambda x: round(float(-x[0] * x[1] + x[2]), 6)     # drifted reality


def rows(n, fn, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 2, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]),
                       "c": float(x[2])}, "target": str(fn(x))} for x in X]


class MCP:
    def __init__(self, root):
        env = dict(os.environ, SOFTMODEL_MODELS_ROOT=str(root),
                   SOFTMODEL_BACKEND="mlp")
        self.p = subprocess.Popen(
            [sys.executable, "-m", "mcp.mcp_server"],
            cwd=ROOT, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self._i = 0
        self._send({"method": "initialize", "params": {}})

    def _send(self, m):
        self._i += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._i,
                                       **m}) + "\n")
        self.p.stdin.flush()
        while True:
            r = json.loads(self.p.stdout.readline())
            if r.get("id") == self._i:
                return r["result"]

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
        c = MCP(tmp)
        # ---- TF-1: create/study/commit/serve ----
        out = c.call("create_model", {"model_id": "tf",
                                      "holdout": rows(50, LAW_A, 1),
                                      "substrate": "transformer"})
        assert out["substrate"] == "transformer"
        for k in range(3):
            c.call("study", {"model_id": "tf",
                             "examples": rows(250, LAW_A, 2 + k),
                             "steps": 150})
        r1 = c.call("commit", {"model_id": "tf"})
        assert r1["promoted"], r1
        probe = {"a": 1.5, "b": 1.5, "c": 1.0}
        o = c.call("infer", {"model_id": "tf", "input": probe})
        assert abs(o["output"] - LAW_A([1.5, 1.5, 1.0])) <= 0.5, o
        print("  TF-1 create/study/gated-commit/serve on transformer: PASS")

        # ---- TF-2: growth via surface + second commit + lineage ----
        rep = c.call("growth_report", {"model_id": "tf"})
        assert rep["candidates"] and "site" in rep["candidates"][0]
        g = c.call("grow", {"model_id": "tf", "k_nodes": 2})
        assert g["grown"] and g["depth"] == 2, g
        for k in range(2):
            c.call("study", {"model_id": "tf",
                             "examples": rows(250, LAW_A, 10 + k),
                             "steps": 150})
        r2 = c.call("commit", {"model_id": "tf", "note": "post-growth"})
        vs = c.call("get_versions", {"model_id": "tf"})
        n_commits = 1 + (1 if r2.get("promoted") else 0)
        assert len(vs["versions"]) == 1 + n_commits    # v0 + commits only
        card = c.call("card", {"model_id": "tf"})
        assert card["learned_shape"]["substrate"] == "transformer"
        print(f"  TF-2 growth via surface (depth 2) + lineage sane "
              f"(v0+{n_commits} commits): PASS")

        # ---- TF-3: trajectory + rollback across growth boundary ----
        suites = [{"name": "s", "X": [[1.5, 1.5, 1.0], [1, 1, 1]],
                   "y": [str(LAW_A([1.5, 1.5, 1.0])), str(LAW_A([1, 1, 1]))]}]
        c.call("evaluate", {"model_id": "tf", "suites": suites})
        c.call("evaluate", {"model_id": "tf", "suites": suites})
        t = c.call("trajectory", {"model_id": "tf"})
        assert t["verdict"] in ("REAL", "STUCK", "INSUFFICIENT")
        c.call("rollback", {"model_id": "tf", "to": r1["version"]})
        assert c.call("get_versions",
                      {"model_id": "tf"})["active"] == r1["version"]
        o2 = c.call("infer", {"model_id": "tf", "input": probe})
        assert abs(o2["output"] - LAW_A([1.5, 1.5, 1.0])) <= 0.5
        print("  TF-3 trajectory + rollback across growth boundary: PASS")
        c.close()

        # ---- TF-4: restart recovery in a fresh process ----
        c2 = MCP(tmp)
        o3 = c2.call("infer", {"model_id": "tf", "input": probe})
        assert abs(o3["output"] - o2["output"]) < 1e-9
        print("  TF-4 fresh process serves the same transformer model: PASS")

        # ---- TF-5: drift on the transformer host ----
        c2.call("add_holdout", {"model_id": "tf",
                                "examples": rows(40, LAW_B, 20)})
        d = c2.call("check_drift", {"model_id": "tf", "recent_n": 40})
        assert d["drifted"] and d["needs_reteach"], d
        c2.call("set_policy", {"model_id": "tf",
                               "updates": {"gate_recent_n": 40}})
        for k in range(3):
            c2.call("study", {"model_id": "tf",
                              "examples": rows(250, LAW_B, 30 + k),
                              "steps": 150})
        r3 = c2.call("commit", {"model_id": "tf", "note": "drift adapt"})
        assert r3["promoted"], r3
        d2 = c2.call("check_drift", {"model_id": "tf", "recent_n": 40})
        assert not d2["drifted"], d2
        c2.close()
        print("  TF-5 drift detect -> session re-teach -> gated commit "
              "clears (transformer host): PASS")
        print("SYSTEM FLOW ON TRANSFORMER HOST: PASS (5/5)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
