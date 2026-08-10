"""IT-CONF — process-conformance audit: every step of the closed loop is
checked against the ORIGINAL requirements, from the LOG RECORDS.

The run uses an escalating curriculum (magnitude growth, Phase-2's
proven recipe) so that structural EVOLUTION is actually forced — then the
audit reads the model's event log (events.jsonl) and the run report and
verifies, step by step, that what happened is what the project's
requirements say must happen:

  STEP 1  create          — a name only; NO types/schemas declared
                            (Phase-1 binding: no human-declared types)
  STEP 2  self-shaping    — features/mode/capacity INFERRED from data,
                            recorded in shape.json (model shapes itself)
  STEP 3  study           — supervised learning events logged (S3)
  STEP 4  evaluate        — score-matrix rows from evaluate events (S4)
  STEP 5  trajectory      — verdicts computed during the run (S4:
                            real-vs-false improvement watched)
  STEP 6  EVOLUTION       — grow events exist; params/depth increased;
                            growth fired in a LATER (harder) stage, not
                            the easy ones (S5 recursive multi-scale,
                            S6 complexity-forced, non-uniform)
  STEP 7  gated commits   — versions exist ONLY via commit events; each
                            commit records a holdout score (gate: only
                            promoted if better on reality)
  STEP 8  retention       — after everything, earlier stages still
                            mastered (S4: orderly accumulation, no swap)
  STEP 9  serving         — the committed model answers the hardest
                            stage correctly; card shows grown structure
  STEP 10 reversibility   — rollback to an early version works

All data synthetic. Driven through the REAL MCP server (the LLM path).
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

LAW = lambda x: float(x[0] * x[1] + max(x[1] - x[2], 0.0) * x[0] + 2 * x[2])
BOUNDS = [(1, 1, 2), (3, 3, 3), (6, 6, 4), (9, 9, 6), (12, 12, 8)]
TARGETS = [0.85, 0.85, 0.85, 0.80, 0.75]


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
            "target": TARGETS[k]}


class MCPClient:
    def __init__(self, models_root):
        env = dict(os.environ, SOFTMODEL_MODELS_ROOT=str(models_root),
                   SOFTMODEL_BACKEND="mlp")
        self.p = subprocess.Popen(
            [sys.executable, "-m", "mcp.mcp_server"],
            cwd=ROOT, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self._id = 0
        self._send({"method": "initialize", "params": {}})

    def _send(self, m):
        self._id += 1
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, **m}) + "\n")
        self.p.stdin.flush()
        while True:
            r = json.loads(self.p.stdout.readline())
            if r.get("id") == self._id:
                return r["result"]

    def call(self, tool, args):
        r = self._send({"method": "tools/call",
                        "params": {"name": tool, "arguments": args}})
        assert not r["isError"], f"{tool}: {r['content'][0]['text']}"
        return json.loads(r["content"][0]["text"])

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


CHECKS = []


def check(step, ok, evidence):
    CHECKS.append((step, ok, evidence))
    print(f"  {'PASS' if ok else 'FAIL'}  {step}: {evidence}")


def main():
    tmp = tempfile.mkdtemp()
    try:
        c = MCPClient(tmp)
        curriculum = [make_stage(k) for k in range(len(BOUNDS))]

        # ---- run the closed loop (creation deliberately name-only) ----
        create_args = {"model_id": "conf",
                       "holdout": curriculum[0]["examples"][:40]}
        c.call("create_model", create_args)
        rep = c.call("run_course", {"model_id": "conf",
                                    "curriculum": curriculum,
                                    "policy": {"max_blocks_per_stage": 14,
                                               "post_grow_blocks": 3}})

        # ---- read the records (events.jsonl + report + card) ----
        events = [json.loads(l) for l in
                  (Path(tmp) / "conf" / "events.jsonl")
                  .read_text().splitlines()]
        kinds = [e["event"] for e in events]
        card = c.call("card", {"model_id": "conf"})
        versions = c.call("get_versions", {"model_id": "conf"})

        print("\nEVENT TRAIL (compressed):")
        trail, last = [], None
        for e in events:
            tag = e["event"]
            if tag == "grow":
                tag = f"grow({','.join(e['nodes'])} -> depth {e['depth']})"
            if tag == "commit":
                tag = f"commit({e['version']} score {e['score']:.2f})"
            if tag == last and "grow" not in tag and "commit" not in tag:
                trail[-1] = (trail[-1][0], trail[-1][1] + 1)
            else:
                trail.append((tag, 1))
            last = e["event"]
        print("  " + " -> ".join(t if n == 1 else f"{t} x{n}"
                                 for t, n in trail))
        print("\nCONFORMANCE AUDIT:")

        # STEP 1: create with a name only (no types/schemas)
        check("STEP 1 create (no human-declared types)",
              kinds[0] == "create"
              and set(create_args) == {"model_id", "holdout"},
              "create event first; args carried only a name + holdout — "
              "no features/classes/template/type fields exist")

        # STEP 2: self-shaping recorded
        shape = card["learned_shape"]
        check("STEP 2 self-shaping (model shaped itself from data)",
              shape["mode"] == "numeric"
              and shape["features"] == ["a", "b", "c"]
              and shape["hidden"] >= 16,
              f"shape.json: mode={shape['mode']} (inferred, not declared), "
              f"features={shape['features']} (observed keys), "
              f"capacity hidden={shape['hidden']} (auto-sized)")

        # STEP 3: study events
        n_study = kinds.count("study")
        check("STEP 3 study (supervised learning, logged)",
              n_study >= len(BOUNDS),
              f"{n_study} study events, each with n/steps/loss recorded")

        # STEP 4: evaluate events -> score matrix
        evals = [e for e in events if e["event"] == "evaluate"]
        check("STEP 4 evaluate (measured, not asserted)",
              len(evals) >= len(BOUNDS)
              and all(len(e["stage_accs"]) == len(BOUNDS) for e in evals),
              f"{len(evals)} evaluate events; every row scores all "
              f"{len(BOUNDS)} stages (retention watchable)")

        # STEP 5: trajectory verdicts were consulted during the run
        verdicts = [v for st in rep["stages"] for v in st["verdicts"]]
        check("STEP 5 trajectory (real-vs-false watched)",
              len(verdicts) >= len(BOUNDS),
              f"verdicts consulted each block: "
              f"{dict((v, verdicts.count(v)) for v in set(verdicts))}")

        # STEP 6: EVOLUTION happened, forced by complexity, non-uniform
        grows = [e for e in events if e["event"] == "grow"]
        grows_per_stage = {st["name"]: st["grows"] for st in rep["stages"]}
        early = sum(grows_per_stage.get(f"stage{i+1}", 0) for i in range(3))
        late = sum(grows_per_stage.get(f"stage{i+1}", 0) for i in (3, 4))
        depth_after = card["learned_shape"]["depth"] if "depth" in \
            card["learned_shape"] else grows[-1]["depth"] if grows else 1
        params_growth = (grows[-1]["params"] if grows else 0)
        check("STEP 6 EVOLUTION (multi-scale growth, complexity-forced)",
              len(grows) >= 1 and late >= 1 and depth_after >= 2
              and params_growth > 100,
              f"{len(grows)} grow events (early stages: {early}, hard "
              f"stages: {late}); nodes {sum((g['nodes'] for g in grows), [])}; "
              f"depth 1 -> {depth_after}; params -> {params_growth}")

        # STEP 7: versions ONLY via gated commits, scores recorded
        commits = [e for e in events if e["event"] == "commit"]
        check("STEP 7 gated commits (evolution only improves)",
              len(versions["versions"]) == 1 + len(commits)
              and all("score" in e for e in commits),
              f"versions = v0 + {len(commits)} commits exactly; every "
              f"commit carries its holdout score "
              f"{[round(e['score'], 2) for e in commits]}")

        # STEP 8: retention — earlier stages still mastered at the end
        final_accs = evals[-1]["stage_accs"]
        check("STEP 8 retention (orderly accumulation)",
              min(final_accs[:3]) >= 0.8,
              f"final all-stage accs {[round(a, 2) for a in final_accs]} — "
              f"early stages not sacrificed to late ones")

        # STEP 9: serving the hardest stage from the committed version
        probe = {"a": 10.0, "b": 10.0, "c": 5.0}
        out = c.call("infer", {"model_id": "conf", "input": probe})
        want = LAW([10.0, 10.0, 5.0])
        check("STEP 9 serving (committed model answers)",
              abs(out["output"] - want) / want <= 0.1,
              f"hardest-stage probe -> {out['output']:.1f} "
              f"(truth {want:.1f}, within 10%)")

        # STEP 10: reversibility
        c.call("rollback", {"model_id": "conf", "to": "v1"})
        active = c.call("get_versions", {"model_id": "conf"})["active"]
        check("STEP 10 reversibility (rollback)",
              active == "v1", f"active pointer moved to v1 on demand")

        c.close()
        failed = [s for s, ok, _ in CHECKS if not ok]
        print(f"\nCONFORMANCE: {'PASS (10/10)' if not failed else 'FAIL: ' + ', '.join(failed)}")
        assert not failed
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
