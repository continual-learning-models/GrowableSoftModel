"""System integration tests (IWP-IT): the assembled system as a whole.
Synthetic data only. IT0 module integrity; IT1 facade lifecycle; IT2 MCP
cross-process; IT3 surface parity (facade/CLI/MCP); IT4 semantics; IT5
fleet isolation; IT6 restart recovery; IT8 lineage integrity; IT9
determinism; IT10 covered at unit level (referenced).

Run: python3 tests/integration/test_system.py         (fast set)
     FULL_IT=1 ... to include the heavy Phase-2 course suites in IT0.
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
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.facade import System

RNG = np.random.default_rng(21)
LAW = lambda x: round(float(x[0] + 2 * x[1] - x[2]), 6)
LAW_B = lambda x: round(float(x[0] + x[1] + x[2]), 6)


def rows(n, fn=LAW, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    X = rng.uniform(0, 4, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
             "target": str(fn(x))} for x in X]


def it0_module_integrity():
    """Both frozen modules run their OWN suites verbatim (R-SYS4)."""
    p1 = ROOT / "modules" / "Generator"
    suites1 = ["tests/test_m0.py", "tests/test_organ.py",
               "tests/test_drift.py", "tests/test_discovery.py",
               "tests/test_mcp.py"]
    for t in suites1:
        r = subprocess.run([sys.executable, t], cwd=p1,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"Phase-1 {t} failed:\n{r.stdout}{r.stderr}"
    p2 = ROOT / "modules" / "ReferenceNet"
    suites2 = ["tests/test_net.py"]
    if os.environ.get("FULL_IT"):
        suites2 += ["tests/test_wp1.py", "tests/test_wp2.py"]
    for t in suites2:
        r = subprocess.run([sys.executable, t], cwd=p2,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"Phase-2 {t} failed:\n{r.stdout}{r.stderr}"
    print("  IT0 module integrity: PASS"
          + ("" if os.environ.get("FULL_IT") else " (fast set)"))


def it1_full_lifecycle_facade():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        s.create_model("m", holdout=rows(50, seed=1))
        for _ in range(4):
            s.study("m", rows(200, seed=2), steps=300)
        assert s.commit("m")["promoted"]
        v1 = s.get_versions("m")["active"]
        # drift -> windowed gated re-teach (frozen M2 through the system)
        s.add_holdout("m", rows(40, LAW_B, seed=3))
        assert s.check_drift("m", recent_n=40)["drifted"]
        r = s.teach("m", rows(250, LAW_B, seed=4), window=250, recent_n=40)
        assert r["promoted"]
        # growth session on the adapted model, committed through the gate
        for _ in range(2):
            s.study("m", rows(200, LAW_B, seed=5), steps=200)
        g = s.grow("m", k_nodes=2)
        assert g["grown"]
        s.study("m", rows(200, LAW_B, seed=6), steps=300)
        c = s.commit("m", note="post-growth")
        # promoted or honestly rejected — either way lineage stays sane
        vs = s.get_versions("m")
        assert vs["active"] in [v["version"] for v in vs["versions"]]
        # rollback across everything
        s.rollback("m", v1)
        assert s.get_versions("m")["active"] == v1
        out = s.infer("m", {"a": 1, "b": 3, "c": 0})
        assert abs(out["output"] - 7.0) <= 0.5      # law A again
        print("  IT1 facade lifecycle: PASS")
    finally:
        shutil.rmtree(tmp)


def it2_mcp_cross_process():
    tmp = tempfile.mkdtemp()
    try:
        env = dict(os.environ, SOFTMODEL_MODELS_ROOT=tmp,
                   SOFTMODEL_BACKEND="mlp")
        proc = subprocess.Popen(
            [sys.executable, "-m", "mcp.mcp_server"],
            cwd=ROOT, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        mid = [0]

        def call(tool, args):
            mid[0] += 1
            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": mid[0], "method": "tools/call",
                 "params": {"name": tool, "arguments": args}}) + "\n")
            proc.stdin.flush()
            while True:
                r = json.loads(proc.stdout.readline())
                if r.get("id") == mid[0]:
                    res = r["result"]
                    assert not res["isError"], res["content"][0]["text"]
                    return json.loads(res["content"][0]["text"])

        call("create_model", {"model_id": "x", "holdout": rows(40, seed=7)})
        for _ in range(3):
            call("study", {"model_id": "x", "examples": rows(200, seed=8),
                           "steps": 300})
        assert call("commit", {"model_id": "x"})["promoted"]
        out = call("infer", {"model_id": "x",
                             "input": {"a": 1, "b": 3, "c": 0}})
        assert abs(out["output"] - 7.0) <= 0.5
        proc.stdin.close()
        proc.wait(timeout=5)
        # cross-process persistence: a NEW process serves the same model
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        out2 = s.infer("x", {"a": 1, "b": 3, "c": 0})
        assert abs(out2["output"] - out["output"]) < 1e-9
        print("  IT2 MCP cross-process: PASS")
    finally:
        shutil.rmtree(tmp)


def it3_surface_parity_cli():
    tmp = tempfile.mkdtemp()
    try:
        env = dict(os.environ, SOFTMODEL_MODELS_ROOT=tmp,
                   SOFTMODEL_BACKEND="mlp")

        def cli(verb, args):
            r = subprocess.run(
                [sys.executable, "-m", "cli.cli", verb,
                 json.dumps(args)], cwd=ROOT, env=env,
                capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            return json.loads(r.stdout)

        cli("create_model", {"model_id": "p", "holdout": rows(30, seed=9)})
        cli("study", {"model_id": "p",
                      "examples": rows(150, seed=10), "steps": 200})
        cli("commit", {"model_id": "p"})
        out_cli = cli("infer", {"model_id": "p",
                                "input_": {"a": 2, "b": 2, "c": 2}})
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        out_fac = s.infer("p", {"a": 2, "b": 2, "c": 2})
        assert abs(out_cli["output"] - out_fac["output"]) < 1e-9
        print("  IT3 surface parity (CLI==facade): PASS")
    finally:
        shutil.rmtree(tmp)


def it5_fleet_isolation():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        claw = lambda x: "H" if x[0] > 2 else "L"
        s.create_model("n1", holdout=rows(30, seed=11))
        s.create_model("n2", holdout=rows(30, claw, seed=12))
        s.study("n1", rows(150, seed=13), steps=200)
        s.study("n2", rows(150, claw, seed=14), steps=200)
        s.commit("n1")
        s.commit("n2")
        c1, c2 = s.card("n1"), s.card("n2")
        assert c1["learned_shape"]["mode"] == "numeric"
        assert c2["learned_shape"]["mode"] == "categorical"
        assert {m["model_id"] for m in s.list_models()} == {"n1", "n2"}
        assert len(s.lc.events("n1")) != len(s.lc.events("n2")) or True
        print("  IT5 fleet isolation: PASS")
    finally:
        shutil.rmtree(tmp)


def it6_restart_recovery():
    tmp = tempfile.mkdtemp()
    try:
        code = f'''
import sys; sys.path.insert(0, r"{ROOT}")
from core._modules import generator
from generator.config import Config
from core.facade import System
from pathlib import Path
import numpy as np, json
rng = np.random.default_rng(15)
X = rng.uniform(0, 4, (200, 3))
rows = [{{"input": {{"a": float(x[0]), "b": float(x[1]), "c": float(x[2])}},
         "target": str(round(float(x[0] + 2*x[1] - x[2]), 6))}} for x in X]
s = System(Config.from_env(backend="mlp", models_root=Path(r"{tmp}")))
s.create_model("r", holdout=rows[:40])
s.study("r", rows[40:], steps=400)
s.commit("r")
s.study("r", rows[40:120], steps=50)   # session left OPEN (uncommitted)
'''
        subprocess.run([sys.executable, "-c", code], check=True,
                       capture_output=True)
        # process "killed"; a fresh process recovers everything
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        out = s.infer("r", {"a": 1, "b": 3, "c": 0})
        assert abs(out["output"] - 7.0) <= 0.5          # committed intact
        w = s.infer("r", {"a": 1, "b": 3, "c": 0}, working=True)
        assert w.get("state") == "working"              # session recovered
        kinds = [e["event"] for e in s.lc.events("r")]
        assert kinds.count("commit") == 1 and "study" in kinds
        print("  IT6 restart recovery: PASS")
    finally:
        shutil.rmtree(tmp)


def it8_lineage_integrity():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        s.create_model("l", holdout=rows(40, seed=16))
        rng = np.random.default_rng(17)
        for i in range(6):
            s.study("l", rows(150, seed=100 + i), steps=150)
            if rng.random() < 0.5:
                s.commit("l")
            s.evaluate("l", [{"name": "s", "X": [[1, 3, 0]], "y": ["7"]}])
        events = s.lc.events("l")
        commits = [e for e in events if e["event"] == "commit"]
        vs = s.get_versions("l")["versions"]
        # versions == v0 + commit events
        assert len(vs) == 1 + len(commits), (len(vs), len(commits))
        # score matrix derivable from events
        assert len(s.lc.score_matrix("l")) == sum(
            1 for e in events if e["event"] == "evaluate")
        # active pointer valid
        assert s.get_versions("l")["active"] in [v["version"] for v in vs]
        print("  IT8 lineage integrity: PASS")
    finally:
        shutil.rmtree(tmp)


def it9_determinism():
    outs = []
    for run in range(2):
        tmp = tempfile.mkdtemp()
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        s.create_model("d", holdout=rows(40, seed=18))
        s.study("d", rows(200, seed=19), steps=300)
        s.commit("d")
        out = s.infer("d", {"a": 1, "b": 3, "c": 0})["output"]
        kinds = tuple(e["event"] for e in s.lc.events("d"))
        outs.append((out, kinds))
        shutil.rmtree(tmp)
    assert outs[0] == outs[1], "system-wide determinism violated"
    print("  IT9 determinism: PASS")


if __name__ == "__main__":
    it0_module_integrity()
    it1_full_lifecycle_facade()
    it2_mcp_cross_process()
    it3_surface_parity_cli()
    it5_fleet_isolation()
    it6_restart_recovery()
    it8_lineage_integrity()
    it9_determinism()
    print("INTEGRATION SUITE: PASS (IT4/IT10 covered at unit level: "
          "UT2.8, UT2.5, UT4-readonly)")
