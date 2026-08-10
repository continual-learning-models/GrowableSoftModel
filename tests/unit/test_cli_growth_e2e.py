"""59C stage 2 (T2b): lib CLI E2E for the growth-control verbs
— REAL subprocess (python -m cli.cli ...), the served user's
actual hands. The verbs reach the CLI through the AUTOMATIC
getattr dispatch (cli.py); this stage's own code is help text +
the 59B 2.4 warning re-emission to stderr."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]


def _cli(tmp_path, verb, args=None):
    env = dict(os.environ,
               SOFTMODEL_MODELS_ROOT=str(tmp_path / "ws"))
    cmd = [sys.executable, "-m", "cli.cli", verb]
    if args is not None:
        cmd.append(json.dumps(args))
    p = subprocess.run(cmd, cwd=REPO, env=env,
                       capture_output=True, text=True,
                       timeout=180)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = None
    return p, out


def _mk(tmp_path, mid="m"):
    p, out = _cli(tmp_path, "create_model",
                  {"model_id": mid,
                   "policy": {"max_params_mult": 50}})
    assert p.returncode == 0 and "refusal" not in out, (
        p.stdout, p.stderr)
    p, out = _cli(tmp_path, "study",
                  {"model_id": mid, "examples": ROWS,
                   "steps": 20})
    assert p.returncode == 0 and "refusal" not in out, p.stdout


def test_c1_cli_deepen_delta_via_describe(tmp_path):
    _mk(tmp_path)
    p, d0 = _cli(tmp_path, "describe", {"model_id": "m"})
    assert p.returncode == 0, p.stderr
    p0 = d0["params_total"]
    H = d0["components"][0]["width"]        # trunk width
    p, r = _cli(tmp_path, "deepen", {"model_id": "m", "m": 4})
    assert p.returncode == 0 and "refusal" not in r, p.stdout
    p, d1 = _cli(tmp_path, "describe", {"model_id": "m"})
    assert d1["params_total"] == p0 + (H * 4 + 4 + H * 4)


def test_c2_cli_scoped_warning_json_and_stderr(tmp_path):
    _mk(tmp_path)
    p, r = _cli(tmp_path, "deepen",
                {"model_id": "m", "scope": [0, 1]})
    assert p.returncode == 0 and "refusal" not in r, p.stdout
    assert any("functionally CLOSED" in t
               for t in r.get("warning", [])), r
    assert "functionally CLOSED" in p.stderr   # channel re-emit


def test_c3_cli_propose_and_plan_file(tmp_path):
    _mk(tmp_path)
    rep = _cli(tmp_path, "propose",
               {"model_id": "m", "move": "deepen",
                "args": {"m": 4}})[1]
    assert rep["would_refuse"] is False
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(
        {"steps": [{"move": "deepen", "args": {"m": 4}},
                   {"move": "deepen", "args": {"m": 2}}],
         "limits": {"max_events": 1}}))
    p, r = _cli(tmp_path, "plan_run",
                {"model_id": "m", "plan": str(pf)})
    assert p.returncode == 0 and "refusal" not in r, p.stdout
    assert len(r["events"]) == 1               # hand count
    assert r["halted"] == "limit:max_events"
    d = _cli(tmp_path, "describe", {"model_id": "m"})[1]
    blocks = [c for c in d["components"]
              if c["kind"] not in ("trunk",)]
    assert len(blocks) == 1                    # one event ran


def test_c4_unknown_verb_refuses_with_usage(tmp_path):
    p, out = _cli(tmp_path, "teleport", {"model_id": "m"})
    assert p.returncode == 1
    assert "unknown verb" in out["error"]
    assert "hint" in out
    # help lists the growth-control verbs (2b: usage block)
    p, h = _cli(tmp_path, "help")
    listed = h["softmodel_verbs (the tool's method)"]
    for v in ("deepen", "remove_block", "remove_grown",
              "propose", "plan_validate", "plan_run", "trial",
              "describe", "assess"):
        assert v in listed, v
    assert any("deepen" in e for e in h["examples"])
    assert any("plan_run" in e for e in h["examples"])
