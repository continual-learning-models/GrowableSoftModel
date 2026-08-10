"""S3 tests: MCP + CLI exposure of the standard family."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from mcp.mcp_server import MCPServer, TOOLS         # noqa: E402

ROOT = REPO / "trained_models" / "standard"
STD = ("standard_create", "standard_train", "standard_evaluate",
       "standard_infer", "standard_save", "standard_load",
       "standard_list")
# snapshot of the softmodel tool table BEFORE this feature —
# these names must remain, unchanged in count
SOFT_COUNT = 45  # 59C stage 3 adds the 9 growth-control tools + store (sanctioned; was 35)


@pytest.fixture(autouse=True)
def _clean():
    shutil.rmtree(ROOT / "x1", ignore_errors=True)
    yield
    shutil.rmtree(ROOT / "x1", ignore_errors=True)


def call(srv, name, args):
    r = srv.handle({"jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
    return json.loads(r["result"]["content"][0]["text"])


def rows(n=48):
    rng = np.random.default_rng(0)
    return [{"x": list(rng.uniform(-2, 2, 3)),
             "y": float(np.sin(rng.uniform(-2, 2)))}
            for _ in range(n)]


def test_tool_table_soft_unchanged_std_added():
    names = [t["name"] for t in TOOLS]
    assert [n for n in names if n.startswith("standard_")] \
        == list(STD)
    assert len([n for n in names
                if not n.startswith("standard_")]) == SOFT_COUNT


def test_all_seven_tools_end_to_end():
    srv = MCPServer()
    data = rows()
    assert call(srv, "standard_create",
                {"name": "x1", "arch": "mlp", "hidden": 8})["ok"]
    out = call(srv, "standard_train",
               {"name": "x1", "examples": data, "steps": 40})
    assert out["ok"] and "holdout_mse" in out and "hint" in out
    assert call(srv, "standard_infer",
                {"name": "x1", "x": data[0]["x"]})["ok"]
    assert call(srv, "standard_evaluate",
                {"name": "x1", "examples": data[:10]})["ok"]
    assert call(srv, "standard_save", {"name": "x1"})["ok"]
    assert call(srv, "standard_load", {"name": "x1"})["ok"]
    lst = call(srv, "standard_list", {})
    assert "x1" in [m["name"] for m in lst["models"]]


def test_refusal_wording_surfaces_through_mcp():
    srv = MCPServer()
    out = call(srv, "standard_create",
               {"name": "x1", "arch": "mlp", "warp": 9})
    assert "unknown standard parameter" in out["refusal"]


def test_cli_help_and_subcommand():
    env = {"PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, str(REPO / "cli/cli.py"),
                        "help"], capture_output=True, text=True)
    d = json.loads(r.stdout)
    assert "standard_verbs (optional industry mode)" in d
    r2 = subprocess.run([sys.executable, str(REPO / "cli/cli.py"),
                         "standard_list"], capture_output=True,
                        text=True)
    assert json.loads(r2.stdout)["ok"]
