"""IWP4 acceptance (UT4.1, 4.2, UT2.9-surface, read-only non-mutation):
one surface, full combined loop via MCP, read/write split."""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.wiring import SysFactory
from core.facade import System
from mcp.mcp_server import MCPServer, TOOLS

RNG = np.random.default_rng(9)
LAW = lambda x: round(float(x[0] + 2 * x[1] - x[2]), 6)


def rows(n, fn=LAW):
    X = RNG.uniform(0, 4, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
             "target": str(fn(x))} for x in X]


def _server(tmp):
    return MCPServer(System(Config.from_env(backend="mlp",
                                            models_root=Path(tmp))))


def _call(s, name, args):
    r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": name, "arguments": args}})
    res = r["result"]
    assert not res["isError"], res["content"][0]["text"]
    return json.loads(res["content"][0]["text"])


def test_ut4_1_protocol_surface():
    tmp = tempfile.mkdtemp()
    try:
        s = _server(tmp)
        r = s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                      "params": {}})
        assert r["result"]["serverInfo"]["name"] == "core"
        r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in r["result"]["tools"]}
        assert len(names) == len(TOOLS) == 52   # 45 softmodel (59C stage 3 adds the 9 growth-control tools + store, 59B sanctioned) + 7 standard_*
        assert {"study", "commit", "grow", "trajectory",
                "teach", "discoveries", "run_course"} <= names
    finally:
        shutil.rmtree(tmp)


def test_ut4_2_full_combined_loop_via_mcp():
    tmp = tempfile.mkdtemp()
    try:
        s = _server(tmp)
        _call(s, "create_model", {"model_id": "m", "holdout": rows(50)})
        for _ in range(4):
            _call(s, "study", {"model_id": "m", "examples": rows(200),
                               "steps": 300})
        # default infer = committed only (v0 -> untrained)
        out = _call(s, "infer", {"model_id": "m",
                                 "input": {"a": 1, "b": 3, "c": 0}})
        assert out["output"] is None
        r = _call(s, "commit", {"model_id": "m"})
        assert r["promoted"]
        out = _call(s, "infer", {"model_id": "m",
                                 "input": {"a": 1, "b": 3, "c": 0}})
        assert abs(out["output"] - 7.0) <= 0.5
        # suites + trajectory + growth machinery over MCP
        suites = [{"name": "s", "X": [[1, 3, 0], [2, 2, 2]],
                   "y": ["7", "4"]}]
        _call(s, "evaluate", {"model_id": "m", "suites": suites})
        _call(s, "evaluate", {"model_id": "m", "suites": suites})
        t = _call(s, "trajectory", {"model_id": "m"})
        assert t["verdict"] in ("REAL", "STUCK", "INSUFFICIENT")
        rep = _call(s, "growth_report", {"model_id": "m"})
        assert rep["candidates"]
        g = _call(s, "grow", {"model_id": "m"})
        assert g["grown"] and g["depth"] == 2
        # gated teach convenience + discoveries on a categorical model
        claw = lambda x: "HIGH" if x[0] * x[1] > 4 else "LOW"
        _call(s, "create_model", {"model_id": "c", "holdout": rows(40, claw)})
        r = _call(s, "teach", {"model_id": "c", "examples": rows(250, claw)})
        assert r["promoted"]
        d = _call(s, "discoveries", {"model_id": "c"})
        assert d["n_rules"] >= 1
        fleet = _call(s, "list_models", {})
        assert {m["model_id"] for m in fleet} == {"m", "c"}
    finally:
        shutil.rmtree(tmp)


def test_ut4_readonly_non_mutation():
    tmp = tempfile.mkdtemp()
    try:
        s = _server(tmp)
        _call(s, "create_model", {"model_id": "m", "holdout": rows(40)})
        _call(s, "study", {"model_id": "m", "examples": rows(150),
                           "steps": 200})
        _call(s, "commit", {"model_id": "m"})
        organ, _ = s.sys.lc._load_working("m")
        w_before = copy.deepcopy(organ.W1)
        versions_before = _call(s, "get_versions", {"model_id": "m"})
        for name, args in [
                ("infer", {"model_id": "m", "input": {"a": 1, "b": 1, "c": 1}}),
                ("trajectory", {"model_id": "m"}),
                ("growth_report", {"model_id": "m"}),
                ("check_drift", {"model_id": "m"}),
                ("card", {"model_id": "m"}),
                ("list_models", {}),
                ("get_versions", {"model_id": "m"})]:
            _call(s, name, args)
        organ2, _ = s.sys.lc._load_working("m")
        assert np.array_equal(organ2.W1, w_before)          # weights intact
        assert _call(s, "get_versions",
                     {"model_id": "m"}) == versions_before  # lineage intact
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_ut4_1_protocol_surface()
    test_ut4_2_full_combined_loop_via_mcp()
    test_ut4_readonly_non_mutation()
    print("iwp4 tests passed")
