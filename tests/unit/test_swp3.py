"""SWP3 unit tests (SUT3.1-3.4): form detection, AI selection interface,
transparent auto-default, advisory recommendations."""
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
from core.substrates.forms import detect_form, interaction_probe
from core.facade import System
from mcp.mcp_server import MCPServer

RNG = np.random.default_rng(0)


def _rows(n, fn):
    X = RNG.uniform(0, 2, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]),
                       "c": float(x[2])}, "target": str(fn(x))} for x in X]


def test_sut3_1_detect_form():
    assert detect_form(_rows(8, lambda x: x[0])) == "vector"
    seq = [{"input": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], "target": "1"}]
    assert detect_form(seq) == "sequence"
    grid = [{"input": np.zeros((12, 12)).tolist(), "target": "0"}]
    assert detect_form(grid) == "grid"
    graph = [{"input": {"nodes": [1, 2], "edges": [[0, 1]]}, "target": "x"}]
    assert detect_form(graph) == "graph"
    assert detect_form([{"input": "text", "target": "1"}]) is None
    mixed = _rows(2, lambda x: x[0]) + seq
    assert detect_form(mixed) is None


def test_sut3_2_create_substrate_choice_and_auto_default():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        # explicit choice recorded
        out = s.create_model("e", holdout=_rows(20, lambda x: x[0]),
                             substrate="mlp")
        assert out["substrate"] == "mlp" and "auto_selection" not in out
        assert s.lc.policy("e")["substrate"] == "mlp"
        # unknown substrate -> refusal
        out = s.create_model("u", substrate="quantum")
        assert "refusal" in out
        # auto-default states its reason
        out = s.create_model("a", holdout=_rows(20, lambda x: x[0]))
        assert out["substrate"] == "mlp" and "auto_selection" in out
        # grid form refused while cnn unregistered
        grid_rows = [{"input": np.zeros((12, 12)).tolist(), "target": "0"}]
        out = s.create_model("g", holdout=grid_rows)
        assert "refusal" in out
    finally:
        shutil.rmtree(tmp)


def test_sut3_3_recommend_substrate_advisory():
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        weak = _rows(150, lambda x: 2 * x[0] + x[1])          # linear
        strong = _rows(150, lambda x: x[0] * x[1] * 3)        # interaction
        r1 = s.recommend_substrate(weak)
        assert r1["recommendations"][0]["substrate"] == "mlp"
        r2 = s.recommend_substrate(strong)
        assert r2["recommendations"][0]["substrate"] == "transformer"
        assert r2["interaction_gain"] > r1["interaction_gain"]
        assert all("reason" in r for r in r2["recommendations"])
        # advisory: no models created, no state
        assert s.list_models() == []
    finally:
        shutil.rmtree(tmp)


def test_sut3_4_mcp_exposure():
    tmp = tempfile.mkdtemp()
    try:
        srv = MCPServer(System(Config.from_env(backend="mlp",
                                               models_root=Path(tmp))))
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in r["result"]["tools"]}
        assert {"list_substrates", "recommend_substrate"} <= names
        assert len(names) == 52   # 45 softmodel (59C stage 3 adds the 9 growth-control tools + store, sanctioned) + 7 standard_*

        def call(tool, args):
            resp = srv.handle({"jsonrpc": "2.0", "id": 2,
                               "method": "tools/call",
                               "params": {"name": tool, "arguments": args}})
            res = resp["result"]
            assert not res["isError"], res["content"][0]["text"]
            return json.loads(res["content"][0]["text"])

        g = call("list_substrates", {})
        assert "mlp" in g["substrates"]
        out = call("create_model", {"model_id": "m",
                                    "holdout": _rows(20, lambda x: x[0]),
                                    "substrate": "mlp"})
        assert out["substrate"] == "mlp"
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_sut3_1_detect_form()
    test_sut3_2_create_substrate_choice_and_auto_default()
    test_sut3_3_recommend_substrate_advisory()
    test_sut3_4_mcp_exposure()
    print("swp3 tests passed")
