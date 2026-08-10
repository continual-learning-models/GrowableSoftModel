"""GSM-I1 tests: the predict_dist facade verb."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))


@pytest.fixture
def ws(monkeypatch):
    with tempfile.TemporaryDirectory() as t:
        monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", t)
        from core.facade import System
        yield System()


def _teach_numeric(sy, mid="pd1"):
    rng = np.random.default_rng(0)
    rows = [{"input": {"a": float(a), "b": float(b)},
             "target": float(2 * a + b)}
            for a, b in rng.uniform(-2, 2, (40, 2))]
    sy.create_model(mid)
    sy.add_holdout(mid, rows[:8])
    sy.teach(mid, rows[8:])
    return rows


def test_1_untrained_kind_none(ws):
    ws.create_model("pd0")
    d = ws.predict_dist("pd0", {"a": 1.0})
    assert d["kind"] == "none" and d["note"] == "untrained"


def test_2_numeric_committed(ws):
    rows = _teach_numeric(ws)
    d = ws.predict_dist("pd1", rows[0]["input"])
    assert d["kind"] == "numeric"
    assert abs(d["value"] - rows[0]["target"]) < 2.0
    # consistency with infer (same serving path)
    out = ws.infer("pd1", rows[0]["input"])
    assert abs(d["value"] - out["output"]) < 1e-9


def test_3_categorical_probs_sum_one(ws):
    rng = np.random.default_rng(1)
    rows = [{"input": {"x": float(x)},
             "target": ("hi" if x > 0 else "lo")}
            for x in rng.uniform(-2, 2, 50)]
    ws.create_model("pdc")
    ws.add_holdout("pdc", rows[:10])
    ws.teach("pdc", rows[10:])
    d = ws.predict_dist("pdc", {"x": 1.5})
    assert d["kind"] == "categorical"
    assert set(d["labels"]) == {"hi", "lo"}
    assert abs(sum(d["probs"]) - 1.0) < 1e-6
    assert d["probs"][d["labels"].index("hi")] > 0.5


def test_4_working_state_dist(ws):
    ws.create_model("pdw")
    rng = np.random.default_rng(2)
    rows = [{"input": {"a": float(a)}, "target": float(3 * a)}
            for a in rng.uniform(-2, 2, 40)]
    ws.study("pdw", rows, steps=200)
    d = ws.predict_dist("pdw", {"a": 1.0}, working=True)
    assert d["kind"] == "numeric" and d["state"] == "working"
    assert abs(d["value"] - 3.0) < 1.5


def test_5_mcp_tool_round_trip(ws, monkeypatch):
    import json
    from mcp.mcp_server import MCPServer
    rows = _teach_numeric(ws, "pdm")
    srv = MCPServer(ws)
    r = srv.handle({"jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": "predict_dist",
                               "arguments": {
                                   "model_id": "pdm",
                                   "input_": rows[0]["input"]}}})
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["kind"] == "numeric"
