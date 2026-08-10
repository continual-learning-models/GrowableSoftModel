"""GSM-I3 S3: the door and the serve path
(EXEC_PLAN_GSM_I3 S3 tests 1-5, design boxes S3.1-S3.5)."""
import json
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


def _rows(n=60, eps=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return [{"input": {"a": float(a), "b": float(b)},
             "target": float(2 * a + b + rng.normal(0, eps))}
            for a, b in rng.uniform(-2, 2, (n, 2))]


def _teach_dist(sy, mid="nd1", rows=None):
    rows = rows or _rows()
    out = sy.create_model(mid, policy={"numeric_head": "dist"})
    assert "refusal" not in out
    sy.add_holdout(mid, rows[:10])
    sy.study(mid, rows[10:], steps=300)
    return rows


def test_e2e_facade_and_mcp_value_and_std(ws, monkeypatch):
    """S3.1: create(policy numeric_head=dist) -> study -> the
    distribution carries value AND std, working and committed,
    facade and MCP tool alike."""
    rows = _teach_dist(ws)
    d = ws.predict_dist("nd1", rows[0]["input"], working=True)
    assert d["kind"] == "numeric_dist"
    assert np.isfinite(d["value"]) and d["std"] > 0.0
    out = ws.commit("nd1")
    assert out.get("promoted"), out
    d2 = ws.predict_dist("nd1", rows[0]["input"])
    assert d2["kind"] == "numeric_dist" and d2["std"] > 0.0
    # MCP tool round trip (the GSM-I1 test_5 pattern)
    from mcp.mcp_server import MCPServer
    srv = MCPServer(ws)
    r = srv.handle({"jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": "predict_dist",
                               "arguments": {
                                   "model_id": "nd1",
                                   "input_": rows[0]["input"]}}})
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["kind"] == "numeric_dist" and "std" in body


def test_door_predict_infer_return_point(ws):
    """S3.2: plain infer on a numeric_dist model returns the point
    (design 3.6: existing consumers see a numeric answer)."""
    rows = _teach_dist(ws, "nd2")
    ws.commit("nd2")
    out = ws.infer("nd2", rows[1]["input"])
    val = out["output"] if isinstance(out, dict) else out
    assert np.isfinite(float(val))
    d = ws.predict_dist("nd2", rows[1]["input"])
    assert abs(float(val) - d["value"]) < 1e-6


def test_existing_numeric_response_shape_unchanged(ws):
    """S3.3: a plain numeric model (no policy) answers EXACTLY the
    pre-I3 dict — same kind, same key set, no std."""
    rows = _rows()
    ws.create_model("plain")
    ws.add_holdout("plain", rows[:10])
    ws.study("plain", rows[10:], steps=200)
    d = ws.predict_dist("plain", rows[0]["input"], working=True)
    assert d["kind"] == "numeric"
    assert set(d.keys()) == {"kind", "value", "state"}
    ws.commit("plain")
    d2 = ws.predict_dist("plain", rows[0]["input"])
    assert d2["kind"] == "numeric"
    assert set(d2.keys()) == {"kind", "value", "version"}


def test_misuse_refusals(ws):
    """S3.4: invalid numeric_head refuses listing valid values;
    numeric_head=dist on sequence data refuses naming the boundary;
    predict_proba on a numeric_dist organ refuses."""
    out = ws.create_model("bad", policy={"numeric_head": "gaussian"})
    assert "refusal" in out and "point" in out["refusal"] \
        and "dist" in out["refusal"]
    ws.create_model("bad2")
    out = ws.set_policy("bad2", numeric_head="gaussian")
    assert "refusal" in out          # the OTHER door (review finding A)
    ws.create_model("seqd", policy={"numeric_head": "dist"})
    seq_rows = [{"input": [[0.1, 0.2]] * 4, "target": 0.5}
                for _ in range(12)]
    out = ws.study("seqd", seq_rows)
    assert "refusal" in out and "sequence" in out["refusal"]
    rows = _teach_dist(ws, "nd3")
    with pytest.raises(AssertionError):
        ws.lc._load_working("nd3")[0].predict_proba(
            np.zeros((1, 2)))


def test_tool_table_pin_unchanged():
    """S3.5: the fence made testable — no new tool, count still 39,
    predict_dist present, nothing numeric_dist-named."""
    from mcp.mcp_server import TOOLS
    names = [t["name"] for t in TOOLS]
    assert len(TOOLS) == 52  # 59C stage 3 adds the 9 growth-control tools + store (59B approved surface change; was 42)
    assert "predict_dist" in names
    assert not any("numeric_dist" in n for n in names)
