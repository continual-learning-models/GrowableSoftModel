"""S9.1 interface-supplement tests (docs/system/7 plan, T19 a-d):
D-G1 substrate_params birth pass-through + D-G3 tol pass-through.
Product path only (create_model/study/infer through the facade).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.facade import System                    # noqa: E402

ROWS = [{"input": {"x": float(i) / 24.0, "y": float(24 - i) / 24.0},
         "target": (float(i) / 24.0) * 0.7 + 0.1}
        for i in range(24)]


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def _mk(ws, mid, sp, steps=20):
    r = ws.create_model(mid, holdout=ROWS[:8],
                        substrate="growable_attention",
                        policy={"substrate_params": sp})
    assert "refusal" not in r, r
    ws.study(mid, ROWS[8:], steps=steps)
    return ws.lc._load_working(mid)[0]


def test_t19a_substrate_params_reach_the_organ(ws):
    """T19a: the born organ carries EXACTLY the requested values."""
    organ = _mk(ws, "t19a", {"d_model": 16, "n_layers": 1,
                             "heads_spec": [[1]], "lr": 5e-3,
                             "seed": 11})
    assert organ.d == 16
    assert organ.L == 1
    assert [len(hs) for hs in organ.heads] == [1]
    assert organ.lr == 5e-3
    assert organ.seed == 11


def test_t19b_per_model_seed_reproducibility(ws):
    """T19b: same substrate_params seed -> bit-identical predictions
    after identical study; different seed -> different."""
    x = {"x": 0.3, "y": 0.4}
    outs = {}
    for mid, seed in (("s7a", 7), ("s7b", 7), ("s9", 9)):
        _mk(ws, mid, {"seed": seed}, steps=10)
        outs[mid] = ws.infer(mid, x, working=True)["output"]
    assert outs["s7a"] == outs["s7b"]          # bitwise (same floats)
    assert outs["s7a"] != outs["s9"]


def test_t19c_unknown_key_refused_loudly(ws):
    """T19c: unknown or birth-derived keys refuse, NAMING the key."""
    r = ws.create_model("t19c", holdout=ROWS[:8],
                        substrate="growable_attention",
                        policy={"substrate_params": {"d_modl": 16}})
    assert "refusal" in r and "d_modl" in r["refusal"]
    r2 = ws.create_model("t19c2", holdout=ROWS[:8],
                         substrate="growable_attention",
                         policy={"substrate_params": {"mode":
                                                      "categorical"}})
    assert "refusal" in r2 and "mode" in r2["refusal"]
    assert "birth-derived" in r2["refusal"]


def test_t19d_tol_reaches_the_gate(ws):
    """T19d: tol threads facade -> lifecycle -> driver. tol=0.0
    refuses growth on a fixture tol=1e9 accepts (the contrast
    proves the parameter reaches the gate in both directions
    without depending on marginal default-tol behavior; the
    default path itself stays pinned by T16)."""
    _mk(ws, "t19d", {"d_model": 8, "n_layers": 1,
                     "heads_spec": [[1]], "seed": 3}, steps=30)
    row_no = ws.grow_attention("t19d", layer=0, tol=0.0)
    assert row_no.get("verdict") in ("refused", "refuse"), row_no
    row_yes = ws.grow_attention("t19d", layer=0, tol=1e9)
    assert row_yes.get("verdict") in ("accepted", "accept"), row_yes


def test_t19e_mcp_grow_attention_dispatch(ws, monkeypatch):
    """S9.1 defect pin: the grow_attention MCP tool was defined
    (census-counted) but had NO dispatch entry since S8 — calling
    it raised 'unknown tool'. Pin: dispatch resolves and returns
    the facade's refusal dict for an unknown model (not
    ValueError), and the tool schema carries tol."""
    import mcp.mcp_server as mserv
    server = mserv.MCPServer()
    r = server.sys.create_model("t19e-mlp", holdout=ROWS[:8],
                                substrate="mlp")
    assert "refusal" not in r, r
    server.sys.study("t19e-mlp", ROWS[8:], steps=5)
    out = server._dispatch("grow_attention",
                           {"model_id": "t19e-mlp"})
    assert isinstance(out, dict) and "refusal" in out
    assert "growable_attention" in out["refusal"]
    tool = next(t for t in mserv.TOOLS
                if t["name"] == "grow_attention")
    assert "tol" in tool["inputSchema"]["properties"]
