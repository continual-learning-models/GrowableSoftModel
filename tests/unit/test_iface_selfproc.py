"""S9.3 self-processing switch (docs/system/6 D-G2; plan doc 7
T21 a-d). The discipline's math is untouched — this is the
product path to the existing _selfproc_on switch plus a
per-model head allow-set.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

import pytest                                          # noqa: E402

from core.facade import System                         # noqa: E402
from core.substrates.growable_attention import (       # noqa: E402
    GrowableAttentionSubstrate)

ROWS = [{"input": {"x": float(i) / 24.0, "y": float(24 - i) / 24.0},
         "target": (float(i) / 24.0) * 0.7 + 0.1}
        for i in range(24)]


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_t21a_constructor_param_sets_switch():
    """T21a: selfproc=True at construction sets the switch (and is
    reachable through substrate_params by D-G1's signature
    filter); default stays False."""
    m = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                   heads_spec=[[1]], selfproc=True)
    assert m._selfproc_on is True
    m2 = GrowableAttentionSubstrate(2, 4, d_model=6, n_layers=1,
                                    heads_spec=[[1]])
    assert m2._selfproc_on is False


def test_t21b_verb_toggles_and_refuses(ws):
    """T21b: the facade verb toggles a working attention model
    (persisted across reloads) and REFUSES on mlp naming the
    boundary."""
    ws.create_model("t21b", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": {
                        "d_model": 8, "n_layers": 1,
                        "heads_spec": [[1]], "seed": 3}})
    ws.study("t21b", ROWS[8:], steps=10)
    r = ws.set_attention_selfproc("t21b", True)
    assert r == {"model": "t21b", "selfproc": True, "heads": "all"}
    organ, _ = ws.lc._load_working("t21b")     # fresh load: persisted
    assert organ._selfproc_on is True
    r2 = ws.set_attention_selfproc("t21b", False)
    assert r2["selfproc"] is False
    assert ws.lc._load_working("t21b")[0]._selfproc_on is False
    ws.create_model("t21b-mlp", holdout=ROWS[:8], substrate="mlp")
    ws.study("t21b-mlp", ROWS[8:], steps=10)
    r3 = ws.set_attention_selfproc("t21b-mlp", True)
    assert "refusal" in r3 and "growable_attention" in r3["refusal"]
    assert "mlp" in r3["refusal"]


def test_t21c_instance_allowset_precedence(ws):
    """T21c: the per-model allow-set gates selfproc_active ahead
    of the module POLICY default (which allows all)."""
    ws.create_model("t21c", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": {
                        "d_model": 8, "n_layers": 1,
                        "heads_spec": [[1, 1]], "seed": 3}})
    ws.study("t21c", ROWS[8:], steps=10)
    ws.set_attention_selfproc("t21c", True, heads=[[0, 1]])
    organ, _ = ws.lc._load_working("t21c")
    organ._t_att = 10_000                     # clear warmup/age gates
    assert organ.selfproc_active(0, 1) is True
    assert organ.selfproc_active(0, 0) is False   # not in allow-set
    ws.set_attention_selfproc("t21c", True)       # heads=None -> all
    organ, _ = ws.lc._load_working("t21c")
    organ._t_att = 10_000
    assert organ.selfproc_active(0, 0) is True


def test_t21d_default_off_and_gates_still_apply(ws):
    """T21d: a fresh model never self-processes unasked; and the
    switch ENABLES, never bypasses — warmup still gates a
    just-switched-on model."""
    ws.create_model("t21d", holdout=ROWS[:8],
                    substrate="growable_attention",
                    policy={"substrate_params": {
                        "d_model": 8, "n_layers": 1,
                        "heads_spec": [[1]], "seed": 3}})
    ws.study("t21d", ROWS[8:], steps=10)
    organ, _ = ws.lc._load_working("t21d")
    organ._t_att = 10_000
    assert organ.selfproc_active(0, 0) is False   # default off
    ws.set_attention_selfproc("t21d", True)
    organ, _ = ws.lc._load_working("t21d")        # young organ:
    assert organ.selfproc_active(0, 0) is False   # warmup gate holds


def test_t21e_mcp_tool_and_dispatch():
    """T21e: tool defined AND dispatch wired in the same step (the
    S8 grow_attention lesson, pinned for this verb from birth)."""
    import mcp.mcp_server as mserv
    tool = next(t for t in mserv.TOOLS
                if t["name"] == "set_attention_selfproc")
    assert tool["inputSchema"]["required"] == ["model_id", "on"]
    import inspect
    src = inspect.getsource(mserv.MCPServer._dispatch)
    assert '"set_attention_selfproc"' in src
