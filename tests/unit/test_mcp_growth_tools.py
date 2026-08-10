"""59C stage 3 (T2c): lib MCP growth-control tools — driven
through MCPServer.handle() with JSON-RPC dicts (the wire form).
M-5 is the PERMANENT terminology sweep (59B 3.6(e)): served
texts never attach "multi-scale" to deepen/insert_layer."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.facade import System                       # noqa: E402
from core.wiring import Config                       # noqa: E402
from mcp.mcp_server import (MCPServer, TOOLS,        # noqa: E402
                            SERVER_INSTRUCTIONS)

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]

NEW_TOOLS = ("deepen", "remove_block", "remove_grown",
             "propose", "plan_validate", "plan_run", "trial",
             "describe", "assess", "store")


def _srv(tmp_path):
    return MCPServer(System(Config.from_env(
        backend="mlp", models_root=tmp_path / "ws")))


def _call(srv, name, args, rid=1):
    resp = srv.handle({"jsonrpc": "2.0", "id": rid,
                       "method": "tools/call",
                       "params": {"name": name,
                                  "arguments": args}})
    assert resp["result"]["isError"] is False, resp
    return json.loads(resp["result"]["content"][0]["text"])


def _mk(srv, mid="m"):
    r = _call(srv, "create_model",
              {"model_id": mid,
               "policy": {"max_params_mult": 50}})
    assert "refusal" not in r, r
    r = _call(srv, "study", {"model_id": mid,
                             "examples": ROWS, "steps": 20})
    assert "refusal" not in r, r


def _hash(net):
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if str(k) in ("gain_ledger", "growth_events",
                              "_growth_policy"):
                    continue
                h.update(str(k).encode())
                walk(o[k])
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, np.ndarray):
            h.update(o.tobytes())
        elif hasattr(o, "__dict__"):
            walk(vars(o))
        else:
            h.update(repr(o).encode())
    walk(vars(net))
    return h.hexdigest()


# ---------------- M-1 tools/list count ----------------

def test_m1_tools_list_ten_additions(tmp_path):
    srv = _srv(tmp_path)
    resp = srv.handle({"jsonrpc": "2.0", "id": 1,
                       "method": "tools/list"})
    names = [t["name"] for t in resp["result"]["tools"]]
    for n in NEW_TOOLS:
        assert n in names, n
    assert len(names) == len(set(names))     # no duplicates
    assert len(names) == 42 + 10             # counted identity


# ---------------- M-2 deepen via the wire ----------------

def test_m2_deepen_delta_via_describe(tmp_path):
    srv = _srv(tmp_path)
    _mk(srv)
    d0 = _call(srv, "describe", {"model_id": "m"})
    H = d0["components"][0]["width"]
    r = _call(srv, "deepen", {"model_id": "m", "m": 4})
    assert "refusal" not in r, r
    d1 = _call(srv, "describe", {"model_id": "m"})
    assert d1["params_total"] == \
        d0["params_total"] + (H * 4 + 4 + H * 4)


# ---------------- M-3 scoped warning ----------------

def test_m3_scoped_deepen_carries_warning(tmp_path):
    srv = _srv(tmp_path)
    _mk(srv)
    with pytest.warns(UserWarning):
        r = _call(srv, "deepen", {"model_id": "m",
                                  "scope": [0, 1]})
    assert any("functionally CLOSED" in t
               for t in r.get("warning", [])), r


# ---------------- M-4 plan file + propose no-mutation ------

def test_m4_plan_file_and_propose_readonly(tmp_path):
    srv = _srv(tmp_path)
    _mk(srv)
    organ = srv.sys.lc._load_working("m")[0]
    pre = _hash(organ)
    rep = _call(srv, "propose", {"model_id": "m",
                                 "move": "deepen",
                                 "args": {"m": 4}})
    assert rep["would_refuse"] is False
    assert _hash(srv.sys.lc._load_working("m")[0]) == pre
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(
        {"steps": [{"move": "deepen", "args": {"m": 4}},
                   {"move": "deepen", "args": {"m": 2}}],
         "limits": {"max_events": 1}}))
    r = _call(srv, "plan_run", {"model_id": "m",
                                "plan": str(pf)})
    assert "refusal" not in r, r
    assert len(r["events"]) == 1             # hand count
    assert r["halted"] == "limit:max_events"
    # remaining new tools respond over the wire (store incl.)
    for name, args in (
            ("plan_validate", {"model_id": "m",
                               "plan": {"steps": []}}),
            ("trial", {"model_id": "m", "move": "deepen",
                       "args": {"m": 2}, "budget_steps": 2,
                       "examples": ROWS}),
            ("remove_block", {"model_id": "m", "k": 0}),
            ("remove_grown", {"model_id": "m", "key": "x"}),
            ("assess", {"model_id": "m"}),
            ("store", {"model_id": "m"})):
        out = _call(srv, name, args)
        assert isinstance(out, dict), (name, out)
    assert _call(srv, "store", {"model_id": "m"})["rows"] >= 0


# ---------------- M-5 terminology sweep (permanent) --------

def _sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))


def test_m5_terminology_sweep_served_texts():
    """59B 3.6: LLM-facing texts never conflate MULTI-SCALE
    (additive family) with DEEPEN (composition axis). Any
    sentence naming both must be a distinction statement."""
    served = [(t["name"], t["description"]) for t in TOOLS]
    served.append(("__instructions__", SERVER_INSTRUCTIONS))
    cli_src = (REPO / "cli" / "cli.py").read_text()
    served.append(("__cli__", cli_src))
    hits = 0
    for name, text in served:
        for s in _sentences(text):
            low = s.lower()
            if "multi-scale" in low and (
                    "deepen" in low or "insert_layer" in low):
                hits += 1
                assert ("separate" in low or "distinct" in low
                        or "adds a processing stage" in low), (
                    name, s)
    # the deepen tool carries the one-line distinction (3.6(c))
    deepen_desc = next(t["description"] for t in TOOLS
                       if t["name"] == "deepen").lower()
    assert "multi-scale growth adds capacity in place" \
        in deepen_desc
    assert "deepen adds a processing stage" in deepen_desc
    # create_model wording fixed (3.6(d))
    cm = next(t["description"] for t in TOOLS
              if t["name"] == "create_model")
    assert "deepening is a separate operation" in cm
    assert hits >= 2                # the sweep saw both texts
