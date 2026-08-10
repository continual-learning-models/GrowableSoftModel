"""MCP tool adapter (M5): expose the SoftModel factory to a general LLM.

A minimal Model Context Protocol server over stdio (newline-delimited
JSON-RPC 2.0) — no SDK dependency. The brain (an MCP client such as Claude
Code / Claude Desktop) discovers the model fleet and drives the full loop as
tool calls: infer (use), teach (evolve, gated), add_holdout, check_drift,
discoveries, versions, rollback.

Run:      python3 -m generator.mcp_server
Register: claude mcp add generator -- python3 -m generator.mcp_server

stdout is the transport — logs go to stderr only.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Optional

from . import __version__
from .config import Config
from .factory import SoftModelFactory
from .spec import ModelSpec

PROTOCOL_VERSION = "2024-11-05"

# Injected into the client model's context at initialize (MCP
# `instructions`). This is the operating manual for the AI: the
# required call order, the contracts, and how to recover from the
# common failure modes.
SERVER_INSTRUCTIONS = """\
Every controllable parameter (model config keys, substrate_params,
growth_params) is cataloged in docs/PARAMETER_REFERENCE.md.

You operate a fleet of SoftModel models: tiny specialized neural
models that you both USE (infer) and TEACH (teach). Each model
carries one domain's learned judgment in its own weights; you supply
all language understanding and feature extraction.

REQUIRED ORDER — the gate needs held-out reality before teaching:
1. create_model (a name is enough; you may pass holdout here).
2. add_holdout with labeled examples KEPT SEPARATE from training.
   Without holdout, teach can never promote (0.0 vs 0.0 tie) and
   infer stays "untrained".
3. teach with training examples -> response says promoted true/false.
4. infer to use the model.

CONTRACTS:
- Feature extraction is YOUR job: read the model's learned_shape
  (from list_models or create_model) and extract exactly those
  feature keys from the user's problem into a flat JSON object of
  numbers/booleans. Same keys at teach and infer time.
- target may be a NUMBER (model self-shapes a regression head) or a
  LABEL string (self-shapes a categorical head). Be consistent per
  model.
- Teaching is safe by construction: a candidate is promoted only if
  it beats the live version on the held-out slice. promoted:false
  means the live model stays — it is an outcome, not an error.
- infer returns confidence, the serving version, and (when a mined
  rule agrees) a citation; returns note:"untrained" if the model has
  no promoted version yet — go back to steps 2-3.

MAINTENANCE LOOP (long-lived models):
- As new ground truth arrives, add_holdout it (the gate judges on
  the most recent slice — reality as it is NOW).
- Periodically check_drift; if needs_reteach is true, collect fresh
  examples and teach with window=N (train on the last N stored
  examples, shedding outdated labels) and recent_n=N (judge the
  gate on the last N held-out examples).
- discoveries returns readable IF/THEN regularities the model mined
  from its own data — domain knowledge you cannot have from
  pretraining; incorporate it into your reasoning and cite it.
- get_versions / rollback: full lineage; reverting is instant.

RECOVERY:
- "unknown model" -> list_models to see the fleet, or create_model.
- promoted:false repeatedly -> holdout missing, too small, or stale;
  add_holdout fresh reality, or pass recent_n after drift.
- Numeric answers are integer-rounded when the taught data was
  integral; exactness beyond the data's resolution is not promised.
"""

_EXAMPLES_SCHEMA = {
    "type": "array",
    "description": "Labeled examples. `input` is a flat feature object whose "
                   "keys are the model's input_features (extract the values "
                   "from your context); `target` is the answer — a NUMBER "
                   "(numeric domains: the model self-shapes a regression "
                   "head) or a LABEL string (categorical domains).",
    "items": {
        "type": "object",
        "properties": {"input": {"type": "object"},
                       "target": {"type": ["string", "number"]}},
        "required": ["input", "target"],
    },
}

TOOLS: list[dict] = [
    {
        "name": "list_models",
        "description": (
            "Discover the fleet of SoftModel models (evolvable specialized "
            "small models you can both USE and TEACH). Returns each model's "
            "id, description, active version, held-out score, drift status, "
            "and learned_shape — the feature space and output form (numeric "
            "value or learned label vocabulary) that the MODEL shaped for "
            "itself from its data. Read learned_shape to know which features "
            "to extract when calling infer."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_model",
        "description": (
            "Create a new SoftModel model. Deliberately minimal — no types, "
            "no schemas: just a name. The model shapes ITSELF from the data "
            "you teach it: feature space, numeric vs categorical output, "
            "output vocabulary, capacity. IMPORTANT: provide `holdout` here "
            "(or call add_holdout) BEFORE the first teach — the promotion "
            "gate judges candidates on held-out reality, and without any "
            "holdout no candidate can ever be promoted."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "description": {"type": "string"},
                "holdout": _EXAMPLES_SCHEMA,
            },
            "required": ["model_id"],
        },
    },
    {
        "name": "infer",
        "description": (
            "USE a SoftModel model: extract the features named in its "
            "learned_shape (see list_models/card) from your context as a "
            "numeric feature object; get back the model's answer — a number, "
            "or a label with confidence, depending on the shape the model "
            "learned from its data. The model carries this domain's own "
            "validated experience — trust its judgment especially where it "
            "disagrees with your general expectations. A response of "
            "note:'untrained' means no version has been promoted yet: "
            "add_holdout, then teach, then retry."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "input": {"type": "object",
                          "description": "feature object, e.g. {\"amount\": 800, \"night\": 0}"},
                "version": {"type": "string"},
            },
            "required": ["model_id", "input"],
        },
    },
    {
        "name": "teach",
        "description": (
            "TEACH a SoftModel model with labeled examples. A candidate "
            "version is trained and promoted ONLY if it beats the live "
            "version on held-out data — teach freely, the gate protects "
            "quality; promoted:false is a safe outcome (live model "
            "untouched), and if it happens repeatedly the usual cause is "
            "missing or stale holdout (fix with add_holdout). After drift, "
            "pass window=N to train on only the most recent N stored "
            "examples (sheds outdated labels) AND recent_n=N to judge the "
            "gate on the most recent N held-out examples (a mixed-era "
            "holdout can tie and block adaptation)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "examples": _EXAMPLES_SCHEMA,
                "window": {"type": "integer"},
                "recent_n": {"type": "integer"},
            },
            "required": ["model_id", "examples"],
        },
    },
    {
        "name": "add_holdout",
        "description": (
            "Append fresh labeled reality to a model's held-out stream. Do "
            "this as new ground truth becomes available — the promotion gate "
            "and drift checks score on the most recent slice, so evolution "
            "is judged against reality as it is now."),
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"},
                           "examples": _EXAMPLES_SCHEMA},
            "required": ["model_id", "examples"],
        },
    },
    {
        "name": "check_drift",
        "description": (
            "Check whether a model's experience has gone stale versus recent "
            "reality. If needs_reteach is true: collect fresh labeled "
            "examples and call teach (consider window=N)."),
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"},
                           "recent_n": {"type": "integer"}},
            "required": ["model_id"],
        },
    },
    {
        "name": "discoveries",
        "description": (
            "Read the regularities a model has mined from its own data "
            "(readable IF/THEN statements with confidence and support). "
            "Available for any model whose data shaped a categorical output. "
            "These are regularities of THIS domain/business that you cannot "
            "know from general knowledge — incorporate them into your "
            "reasoning."),
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"},
                           "version": {"type": "string"}},
            "required": ["model_id"],
        },
    },
    {
        "name": "get_versions",
        "description": "A model's version lineage with per-version held-out scores and the active pointer.",
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
    },
    {
        "name": "rollback",
        "description": "Set a model's active version (instant, reversible).",
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"},
                           "to": {"type": "string"}},
            "required": ["model_id", "to"],
        },
    },
]


class MCPServer:
    """Transport-independent MCP core: handle(message) -> response | None."""

    def __init__(self, factory: Optional[SoftModelFactory] = None):
        self.factory = factory or SoftModelFactory(Config.from_env())

    # ---------- JSON-RPC ----------
    def handle(self, msg: dict) -> Optional[dict]:
        method = msg.get("method")
        msg_id = msg.get("id")
        if method is None:
            return self._error(msg_id, -32600, "invalid request: no method")
        if method.startswith("notifications/"):
            return None                                   # notifications: no response
        try:
            if method == "initialize":
                params = msg.get("params") or {}
                return self._result(msg_id, {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "generator", "version": __version__},
                    "instructions": SERVER_INSTRUCTIONS,
                })
            if method == "ping":
                return self._result(msg_id, {})
            if method == "tools/list":
                return self._result(msg_id, {"tools": TOOLS})
            if method == "resources/list":                # graceful: none served
                return self._result(msg_id, {"resources": []})
            if method == "prompts/list":
                return self._result(msg_id, {"prompts": []})
            if method == "tools/call":
                params = msg.get("params") or {}
                return self._tool_call(msg_id, params.get("name"),
                                       params.get("arguments") or {})
            return self._error(msg_id, -32601, f"method not found: {method}")
        except Exception as e:                            # defensive: never crash the loop
            return self._error(msg_id, -32603, f"internal error: {e}")

    def _tool_call(self, msg_id, name: str, args: dict) -> dict:
        try:
            data = self._dispatch(name, args)
            text = json.dumps(data, ensure_ascii=False, indent=2)
            return self._result(msg_id, {
                "content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:
            hint = self._recovery_hint(str(e))
            text = f"error: {e}" + (f"\nhint: {hint}" if hint else "")
            return self._result(msg_id, {
                "content": [{"type": "text", "text": text}],
                "isError": True})

    @staticmethod
    def _recovery_hint(err: str) -> str:
        """Actionable next step for the calling AI on common failures."""
        e = err.lower()
        if ("unknown model" in e or "no such model" in e
                or "not found" in e
                or ("no such file" in e and "registry" in e)):
            return ("model does not exist here — call list_models to see "
                    "the fleet, or create_model to create it")
        if "holdout" in e:
            return "call add_holdout with labeled examples first"
        if "keyerror" in e or "missing" in e or "required" in e:
            return ("check required arguments against the tool schema; "
                    "examples rows need {input: {...}, target: ...}")
        return ""

    # ---------- tool dispatch ----------
    def _dispatch(self, name: str, a: dict) -> Any:
        f = self.factory
        if name == "list_models":
            return f.list_models()
        if name == "create_model":
            return f.create(ModelSpec(
                model_id=a["model_id"],
                description=a.get("description", ""),
                holdout=a.get("holdout", [])))
        if name == "infer":
            return f.infer(a["model_id"], a["input"], version=a.get("version"))
        if name == "teach":
            return f.teach(a["model_id"], a["examples"], window=a.get("window"),
                           recent_n=a.get("recent_n"))
        if name == "add_holdout":
            return f.add_holdout(a["model_id"], a["examples"])
        if name == "check_drift":
            return f.check_drift(a["model_id"], recent_n=a.get("recent_n"))
        if name == "discoveries":
            return f.discoveries(a["model_id"], version=a.get("version"))
        if name == "get_versions":
            return f.versions(a["model_id"])
        if name == "rollback":
            return f.rollback(a["model_id"], a["to"])
        raise ValueError(f"unknown tool: {name}")

    # ---------- helpers ----------
    @staticmethod
    def _result(msg_id, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}


def main() -> int:
    server = MCPServer()
    print(f"generator MCP server v{__version__} (stdio)", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue
        resp = server.handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
