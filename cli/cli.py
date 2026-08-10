"""System CLI (IWP4/S4.3): thin mirror of the facade. JSON in/out.

Usage: python3 -m cli.cli <verb> '<json-args>'
Example: ... infer '{"model_id":"m","input":{"a":1,"b":3,"c":0}}'
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys

from core.facade import System


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        print(json.dumps({"error": "usage: cli <verb> '<json-args>'"}))
        return 1
    verb = argv[0]
    if verb in ("help", "--help", "-h"):
        import standard_methods as std
        soft = sorted(v for v in dir(System)
                      if not v.startswith("_"))
        std_verbs = ["standard_" + v for v in
                     ("create", "train", "evaluate", "infer",
                      "save", "load", "list")]
        print(json.dumps({
            "usage": "cli <verb> '<json-args>'",
            "softmodel_verbs (the tool's method)": soft,
            "standard_verbs (optional industry mode)": std_verbs,
            "examples": [
                "cli create_model '{\"model_id\": \"m1\"}'",
                "cli standard_create '{\"name\": \"m2\", "
                "\"arch\": \"mlp\"}'",
                "cli standard_train '{\"name\": \"m2\", "
                "\"examples\": [{\"x\": [1,2], \"y\": 3}]}'",
                "cli deepen '{\"model_id\": \"m1\", "
                "\"m\": 4}'  (delta: add a processing stage; "
                "position/scope optional)",
                "cli plan_run '{\"model_id\": \"m1\", "
                "\"plan\": \"/path/to/plan.json\"}'  "
                "(rule-file execution; plan may also be an "
                "inline dict)",
            ]}, indent=1))
        return 0
    args = json.loads(argv[1]) if len(argv) > 1 else {}
    if verb.startswith("standard_"):
        import standard_methods as std
        short = {"standard_list": "list_models"}.get(
            verb, verb[len("standard_"):])
        fn = getattr(std, short, None)
        if fn is None:
            print(json.dumps({"error": f"unknown verb: {verb}",
                              "hint": "cli help lists all verbs"}))
            return 1
        print(json.dumps(fn(**args)))
        return 0
    system = System()
    fn = getattr(system, verb, None)
    if fn is None or verb.startswith("_"):
        print(json.dumps({"error": f"unknown verb: {verb}",
                          "hint": "cli help lists all verbs"}))
        return 1
    try:
        out = fn(**args)
    except Exception as exc:                      # noqa: BLE001
        # 60A L4 last line: an unexpected internal error is a
        # STRUCTURED reply (never a raw crash); the traceback
        # stays on stderr — nothing is swallowed
        import traceback
        traceback.print_exc()
        print(json.dumps({"error":
                          f"{type(exc).__name__}: {exc}"}))
        return 1
    # 59B 2.4: the facade captures lib warnings into the
    # response; this CHANNEL re-emits them on its stderr so a
    # shell user sees them even under -W ignore.
    if isinstance(out, dict):
        for w in out.get("warning") or []:
            print(f"warning: {w}", file=sys.stderr)
    try:
        print(json.dumps(out))
    except TypeError:
        # programmatic-object verbs (e.g. store) have no JSON
        # form on this channel — refuse loudly, never crash
        print(json.dumps({"refusal":
                          f"verb {verb!r} returns a "
                          "programmatic object with no JSON "
                          "form on the CLI channel",
                          "instead": "use the MCP tool or the "
                                     "facade directly"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
