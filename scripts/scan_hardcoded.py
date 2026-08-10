#!/usr/bin/env python3
"""Standing acceptance gate: hardcoded behavioral-constant census.

Scans core/ and modules/ (product code only) for module-level and
class-level ALL_CAPS numeric constants. Every finding must be
listed in scripts/scan_manifest.json under exactly one of:
  "censused"  — behavioral parameters exposed via an interface
                (the constant remains as the default value);
  "allowlist" — non-parameters (mathematical properties, protocol
                versions, numeric-stability epsilons).
Anything unlisted fails the gate (exit 1): either a new hardcoded
behavioral constant (defect — parameterize it) or a new
non-parameter (classify it in the manifest with a reason).

Doc: docs/system/21-23 (parameter-interface completion, 2026-07-19).
Usage: python scripts/scan_hardcoded.py [--write-baseline]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "scan_manifest.json"
SCAN_DIRS = ("core", "modules")
EXCLUDE_PARTS = {"tests", "test", "scripts", "__pycache__",
                 "egg-info", ".git", "students", "trained_models"}


def _numeric(node):
    if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)) and not isinstance(
            node.value, bool):
        return node.value
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)):
        inner = _numeric(node.operand)
        return None if inner is None else -inner
    return None


def _collect(tree):
    out = []
    def visit_body(body, scope):
        for st in body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                t = st.targets[0]
                if (isinstance(t, ast.Name) and t.id.isupper()
                        and len(t.id) >= 3):
                    v = _numeric(st.value)
                    if v is not None:
                        out.append((scope, t.id, v))
            elif isinstance(st, ast.ClassDef):
                visit_body(st.body, st.name)
    visit_body(tree.body, "")
    return out


def scan():
    found = []
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_PARTS or part.endswith(
                    ".egg-info") for part in rel.parts):
                continue
            try:
                tree = ast.parse(p.read_text())
            except SyntaxError as e:  # loud, never silent
                print(f"SYNTAX ERROR {rel}: {e}")
                sys.exit(2)
            for scope, name, val in _collect(tree):
                found.append({"file": str(rel), "scope": scope,
                              "name": name, "value": val})
    return found


def main():
    found = scan()
    if "--write-baseline" in sys.argv:
        # First run only: classify by hand afterwards.
        MANIFEST.write_text(json.dumps(
            {"censused": found, "allowlist": []}, indent=1))
        print(f"baseline written: {len(found)} findings")
        return
    man = json.loads(MANIFEST.read_text())
    known = {(e["file"], e["scope"], e["name"])
             for k in ("censused", "allowlist") for e in man[k]}
    unknown = [f for f in found
               if (f["file"], f["scope"], f["name"]) not in known]
    gone = [e for k in ("censused", "allowlist") for e in man[k]
            if (e["file"], e["scope"], e["name"]) not in
            {(f["file"], f["scope"], f["name"]) for f in found}]
    for f in unknown:
        print("UNCLASSIFIED:", f)
    for e in gone:
        print("STALE MANIFEST ENTRY:", e)
    if unknown or gone:
        sys.exit(1)
    print(f"census gate clean: {len(found)} constants, "
          f"{len(man['censused'])} censused, "
          f"{len(man['allowlist'])} allowlisted")


if __name__ == "__main__":
    main()
