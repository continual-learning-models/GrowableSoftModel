"""CI guard: language scan, single-shim rule, no benchmark code,
no network code. Importable functions for self-tests.

Run: python3 scripts/ci_guard.py   (exit 0 = all guards pass)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def language_scan(paths) -> list[str]:
    cjk = re.compile("[\\u4e00-\\u9fff\\u0400-\\u04ff]")
    v = []
    for base in paths:
        for f in Path(base).rglob("*"):
            # bilingual research records (EN+ZH) are exempt from the
            # English-only rule.
            if "research" in f.parts:
                continue
            if f.suffix in (".py", ".md") and f.is_file():
                if cjk.search(f.read_text(errors="ignore")):
                    v.append(f"non-English text in {f}")
    return v


def single_shim_rule() -> list[str]:
    v = []
    for f in (ROOT / "core").rglob("*.py"):
        if f.name == "_modules.py":
            continue
        if "sys.path" in f.read_text():
            v.append(f"sys.path manipulation outside the shim: {f}")
    return v


def substrate_isolation() -> list[str]:
    """No concrete substrate class names outside the substrates package
    (upstream must use the registry)."""
    banned = ("MLPSubstrate", "TransformerSubstrate", "SequenceSubstrate")
    v = []
    for f in (ROOT / "core").rglob("*.py"):
        if "substrates" in f.parts:
            continue
        text = f.read_text()
        for b in banned:
            if b in text:
                v.append(f"concrete substrate name '{b}' outside "
                         f"substrates package: {f}")
    return v


def no_network_code() -> list[str]:
    """Owner controllability rule: the model must never reach the
    network on its own. No network library may appear anywhere in the
    system or the modules (learning material arrives only through the
    user/brain or the model's own store)."""
    banned = ("import socket", "import urllib", "import requests",
              "import http.client", "import aiohttp", "import httpx",
              "from urllib", "from http import", "from socket import")
    v = []
    for base in (ROOT / "core", ROOT / "modules" / "Generator",
                 ROOT / "modules" / "ReferenceNet"):
        for f in base.rglob("*.py"):
            if ".git" in f.parts:
                continue
            text = f.read_text(errors="ignore")
            for b in banned:
                if b in text:
                    v.append(f"network import '{b}' in {f}")
    return v


def no_benchmark_code() -> list[str]:
    banned = ("tabpfn", "torchvision", "medmnist")
    v = []
    for f in (ROOT / "core").rglob("*.py"):
        text = f.read_text()
        for b in banned:
            if b in text:
                v.append(f"benchmark reference '{b}' in {f}")
    return v


def main() -> int:
    violations = []
    # Modules are now normal maintained code (unfrozen for release);
    # the former pinned-hash / additive-only integrity check is retired.
    violations += language_scan([ROOT / "core", ROOT / "tests",
                                 ROOT / "docs", ROOT / "scripts"])
    violations += single_shim_rule()
    violations += substrate_isolation()
    violations += no_benchmark_code()
    violations += no_network_code()
    for x in violations:
        print("GUARD FAIL:", x)
    print("CI GUARD:", "PASS" if not violations else f"{len(violations)} violation(s)")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
