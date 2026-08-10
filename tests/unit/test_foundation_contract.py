"""A0 boxes (plan 53 v1.2 step A0; 54 v1.10 section 3):

 V-A0-1 contract conformance — Network and AttentionBody
        satisfy the REQUIRED set with an EMPTY missing list;
        a deliberately deficient dummy lists exactly its
        missing members, in REQUIRED order.
 V-A0-2 import-direction linter — foundation/* imports only
        numpy / engine backends / stdlib (never hosts, never
        the method tier, never growthpolicy); method/* (once
        present) imports only foundation + numpy + stdlib.
        Permanent regression box (AR-2).
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RN = REPO / "modules" / "ReferenceNet" / "reference_net"
for p in (REPO, REPO / "modules" / "Engine",
          REPO / "modules" / "ReferenceNet"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reference_net.foundation import REQUIRED, conforms   # noqa: E402
from reference_net.net import Network                     # noqa: E402
from reference_net.attention_body import AttentionBody    # noqa: E402


# ---------------- V-A0-1 conformance ----------------

def test_a0_network_conforms():
    net = Network(3, 8, lr=1e-2, seed=7)
    assert conforms(net) == []


def test_a0_attention_body_conforms():
    body = AttentionBody(3, seed=5)
    assert conforms(body) == []


def test_a0_deficient_dummy_lists_missing_members():
    class Deficient:
        def predict(self, X):          # has ONE required member
            return X
    missing = conforms(Deficient())
    # exactly the absent REQUIRED members, in REQUIRED order
    # (__getstate__ exists on object since Python 3.11 —
    # membership is trivially present; the contract records
    # that behavior is adjudicated by the serialization boxes)
    expected = [n for n in REQUIRED
                if n not in ("predict", "__getstate__")]
    assert missing == expected


def test_a0_conformance_failure_names_are_actionable():
    missing = conforms(object())
    assert "n_params" in missing and "predict" in missing


# ---------------- V-A0-2 import-direction linter ----------------

def _import_roots(pyfile):
    tree = ast.parse(pyfile.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue                       # "from . import x"
            if node.level:                     # relative: stays
                roots.add("__relative__")      # inside own package
            else:
                roots.add(node.module.split(".")[0])
    return roots


_STDLIB = {"sys", "os", "json", "math", "time", "types",
           "pathlib", "hashlib", "pickle", "warnings",
           "dataclasses", "typing", "collections", "functools",
           "itertools", "sqlite3", "uuid", "abc", "copy"}

FOUNDATION_ALLOWED = _STDLIB | {"numpy", "engine",
                                "__relative__"}
METHOD_ALLOWED = FOUNDATION_ALLOWED | {"reference_net"}
METHOD_FORBIDDEN_MODULES = {"reference_net.net",
                            "reference_net.growth_port",
                            "reference_net.bodies",
                            "reference_net.growthpolicy"}


def test_a0_foundation_imports_are_neutral():
    files = sorted((RN / "foundation").glob("*.py"))
    assert files, "foundation package must exist"
    for f in files:
        bad = _import_roots(f) - FOUNDATION_ALLOWED
        assert not bad, f"{f.name} imports {sorted(bad)}"


def test_a0_method_imports_foundation_only():
    mdir = RN / "method"
    if not mdir.exists():                 # arrives at step A2
        return
    for f in sorted(mdir.glob("*.py")):
        roots = _import_roots(f)
        bad = roots - METHOD_ALLOWED
        assert not bad, f"{f.name} imports {sorted(bad)}"
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                assert not any(mod.startswith(x) for x in
                               METHOD_FORBIDDEN_MODULES), \
                    f"{f.name}: forbidden import {mod}"
                if mod.startswith("reference_net"):
                    assert mod.startswith(
                        "reference_net.foundation"), \
                        f"{f.name}: {mod} is not foundation"


# ---------------- V-A5-1: facade import census ----------------

def test_a5_growth_port_facade_reexports_all_public_names():
    """A5 (doc 52 s6.3, RK-3): every historical public name of
    growth_port stays importable from growth_port — external
    imports never break. Legacy bodies moved VERBATIM to
    legacy_compat (byte-diff verified at the step gate)."""
    from reference_net import growth_port as gp
    for name in ("make_port_body", "grow_ffn_body", "PortSite",
                 "LegacyScalarPort", "PORT_TYPES",
                 "legacy_attach", "legacy_handoff",
                 "legacy_attach_layer", "legacy_collect_layer"):
        assert callable(getattr(gp, name)) or \
            isinstance(getattr(gp, name), tuple), name
    from reference_net import legacy_compat as lc
    for name in ("legacy_attach", "legacy_handoff",
                 "legacy_attach_layer", "legacy_collect_layer",
                 "LegacyScalarPort"):
        assert hasattr(lc, name), name
