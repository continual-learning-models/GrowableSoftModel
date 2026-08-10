"""One-command product verification — the clean acceptance entry.

Runs everything a release user needs to trust the product:
guard -> unit -> plasticity -> integration -> substrate kit -> module
suites. Exit 0 = the product is healthy.

Usage:  python3 scripts/check_all.py [--fast]
        --fast skips the slow integration scripts and module suites.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAST = "--fast" in sys.argv


def run(label, cmd, cwd=ROOT):
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        print(f"CHECK FAILED at: {label}")
        sys.exit(1)


run("CI guard (integrity / isolation / language / no-network)",
    [sys.executable, "scripts/ci_guard.py"])
run("Unit + plasticity suites",
    [sys.executable, "-m", "pytest", "tests/unit", "tests/plasticity",
     "-q"])
if not FAST:
    for script in ["test_system.py", "test_llm_callable.py",
                   "test_conformance.py", "test_system_transformer.py",
                   "test_transformer_growth_audit.py"]:
        run(f"Integration: {script}",
            [sys.executable, f"tests/integration/{script}"])
    run("Substrate compatibility kit (all registry entries)",
        [sys.executable, "tests/substrate_kit/kit.py", "--all"])
    run("Module suite: Generator",
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT / "modules" / "Generator")
    run("Module suite: ReferenceNet",
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT / "modules" / "ReferenceNet")
print("\nALL CHECKS PASSED — the product is healthy.")
