"""Box T-0 (doc 37 0b + W0): golden fixtures exist, are
complete, and the capture script is WRITE-ONCE — it refuses a
non-empty output directory, so no previously captured golden
(or any other fixture) can ever be overwritten in place."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "tests" / "unit" / "fixtures" / \
    "classical_goldens"


def test_t0_goldens_complete():
    manifest = json.loads((GOLD / "MANIFEST.json").read_text())
    assert set(manifest["configs"]) == {
        f"F{i}" for i in range(1, 9)}
    for name in manifest["configs"]:
        assert (GOLD / f"{name}.npz").exists(), name
    assert manifest["adam_steps"] == 100
    assert manifest["sgd_steps"] == 20


def test_t0_capture_is_write_once():
    before = sorted(p.name for p in GOLD.iterdir())
    r = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "capture_classical_goldens.py")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "WRITE-ONCE REFUSAL" in (r.stdout + r.stderr)
    after = sorted(p.name for p in GOLD.iterdir())
    assert after == before          # file census unchanged
