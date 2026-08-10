"""P0 guard boxes (plan 53 v1.2 step P0; 54 v1.10 section 2,
T-0' row of the coverage matrix, C22-adjacent):

 - the adjustment_baseline fixture set is COMPLETE (manifest
   lists FX-1..FX-4, every listed file exists, every sha256
   matches — nothing silently altered);
 - the capture script is WRITE-ONCE (refuses a non-empty
   output directory);
 - the capture is REPRODUCIBLE: a fresh run into a scratch
   directory yields the identical scenario state hashes
   (the baseline is a property of the code at growthport-v1,
   not of the day it was captured).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "unit" / "fixtures" / \
    "adjustment_baseline"
SCRIPT = REPO / "scripts" / "capture_adjustment_baseline.py"


def _manifest():
    return json.loads((FIX / "MANIFEST.json").read_text())


def test_p0_fixture_set_complete():
    man = _manifest()
    assert man["tag"] == "growthport-v1"
    assert set(man["fixtures"]) == {"FX-1", "FX-2", "FX-3",
                                    "FX-4"}
    assert set(man["fixtures"]["FX-4"]) == {
        "duplicate", "port_type", "guard_refuse"}
    for name, sha in man["files"].items():
        f = FIX / name
        assert f.exists(), name
        assert hashlib.sha256(
            f.read_bytes()).hexdigest() == sha, name


def test_p0_capture_is_write_once():
    r = subprocess.run([sys.executable, str(SCRIPT)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "REFUSED" in (r.stdout + r.stderr)


def test_p0_capture_reproducible(tmp_path):
    out = tmp_path / "recap"
    r = subprocess.run([sys.executable, str(SCRIPT), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    man0, man1 = _manifest(), json.loads(
        (out / "MANIFEST.json").read_text())
    # scenario state hashes (ledger-exclusive, see capture
    # script) must match EXACTLY — the numeric bit-identity
    # verdict of the whole scenario set
    assert man0["fixtures"] == man1["fixtures"]
    # numeric record files byte-identical; pickle artifacts
    # embed the gain ledger, whose record shape is the ONE
    # declared amendment (52 SR-22) — their state is judged
    # by the hashes above and the V-A1/V-A2 boxes instead
    npz0 = {k: v for k, v in man0["files"].items()
            if k.endswith(".npz")}
    npz1 = {k: v for k, v in man1["files"].items()
            if k.endswith(".npz")}
    assert npz0 == npz1
