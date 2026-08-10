"""Standing census gate in the suite (docs/system/22 6):
scripts/scan_hardcoded.py must run clean on every suite run, so a
new unclassified behavioral constant fails acceptance here, not
only when the script is run by hand."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_census_scanner_clean():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_hardcoded.py")],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "census gate clean" in proc.stdout
