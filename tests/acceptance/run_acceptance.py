"""Product acceptance entry — runs A (generator), B (growth),
C (system). Exit 0 = all pass. See tests/acceptance/README.md."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED = []
for part in ["phase1_generator.py", "phase2_growth.py",
             "phase3_system.py"]:
    print(f"\n===== {part} =====")
    r = subprocess.run([sys.executable, str(HERE / part)])
    if r.returncode != 0:
        FAILED.append(part)
print("\nPRODUCT ACCEPTANCE:",
      "PASS" if not FAILED else f"FAIL ({FAILED})")
sys.exit(0 if not FAILED else 1)
