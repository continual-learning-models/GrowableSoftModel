"""SPU S5 capability test: committed verdicts match committed
artifacts (artifact-pinned; no heavy re-run in CI)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
RES = ROOT / "experiments" / "PS_spu" / "results"


def load():
    return (json.loads((RES / "summary.json").read_text()),
            json.loads((RES / "verdicts.json").read_text()))


def test_verdicts_match_summary():
    s, v = load()
    noisy = s["scenes"]["noisy"]
    ps1 = sum(1 for r in noisy
              if r["with"]["mse_clean"] < r["without"]["mse_clean"])
    assert (ps1 >= 2) == v["PS1_value"]
    ps2 = sum(1 for r in noisy
              if r["with"]["degradation"] < r["without"]["degradation"])
    assert (ps2 >= 2) == v["PS2_robustness"]
    ps3 = all(r["with"]["ps3_final_ok"] and r["with"]["ps3_events_ok"]
              for sc in s["scenes"].values() for r in sc)
    assert ps3 == v["PS3_no_collapse"]


def test_ps4_mechanics_clauses_in_artifacts():
    s, v = load()
    noisy = s["scenes"]["noisy"]
    assert all(r["with"]["ps4_budget_ok"] for r in noisy)
    assert all(r["with"]["ps4_summaries_ok"] for r in noisy)
    over = all((r["with"]["wall_s"] / r["with"]["steps"])
               <= 1.25 * (r["without"]["wall_s"] / r["without"]["steps"])
               for r in noisy)
    assert over == v["PS4_mechanics"]


def test_budget_fairness_was_granted():
    s, _ = load()
    for sc in s["scenes"].values():
        for r in sc:
            assert r["without"]["steps"] > r["with"]["steps"]
            assert r["with"]["budget"]["extra_steps_granted"] > 0


def test_event_artifacts_exist_and_wellformed():
    for f in RES.glob("events_*.jsonl"):
        lines = f.read_text().strip().splitlines()
        assert lines and all(json.loads(x) for x in lines)
    assert len(list(RES.glob("events_*.jsonl"))) == 6
