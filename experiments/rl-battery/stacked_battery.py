"""J-1 stacked battery — executes BATTERY_SPEC.md §J-1."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("RLTrainer", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

from reference_net.growthpolicy import preference as prf   # noqa
from rl_trainer.eval_provider import (EvalEpisodeProvider,  # noqa
                                      gate_adjudicate)
from rl_trainer.runner import OrganPPORunner               # noqa
from rl_trainer.worlds import StagedExpansionWorld         # noqa

OUT = Path(__file__).parent / "runs"


class _A:
    def __init__(self, ad):
        self.ad = ad

    def act_probs(self, s):
        return self.ad.probs(np.asarray(s, dtype=float)[None])[0]


def _score(run, seed, base):
    prov = EvalEpisodeProvider(StagedExpansionWorld,
                               world_seed=seed,
                               eval_seed_base=base)
    a = _A(run.policy_adapter)
    return prov.evaluate_pair(
        a, a, {"rl.eval_episode_budget": 6,
               "rl.eval_window": 6})["score_inc"]


def one_seed(seed):
    run = OrganPPORunner(StagedExpansionWorld(seed=seed),
                         seed=500 + seed, hidden=10)
    pref = prf.GrowthPreference({"seed": seed,
                                 "preference.rule": "mean_clip",
                                 "preference.bucket_spec": "b0",
                                 "preference.min_count": 1})
    run.train_rounds(10)
    ctxs = [{"move": "grow", "slope": None},
            {"move": "deepen", "slope": None}]
    out = prf.rank_with_preference(pref, [0.5, 0.5], ctxs)
    move = ctxs[int(np.argmax(out["scores_adj"]))]["move"]
    cand = copy.deepcopy(run)
    organ = cand.policy_adapter.organ
    (organ.grow(0, hidden=6, force=True) if move == "grow"
     else organ.deepen(m=4, force=True))
    cand.train_rounds(2)
    prov = EvalEpisodeProvider(StagedExpansionWorld,
                               world_seed=seed,
                               eval_seed_base=80 + seed)
    v = gate_adjudicate(prov, _A(run.policy_adapter),
                        _A(cand.policy_adapter),
                        {"rl.eval_episode_budget": 6,
                         "rl.eval_window": 6})
    audited = "audit" in v and "episodes" in v["audit"]
    if v["adopt"]:
        run = cand
    gain = float(v["score_cand"] - v["score_inc"])
    pref.credit({"event_id": f"j1-{seed}", "bucket": move,
                 "move": move, "batch": 10, "quoted_gain": 0.0,
                 "window_gains": [gain], "credited_gain": gain,
                 "advantage": gain})
    run.train_rounds(10)
    fin = _score(run, seed, base=seed)

    class _U:
        def act_probs(self, s):
            return np.full(3, 1.0 / 3.0)
    base = EvalEpisodeProvider(
        StagedExpansionWorld, world_seed=seed,
        eval_seed_base=seed).evaluate_pair(
            _U(), _U(), {"rl.eval_episode_budget": 6,
                         "rl.eval_window": 6})["score_inc"]
    w_after = pref.snapshot()["stats"][move]["w"]
    return {"seed": seed, "move": move, "adopt": v["adopt"],
            "audited": bool(audited), "final": fin,
            "baseline": base, "pref_w": w_after}


def main():
    rows = [one_seed(s) for s in (0, 1, 2)]
    for r in rows:
        print(f"[j1] s{r['seed']} move={r['move']} "
              f"adopt={r['adopt']} audited={r['audited']} "
              f"final={r['final']:.2f} base={r['baseline']:.2f} "
              f"pref_w={r['pref_w']:.2f}", flush=True)
    verdict = {
        "L-J1-a": len(rows) == 3,
        "L-J1-b": all(r["pref_w"] > 0 for r in rows),
        "L-J1-c": sum(1 for r in rows
                      if r["final"] > r["baseline"]) >= 2,
        "L-J1-d": all(r["audited"] for r in rows),
    }
    verdict["ALL"] = all(verdict.values())
    (OUT / "J1_RESULTS.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict},
                   indent=1, default=float))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
