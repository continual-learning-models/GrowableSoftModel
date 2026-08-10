"""R2 arms — executes BATTERY_SPEC.md §R2 verbatim."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("RLTrainer", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

from rl_trainer.eval_provider import EvalEpisodeProvider   # noqa
from rl_trainer.regime import RegimeDispatcher             # noqa
from rl_trainer.runner import OrganPPORunner               # noqa
from rl_trainer.worlds import StationaryWorld              # noqa

OUT = Path(__file__).parent / "runs"


class _A:
    def __init__(self, ad):
        self.ad = ad

    def act_probs(self, s):
        return self.ad.probs(np.asarray(s, dtype=float)[None])[0]


def _final(run, seed):
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=seed,
                               eval_seed_base=seed)
    a = _A(run.policy_adapter)
    return prov.evaluate_pair(a, a, {"rl.eval_episode_budget": 6,
                                     "rl.eval_window": 6})["score_inc"]


def _baseline(seed):
    class _U:
        def act_probs(self, s):
            return np.full(3, 1.0 / 3.0)
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=seed,
                               eval_seed_base=seed)
    return prov.evaluate_pair(_U(), _U(),
                              {"rl.eval_episode_budget": 6,
                               "rl.eval_window": 6})["score_inc"]


def _params(run):
    o = run.policy_adapter.organ
    return np.concatenate([np.asarray(o.W1).ravel(),
                           np.asarray(o.W2).ravel()])


def arm_kl(seed):
    out = {}
    for tag, beta in (("free", 0.0), ("anchored", 0.5)):
        run = OrganPPORunner(StationaryWorld(seed=seed),
                             seed=600 + seed, hidden=10,
                             policy={"rl.kl_ref_coef": beta})
        run.train_rounds(10)
        snap = _params(run)                 # round-10 snapshot
        run.train_rounds(10)
        out[tag] = {"final": _final(run, seed),
                    "drift": float(np.abs(_params(run)
                                          - snap).sum())}
    out["baseline"] = _baseline(seed)
    return out


def arm_il():
    labeled = [{"input": [0.1], "target": 1.0}]
    rewards = [{"source": "env_return", "value": 1.0}]
    d = RegimeDispatcher(policy={"rl.interleave": [2, 1]})
    seq = [d.dispatch(labeled_rows=labeled,
                      reward_records=rewards)
           for _ in range(12)]
    want = (["teach", "teach", "rl"] * 4)
    rules = {e["rule"] for e in d.audit}
    return {"seq_ok": seq == want,
            "rule_named": rules == {"interleave:2:1"}}


def main():
    res = {"arm_kl": [], "arm_il": arm_il()}
    for sd in (0, 1, 2):
        r = arm_kl(sd)
        res["arm_kl"].append({"seed": sd, **r})
        print(f"[ARM-KL] s{sd} free final={r['free']['final']:.2f} "
              f"drift={r['free']['drift']:.3f} | anchored "
              f"final={r['anchored']['final']:.2f} "
              f"drift={r['anchored']['drift']:.3f} | "
              f"base={r['baseline']:.2f}", flush=True)
    need = 2
    ka = sum(1 for x in res["arm_kl"]
             if x["free"]["final"] > x["baseline"]) >= need and \
        sum(1 for x in res["arm_kl"]
            if x["anchored"]["final"] > x["baseline"]) >= need
    kb = sum(1 for x in res["arm_kl"]
             if x["anchored"]["drift"] < x["free"]["drift"]) >= need
    verdict = {"L-R2K-a": ka, "L-R2K-b": kb,
               "L-R2I": bool(res["arm_il"]["seq_ok"]
                             and res["arm_il"]["rule_named"])}
    verdict["ALL"] = all(verdict.values())
    res["verdict"] = verdict
    (OUT / "R2_ARMS_RESULTS.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
