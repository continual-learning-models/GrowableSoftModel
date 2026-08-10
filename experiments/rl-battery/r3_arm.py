"""R3 arm v2 — executes BATTERY_SPEC.md §R3 revision (ARM-ILT-v2) verbatim.
Caller-scripted training interleave (LAW-3 ownership): the
dispatcher names phases; this script executes them."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("RLTrainer", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

from rl_trainer.regime import RegimeDispatcher             # noqa
from rl_trainer.runner import OrganPPORunner               # noqa
from rl_trainer.worlds import StationaryWorld              # noqa

OUT = Path(__file__).parent / "runs"
LABELED = [{"input": [0.1], "target": 1.0}]
REWARDS = [{"source": "env_return", "value": 1.0}]


def _probe_rows(seed):
    """v2 (spec R3 revision): TASK-CONSISTENT evidence —
    oracle one-hot labels of the SAME world at stage 0."""
    w = StationaryWorld(seed=seed)
    X = np.stack([w.sample_state(i) for i in range(32)])
    y = np.eye(3)[[w.oracle(x) for x in X]]
    return X, y


def _teach(run, X, y, steps):
    for _ in range(steps):
        run.policy_adapter.organ.train_step(X, y)


def _mse(run, X, y):
    return float(np.mean(
        (run.policy_adapter.outputs(X) - y) ** 2))


def arm_ilt(seed):
    X, y = _probe_rows(seed)

    def _mk():
        run = OrganPPORunner(StationaryWorld(seed=seed),
                             seed=300 + seed, hidden=10)
        _teach(run, X, y, 30)               # initial phase
        return run

    base = _mk()
    base_ret = base.mean_recent_return()    # untrained-rl floor

    inter = _mk()
    d = RegimeDispatcher(policy={"rl.interleave": [1, 1]})
    for _ in range(12):                     # 6 teach + 6 rl
        ph = d.dispatch(labeled_rows=LABELED,
                        reward_records=REWARDS)
        if ph == "teach":
            _teach(inter, X, y, 5)
        else:
            inter.train_rounds(1, horizon=64)

    only = _mk()
    for _ in range(6):                      # same rl budget
        only.train_rounds(1, horizon=64)

    return {"seed": seed,
            "mse_inter": _mse(inter, X, y),
            "mse_only": _mse(only, X, y),
            "ret_inter": inter.mean_recent_return(),
            "ret_only": only.mean_recent_return(),
            "ret_floor": base_ret}


def main():
    res = [arm_ilt(sd) for sd in (0, 1, 2)]
    for r in res:
        print(f"[ARM-ILT] s{r['seed']} mse inter="
              f"{r['mse_inter']:.4f} only={r['mse_only']:.4f}"
              f" | ret inter={r['ret_inter']:.2f} "
              f"only={r['ret_only']:.2f} "
              f"floor={r['ret_floor']:.2f}", flush=True)
    a = sum(1 for r in res
            if r["mse_inter"] < r["mse_only"]) >= 2
    b = (sum(1 for r in res
             if r["ret_inter"] > r["ret_floor"]) >= 2
         and sum(1 for r in res
                 if r["ret_only"] > r["ret_floor"]) >= 2)
    verdict = {"L-R3T-a": a, "L-R3T-b": b, "ALL": a and b}
    out = {"arms": res, "verdict": verdict}
    (OUT / "R3_ARM_V2_RESULTS.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
