"""Track-B D-P5 battery — executes BATTERY_SPEC.md verbatim
(registered criteria; caller-side experiment code — the lib
is consumed through its public modules only). FAST=1 runs a
plumbing preflight (1 seed, 6 rounds, verdicts not binding).
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("RLTrainer", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

from rl_trainer.eval_provider import (EvalEpisodeProvider,   # noqa
                                      gate_adjudicate)
from rl_trainer.runner import OrganPPORunner                 # noqa
from rl_trainer.worlds import (StagedExpansionWorld,         # noqa
                               StationaryWorld,
                               SensorArrivalWorld)

FAST = os.environ.get("FAST") == "1"
SEEDS = [0] if FAST else [0, 1, 2]
ROUNDS = 6 if FAST else 30
HORIZON = 256
WORLDS = {"staged_expansion": StagedExpansionWorld,
          "stationary": StationaryWorld,
          "sensor_arrival": SensorArrivalWorld}
OUT = Path(__file__).parent / "runs"
OUT.mkdir(exist_ok=True)


class _OrganActor:
    def __init__(self, adapter):
        self.adapter = adapter

    def act_probs(self, s):
        return self.adapter.probs(
            np.asarray(s, dtype=float)[None])[0]


def _baseline_score(world_cls, wseed, base):
    """Untrained-baseline actor on the SAME eval seeds."""
    class _U:
        def act_probs(self, s):
            k = world_cls(seed=wseed).n_actions
            return np.full(k, 1.0 / k)
    prov = EvalEpisodeProvider(world_cls, world_seed=wseed,
                               eval_seed_base=base)
    out = prov.evaluate_pair(_U(), _U(),
                             {"rl.eval_episode_budget": 6,
                              "rl.eval_window": 6})
    return out["score_inc"]


def _final_score(run, world_cls, wseed, base):
    prov = EvalEpisodeProvider(world_cls, world_seed=wseed,
                               eval_seed_base=base)
    a = _OrganActor(run.policy_adapter)
    out = prov.evaluate_pair(a, a,
                             {"rl.eval_episode_budget": 6,
                              "rl.eval_window": 6})
    return out["score_inc"]


def run_cell(trainer, wname, seed):
    world_cls = WORLDS[wname]
    run = OrganPPORunner(world_cls(seed=seed), seed=100 + seed,
                         hidden=10,
                         policy={"rl.trainer": trainer})
    curve = []
    for _ in range(ROUNDS):
        run.train_rounds(1, horizon=HORIZON)
        curve.append(run.mean_recent_return())
    fin = _final_score(run, world_cls, seed, base=seed)
    base = _baseline_score(world_cls, seed, base=seed)
    return {"trainer": trainer, "world": wname, "seed": seed,
            "curve": curve, "final": fin, "baseline": base,
            "run": run}


def growth_arm(seed):
    """L-B2: gate-adjudicated growth at the staged boundary
    vs the no-growth twin (same seeds)."""
    w = StagedExpansionWorld
    grow = OrganPPORunner(w(seed=seed), seed=200 + seed,
                          hidden=10)
    plain = OrganPPORunner(w(seed=seed), seed=200 + seed,
                           hidden=10)
    half = ROUNDS // 2
    adoptions = 0
    for phase, runs in ((0, half), (1, ROUNDS - half)):
        for _ in range(runs):
            grow.train_rounds(1, horizon=HORIZON)
            plain.train_rounds(1, horizon=HORIZON)
        if phase == 0:                     # boundary reached
            cand = copy.deepcopy(grow)
            cand.policy_adapter.organ.grow(0, hidden=6,
                                           force=True)
            cand.train_rounds(2, horizon=HORIZON)
            prov = EvalEpisodeProvider(w, world_seed=seed,
                                       eval_seed_base=50 + seed)
            prov.align_to(grow.world)   # 96 E-9 (G-7): judge
            v = gate_adjudicate(        # in the CURRENT regime
                prov, _OrganActor(grow.policy_adapter),
                _OrganActor(cand.policy_adapter),
                {"rl.eval_episode_budget": 6,
                 "rl.eval_window": 6})
            if v["adopt"]:
                grow = cand
                adoptions += 1
    fg = _final_score(grow, w, seed, base=seed)
    fp = _final_score(plain, w, seed, base=seed)
    return {"seed": seed, "grow_final": fg, "plain_final": fp,
            "adoptions": adoptions}


def silence_arm(seed):
    run = OrganPPORunner(StationaryWorld(seed=seed),
                         seed=300 + seed, hidden=8)
    run.train_rounds(2, horizon=HORIZON)
    inc = _OrganActor(run.policy_adapter)
    cand = _OrganActor(copy.deepcopy(run.policy_adapter))
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=seed,
                               eval_seed_base=70 + seed)
    wins = 0
    for _ in range(5):
        if gate_adjudicate(prov, inc, cand,
                           {"rl.eval_episode_budget": 4,
                            "rl.eval_window": 4})["adopt"]:
            wins += 1
    return {"seed": seed, "clone_wins": wins}


def mixed_arm(seed):
    w = StationaryWorld(seed=seed)
    run = OrganPPORunner(w, seed=400 + seed, hidden=12)
    probe = np.stack([w.sample_state(i) for i in range(64)])
    labels = np.array([w.oracle(s) for s in probe])
    onehot = np.eye(w.n_actions)[labels] * 2.0 - 1.0
    chance = 1.0 / w.n_actions

    def agree():
        pred = np.argmax(run.policy_logits(probe), axis=1)
        return float(np.mean(pred == labels))
    for _ in range(60):
        run.policy_adapter.organ.train_step(probe, onehot)
    a_teach = agree()
    run.train_rounds(10 if not FAST else 3, horizon=HORIZON)
    a_after_rl = agree()
    for _ in range(30):
        run.policy_adapter.organ.train_step(probe, onehot)
    run.collect(128)
    ret_close = float(np.mean(
        [r["value"] for r in run.records[-4:]]))
    return {"seed": seed, "agree_teach": a_teach,
            "agree_after_rl": a_after_rl,
            "ret_close": ret_close, "chance": chance,
            "floor_ret": 0.8 * chance * w.ep_len}


def main():
    res = {"cells": [], "growth": [], "silence": [],
           "mixed": []}
    for tr in ("ppo", "grpo"):
        for wn in WORLDS:
            for sd in SEEDS:
                c = run_cell(tr, wn, sd)
                c.pop("run")
                res["cells"].append(c)
                print(f"[cell] {tr}/{wn}/s{sd} final="
                      f"{c['final']:.2f} base="
                      f"{c['baseline']:.2f}", flush=True)
    for sd in SEEDS:
        g = growth_arm(sd)
        res["growth"].append(g)
        print(f"[growth] s{sd} grow={g['grow_final']:.2f} "
              f"plain={g['plain_final']:.2f} "
              f"adoptions={g['adoptions']}", flush=True)
        s = silence_arm(sd)
        res["silence"].append(s)
        print(f"[silence] s{sd} clone_wins={s['clone_wins']}",
              flush=True)
        m = mixed_arm(sd)
        res["mixed"].append(m)
        print(f"[mixed] s{sd} agree {m['agree_teach']:.2f}->"
              f"{m['agree_after_rl']:.2f} ret_close="
              f"{m['ret_close']:.2f}", flush=True)
    # ---------- registered verdicts ----------
    need = 2 if not FAST else 1
    lb1 = {}
    for tr in ("ppo", "grpo"):
        for wn in WORLDS:
            wins = sum(1 for c in res["cells"]
                       if c["trainer"] == tr and c["world"] == wn
                       and c["final"] > c["baseline"])
            lb1[f"{tr}/{wn}"] = wins
    v1 = all(w >= need for w in lb1.values())
    v2 = sum(1 for g in res["growth"]
             if g["grow_final"] >= g["plain_final"]) >= need
    v3 = all(s["clone_wins"] == 0 for s in res["silence"])
    v4 = sum(1 for m in res["mixed"]
             if m["agree_after_rl"] > m["chance"] + 0.10
             and m["ret_close"] > m["floor_ret"]) >= need
    verdict = {"L-B1": v1, "L-B1 detail": lb1, "L-B2": v2,
               "L-B3": v3, "L-B4": v4,
               "ALL": bool(v1 and v2 and v3 and v4),
               "fast_preflight": FAST}
    res["verdict"] = verdict
    tag = "FAST" if FAST else "FULL"
    (OUT / f"RESULTS_{tag}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
