"""PS driver — executes spec.md exactly. Artifacts to results/."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.spu.spu_network import SPUNetwork          # noqa: E402
from engine.spu.spu_policy import DEFAULT_SPU_POLICY   # noqa: E402
from engine.spu.spu_report import (                    # noqa: E402
    build_report, write_events_jsonl)

OUT = Path(__file__).parent / "results"
SEEDS = (7, 8, 9)
STEPS, GROW_AT, N_TRAIN, N_EVAL = 700, (150, 300), 128, 256


def law(X):
    return (np.sin(3 * X[:, :1]) + 0.6 * np.cos(5 * X[:, 1:2])
            + 0.3 * X[:, 2:3] * X[:, 3:4])


def make_data(seed, noisy):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-2, 2, (N_TRAIN, 4))
    ytr = law(Xtr)
    if noisy:
        drop = rng.random(Xtr.shape) < 0.2
        Xtr = np.where(drop, 0.0, Xtr)
        ytr = ytr + rng.normal(0, 0.1, ytr.shape)
    Xev = np.random.default_rng(seed + 1000).uniform(-2, 2, (N_EVAL, 4))
    yev = law(Xev)
    mask_ev = np.random.default_rng(seed + 2000).random(Xev.shape) < 0.2
    Xev_cor = np.where(mask_ev, 0.0, Xev)
    return Xtr, ytr, Xev, yev, Xev_cor


def run_arm(seed, noisy, with_spu, extra_steps=0):
    Xtr, ytr, Xev, yev, Xev_cor = make_data(seed, noisy)
    net = SPUNetwork(d_in=4, hidden=6, lr=1e-2, seed=seed)
    if with_spu:
        net.set_spu_policy({"spu_enabled": True})
    t0 = time.perf_counter()
    for step in range(STEPS + extra_steps):
        if step in GROW_AT and (j := 1 if step == GROW_AT[0] else 2) \
                not in net.inner:
            net.grow(j, hidden=5)
        net.train_step(Xtr, ytr)
    wall = time.perf_counter() - t0
    mse_clean = float(((net.predict(Xev) - yev) ** 2).mean())
    mse_cor = float(((net.predict(Xev_cor) - yev) ** 2).mean())
    inner_stds = {f"root/{j}": float(np.std(inner.predict(Xev)[:, 0]))
                  for j, inner in net.inner.items()}
    return net, {"seed": seed, "with_spu": with_spu,
                 "steps": STEPS + extra_steps, "wall_s": wall,
                 "mse_clean": mse_clean, "mse_corrupted": mse_cor,
                 "degradation": mse_cor / mse_clean,
                 "inner_net_eval_std": inner_stds}


def analytic_ratio(net):
    pol = DEFAULT_SPU_POLICY
    p_root = net.n_params()
    p_leaf = (5 * 4 + 5 + 5 + 1)                      # hidden=5, d=4
    return (pol["spu_S_max"] * (pol["spu_K"] + 2) * p_leaf
            + 2 * p_root) / (3 * p_root)


def main():
    OUT.mkdir(exist_ok=True)
    summary = {"scenes": {}}
    for scene, noisy in (("noisy", True), ("clean", False)):
        rows = []
        for seed in SEEDS:
            net_w, rw = run_arm(seed, noisy, True)
            rep = build_report(net_w)
            processed = rep["processed_steps"]
            r = analytic_ratio(net_w)
            extra = math.ceil(processed * r)
            _, ro = run_arm(seed, noisy, False, extra_steps=extra)
            rw["report"] = {"processed_steps": processed,
                            "skips": rep["skips"],
                            "interference": rep["interference"]}
            rw["budget"] = {"ratio": r, "extra_steps_granted": extra}
            # PS3 event-level check + PS4 budget check
            evs = [e for e in net_w.spu_events
                   if e.get("skip") is None and "steps" in e]
            floor = DEFAULT_SPU_POLICY["spu_rho_floor"] / 2
            rw["ps3_events_ok"] = all(
                e["s_after"] >= floor * e["s_entry"] for e in evs)
            rw["ps3_final_ok"] = all(v > 1e-9
                                     for v in rw["inner_net_eval_std"].values())
            rw["ps4_budget_ok"] = all(
                e["steps"] <= DEFAULT_SPU_POLICY["spu_S_max"]
                for e in evs)
            rw["ps4_summaries_ok"] = (
                rep["processed_steps"]
                == sum(1 for e in net_w.spu_events
                       if e.get("path") == "__step__"))
            write_events_jsonl(OUT / f"events_{scene}_s{seed}.jsonl",
                               net_w.spu_events)
            rows.append({"with": rw, "without": ro})
        summary["scenes"][scene] = rows
    # verdicts
    def wins(scene, key, smaller_better=True):
        return sum(
            1 for r in summary["scenes"][scene]
            if (r["with"][key] < r["without"][key]) == smaller_better)
    noisy_rows = summary["scenes"]["noisy"]
    ps1 = wins("noisy", "mse_clean") >= 2
    ps2 = wins("noisy", "degradation") >= 2
    ps3 = all(r["with"]["ps3_final_ok"] and r["with"]["ps3_events_ok"]
              for sc in summary["scenes"].values() for r in sc)
    ps4 = (all(r["with"]["ps4_budget_ok"] and r["with"]["ps4_summaries_ok"]
               for r in noisy_rows)
           and all((r["with"]["wall_s"] / r["with"]["steps"])
                   <= 1.25 * (r["without"]["wall_s"] / r["without"]["steps"])
                   for r in noisy_rows))
    control_ps1 = wins("clean", "mse_clean") >= 2
    verdicts = {"PS1_value": bool(ps1), "PS2_robustness": bool(ps2),
                "PS3_no_collapse": bool(ps3), "PS4_mechanics": bool(ps4),
                "control_clean_PS1_bonus": bool(control_ps1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
