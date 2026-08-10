"""T6 exam driver — executes spec.md exactly (committed first)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.spu.spu_network import SPUNetwork      # noqa: E402
from engine.spu.spu_policy import DEFAULT_SPU_POLICY  # noqa: E402
from engine.spu.spu_report import build_report     # noqa: E402

OUT = Path(__file__).parent / "results"
SEEDS = (7, 8, 9)
STEPS = 850
FLOOR2 = DEFAULT_SPU_POLICY["spu_rho_floor"] / 2
S_MAX = DEFAULT_SPU_POLICY["spu_S_max"]


def law(X):
    keys = X[:, 3:6]
    w = np.exp(keys - keys.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (w * X[:, 0:3]).sum(1, keepdims=True)


def data(seed):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-2, 2, (128, 6))
    ytr = law(Xtr)
    Xtr = np.where(rng.random(Xtr.shape) < 0.2, 0.0, Xtr)
    ytr = ytr + rng.normal(0, 0.1, ytr.shape)
    Xev = np.random.default_rng(seed + 1000).uniform(-2, 2, (256, 6))
    return Xtr, ytr, Xev, law(Xev)


def run(seed, with_spu, extra=0):
    Xtr, ytr, Xev, yev = data(seed)
    net = SPUNetwork(6, 6, lr=1e-2, seed=seed)
    if with_spu:
        net.set_spu_policy({"spu_enabled": True})
    for step in range(STEPS + extra):
        if step == 150:
            net.grow(1, body_type="attention")
        if step == 300:
            net.grow(2, body_type="attention")
        net.train_step(Xtr, ytr)
    yhat = net.predict(Xev)
    return net, float(((yhat - yev) ** 2).mean()), yhat


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for seed in SEEDS:
        nw, mse_w, yh = run(seed, True)
        rep = build_report(nw)
        # denominator = the FULL model's step cost, the PS
        # battery's analytic_ratio convention (net.n_params()
        # includes the grown bodies, which the hand-off trains
        # every step). The first run hardcoded the bare root (49
        # params), over-granting the control ~28x and timing out
        # before any result existed — fixed to the house
        # convention, documented here.
        p_total = nw.n_params()
        p_body = nw.inner[1].n_params()
        pol = DEFAULT_SPU_POLICY
        r = (pol["spu_S_max"] * (pol["spu_K"] + 2) * p_body * 2
             + 2 * p_total) / (3 * p_total)
        extra = math.ceil(rep["processed_steps"] * r)
        _, mse_o, _ = run(seed, False, extra=extra)
        evs = [e for e in nw.spu_events
               if e.get("skip") is None and "steps" in e]
        att = [e for e in evs if e.get("body_type") == "attention"]
        b1 = sum(1 for e in nw.spu_events
                 if e.get("path") == "root/1"
                 and e.get("body_type") == "attention")
        b2 = sum(1 for e in nw.spu_events
                 if e.get("path") == "root/2"
                 and e.get("body_type") == "attention")
        rows.append({
            "seed": seed, "mse_with": mse_w,
            "mse_without_budget_fair": mse_o,
            "a_wins": bool(mse_w < mse_o),
            "extra_steps": extra, "ratio": r,
            "events_root1": b1, "events_root2": b2,
            "exec_ok": bool(b1 > 0 and b2 > 0
                            and len(att) == len(evs)),
            "p3_ok": bool(all(e["s_after"] >= FLOOR2 * e["s_entry"]
                              for e in evs)
                          and float(np.std(yh)) > 1e-9),
            "p4_ok": bool(all(e["steps"] <= S_MAX for e in evs)
                          and bool(np.isfinite(yh).all()))})
    void = not all(r_["exec_ok"] for r_ in rows)
    verdicts = {
        "AT1_value": ("VOID" if void
                      else sum(r_["a_wins"] for r_ in rows) >= 2),
        "AT3_no_collapse": all(r_["p3_ok"] for r_ in rows),
        "AT4_mechanics": all(r_["p4_ok"] and r_["exec_ok"]
                             for r_ in rows)}
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows}, indent=1))
    (OUT / "verdicts.json").write_text(json.dumps(verdicts,
                                                  indent=1))
    print(json.dumps(verdicts, indent=1))
    for r_ in rows:
        print(f"s{r_['seed']}: with={r_['mse_with']:.4f} "
              f"without+{r_['extra_steps']}steps="
              f"{r_['mse_without_budget_fair']:.4f} "
              f"win={r_['a_wins']} "
              f"events={r_['events_root1']}/{r_['events_root2']}")


if __name__ == "__main__":
    main()
