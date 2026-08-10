"""R2 driver — executes spec.md exactly (committed first).
Parallel template: 12 independent (scene, seed, arm) runs in a
process pool; per-run lines print as they complete."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.attention_body import AttentionBody   # noqa: E402
from reference_net.net import Network                    # noqa: E402

OUT = Path(__file__).parent / "results"
SEEDS = (7, 8, 9)
STEPS = 2500
MATCHED_HIDDEN = 46                                 # frozen in spec


def law_s16(X):
    keys = X[:, 8:16]
    w = np.exp(keys - keys.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (w * X[:, 0:8]).sum(1, keepdims=True)


def law_p16(X):
    return (np.sin(3 * X[:, :1]) + 0.6 * np.cos(5 * X[:, 1:2])
            + 0.3 * X[:, 2:3] * X[:, 3:4])


LAWS = {"S16": law_s16, "P16": law_p16}


def data(seed, law):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-2, 2, (1024, 16))
    ytr = law(Xtr)
    Xtr = np.where(rng.random(Xtr.shape) < 0.2, 0.0, Xtr)
    ytr = ytr + rng.normal(0, 0.1, ytr.shape)
    Xev = np.random.default_rng(seed + 1000).uniform(
        -2, 2, (2048, 16))
    return Xtr, ytr, Xev, law(Xev)


def run_one(scene, seed, body_type):
    law = LAWS[scene]
    Xtr, ytr, Xev, yev = data(seed, law)
    net = Network(16, 6000, lr=1e-2, seed=seed)
    e100 = None
    shaped = None
    entry_exact = True
    for step in range(STEPS):
        if step == 100:
            e100 = net.residual_energy()
        for at, j in ((500, 1), (900, 2)):
            if step == at:
                if at == 500:
                    shaped = bool(net.residual_energy()
                                  <= 0.40 * e100)
                before = net.predict(Xev)
                net.grow(j, hidden=MATCHED_HIDDEN,
                         body_type=body_type)
                entry_exact &= bool(np.array_equal(
                    net.predict(Xev), before))
        net.train_step(Xtr, ytr)
    yhat = net.predict(Xev)
    expected = (AttentionBody if body_type == "attention"
                else Network)
    exec_ok = all(type(b) is expected
                  for b in net.inner.values())
    if body_type == "attention":
        exec_ok &= any(e["event"] == "refine[attention]"
                       for e in net.gain_ledger)
    return {"scene": scene, "seed": seed, "body_type": body_type,
            "mse": float(((yhat - yev) ** 2).mean()),
            "entry_exact": entry_exact,
            "shaped": bool(shaped),
            "finite": bool(np.isfinite(yhat).all()),
            "exec_ok": bool(exec_ok),
            "scale_violations": len(getattr(net, "_scale_events",
                                            []))}


def main():
    OUT.mkdir(exist_ok=True)
    cells = [(sc, sd, bt) for sc in ("S16", "P16") for sd in SEEDS
             for bt in ("attention", "reference")]
    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, *c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"done {r['scene']} s{r['seed']} "
                  f"{r['body_type']}: mse={r['mse']:.5f} "
                  f"shaped={r['shaped']} "
                  f"viol={r['scale_violations']}", flush=True)
    idx = {(r["scene"], r["seed"], r["body_type"]): r
           for r in results}
    rows = []
    for sc in ("S16", "P16"):
        for sd in SEEDS:
            a, b = idx[(sc, sd, "attention")], \
                idx[(sc, sd, "reference")]
            rows.append({
                "scene": sc, "seed": sd,
                "mse_attention": a["mse"],
                "mse_reference": b["mse"],
                "a_wins": bool(a["mse"] < b["mse"]),
                "within_band": bool(a["mse"] <= 1.20 * b["mse"]),
                "ok": bool(a["entry_exact"] and b["entry_exact"]
                           and a["finite"] and b["finite"]
                           and a["exec_ok"] and b["exec_ok"]
                           and a["shaped"] and b["shaped"]
                           and a["scale_violations"] == 0
                           and b["scale_violations"] == 0)})
    s = [r for r in rows if r["scene"] == "S16"]
    p = [r for r in rows if r["scene"] == "P16"]
    void = not all(r["ok"] for r in rows)
    verdicts = {
        "RB1_selection_value": ("VOID" if void else
                                sum(r["a_wins"] for r in s) >= 2),
        "RB2_no_regression": ("VOID" if void else
                              sum(r["within_band"]
                                  for r in p) >= 2),
        "RB3_mechanics": all(r["ok"] for r in rows)}
    (OUT / "summary.json").write_text(json.dumps(
        {"rows": rows}, indent=1))
    (OUT / "verdicts.json").write_text(json.dumps(verdicts,
                                                  indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
