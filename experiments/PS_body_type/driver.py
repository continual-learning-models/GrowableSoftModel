"""T4 driver — executes spec.md exactly (committed first)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.attention_body import AttentionBody   # noqa: E402
from reference_net.net import Network                    # noqa: E402

OUT = Path(__file__).parent / "results"
SEEDS = (7, 8, 9)
STEPS = 850
MATCH_HIDDEN = {"S": 84, "P": 107}                  # frozen in spec


def law_s(X):
    keys = X[:, 3:6]
    w = np.exp(keys - keys.max(1, keepdims=True))
    w = w / w.sum(1, keepdims=True)
    return (w * X[:, 0:3]).sum(1, keepdims=True)


def law_p(X):
    return (np.sin(3 * X[:, :1]) + 0.6 * np.cos(5 * X[:, 1:2])
            + 0.3 * X[:, 2:3] * X[:, 3:4])


SCENES = {"S": (law_s, 6), "P": (law_p, 4)}


def data(seed, law, d_in):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-2, 2, (128, d_in))
    ytr = law(Xtr)
    Xtr = np.where(rng.random(Xtr.shape) < 0.2, 0.0, Xtr)
    ytr = ytr + rng.normal(0, 0.1, ytr.shape)
    Xev = np.random.default_rng(seed + 1000).uniform(
        -2, 2, (256, d_in))
    return Xtr, ytr, Xev, law(Xev)


def run(scene, seed, body_type):
    law, d_in = SCENES[scene]
    Xtr, ytr, Xev, yev = data(seed, law, d_in)
    net = Network(d_in, 6, lr=1e-2, seed=seed)
    hidden = MATCH_HIDDEN[scene]
    entry_exact = True
    for step in range(STEPS):
        for at, j in ((150, 1), (300, 2)):
            if step == at:
                before = net.predict(Xev)
                net.grow(j, hidden=hidden, body_type=body_type)
                entry_exact &= bool(np.array_equal(
                    net.predict(Xev), before))
        net.train_step(Xtr, ytr)
    yhat = net.predict(Xev)
    expected = (AttentionBody if body_type == "attention"
                else Network)
    exec_ok = all(type(b) is expected for b in net.inner.values())
    if body_type == "attention":
        exec_ok &= any(e["event"] == "refine[attention]"
                       for e in net.gain_ledger)
    return {"mse": float(((yhat - yev) ** 2).mean()),
            "entry_exact": entry_exact,
            "finite": bool(np.isfinite(yhat).all()),
            "exec_ok": bool(exec_ok),
            "grown_params": int(sum(b.n_params()
                                    for b in net.inner.values()))}


def main():
    OUT.mkdir(exist_ok=True)
    out = {"scenes": {}}
    for scene in ("S", "P"):
        rows = []
        for seed in SEEDS:
            a = run(scene, seed, "attention")
            b = run(scene, seed, "reference")
            rows.append({"seed": seed,
                         "mse_attention": a["mse"],
                         "mse_reference": b["mse"],
                         "ratio_a_over_b": a["mse"] / b["mse"],
                         "a_wins": bool(a["mse"] < b["mse"]),
                         "within_band": bool(a["mse"]
                                             <= 1.20 * b["mse"]),
                         "entry_exact": a["entry_exact"]
                         and b["entry_exact"],
                         "finite": a["finite"] and b["finite"],
                         "exec_ok": a["exec_ok"] and b["exec_ok"],
                         "params_a": a["grown_params"],
                         "params_b": b["grown_params"]})
        out["scenes"][scene] = rows
    s, p = out["scenes"]["S"], out["scenes"]["P"]
    void = not all(r["exec_ok"] for r in s + p)
    verdicts = {
        "BT1_predicted_regime_value":
            ("VOID" if void else sum(r["a_wins"] for r in s) >= 2),
        "BT2_no_regression_band":
            ("VOID" if void else
             sum(r["within_band"] for r in p) >= 2),
        "BT3_mechanics": all(r["entry_exact"] and r["finite"]
                             and r["exec_ok"] for r in s + p)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    (OUT / "verdicts.json").write_text(json.dumps(verdicts,
                                                  indent=1))
    print(json.dumps(verdicts, indent=1))
    for scene in out["scenes"]:
        for r in out["scenes"][scene]:
            print(f"{scene} s{r['seed']}: A={r['mse_attention']:.4f}"
                  f" B={r['mse_reference']:.4f}"
                  f" ratio={r['ratio_a_over_b']:.2f}"
                  f" win={r['a_wins']}")


if __name__ == "__main__":
    main()
