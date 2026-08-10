"""W0 golden capture (docs/system/36 v1.4 W0(b); 37 v1.3
Part A + 0b).

Captures the BEFORE side of the fixed-net before/after
comparison from the PRISTINE pre-reform tree: for each config
F1..F8 (all UNGROWN networks), records pinned inputs,
per-step losses (100 adam + 20 sgd), outputs at steps
{0, 50, 100}, and the final organ state (default-policy
pickle bytes) into tests/unit/fixtures/classical_goldens/.

WRITE-ONCE (owner data-preservation rule, doc 37 0b): the
script REFUSES to run if the output directory already
contains files. Re-capture requires an explicit NEW versioned
directory passed as argv[1] — never an in-place overwrite.
"""
import json
import pickle   # safe: locally produced organ states only
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from core.substrates import get_substrate                # noqa: E402
from reference_net.net import Network                    # noqa: E402

ADAM_STEPS, SGD_STEPS, SGD_LR = 100, 20, 0.01
OUT_STEPS = (0, 50, 100)


def _vec_data(seed, n, d):
    rng = np.random.default_rng(seed + 100)
    X = rng.normal(size=(n, d))
    # (n, 1) targets: the strictest host contract (reference
    # Network); every numeric host accepts this shape
    y = rng.normal(size=(n, 1))
    return X, y


def _cat_data(seed, n, d, vocab):
    rng = np.random.default_rng(seed + 200)
    X = rng.normal(size=(n, d))
    y = np.array([vocab[int(v)] for v in
                  rng.integers(0, len(vocab), n)])
    return X, y


def _seq_data(seed, n, T, vocab):
    rng = np.random.default_rng(seed + 300)
    X = np.zeros((n, T, 64))
    seq = rng.integers(3, 3 + max(2, len(vocab) - 1),
                       size=(n, T))
    for i in range(n):
        X[i, np.arange(T), seq[i]] = 1.0
    y = np.array([vocab[int(v)] for v in
                  rng.integers(0, len(vocab), n)])
    return X, y


def build_configs():
    ga = get_substrate("growable_attention")
    tr = get_substrate("transformer")
    mlp = get_substrate("mlp")
    v5 = [f"t{i}" for i in range(4)] + ["EOS"]
    v10 = [f"t{i}" for i in range(10)] + ["EOS"]
    return {
        "F1": dict(kind="numeric",
                   make=lambda: mlp(6, 12, mode="numeric",
                                    lr=1e-2, seed=0),
                   data=lambda: _vec_data(0, 16, 6)),
        "F2": dict(kind="numeric",
                   make=lambda: Network(8, 10, lr=1e-2, seed=1),
                   data=lambda: _vec_data(1, 16, 8)),
        "F3": dict(kind="numeric",
                   make=lambda: ga(6, 8, mode="numeric",
                                   lr=3e-3, seed=0, d_model=16,
                                   n_layers=2,
                                   heads_spec=[[1, 3], [2, 1]]),
                   data=lambda: _vec_data(2, 16, 6)),
        "F4": dict(kind="categorical",
                   make=lambda: ga(64, 8, mode="categorical",
                                   vocab=list(v5), lr=3e-3,
                                   seed=2, d_model=16,
                                   n_layers=1,
                                   heads_spec=[[2, 2]],
                                   causal=True, window=12),
                   data=lambda: _seq_data(2, 16, 10, v5)),
        "F5": dict(kind="categorical",
                   make=lambda: ga(64, 20, mode="categorical",
                                   vocab=list(v10), lr=3e-3,
                                   seed=3, d_model=64,
                                   n_layers=4,
                                   heads_spec=[[4, 4]] * 4,
                                   causal=True, window=64),
                   data=lambda: _seq_data(3, 16, 12, v10)),
        "F6": dict(kind="numeric",
                   make=lambda: tr(6, 10, mode="numeric",
                                   lr=1e-2, seed=4, d_model=32,
                                   n_layers=2, n_heads=2),
                   data=lambda: _vec_data(4, 16, 6)),
        "F7": dict(kind="categorical",
                   make=lambda: tr(6, 10, mode="categorical",
                                   vocab=list(v5), lr=1e-2,
                                   seed=5, d_model=32,
                                   n_layers=2, n_heads=2),
                   data=lambda: _cat_data(5, 16, 6, v5)),
        "F8": dict(kind="categorical", add_class_at=50,
                   make=lambda: mlp(6, 12, mode="categorical",
                                    vocab=list(v5), lr=1e-2,
                                    seed=6),
                   data=lambda: _cat_data(6, 16, 6, v5)),
    }


def outputs_of(m, kind, X):
    return (m.predict(X) if kind == "numeric"
            else m.predict_proba(X))


def run_config(name, spec):
    m = spec["make"]()
    X, y = spec["data"]()
    rec = {"X": X, "y_repr": np.asarray(y).astype("U16")}
    rec["out0"] = outputs_of(m, spec["kind"], X)
    losses = []
    for i in range(ADAM_STEPS):
        losses.append(float(m.train_step(X, y)))
        if spec.get("add_class_at") == i + 1:
            m.add_class("t_new")
        if (i + 1) in OUT_STEPS:
            rec[f"out{i + 1}"] = outputs_of(m, spec["kind"], X)
    rec["losses_adam"] = np.asarray(losses)
    sgd_losses = [float(m.train_step(X, y, sgd_lr=SGD_LR))
                  for _ in range(SGD_STEPS)]
    rec["losses_sgd"] = np.asarray(sgd_losses)
    rec["final_state"] = np.frombuffer(
        pickle.dumps(m), dtype=np.uint8)
    return rec


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else \
        "classical_goldens"
    out = REPO / "tests" / "unit" / "fixtures" / sub
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(
            f"WRITE-ONCE REFUSAL (doc 37 0b): {out} is not "
            "empty; re-capture requires a NEW versioned "
            "directory argument, never an overwrite.")
    manifest = {"adam_steps": ADAM_STEPS,
                "sgd_steps": SGD_STEPS, "sgd_lr": SGD_LR,
                "out_steps": list(OUT_STEPS), "configs": {}}
    for name, spec in build_configs().items():
        rec = run_config(name, spec)
        np.savez_compressed(out / f"{name}.npz", **rec)
        manifest["configs"][name] = {
            "kind": spec["kind"],
            "add_class_at": spec.get("add_class_at"),
            "loss0": float(rec["losses_adam"][0]),
            "loss_last": float(rec["losses_sgd"][-1])}
        print(name, "captured | loss0=%.6f -> sgd_last=%.6f"
              % (rec["losses_adam"][0], rec["losses_sgd"][-1]))
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1))
    print("goldens complete:", out)


if __name__ == "__main__":
    main()
