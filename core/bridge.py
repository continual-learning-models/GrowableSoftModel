"""Phase-1 artifact bridge (IWP2/S2.3): organ.npz -> MSOrgan, by
DISTILLATION.

Honest engineering note: exact weight-copy is impossible — the frozen
TinyMLP uses ReLU, the recursive substrate uses GELU (discovered at
implementation; plan's atol-1e-9 expectation revised to behavioral
equivalence, recorded here). The bridge therefore DISTILLS: a fresh
MSOrgan learns to match the source model's own predictions on the source
model's own training store — importing a model IS teaching (the system's
own semantics), robust to any internal architecture difference.

One-way, artifact-level (R-SYS3). Acceptance: behavioral equivalence on
the source store — numeric max|delta| <= 0.05 * y_sigma; categorical
label agreement >= 0.99.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core._modules import generator  # noqa: F401
from generator.nets import TinyMLP
from generator.data import featurize, read_jsonl

from core.substrate import MSOrgan


def _fit(organ, X, y, epochs, seed, bs=32):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            organ.train_step(X[order[i:i + bs]], y[order[i:i + bs]])


def import_phase1_artifact(weights_dir, epochs=300, seed=0):
    wdir = Path(weights_dir)
    tiny = TinyMLP.load(wdir)
    shape = json.loads((wdir / "shape.json").read_text())
    store = read_jsonl(wdir / "train_store.jsonl")
    feats = shape["features"]
    X = np.array([featurize(r["input"], feats) for r in store])

    organ = MSOrgan(len(feats), max(16, tiny.hidden_sizes[0]),
                    mode=shape["mode"], vocab=shape.get("vocab"), seed=seed)
    if shape["mode"] == "numeric":
        teacher = tiny.predict_value(X).reshape(-1, 1)
        _fit(organ, X, teacher, epochs, seed)
        delta = float(np.max(np.abs(organ.predict(X) - teacher)))
        ok = delta <= 0.05 * (float(tiny.y_sigma) + 1e-9)
        report = {"mode": "numeric", "max_delta": delta, "ok": bool(ok)}
    else:
        proba = tiny.predict_proba(X)
        teacher = np.array([shape["vocab"][i] for i in proba.argmax(1)])
        _fit(organ, X, teacher, epochs, seed)
        ours = np.array(organ.predict_label(X)[0])
        agree = float(np.mean(ours == teacher))
        report = {"mode": "categorical", "agreement": agree,
                  "ok": bool(agree >= 0.99)}
    return organ, shape, report
