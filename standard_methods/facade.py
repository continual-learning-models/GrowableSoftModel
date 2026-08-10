"""The standard-family facade: seven industry verbs.

Design: docs/DESIGN_STANDARD_METHODS.md v2.1. Reuses the released
substrates READ-ONLY as the model bodies (they are standard
architectures; "fixed" = growth is simply never called — no
evolution verb exists in this module). Storage:
trained_models/standard/<name>/ (override: STANDARD_MODELS_ROOT).
Every response carries ok/refusal, a path where relevant, and a
`hint` (friendliness charter).
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

_ARCH_PARAMS = {
    "transformer": ("n_layers", "d_model", "n_heads", "ffn_hidden",
                    "lr"),
    "mlp": ("hidden", "lr"),
}
_DEFAULTS = {"n_layers": 2, "d_model": 32, "n_heads": 2,
             "ffn_hidden": 32, "hidden": 32, "lr": 1e-2}
_CACHE = {}


def _ckey(name):
    """Cache key includes the backend identity (E2E report F3):
    switching the compute policy must never hand back a model
    living on another device."""
    from engine.backends import current_backend
    bk = current_backend()
    return (name, bk.name, getattr(bk, "device", "cpu"),
            str(getattr(bk, "dtype", "")))


def _root():
    return Path(os.environ.get("STANDARD_MODELS_ROOT",
                               _REPO / "trained_models" / "standard"))


def _mdir(name):
    return _root() / name


def _manifest(name):
    f = _mdir(name) / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else None


def _write_manifest(name, man):
    _mdir(name).mkdir(parents=True, exist_ok=True)
    (_mdir(name) / "manifest.json").write_text(
        json.dumps(man, indent=1))


def _persist(name, model):
    with open(_mdir(name) / "model.pkl", "wb") as f:
        pickle.dump(model, f)          # device-free (backend rule)


def _restore(name):
    k = _ckey(name)
    if k in _CACHE:
        return _CACHE[k]
    f = _mdir(name) / "model.pkl"
    if f.exists():
        with open(f, "rb") as f2:
            _CACHE[k] = pickle.load(f2)
        return _CACHE[k]
    return None


def _predict_np(model, X):
    """Every prediction output crosses the backend's to_numpy
    edge HERE (DESIGN_BACKEND 2b) — the standard facade's single
    choke point, so no verb can reintroduce the device-tensor/
    numpy mixing that broke mps (E2E report F1). Identity on the
    numpy judge."""
    bk = getattr(model, "_bk", None)
    p = model.predict(X)
    return bk.to_numpy(p) if bk is not None else p


def _rows_to_xy(examples):
    X = np.asarray([r["x"] for r in examples], float)
    ys = [r["y"] for r in examples]
    return X, ys


def create(name, arch=None, examples=None, mode="numeric",
           **standard_params):
    """Create a standard model. arch: "transformer"|"mlp"|None
    (None + examples -> the tool's data-form default rule; None
    alone -> "transformer"). mode: "numeric"|"categorical".
    standard_params: industry hyperparameters only."""
    if _manifest(name) is not None:
        return {"ok": True, "existing": True,
                "note": "name exists — loaded the existing model, "
                        "not newly created; use a new name for a "
                        "new model",
                "path": str(_mdir(name)),
                "hint": "standard_train to continue training it, "
                        "or create with a different name"}
    note = None
    if arch is None:
        if examples:
            from core.substrates.forms import detect_form
            form = detect_form(list(examples)) or "vector"
            arch = "mlp" if form == "vector" else None
            if arch is None:
                return {"refusal": f"data form '{form}' has no "
                        "standard-family architecture; the "
                        "softmodel family may support it"}
            note = (f"architecture auto-selected: '{arch}' because "
                    f"the detected data form is '{form}'")
        else:
            arch = "transformer"
            note = ("architecture defaulted to 'transformer' "
                    "(pass arch= or sample examples= to choose)")
    if arch not in _ARCH_PARAMS:
        return {"refusal": f"unknown arch {arch!r}; valid: "
                           f"{sorted(_ARCH_PARAMS)}"}
    if mode not in ("numeric", "categorical"):
        return {"refusal": f"unknown mode {mode!r}; valid: "
                           "numeric | categorical"}
    bad = [k for k in standard_params if k not in _ARCH_PARAMS[arch]]
    if bad:
        return {"refusal": f"unknown standard parameter(s) {bad}; "
                           f"valid for {arch}: "
                           f"{list(_ARCH_PARAMS[arch])}"}
    params = {k: standard_params.get(k, _DEFAULTS[k])
              for k in _ARCH_PARAMS[arch]}
    man = {"name": name, "method": "standard", "arch": arch,
           "mode": mode, "params": params, "trained_steps": 0}
    _write_manifest(name, man)
    out = {"ok": True, "created": True, "arch": arch, "mode": mode,
           "params": params, "path": str(_mdir(name)),
           "hint": "standard_train with your examples "
                   "[{'x': [...], 'y': ...}, ...] — the input "
                   "width shapes itself from your first batch"}
    if note:
        out["auto_selection"] = note
    return out


def _build(man, X, ys):
    d_in = X.shape[1]
    p = man["params"]
    if man["arch"] == "transformer":
        from core.substrates.transformer import TransformerSubstrate
        vocab = (sorted({str(y) for y in ys})
                 if man["mode"] == "categorical" else None)
        return TransformerSubstrate(
            d_in, p["ffn_hidden"], mode=man["mode"], vocab=vocab,
            lr=p["lr"], d_model=p["d_model"],
            n_layers=p["n_layers"], n_heads=p["n_heads"])
    from core.substrates.mlp import MLPSubstrate
    vocab = (sorted({str(y) for y in ys})
             if man["mode"] == "categorical" else None)
    return MLPSubstrate(d_in, p["hidden"], mode=man["mode"],
                        vocab=vocab, lr=p["lr"])


def train(name, examples, steps=200):
    """Plain supervised training (industry standard). Reports
    before/after training error and a held-out score."""
    man = _manifest(name)
    if man is None:
        return {"refusal": f"no standard model named {name!r}; "
                "standard_create it first",
                "hint": "standard_create(name)"}
    examples = list(examples)
    if len(examples) < 2:
        return {"refusal": "need at least 2 examples"}
    X, ys = _rows_to_xy(examples)
    model = _restore(name)
    if model is None:
        model = _build(man, X, ys)
    # deterministic 80/20 split for the held-out report
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    cut = max(1, n // 5) if n >= 10 else 0
    tr, ho = (idx[cut:], idx[:cut]) if cut else (idx, idx[:0])
    if man["mode"] == "numeric":
        ytr = np.asarray(ys, float).reshape(-1, 1)
        first = last = None
        for _ in range(int(steps)):
            m = model.train_step(X[tr], ytr[tr])
            first = m if first is None else first
            last = m
        ho_score = (float(((_predict_np(model, X[ho])
                            - ytr[ho]) ** 2).mean())
                    if cut else None)
        report = {"train_error_before": first,
                  "train_error_after": last,
                  "holdout_mse": ho_score}
    else:
        labels = [str(y) for y in ys]
        for lbl in sorted(set(labels)):
            if lbl not in model.vocab:
                model.add_class(lbl)
        ytr = np.asarray(labels, object)
        first = last = None
        for _ in range(int(steps)):
            m = model.train_step(X[tr], ytr[tr])
            first = m if first is None else first
            last = m
        if cut:
            pred, _ = model.predict_label(X[ho])
            acc = float(np.mean([p == t for p, t in
                                 zip(pred, ytr[ho])]))
        else:
            acc = None
        report = {"train_loss_before": first,
                  "train_loss_after": last,
                  "holdout_accuracy": acc}
    _CACHE[_ckey(name)] = model
    man["trained_steps"] += int(steps)
    _write_manifest(name, man)
    _persist(name, model)
    return {"ok": True, **report, "steps": int(steps),
            "total_steps": man["trained_steps"],
            "path": str(_mdir(name)),
            "hint": "standard_evaluate to score it, or "
                    "standard_infer to use it"}


def evaluate(name, examples):
    model = _restore(name)
    man = _manifest(name)
    if model is None or man is None:
        return {"refusal": f"no trained standard model {name!r}",
                "hint": "standard_create then standard_train"}
    X, ys = _rows_to_xy(list(examples))
    if man["mode"] == "numeric":
        y = np.asarray(ys, float).reshape(-1, 1)
        err = _predict_np(model, X) - y
        return {"ok": True, "mse": float((err ** 2).mean()),
                "mae": float(np.abs(err).mean()),
                "n": len(X), "hint": "standard_infer to use it"}
    pred, _ = model.predict_label(X)
    acc = float(np.mean([p == str(t) for p, t in zip(pred, ys)]))
    return {"ok": True, "accuracy": acc, "n": len(X),
            "hint": "standard_infer to use it"}


def infer(name, x):
    model = _restore(name)
    man = _manifest(name)
    if model is None or man is None:
        return {"refusal": f"no trained standard model {name!r}",
                "hint": "standard_create then standard_train"}
    X = np.asarray([x], float)
    if man["mode"] == "numeric":
        return {"ok": True,
                "prediction": float(_predict_np(model, X)[0, 0])}
    labels, conf = model.predict_label(X)
    return {"ok": True, "prediction": labels[0],
            "confidence": float(conf[0])}


def save(name):
    model = _restore(name)
    if model is None:
        return {"refusal": f"no trained standard model {name!r}"}
    _persist(name, model)
    return {"ok": True, "path": str(_mdir(name)),
            "hint": "standard_load restores it anytime"}


def load(name):
    _CACHE.pop(_ckey(name), None)
    model = _restore(name)
    man = _manifest(name)
    if man is None:
        return {"refusal": f"no standard model named {name!r}"}
    return {"ok": True, "loaded": model is not None, **man,
            "path": str(_mdir(name)),
            "hint": "standard_infer to use it"}


def list_models():
    root = _root()
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if (d / "manifest.json").exists():
                man = json.loads((d / "manifest.json").read_text())
                out.append({"name": man["name"], "arch": man["arch"],
                            "mode": man["mode"],
                            "trained_steps": man["trained_steps"],
                            "path": str(d)})
    return {"ok": True, "models": out,
            "hint": "standard_create to add one" if not out
            else "standard_infer/standard_train by name"}
