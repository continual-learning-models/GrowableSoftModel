"""P0 baseline capture for the Base-Design Adjustment batch
(next-dev-docs/53 v1.2 step P0; 54 v1.10 section 2).

Captures, at tag growthport-v1 BEFORE any adjustment code, the
four write-once fixtures that anchor the batch's bitwise
behavior-preservation gates:

  FX-1  grown+deepened artifact + recorded predict/train
        outputs                          -> box V-A1-2
  FX-2  multi-body site scenario (ONE shared site, TWO bodies,
        4 recorded train steps — exposes the site-shared Adam
        counter, 52 SR-20/SR-24)         -> box V-A1-3
  FX-3  per-preset scenarios rho / delta / site-host with
        per-step output records          -> box V-A2-1
  FX-4  refused-growth state snapshots (duplicate site,
        port-type, guard-refuse): state hash must be untouched
        by a refusal                     -> box V-A2-4

WRITE-ONCE (owner data-preservation rule; T-0 pattern): the
script REFUSES to run if the output directory already contains
files. Re-capture requires an explicit NEW versioned directory
passed as argv[1] — never an in-place overwrite.
"""
import hashlib
import json
import pickle   # safe: locally produced organ states only
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                           # noqa: E402
from reference_net.net import Network                    # noqa: E402
from reference_net.growthpolicy import \
    DEFAULT_GROWTH_POLICY                                # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "tests" / "unit" / "fixtures" / "adjustment_baseline"

WARM = 120          # train steps before growth (guard-quiet)
REC = 4             # recorded post-growth steps


def _data(seed, n=32, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    Xe = rng.normal(size=(8, d))
    return X, y, Xe


def state_hash(net):
    """Canonical sha256 over the device-free state tree —
    arrays by exact bytes, scalars by repr, dict order by key.
    EXCLUDES the gain ledger: its record shape is the one
    DECLARED amendment of the batch (specs field, 52 SR-22,
    owner-approved), so it is verified by its own boxes
    (V-A2-2), never by this hash; everything numeric stays."""
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if str(k) == "gain_ledger":
                    continue
                h.update(str(k).encode()); walk(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"["); [walk(v) for v in o]; h.update(b"]")
        elif isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
            h.update(str(o.dtype).encode())
            h.update(str(o.shape).encode())
        elif hasattr(o, "__getstate__") and not isinstance(
                o, (str, bytes, int, float, bool, type(None))):
            walk(o.__getstate__())
        else:
            h.update(repr(o).encode())
    walk(net.__getstate__())
    return h.hexdigest()


def _run_recorded(net, X, y, Xe, steps):
    losses, preds = [], []
    for _ in range(steps):
        losses.append(float(net.train_step(X, y)))
        preds.append(np.asarray(net._bk.to_numpy(net.predict(Xe)),
                                dtype=np.float64))
    return np.array(losses, dtype=np.float64), np.stack(preds)


def fx1():
    X, y, Xe = _data(101)
    net = Network(3, 8, lr=1e-2, seed=7)
    for _ in range(WARM):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
        net.grow(1, hidden=4)
        net.deepen()
    losses, preds = _run_recorded(net, X, y, Xe, REC)
    np.savez(OUT / "fx1_records.npz", X=X, y=y, Xe=Xe,
             losses=losses, preds=preds,
             final_pred=preds[-1])
    (OUT / "fx1_artifact.pkl").write_bytes(pickle.dumps(net))
    return {"scenario": "grow(0,4)+grow(1,4)+deepen after "
            f"{WARM} warm steps, {REC} recorded steps",
            "state_hash": state_hash(net)}


def fx2():
    X, y, Xe = _data(202)
    net = Network(3, 8, lr=1e-2, seed=11)
    for _ in range(WARM):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
        net.grow(2, hidden=4)
    assert len(net._port_site.bodies) == 2, "need TWO bodies"
    (OUT / "fx2_pre_train.pkl").write_bytes(pickle.dumps(net))
    losses, preds = _run_recorded(net, X, y, Xe, REC)
    np.savez(OUT / "fx2_records.npz", X=X, y=y, Xe=Xe,
             losses=losses, preds=preds)
    return {"scenario": "ONE shared site, TWO bodies (j=0, j=2), "
            f"{REC} recorded steps (site-shared Adam counter)",
            "state_hash": state_hash(net)}


def fx3():
    out = {}
    # (a) rho
    X, y, Xe = _data(303)
    net = Network(3, 8, lr=1e-2, seed=21)
    for _ in range(WARM):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=5)
    losses, preds = _run_recorded(net, X, y, Xe, REC)
    np.savez(OUT / "fx3_rho.npz", X=X, y=y, Xe=Xe,
             losses=losses, preds=preds)
    out["rho"] = state_hash(net)
    # (b) delta
    X, y, Xe = _data(304)
    net = Network(3, 8, lr=1e-2, seed=22)
    for _ in range(WARM):
        net.train_step(X, y)
    net.deepen()
    losses, preds = _run_recorded(net, X, y, Xe, REC)
    np.savez(OUT / "fx3_delta.npz", X=X, y=y, Xe=Xe,
             losses=losses, preds=preds)
    out["delta"] = state_hash(net)
    # (c) site host (GA)
    X, y, Xe = _data(305)
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])
    for _ in range(10):
        ga.train_step(X, y)
    ga.grow_site("layer0/ffn[1]", hidden=6)
    losses, preds = [], []
    for _ in range(REC):
        losses.append(float(ga.train_step(X, y)))
        preds.append(np.asarray(ga.predict(Xe), dtype=np.float64))
    np.savez(OUT / "fx3_site.npz", X=X, y=y, Xe=Xe,
             losses=np.array(losses, dtype=np.float64),
             preds=np.stack(preds))
    (OUT / "fx3_site_artifact.pkl").write_bytes(pickle.dumps(ga))
    return out


def fx4():
    out = {}
    X, y, _ = _data(404)
    # (a) duplicate site
    net = Network(3, 8, lr=1e-2, seed=31)
    for _ in range(WARM):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
    (OUT / "fx4_dup_pre.pkl").write_bytes(pickle.dumps(net))
    pre = state_hash(net)
    try:
        net.grow(0, hidden=4)
        raise AssertionError("duplicate grow must refuse")
    except ValueError as e:
        msg = str(e)
    assert state_hash(net) == pre, "refusal mutated state"
    out["duplicate"] = {"state_hash": pre, "error_head": msg[:60]}
    # (b) port-type refusal
    net = Network(3, 8, lr=1e-2, seed=32)
    for _ in range(WARM):
        net.train_step(X, y)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              grow_port_type="legacy_scalar")
    (OUT / "fx4_ptype_pre.pkl").write_bytes(pickle.dumps(net))
    pre = state_hash(net)
    try:
        net.grow(0, hidden=4)
        raise AssertionError("legacy_scalar grow must refuse")
    except ValueError as e:
        msg = str(e)
    assert state_hash(net) == pre, "refusal mutated state"
    out["port_type"] = {"state_hash": pre, "error_head": msg[:60]}
    # (c) scale-guard refuse mode (fresh net: min_steps violated)
    net = Network(3, 8, lr=1e-2, seed=33)
    for _ in range(5):
        net.train_step(X, y)
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              grow_scale_guard="refuse")
    (OUT / "fx4_guard_pre.pkl").write_bytes(pickle.dumps(net))
    pre = state_hash(net)
    pre_seed = int(net._seed_counter)
    try:
        net.grow(0, hidden=4)
        raise AssertionError("guard refuse-mode must refuse")
    except ValueError as e:
        msg = str(e)
    # BASELINE FACT (captured, to be preserved bitwise): the
    # scale-guard refusal fires AFTER the seed-counter
    # increment (net.py grow: duplicate check -> seed += 1 ->
    # build body -> guard), so unlike the duplicate/port-type
    # refusals it leaves seed_counter incremented. The
    # invariant is the RECORDED post-refusal state, plus the
    # deterministic delta: exactly seed_counter + 1.
    post = state_hash(net)
    assert post != pre, "expected the documented seed bump"
    assert net._seed_counter == pre_seed + 1, \
        "delta must be seed_counter + 1 only"
    out["guard_refuse"] = {"state_hash_pre": pre,
                           "state_hash_post": post,
                           "post_seed_counter": pre_seed + 1,
                           "error_head": msg[:60]}
    return out


def main():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"REFUSED: {OUT} is not empty (write-once; "
                 "pass a NEW versioned directory as argv[1])")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": "growthport-v1", "purpose":
                "adjustment-batch baseline (53 P0 / 54 s2)",
                "fixtures": {}}
    manifest["fixtures"]["FX-1"] = fx1()
    manifest["fixtures"]["FX-2"] = fx2()
    manifest["fixtures"]["FX-3"] = fx3()
    manifest["fixtures"]["FX-4"] = fx4()
    files = {}
    for f in sorted(OUT.iterdir()):
        if f.name == "MANIFEST.json":
            continue
        files[f.name] = hashlib.sha256(
            f.read_bytes()).hexdigest()
    manifest["files"] = files
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    print(f"captured {len(files)} fixture files -> {OUT}")


if __name__ == "__main__":
    main()
