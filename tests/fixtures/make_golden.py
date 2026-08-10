"""Golden-fixture generator (DEV_PLAN S0).

Builds a seeded reference Network (one grown inner scope), records
(a) predict() outputs on 256 fixed inputs and (b) the loss sequence
of 50 train_step calls — the bit-identity references guarding the
zero-block code paths through the delta round. Regenerate ONLY by
explicit owner decision; the whole point is that these bytes do not
move.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.net import Network  # noqa: E402


def build_reference():
    rng = np.random.default_rng(20260706)
    X = rng.normal(size=(200, 3))
    y = (np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2
         - 0.3 * X[:, 2]).reshape(-1, 1)
    net = Network(d_in=3, hidden=8, lr=1e-2, seed=7)
    for _ in range(120):
        net.train_step(X, y)
    net.grow(2, hidden=4)
    for _ in range(80):
        net.train_step(X, y)
    return net, X, y


def main():
    # Growth Interface Reform re-pin (doc 36 W3(c)): grow() now
    # produces FULLWIDTH port bodies, so the grown-context golden
    # is re-captured under NEW *_gp1 names. The pre-reform files
    # (golden_predict.npz / golden_train.npz) are PRESERVED as
    # historical record (owner data-preservation rule) and no
    # longer read by tests. Correctness of the fullwidth path is
    # proven independently (Part B enlarged-net equivalence);
    # these bytes only guard future drift. WRITE-ONCE.
    net, X, y = build_reference()
    Xq = np.random.default_rng(99).normal(size=(256, 3))
    pred = net.predict(Xq)
    losses = np.array([net.train_step(X, y) for _ in range(50)])
    out = Path(__file__).parent
    for name in ("golden_predict_gp1.npz", "golden_train_gp1.npz"):
        if (out / name).exists():
            raise SystemExit(f"WRITE-ONCE refusal: {name} exists")
    np.savez(out / "golden_predict_gp1.npz", Xq=Xq, pred=pred)
    np.savez(out / "golden_train_gp1.npz", losses=losses)
    print("fixtures written:", pred.shape, losses.shape)


if __name__ == "__main__":
    main()
