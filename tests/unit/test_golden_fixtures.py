"""Bit-identity guards for the zero-block code paths (DEV_PLAN S0).

W3 re-pin (doc 36 W3(c)): grow() is fullwidth now; the grown-context
golden reads the *_gp1 capture. Pre-reform files preserved unread.

These must pass unchanged at EVERY step of the delta round: the
instrumentation (S1) and the train_step rewrite (S3) may not move a
single bit on networks that have no composition blocks.
"""
from pathlib import Path

import numpy as np

from tests.fixtures.make_golden import build_reference

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_golden_predict_bit_identity():
    net, _, _ = build_reference()
    ref = np.load(FIX / "golden_predict_gp1.npz")
    pred = net.predict(ref["Xq"])
    assert pred.shape == ref["pred"].shape
    assert np.array_equal(pred, ref["pred"])


def test_golden_train_bit_identity():
    net, X, y = build_reference()
    ref = np.load(FIX / "golden_train_gp1.npz")["losses"]
    losses = np.array([net.train_step(X, y) for _ in range(50)])
    assert np.array_equal(losses, ref)
