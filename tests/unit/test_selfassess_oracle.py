"""Selfassess W2 — ANALYTIC ORACLES (doc 27 7 layer 2, full
spec verbatim). Ground truth comes from mathematics computed
in-test, never from the component. A disagreement here is a
component defect BY DEFINITION; tolerances are frozen in doc
27 and must not be tuned."""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.selfassess import (                          # noqa: E402
    MDLCriterion, SliceLedger, V27_DEFAULTS)


def _cfg(**over):
    base = dict(V27_DEFAULTS)
    base.update(over)
    return base


class TestQA2EntropyConvergence:
    """iid categorical source, m=4, q=(0.4,0.3,0.2,0.1),
    n=100k, fixed seed; declared 3-sigma analytic band."""

    Q = np.array([0.4, 0.3, 0.2, 0.1])
    N = 100_000

    def _stream_ledger(self, pred):
        """Feed the ledger the codelengths of predictor
        `pred` on draws from Q; batches of 1000."""
        rng = np.random.default_rng(42)
        draws = rng.choice(4, size=self.N, p=self.Q)
        led = SliceLedger(hist_len=4)
        nats = -np.log(pred[draws])
        for i in range(0, self.N, 1000):
            chunk = nats[i:i + 1000]
            led.record("s", float(np.mean(chunk)), 1000)
        return led.close_cycle()["s"]

    def test_calibrated_predictor_converges_to_entropy(self):
        H = float(-np.sum(self.Q * np.log(self.Q)))
        var = float(np.sum(self.Q * np.log(self.Q) ** 2)
                    - H ** 2)
        band = 3.0 * math.sqrt(var / self.N)
        mean = self._stream_ledger(self.Q)
        assert abs(mean - H) <= band, (mean, H, band)

    def test_miscalibrated_converges_to_cross_entropy(self):
        Qp = np.array([0.25, 0.25, 0.25, 0.25])
        CE = float(-np.sum(self.Q * np.log(Qp)))
        var = float(np.sum(self.Q * np.log(Qp) ** 2)
                    - CE ** 2)
        band = max(3.0 * math.sqrt(var / self.N), 1e-9)
        mean = self._stream_ledger(Qp)
        assert abs(mean - CE) <= band, (mean, CE, band)
        # and the correct quantity is CROSS-entropy, not H
        H = float(-np.sum(self.Q * np.log(self.Q)))
        assert mean > H


# ---------------- QA3: Markov order selection ----------------

TABLE2 = {(0, 0): 0.9, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.4}
TABLE1 = {(0, 0): 0.8, (0, 1): 0.3, (1, 0): 0.8, (1, 1): 0.3}
# TABLE1: rows equal pairwise on the older symbol -> a true
# order-1 source expressed in the same order-2 machinery.


def _gen(table, n, seed):
    rng = np.random.default_rng(seed)
    x = [int(rng.random() < 0.5), int(rng.random() < 0.5)]
    for _ in range(n - 2):
        p1 = table[(x[-2], x[-1])]
        x.append(int(rng.random() < p1))
    return np.array(x)


def _fit_ce(x, order):
    """Laplace-smoothed plug-in conditional frequencies of the
    given order; in-sample cross-entropy over positions with a
    full context (t >= 2 for BOTH orders — same scoring set)."""
    counts = {}
    for t in range(2, len(x)):
        ctx = tuple(x[t - order:t])
        c = counts.setdefault(ctx, [1, 1])   # Laplace
        c[x[t]] += 1
    tot, n = 0.0, 0
    for t in range(2, len(x)):
        ctx = tuple(x[t - order:t])
        c = counts[ctx]
        tot -= math.log(c[x[t]] / (c[0] + c[1]))
        n += 1
    return tot / n, n


def _oracle_selects_order2(x):
    """Textbook two-part/BIC comparison computed independently
    of the component: L_j = n*CE_j + (k_j/2) ln n."""
    ce1, n = _fit_ce(x, 1)
    ce2, _ = _fit_ce(x, 2)
    L1 = n * ce1 + (2 / 2) * math.log(n)
    L2 = n * ce2 + (4 / 2) * math.log(n)
    return L2 < L1, ce1, ce2, n


def _component_accepts(ce1, ce2, n, cfg):
    """The component's acceptance on the same quantities:
    incumbent = order-1 (k=2), candidate = order-2 (k=4),
    added_params = 2."""
    r = MDLCriterion().score_trial(
        {"s": ce1}, {"s": ce2}, "s", added_params=2,
        n_positions=n, lifetime_n=n, cfg=cfg)
    return r["slice_ok"]


class TestQA3OrderSelection:
    def test_a_order2_data_selected(self):
        x = _gen(TABLE2, 50_000, seed=7)
        oracle, ce1, ce2, n = _oracle_selects_order2(x)
        assert oracle is True
        assert _component_accepts(
            ce1, ce2, n,
            _cfg(innovation_cost_form="half_log_n")) is True

    def test_b_order1_data_rejected(self):
        x = _gen(TABLE1, 50_000, seed=7)
        oracle, ce1, ce2, n = _oracle_selects_order2(x)
        assert oracle is False
        assert _component_accepts(
            ce1, ce2, n,
            _cfg(innovation_cost_form="half_log_n")) is False

    def test_c_n_sweep_agreement_both_sources(self):
        for table, seed in ((TABLE2, 3), (TABLE1, 3)):
            for n in (500, 2000, 10_000, 50_000):
                x = _gen(table, n, seed)
                oracle, ce1, ce2, neff = \
                    _oracle_selects_order2(x)
                ours = _component_accepts(
                    ce1, ce2, neff,
                    _cfg(innovation_cost_form="half_log_n"))
                assert ours == oracle, (table is TABLE2, n)

    def test_d_const_form_agrees_at_calibration_point(self):
        x = _gen(TABLE2, 50_000, seed=7)
        oracle, ce1, ce2, n = _oracle_selects_order2(x)
        c0 = 0.5 * math.log(n)          # calibration point
        ours = _component_accepts(
            ce1, ce2, n,
            _cfg(innovation_cost_form="const",
                 innovation_cost_per_param=c0))
        assert ours == oracle
        # and the same on the null source
        y = _gen(TABLE1, 50_000, seed=7)
        o2, d1, d2, m = _oracle_selects_order2(y)
        ours2 = _component_accepts(
            d1, d2, m,
            _cfg(innovation_cost_form="const",
                 innovation_cost_per_param=0.5 * math.log(m)))
        assert ours2 == o2 is False
