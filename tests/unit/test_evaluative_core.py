"""TB-C01..05 — shared Evaluative Update Core referee boxes
(docs 86 v1.23 §3.0/§6; plan 84 v2.13 D-2, RED before D-3).

These boxes referee the L0 OBJECT-AGNOSTIC kernel serving BOTH
loops (S-loop preference + P-loop trainers). Every expected value
below is HAND-DERIVED from the normative equations (T-AX4
worksheets in each docstring); the module under test
(reference_net.growthpolicy.evaluative_core) does not exist yet —
each box is RED via ImportError until D-3's core-first sub-step.

T-AX3 method table (>= 2 independent routes per core numeric):
  advantage      : hand literal (here) + replay identity (TP-01)
  clip envelope  : hand literal (here) + system algebra (TI-01)
  credit weights : hand literal (here) + dual-form (TB-C03 both
                   K-window and exponential forms from one utility)
  ema_fold       : hand literal (TB-01) + rebuild identity (TB-13)
  seeded_draw    : hand literal (here) + two-process replay (TP-01)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules" / "ReferenceNet"))


def _core():
    from reference_net.growthpolicy import evaluative_core
    return evaluative_core


def test_tb_c01_advantage_hand_computed():
    """TB-C01: advantage(realized, baseline) = elementwise
    realized - baseline (doc 86 §3.0 ADVANTAGE; plain arrays).
    WORKSHEET: realized [0.5, 0.2], baseline [0.3, 0.4]
      -> [0.5-0.3, 0.2-0.4] = [0.2, -0.2].
    """
    core = _core()
    out = core.advantage([0.5, 0.2], [0.3, 0.4])
    assert np.allclose(out, [0.2, -0.2], atol=1e-15)


def test_tb_c02_normalized_multiplier_reference_law():
    """TB-C02 (v1.17): normalized_multiplier(raw, mu, sd, lo, hi)
    = clip(exp((raw-mu)/sd), lo, hi) — the raw score compared
    against the OBSERVED ADVANTAGE DISTRIBUTION (magnitude-
    aware; per-value independent).
    WORKSHEET mu=0.2, sd=0.1:
      raw 0.25 -> z=+0.5 -> exp = 1.6487212707001  (graded!)
      raw 0.05 -> z=-1.5 -> exp = 0.2231301601484 -> clip 0.5
      raw 0.20 -> z= 0   -> 1.0
      raw 0.35 -> z=+1.5 -> 4.4816890703381 -> clip 2.0
    Magnitude-awareness referee: raws 0.201 vs 0.35 differ in
    output (0.201 -> exp(0.01)=1.01005) — small edges give
    small multipliers, unlike any set-based z."""
    core = _core()
    f = core.normalized_multiplier
    assert abs(f(0.25, 0.2, 0.1, 0.5, 2.0)
               - 1.6487212707001282) < 1e-12
    assert f(0.05, 0.2, 0.1, 0.5, 2.0) == 0.5
    assert f(0.20, 0.2, 0.1, 0.5, 2.0) == 1.0
    assert f(0.35, 0.2, 0.1, 0.5, 2.0) == 2.0
    assert abs(f(0.201, 0.2, 0.1, 0.5, 2.0)
               - 1.0100501670841679) < 1e-12


def test_tb_c02b_normalized_multiplier_tolerance_edge():
    """TB-C02 edge (v1.17 tolerance law — NEVER exact-zero float
    tests): sd <= 1e-9 * max(1, |mu|) => 1.0 exactly. Includes
    the float-dust case (sd ~ 1e-17 from equal-value summation)
    and the relative-tolerance case (|mu|=1e6, sd=1e-4)."""
    core = _core()
    f = core.normalized_multiplier
    assert f(0.9, 0.4, 0.0, 0.5, 2.0) == 1.0
    assert f(0.9, 0.4, 5.5e-17, 0.5, 2.0) == 1.0     # dust
    assert f(2e6, 1e6, 1e-4, 0.5, 2.0) == 1.0        # relative
    assert f(0.9, 0.4, 1e-3, 0.5, 2.0) == 2.0        # real sd


def test_tb_c03_credit_weights_and_folds_dual_form():
    """TB-C03: ONE weighting utility yields both credit-weight
    families (doc 86 §3.0 credit_weights), and credit_fold does
    both folds (normalize flag; L1 decides what gets folded).
    WORKSHEET (K-window, normalized — doc 83 §4.1 credited_gain):
      weights (1, .5, .25); gains [0.4, 0.2, 0.1]
      credited = (0.4 + 0.1 + 0.025)/1.75 = 0.525/1.75 = 0.3
    WORKSHEET (exponential/GAE form, unnormalized sum):
      base 0.855, n=4 -> [1, 0.855, 0.731025, 0.62502637...]
      values [0.5, -0.1, 0.2, 0.0]
      sum-fold = 0.5 - 0.0855 + 0.146205 + 0 = 0.560705
    """
    core = _core()
    w1 = core.credit_weights({"kind": "list",
                              "weights": [1, 0.5, 0.25]})
    assert np.allclose(w1, [1, 0.5, 0.25], atol=1e-15)
    assert abs(core.credit_fold([0.4, 0.2, 0.1], w1,
                                normalize=True) - 0.3) < 1e-12
    w2 = core.credit_weights({"kind": "exp", "base": 0.855,
                              "n": 4})
    assert np.allclose(
        w2, [1.0, 0.855, 0.731025, 0.62502637], atol=1e-7)
    assert abs(core.credit_fold([0.5, -0.1, 0.2, 0.0], w2,
                                normalize=False)
               - 0.560705) < 1e-9


def test_tb_c04_seeded_draw_determinism_and_worksheet():
    """TB-C04: seeded_draw(rng, mean, se) = mean +
    se * rng.standard_normal(); se <= 0 => mean exactly, rng NOT
    consumed (degenerate edge, doc 83 M4).
    WORKSHEET (numpy Generator stream, seed 12345; stream is
    version-stable by numpy guarantee):
      standard normals: -1.42382504, 1.26372846, -0.87066174
      draws m=0.2 se=0.15:
        [-0.0135737557, 0.3895592687, 0.0694007393]
    """
    core = _core()
    rng = np.random.default_rng(12345)
    got = [core.seeded_draw(rng, 0.2, 0.15) for _ in range(3)]
    assert np.allclose(
        got, [-0.0135737557, 0.3895592687, 0.0694007393],
        atol=1e-9)
    rng2 = np.random.default_rng(12345)
    a = core.seeded_draw(rng2, 0.4, 0.0)      # must not consume
    b = core.seeded_draw(rng2, 0.2, 0.15)
    assert a == 0.4
    assert abs(b - (-0.0135737557)) < 1e-9    # stream undisturbed


def test_tb_c03b_fold_stable_form_equivalence_and_stability():
    """TB-C03b (v1.17): the implemented West recursion is
    algebraically IDENTICAL to the sufficient-statistics
    semantics (referee: 200 random folds match < 1e-12), and
    remains accurate where the naive second-moment form
    catastrophically cancels (values ~1e8 with sd ~ 0.01)."""
    core = _core()
    rng = np.random.default_rng(90210)
    for _ in range(200):
        w, m, v = (float(rng.uniform(0, 9)),
                   float(rng.normal(0, 2)),
                   float(rng.uniform(0, 4)))
        x, g = float(rng.normal(0, 3)), float(rng.uniform(.6, 1))
        w2, m2, v2 = core.ema_fold((w, m, v), x, g)
        w2r = g * w + 1
        m2r = (g * w * m + x) / w2r
        v2r = max((g * w * (v + m * m) + x * x) / w2r
                  - m2r * m2r, 0.0)
        assert abs(w2 - w2r) < 1e-12
        assert abs(m2 - m2r) < 1e-12
        assert abs(v2 - v2r) < 1e-9 * max(1.0, v2r)
    # stability at |x| >> sd: exact variance of the 3-fold at
    # decay 1.0 over [B, B+0.01, B+0.02] is sd^2 of the values
    B = 1e8
    st = (0.0, 0.0, 0.0)
    for x in (B, B + 0.01, B + 0.02):
        st = core.ema_fold(st, x, 1.0)
    true_v = np.var([B, B + 0.01, B + 0.02])   # 6.66e-05
    assert st[2] > 0.0                          # NOT clamped-0
    assert abs(st[2] - true_v) < 1e-6


def test_tb_c05_l0_purity_static_import_check():
    """TB-C05: L0 PURITY (doc 86 §3.0 layering law, static):
    the core module's SOURCE imports no object types — no
    substrate/preference/context/trainer symbols; allowed imports
    only numpy/stdlib. Dependency direction L2 -> L1 -> L0 is
    asserted by this source scan (the reverse direction cannot
    exist if the source never names the upper layers)."""
    import ast
    src_path = (ROOT / "modules" / "ReferenceNet" / "reference_net"
                / "growthpolicy" / "evaluative_core.py")
    assert src_path.exists(), "L0 core module missing (RED)"
    tree = ast.parse(src_path.read_text())
    banned = {"net", "preference", "trainer", "interfaces",
              "combiner_threshold", "pricer_zero_attach",
              "growth_store", "sms", "torch"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            parts = set(n.split("."))
            assert not (parts & banned), (
                f"L0 core imports object-layer module: {n}")
