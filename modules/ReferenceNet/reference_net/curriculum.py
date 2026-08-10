"""Pilot specialization curriculum (PLAN Part I S6, R2, R8) — v2.

ONE discipline law, fixed forever (no contradictory labels); stages are
NESTED INPUT REGIONS of increasing extent and structural difficulty —
literally "new material contains the old" (S1): every later stage\'s region
contains every earlier one. In small regions the law is dominated by its
linear part; in larger regions the product and conditional-composition
terms dominate — the same law demands ever more capacity, like arithmetic
from single digits to large numbers.

    LAW(a,b,c) = a*b + max(b-c, 0)*a + 2*c

Stage regions (a,b,c upper bounds), nested: (1,1,2) < (2,2,3) < (4,4,4)
< (6,6,4). Eval suites measure each stage\'s FRONTIER (points outside the
previous region) so stage-k accuracy reflects the new difficulty, while
earlier suites measure retention. Three-way disjoint study/practice/eval
splits inside every stage. Exact verifier; seeded; deterministic.
"""
from __future__ import annotations

import numpy as np

TOL = 0.5


def law(X):
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    return a * b + np.maximum(b - c, 0.0) * a + 2 * c


# Nested regions; difficulty ESCALATES so that growth is forced by problem
# complexity (owner's principle), never by an artificially small student.
BOUNDS = [(1, 1, 2), (2, 2, 3), (4, 4, 4), (6, 6, 4),
          (9, 9, 6), (12, 12, 8)]
SKILLS = [{"linear"},
          {"linear", "product"},
          {"linear", "product", "conditional"},
          {"linear", "product", "conditional", "composition"},
          {"linear", "product", "conditional", "composition", "magnitude2"},
          {"linear", "product", "conditional", "composition", "magnitude2",
           "magnitude3"}]
NAMES = ["stage1_small", "stage2_medium", "stage3_large", "stage4_frontier",
         "stage5_extended", "stage6_deep"]


def containment_ok() -> bool:
    """R8: nested regions + nested skill declarations."""
    for k in range(1, len(BOUNDS)):
        if not all(BOUNDS[k][i] >= BOUNDS[k - 1][i] for i in range(3)):
            return False
        if not SKILLS[k - 1] < SKILLS[k]:
            return False
    return True


def _inside(X, bound):
    return np.all(X <= np.array(bound), axis=1)


def make_splits(seed: int = 20260702, n_study=400, n_practice=100, n_eval=80):
    rng = np.random.default_rng(seed)
    out = []
    for k, bound in enumerate(BOUNDS):
        n = n_study + n_practice + n_eval
        X = rng.uniform(0, 1, (n * 3, 3)) * np.array(bound)
        if k > 0:  # eval/study emphasize the frontier: keep points outside
            frontier = ~_inside(X, BOUNDS[k - 1])
            X = np.vstack([X[frontier], X[~frontier][: n // 3]])
        rng.shuffle(X)
        X = X[:n]
        y = law(X).reshape(-1, 1)
        out.append({
            "name": NAMES[k], "skills": SKILLS[k], "bound": bound,
            "X": {"study": X[:n_study],
                  "practice": X[n_study:n_study + n_practice],
                  "eval": X[n_study + n_practice:]},
            "y": {"study": y[:n_study],
                  "practice": y[n_study:n_study + n_practice],
                  "eval": y[n_study + n_practice:]},
        })
    return out


def accuracy(pred, truth, tol=TOL) -> float:
    # tol: numeric-match tolerance (param-interface batch S3;
    # MIRROR of the lifecycle gate_tol — one rule, two surfaces)
    return float(np.mean(np.abs(pred - truth) <= tol))
