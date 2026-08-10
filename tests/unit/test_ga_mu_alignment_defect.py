"""mu/nu WIDEN-ALIGNMENT CONTRACT (doc 34 5).

History: these two tests were born as the DEFECT PROOF
(owner-ordered): they asserted the correct contract and
FAILED on the historical ravel-tail extension — confirmed on
the untouched pre-gabackend code (coordinate (1,0) read 12
instead of its own 11; the newborn column read [11,22,0,0]).
The owner then ruled FIX NOW (2026-07-21); head_widen uses
position-correct padding since, and these tests stand as the
permanent contract guard.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

from core.substrates.growable_attention import (            # noqa: E402
    GrowableAttentionSubstrate as GA, _HMATS)

D, DH, M_COLS = 4, 2, 1


def _organ_with_coordinate_coded_mu():
    """mu[Wq][r, c] = 10*r + c + 1 — every entry names its own
    coordinate, so any relocation is visible."""
    m = GA(4, 6, mode="numeric", lr=1e-2, seed=0, d_model=D,
           n_layers=1, heads_spec=[[DH]])
    HS = m.heads[0][0]
    pattern = (10.0 * np.arange(D)[:, None]
               + np.arange(DH)[None, :] + 1.0)      # (4, 2)
    HS.mu = {nm: (pattern.copy() if nm == "Wq"
                  else np.zeros_like(HS.mu[nm]))
             for nm in _HMATS}
    return m, HS, pattern


def test_defect_mu_stays_attached_to_coordinates():
    """CORRECT contract: after widen, mu[Wq][r, c] for every
    pre-existing coordinate (r, c < d_h_old) keeps its value."""
    m, HS, pattern = _organ_with_coordinate_coded_mu()
    m.head_widen(0, 0, m=M_COLS)
    got = np.asarray(HS.mu["Wq"])                   # (4, 3)
    assert got.shape == (D, DH + M_COLS)
    assert np.array_equal(got[:, :DH], pattern), (
        "old coordinates lost their own bookkeeping values:\n"
        f"expected first {DH} columns\n{pattern}\ngot\n"
        f"{got[:, :DH]}")


def test_defect_new_column_reads_zero_activity():
    """CORRECT contract: a just-born, never-updated column must
    read ZERO accumulated update activity."""
    m, HS, _ = _organ_with_coordinate_coded_mu()
    m.head_widen(0, 0, m=M_COLS)
    got = np.asarray(HS.mu["Wq"])
    new_col = got[:, DH:]
    assert not new_col.any(), (
        "the newborn column carries someone else's history: "
        f"{new_col.ravel()}")


def test_context_widen_itself_preserves_function():
    """CONTEXT (passes): the defect is bookkeeping-only — the
    widen operator's function preservation is intact."""
    m, HS, _ = _organ_with_coordinate_coded_mu()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    m.train_step(X, rng.normal(size=6))
    before = m.predict(X)
    m.head_widen(0, 0, m=M_COLS)
    assert np.abs(before - m.predict(X)).max() <= 1e-12
