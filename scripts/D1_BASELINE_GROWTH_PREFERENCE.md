# D-1 Baseline Record — Unified RL Capability, Track A
2026-07-28 · plan doc 84 v2.13 D-1 · branch feature/growth-preference

BRANCH POINT: depthgrowth-v1.7 (annotated tag) = main = db7acc2
(verified identical; doc 74's S-5 moved the tag to the fixed head,
so tag and main coincide — no deviation from the plan's "branch
from the depthgrowth-v1.7 tag").

BASELINE SUITE (before any edit, this tree):
  python 3.14 (homebrew) / pytest 9.1.1 / numpy 2.4.6 (unit-suite
  runner; tests self-insert paths, no venv dependency)
  RESULT: 1102 passed, 0 failed (421.8s)

KNOWN SUITE SIDE-EFFECT (recorded): the full suite appends to the
TRACKED log tests/logs/simulation_adaptive.jsonl (simulation layer).
Restored via git checkout after the baseline run; any tree-clean
assertion (SMS t30c) must run after this restore. This is
pre-existing behavior, not a change of this branch.

TI-02 BASELINE OBLIGATION: the byte-identity scripted-life capture
is authored with the TI-02 box at D-2 and MUST run against code
with zero implementation edits (this commit's tree state) before
D-3 begins. The gate for D-3 includes that capture existing.

ENVIRONMENT NOTE (0728 working tree): SMS .venv rebuilt fresh
(python 3.14.3, ENV_LOCK pins incl. torch 2.13.0/numpy 2.5.1,
editable installs → THIS tree). The copied venv inherited stale
absolute paths (shebang → 20260712 tree, editable finder →
20260723 tree); the 20260712 venv touched during diagnosis was
restored to its own self-consistent state (verified). Stale
__pycache__ from the copy purged in both repos.
