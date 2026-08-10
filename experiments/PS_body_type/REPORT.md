# T4 exam report (verdicts verbatim; spec committed first)

## Verdicts — all three bars PASS
- BT1 (predicted-regime value): **PASS 3/3.** On the selection
  law (softmax pairing — the regime the theory predicted for
  attention), attention bodies beat param-matched reference
  bodies on every seed: 0.425/0.472, 0.329/1.049 (3.2x), 0.503/
  0.586, at 673 vs 673 grown parameters.
- BT2 (no regression): **PASS 3/3** within the pre-frozen 1.20
  band on the plain law: ratios 0.28 (attention much better),
  1.18, 1.03.
- BT3 (mechanics): **PASS** — host output bitwise unchanged at
  every grow event in every run (both halves of the exact-entry
  convention working); all outputs finite; every grown body of
  the scripted type with the refine[attention] ledger tag
  present.

## Reading
The theoretical analysis is now evidence-backed end to end: the
growth algebra is type-agnostic (mechanics), attention bodies pay
exactly where component-selection structure exists (BT1), and at
matched parameters they do not seriously regress on a plain law
(BT2) — on one seed they won there too. User guidance: the mode
is problem-dependent by design; select "attention" when local
laws plausibly contain input-dependent selection/pairing.

## Addendum (owner ruling, scale hierarchy)
Both arms grew bodies LARGER than the 49-param host — internally fair (equally oversized), so the relative conclusion stands, but the configuration violates the scale-hierarchy principle (DESIGN 4c: host >= 100x body AND >= 100k own params) and must not be read as usage guidance.

## OWNER RULING (2026-07-08): SCALE-VOID
The test scale was too small to be meaningful (49-param host; far below the 100k floor and the 100x hierarchy). ALL value verdicts in this report are VOID as usage evidence; only the mechanics results (exact entry, execution checks) retain standing. A scale-sound re-exam is required before any value conclusion.
