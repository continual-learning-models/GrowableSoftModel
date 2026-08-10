# T4 pre-registered exam — attention inner bodies (grow_body_type)

Committed BEFORE the driver. Evaluation only. FAIL branches
recorded verbatim; no bar renegotiated after the driver runs.

## Scene S (selection law — attention's PREDICTED regime)
- Law: y = sum_i softmax(x[3:6])_i * x[0:3]_i (keys = features
  4..6, values = features 1..3 — an attention computation by
  construction), d_in = 6, x ~ U(-2, 2).
- Corruption: feature dropout p=0.2 + label noise sigma=0.1;
  held-out eval 256 CLEAN samples, distinct stream.
- n=128, 850 steps, seeds 7/8/9, host hidden=6, lr=1e-2.
- Scripted growth: rho at root:1 @150 and root:2 @300.

## Scene P (plain law — no-regression clause)
- Law: y = sin(3*x1) + 0.6*cos(5*x2) + 0.3*x3*x4 (the house PS
  law), d_in = 4; same corruption/eval/steps/seeds/growth script.

## Arms (param-fair capacity, values FROZEN here)
- Arm A: grown bodies are ATTENTION at defaults
  (d_model=8, layers=1, heads=2, ffn=16).
- Arm B: grown bodies are REFERENCE with hidden matched to the
  attention body's n_params within +/-10%:
  * Scene S (d_in=6): attention = 673 params -> hidden = 84
    (673 params, +0.0%).
  * Scene P (d_in=4): attention = 641 params -> hidden = 107
    (643 params, +0.3%).
- Same steps, same data, same growth script; SPU off in both arms
  (single-variable exam: body type only).

## Bars
- BT1 (predicted-regime value): A beats B on held-out clean MSE
  in >= 2/3 seeds on Scene S.
- BT2 (no regression): on Scene P, A's MSE <= 1.20 x B's MSE in
  >= 2/3 seeds (band frozen now, before any run).
- BT3 (mechanics): host output bitwise unchanged at both grow
  events in every run; all outputs finite; execution check —
  every grown body is of the scripted type (type introspection +
  the refine[attention] ledger tag), else the scene is VOID.

## Pre-registered interpretations
- BT1 FAIL -> "attention bodies do not pay even in their
  predicted regime at matched parameters" — recorded verbatim;
  the mode remains user-selectable (mechanics stand on their
  own), with the finding governing the default recommendation.
- BT1 PASS + BT2 FAIL -> the mode pays only in-regime and costs
  out-of-regime: the user guidance must say so explicitly.
