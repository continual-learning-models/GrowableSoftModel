# T6 pre-registered exam — SPU value on attention bodies

Committed BEFORE the driver. Evaluation only; FAIL branches
verbatim; no renegotiation.

## Scene
- Law: y = sum_i softmax(x[3:6])_i * x[0:3]_i (the selection law
  — the attention bodies' own regime), d_in = 6.
- Corruption: feature dropout 0.2 + label noise 0.1; held-out
  eval 256 CLEAN samples, distinct stream.
- n=128, 850 steps, seeds 7/8/9, host hidden=6, lr=1e-2.
- Scripted growth: ATTENTION bodies (defaults) at root:1 @150 and
  root:2 @300.
- Arms: A = SPU on (defaults; attention spu_step live); B = SPU
  off, granted budget-fair EXTRA train steps — analytic ratio
  with the attention body's true parameter count:
  r = (S_max*(K+2)*p_body*2 + 2*p_root) / (3*p_root),
  extra = ceil(processed_steps * r).

## Bars
- AT1 (value): A beats B on held-out clean MSE in >= 2/3 seeds.
- AT3 (no collapse): no processed event ends with s_after below
  (rho_floor/2) x s_entry; final outputs non-constant.
- AT4 (mechanics + execution): zero budget overruns; outputs
  finite; execution check — processed events with
  body_type="attention" on BOTH bodies in their expected windows
  ([250,450) and [400,600) holder steps), else VOID.

## Pre-registered interpretation
AT1 FAIL -> "self-processing does not pay on attention bodies at
matched budget on this scene" — recorded verbatim with equal
prominence (the PT1 precedent shows this is a live possibility);
mechanics bars stand on their own.
