# Track-B D-P0 baseline record (2026-07-29)

Branch: feature/rl-trainer, cut from feature/growth-preference
@ ec9d75d (Track A D-1..D-4 closed + C-1..C-4 correction
rounds + audit-field fix; Track A L0 EvaluativeCore is the
shared core D-P2 reuses).

Baselines at branch point:
- lib full suite: 1138 passed (run of 2026-07-29, 406s)
- SMS full suite: 237 passed (pin 4959796 -> ec9d75d refresh)
- referee programs: verify_rl_math 22/22, verify_rl_vs_industry
  14/14, verify_quasistatic_full 16/16,
  verify_vs_authoritative_libs 9/9 ALL PASS (e71db02 report)

EXISTING-CODE FENCE (plan 84 v2.17 D-P0): Track B may touch
ONLY (i) the gate evaluation-stream dispatch seam and
(ii) core/facade.py rl.*/gate.eval_stream registration.
Everything else existing = off-limits; further need = STOP.

Blueprint of record (BLUEPRINT-PARITY GATE, D-P2):
scripts/verify_vs_authoritative_libs.py NumpyPPO/GRPO drivers
(9/9 vs SB3/gymnasium/MABWiser, zero tolerance, multi-seed
means) — port, don't re-derive.
