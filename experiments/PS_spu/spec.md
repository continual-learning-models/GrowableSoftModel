# PS — SPU capability battery (pre-registered)

Committed BEFORE the driver runs (house rule). Bars are BINDING;
FAIL branches are applied verbatim in REPORT.md.

## Scenes (hidden-law protocol, seeded, CPU)

- NOISY (the mechanism's habitat), d_in=4:
      y = sin(3 x1) + 0.6 cos(5 x2) + 0.3 x3 x4
  Training set (n=128 per seed): inputs with FEATURE DROPOUT
  (each coordinate zeroed independently, p=0.2) + label noise
  sigma=0.1. Held-out eval set (n=256): clean inputs, true law.
- CLEAN control: same law, no corruption anywhere.
- Seeds: 7, 8, 9.

## Protocol (identical for both arms)

SPUNetwork(d_in=4, hidden=6, seed), full-batch train_step on the
(corrupted) training set; fixed growth script: rho at node 1 at
step 150, rho at node 2 at step 300 (leaf hidden=5); 700 steps
total. WITH arm: spu policy defaults with spu_enabled=True from
step 0 (juvenile window 200 covers the newborns). WITHOUT arm:
spu off, plus EXTRA plain steps for budget fairness.

## Budget fairness (analytic, deterministic)

extra_steps = ceil(P * r), where P = number of processed steps
in the WITH arm and r = spu cost per processed step as a fraction
of one global step, computed from parameter counts as FLOP
proxies:
    r = (S_max (K+2) p_leaf + 2 p_root) / (3 p_root)
(leaf loop: S_max iterations of K masked forwards + ~2
forward-equivalents of backward on the leaf; 2 root predicts for
the interference record; a global step ~ 3 root-forward
equivalents). Wall-clock is reported as observation only.

## Bars (binding)

- PS1 (value): on NOISY, WITH beats WITHOUT on held-out clean
  MSE in >= 2/3 seeds.
- PS2 (robustness): on NOISY, under eval-time feature dropout
  (p=0.2, seeded), the degradation ratio
  mse_corrupted / mse_clean is smaller for WITH in >= 2/3 seeds.
- PS3 (no collapse): across all WITH runs, no recorded event ends
  with s below (rho_floor/2) x s_entry; and on the eval batch no
  final model has a constant-output leaf (batch std <= 1e-9).
- PS4 (mechanics): zero budget overruns (every event's steps <=
  S_max); every processed step has its summary event; measured
  wall-clock overhead of WITH <= 25% per step at defaults.
- CONTROL reading (pre-registered interpretation): PS1 is NOT
  expected to pass on CLEAN; the asymmetry (pays under
  corruption, neutral when clean) is the mechanism pricing
  robustness. A CLEAN pass is a bonus, not a bar; a CLEAN fail
  is not a failure of the battery.

## Interpretation of failure (fixed now)

If PS1 fails: the local process objective does not convert
functional robustness into held-out value at this scale — an
honest NOT-MET record; the mechanics (PS3/PS4) stand on their
own. If PS2 fails: the invariance trained at the leaves does not
transfer to input-level corruption — recorded as the finding.
No bar is renegotiated after the driver runs.
