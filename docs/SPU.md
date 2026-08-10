# SPU — Self-Processing Units (usage guide)

## What it is

During evolution steps (teach/practice-style train_step calls),
each NEWBORN inner network — a node grown into a subnetwork by
ρ, inside its self-processing window
`[spu_warmup_steps, spu_newborn_steps)` of its own training
(default [100, 300): rough shape first, then polish; both ends
user-adjustable per problem) (paper
vocabulary: inner network, newborn) — runs a
small bounded solving loop on its own weights before
participating: up to `spu_S_max` plain-SGD steps (a
user-modifiable system coefficient; default 4) toward
FUNCTIONAL CONSISTENCY — stability of its output under internal
hidden-unit masking (J_inv), guarded against collapse by a
relative spread floor (J_disc). Serving (`predict`) never runs
any of this; the committed version stays a pure function of its
store. With no policy attached the class is bit-identical to the
released Network.

## Use

    from reference_net.spu.spu_network import install_spu_policy
    install_spu_policy(model, {"spu_enabled": True})  # any holder

Holders (v2.0 architecture): any reference-family root (Network,
MSOrgan/mlp hosts — they subclass Network) runs the pre-forward
walk from inside train_step itself; the attention hosts
(transformer/sequence) process their grown inner bodies at the
top of their training step against the previous training
forward's cached layer inputs (one-step-stale observation,
disclosed; predict never writes the cache). SPUNetwork remains as
a back-compat facade (set_spu_policy delegates to install). The
policy lives ONLY on the holder; children never re-walk.

    from engine.spu.spu_report import build_report
    report = build_report(model)                  # readiness field

### Perturbation scale (revised 2026-07-09/10)

spu_p_mask defaults to 0.10 (was 0.25 — on small subnets a
quarter-drop is amputation, not bounded perturbation). There is
deliberately NO node floor (owner ruling): the mechanism is
probabilistic and self-governing — small units receive
proportionally weaker perturbation signal (an empty draw
contributes the unperturbed response to the consistency mean;
across K draws a fully idle step is rare and harmless). A hard
floor would replace that probability with a deterministic
cliff.

## Guarantees (tested)

- spu off -> bit-identical to the released Network; predict is
  read-only always; probes never self-process; widen-only growth
  mode forces the pre-pass off.
- Loop bounded by spu_S_max; per-step update-norm cap spu_clip;
  Adam moments never touched; bit-replay deterministic.
- Anti-collapse: relative floor inside the objective; the gate
  remains the system-level line.

## Verdict record (PS battery, seeds 7/8/9, verbatim)

PS1 value PASS 3/3 (held-out MSE 2-3x better than a control
granted 1107 extra steps, on corrupted training data); PS2
ratio-metric FAIL 0/3 as pre-registered (absolute corrupted
error better 3/3 — the ratio penalizes the lower clean
denominator); PS3 no-collapse PASS (event-level + final-state);
PS4 FAIL on the wall-clock cost ceiling (+64% vs the declared
25%; budget/logging clauses PASS). Details:
experiments/PS_spu/REPORT.md.

## Body types (grow_body_type)

Growth-policy selectable (P8: user choice, never self-switched):

    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    DEFAULT_GROWTH_POLICY["grow_body_type"] = "attention"
    # or per call: net.grow(j, body_type="attention")

"reference" (default) grows the single-hidden-layer body;
"attention" grows a complete small transformer (sizes from the
grow_attention_* keys, validated loudly). Exact entry holds for
both. v1 boundaries (disclosed): structural operators and the
SPU do not yet apply to attention bodies (SPU shows the
body_type_unsupported counter); selector pricing remains
reference-calibrated. Exam verdicts (experiments/PS_body_type):
BT1 PASS 3/3 in the selection regime at matched parameters, BT2
no-regression PASS 3/3, BT3 mechanics PASS.

### SPU on attention bodies (T6)

Realized and user-available (the eligibility gate accepts
attention bodies; token-mask perturbation, seeded backprop, same
loop discipline). HONEST STATUS: the pre-registered exam FAILED
at default policy values (AT1 0/3; AT3 chronic hinge activation —
token masking is a stronger perturbation than node masking and
the defaults over-process). Until a calibration round passes,
prefer spu off on attention bodies or adjust
spu_eta/spu_p_mask/spu_gamma deliberately; verdicts:
experiments/PS_spu_attention/REPORT.md.

### Scale hierarchy (binding)

grow_min_host_ratio (100x) and grow_min_host_params (100,000):
a host grows inner bodies only when its OWN parameters clear both
the relative and the absolute bar — type-independent.
grow_scale_guard: "warn" (default; recorded exemption for
toy-scale tests) | "refuse" (production).

### Loop-carrying units (staging)

Units carrying a lambda loop block are DISCLOSED-skipped
(`loop_block_unsupported` in the skip ledger) until the stage-2
realization (forward-under-mask through the loop kernels — the
delta-chain precedent). Their un-looped siblings process
normally.


## Analytic objective (B1, attention-build S7)

`spu_objective` is an OPTIONAL policy key: `"mask"` (default; the
original masked-copies estimate, unchanged) or `"analytic"` — the
EXACT closed form of the mask objective's expectation under its
truncated mask law (leaf subnets only; with composition blocks the
loop falls back to the mask branch). Same thresholds carry over
(the returned J~ is (1-1/K)-scaled); zero sampling noise. Which
objective becomes the operating default is an evaluation decision
after the experiments (owner ruling).
