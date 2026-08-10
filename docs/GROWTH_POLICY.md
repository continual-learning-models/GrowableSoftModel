# Growth in Two Directions: the Growth-Direction Policy

This guide covers the delta operator (scope deepening) and the
signal-based selection between the two growth directions — widen
(rho/omega, term addition) and deepen (composition blocks). Design
rationale and literature grounding are recorded in the
development repository's design notes (not shipped here).

## Concepts

- **Scope**: the root network or any inner network in the inclusion
  tree. Every scope can grow in two directions.
- **Widen** (existing): add units — refine a node (rho) or widen the
  layer (omega). Term addition; gains are typically front-loaded
  (Jones 1992; Barron 1993).
- **Deepen** (delta): append a zero-initialized residual composition
  block to the scope's body:
  `H <- H + gelu(H @ Bin.T + bb) @ Bout.T`, `Bout = 0` at creation.
  Exact at application (bitwise), removable, repeatable (unbounded
  layer depth inside the scope), and it never touches anything
  downstream. Compositional gains are typically back-loaded.
- **Three tiers**: predict first (history signals gated by a
  predictability certificate), probe to verify (zero-attach probes
  whose curves are read through the extrapolator's asymptote — never
  as raw scores; short-horizon bias, Wu et al. 2018), the gate
  adopts (factory wiring, next round).
- **Parts**: every algorithmic role is a replaceable part behind a
  registry — extrapolator, forecastability, changepoint, backtest,
  pricer, combiner. Selection is an explicit operator act via the
  policy dict; the machinery never switches parts on its own.

## Configuration

`reference_net.growthpolicy.DEFAULT_GROWTH_POLICY` is the single edit
point. Field-by-field:

| key | default | meaning (algorithm it parameterizes) |
|---|---|---|
| extrapolator | domhan2015 | curve-family asymptote extrapolation (Domhan et al. 2015) |
| forecastability | spectral_entropy | structure score of the energy series (ForeCA, Goerg 2013) |
| changepoint | bocpd | regime validity (Adams & MacKay 2007) |
| backtest | rolling_origin | forecaster skill gate (Tashman 2000) |
| pricer | zero_attach_v1 | two-arm probe curves (same units) |
| combiner | threshold_policy | the decision flow itself |
| seed | 0 | bootstrap / decision determinism |
| forecastability_min | 0.25 | pass bar (power-law curves ~0.32; noise < 0.2) |
| changepoint_max | 0.2 | max recent-changepoint probability |
| backtest_max_err | 0.25 | max relative extrapolation error |
| max_rel_ci_width | 0.5 | asymptote-CI confidence bar |
| stall_k / stall_eps | 3 / 0.05 | ledger stall rule (last k gains < eps) |
| saturation_norm | 1e-4 | median update-norm floor (class exhaustion) |
| energy_floor | 1e-3 | residual energy that still counts as "high" |
| min_energy_points | 64 | certificate minimum data |
| min_ledger_events | 3 | Tier-1 minimum growth history |
| min_window_rows | 64 | probe minimum training material |
| probe_steps / probe_lr | 300 / 0.05 | probe budget |
| refine_hidden | 8 | rho inner width when widen applies |
| max_blocks | 8 | per-scope deepen budget (refusals logged) |
| grow_body_type | "reference" | type of a node's grown inner body: `reference` or `attention` (a small transformer) |
| grow_attention_d_model / _layers / _heads / _ffn | 8 / 1 / 2 / 16 | attention-body sizes (used only when `grow_body_type="attention"`; validated loudly) |
| grow_min_host_ratio | 100.0 | scale guard: host's own params must be >= this x the grown body's params |
| grow_min_host_params | 100000 | scale guard: absolute floor — a host below it does not grow inner bodies |
| grow_min_host_steps | 100 | scale guard: grow only after the base has trained into shape (slow, small topology change) |
| grow_scale_guard | "warn" | scale-guard mode: `warn` (proceed + record) or `refuse` (raise) |
| loop_enabled | False | master switch for the LOOP (lambda) operator — grown, governed directed cycles; OFF = the default world carries no cycles |
| loop_m | 8 | lambda width |
| loop_K_max | 32 | Picard iteration budget (also bounds inference cost on looped scopes) |
| loop_tol | 1e-6 | convergence tolerance |
| loop_rho_max | 0.6 | contraction cap, enforced by post-step projection (audited counter). Memory note: training a looped scope caches up to K_max z-iterates, O(k*n*H) — ~1.5 GB on a 100k-param scope with n=1024; keep loops on small load-bearing scopes |
| eta_target | 0.5 | target learning-progress rate for the controller |
| instrument_min_len | 64 | minimum instrument history before decisions |
| bocpd_recent | 32 | recent-window length for the BOCPD drift trigger |
| growth_mode | "adaptive" | "adaptive" or "widen_only" (see the growth-mode section below) |
| grow_port_type | "fullwidth" | inner-body port form (load-only key; `grow` refuses it loudly) |
| grow_body_out_width | 1 | grown inner-body output width |
| log_path | None | JSONL decision log (set to enable) |

Override per call: `decide(scope, {"probe_steps": 100})`.

Self-processing (SPU) has its own separate policy dict
(`spu_enabled` and the `spu_*` knobs) installed per model —
off by default; see `docs/SPU.md`. The compute backend/device
(CPU or GPU) is a separate policy too (`set_compute_policy`);
see `docs/COLAB_RUNBOOK.md`.

## System control: growth_mode

One system-level switch selects the growth regime — callable by
the user today and by the operating LLM once the factory surface
exposes it (next round):

```python
from reference_net import growthpolicy as gp
gp.get_growth_mode()                           # "adaptive"
gp.set_growth_mode(gp.GROWTH_MODE_WIDEN_ONLY)  # returns {old, new}
gp.set_growth_mode(gp.GROWTH_MODE_ADAPTIVE)
```

- `GROWTH_MODE_ADAPTIVE` (default): the full two-direction
  machinery described in this guide.
- `GROWTH_MODE_WIDEN_ONLY`: the previous pure-linear regime — the
  machinery never calls deepen, no certificate or probes run (the
  decision short-circuits before any part is assembled, at zero
  extra cost), and siting is the legacy u_j rule. Models governed
  under this mode stay in the pure linear-superposition class.
- Enforcement sits in `decide()` itself, above the replaceable
  parts, so no combiner swap can bypass it; a per-call override
  (`decide(net, {"growth_mode": ...})`) never touches the global.
- The active mode travels in every decision record's
  `policy_snapshot`; direct operator calls (`net.deepen()`)
  remain legal in every mode.

## Selecting and discovering parts

```python
from reference_net import growthpolicy as gp
gp.list_available("extrapolator")      # -> ['domhan2015']
gp.decide(net, {"forecastability": "my_part"})
```
An unknown name returns a refusal dict naming the available parts —
never an exception. v1 ships exactly one implementation per role;
the machinery never self-switches (constitution P8).

## Reading a decision record

Every `decide()` returns (and, with `log_path`, appends) one record:

- `arm` (widen | deepen), `site` (node index for rho; None for
  omega/deepen), `apply_as` (rho | omega | None),
- `tier_used`: cold_start_default | tier1 | tier2 | tier2_refused,
- `reasons`: ordered human-readable justification of every branch
  taken,
- `certificate`: the four checks' full outputs (forecastability
  score, changepoint probability/location, backtest error,
  extrapolation fit with asymptote/CI/p_useful),
- `signals`: stall flag, saturation flag, recent gains, median
  update norm (Tier 1 only),
- `prices`: both probe curves' extrapolations (Tier 2 only),
- `parts`, `policy_snapshot` (the seed rides inside the
  snapshot): full audit context,
- `gate_verdict`: null this round; the factory gate fills it when
  wired (next round).

## Reading the gain ledger

`net.gain_ledger` — one record per structural event:
`{event: refine|deepen|prune_block, site, params_added, E_before,
E_after, gain, step, due}`. `gain` is the RELATIVE residual-energy
reduction at the fixed horizon `net.gain_horizon` (default 200
steps): comparable across events by construction; asymptote
questions belong to the extrapolator, not the ledger. The stall
rule reads the last `stall_k` gains.

## Worked example

```python
import numpy as np
from reference_net.net import Network
from reference_net import growthpolicy as gp

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 2))
y = np.sin(2 * X[:, :1])

net = Network(d_in=2, hidden=4, lr=1e-2, seed=1)
for _ in range(50):
    net.train_step(X, y)

decision = gp.grow_with_policy(net, {"log_path": "decisions.jsonl"})
print(decision["arm"], decision["tier_used"], decision["applied"])
print(decision["reasons"])
for _ in range(250):                    # past the gain horizon
    net.train_step(X, y)
print(net.gain_ledger[-1])              # the event's realized gain
```

## FAQ

**When does it deepen?** When the history certifies class
exhaustion (stall + saturation with a confident extrapolation), or
when the deepen probe's extrapolated asymptote beats the widen
probe's outside CI overlap. Cold starts and ties default to widen
(additive default, constitution P1).

**How do I force a direction?** Policy is the lever: e.g.
`{"min_energy_points": 10**9}` routes every decision through the
probes; applying an operator directly (`net.deepen()`,
`net.grow(j)`) is always available and is ledgered either way.

**How do I add a new part?** Implement the role's contract
(`growthpolicy/interfaces.py`), `register(role, name, cls)` at
import, select it by name in policy. Contracts are validated
against one named alternative per role (DESIGN section 6), so a
conforming part drops in without pipeline changes.

**Does a decision change the model?** `decide()` never mutates the
scope (fingerprint-asserted, probes run on copies).
`grow_with_policy()` applies rho/deepen; omega application and
gate adjudication are factory wiring (next round).
