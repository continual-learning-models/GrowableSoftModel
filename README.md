# Growable Soft Model

An evolvable domain extension for a generative-AI brain. A **soft
model** is a small neural model that a general LLM both **uses**
(`infer`) and **teaches** (`teach`): it shapes itself from data,
learns under a reality gate that only ever promotes improvements,
and **grows its own topology** — wider, deeper, new input features,
or re-founded — while every scale stays trainable for life. No
foundation model inside; the calling LLM supplies general capability.

Two method families, one tool:

- **softmodel** (the tool's method, default): everything below —
  gated teach/use/adapt, topology growth, self-processing.
- **standard mode** (optional secondary): a fixed-architecture
  standard transformer/MLP trained from scratch on your data —
  `standard_*` tools/verbs; see docs/STANDARD_METHODS_USAGE.md.

Three channels: MCP (`claude mcp add growable -- python3 -m
mcp.mcp_server`; the primary, AI-operated), the agent/human CLI
(`python3 cli/cli.py help`), and the Python API. Quickstarts:
docs/QUICKSTART_HUMAN.md, docs/QUICKSTART_AI.md. Trained models
live in `trained_models/` (per-family subfolders; every response
carries the exact path). GPU: optional torch backend —
docs/COLAB_RUNBOOK.md. Note: the local package `mcp/` shadows the
PyPI `mcp` SDK if both are imported in one process; the server
runs as a subprocess, so normal use is unaffected.

---

## 1. Install

Use a virtual environment (recommended):

```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install as a package (code + dependencies in one command):

```
pip install git+https://github.com/continual-learning-models/GrowableSoftModel
```

Or from a clone, to run in place:

```
git clone https://github.com/continual-learning-models/GrowableSoftModel
cd GrowableSoftModel
pip install -r requirements.txt
```

## 2. The mental model

- The **LLM is the brain**: it understands language, decides when a
  specialized judgment is needed, and extracts features from a
  problem.
- A **soft model** carries one domain's learned judgment in its own
  weights. No language ability, no foundation model — it consumes a
  handful of features and returns a judgment with a confidence.
- You never declare a type or schema. Create a model with a **name
  only**; it infers its own feature space and output head (numeric
  regression, or categorical over a learned vocabulary) from the
  data it is taught.
- Teaching is **gated against reality**: a new version goes live
  only if it beats the current one on a held-out set. Teaching can
  never make the live model worse.

Data is JSONL, one `{input, target}` per line. `input` is a flat
object of named features; `target` is a number (→ regression) or a
label (→ classification).

## 3. Factory: teach and use (command line)

A soft model is created with only a name; teaching is gated, so add
a held-out set first.

```
softmodel create risk
softmodel add-holdout risk --data holdout.jsonl   # held-out reality (the gate)
softmodel teach       risk --data train.jsonl     # trains + promotes only if better
softmodel infer       risk '{"amount": 880, "night": 1}'
```

`teach` reports whether it promoted:

```json
{ "candidate_version": "v3", "candidate_metric": 1.0,
  "promoted": true, "active_version": "v3", "n_examples": 4 }
```

`infer` returns the judgment, confidence, version, and any mined
rule it agrees with:

```json
{ "output": "risk", "confidence": 0.999,
  "rule": "IF night == 1 THEN risk  (confidence 1.00, support 2)",
  "version": "v3" }
```

Other factory commands:

- `softmodel discoveries risk` — the interpretable decision list the
  model mined from its own data (regularities of *this* domain the
  LLM could not know from pretraining).
- `softmodel drift risk` — compares recent held-out performance
  against the promotion-time baseline; when `needs_reteach` turns
  `true`, collect fresh examples and `teach` again.
- `softmodel versions|card|eval|rollback risk` — lineage, scores,
  evaluation, and reverting the active version.

Running from a clone, the built-in end-to-end demo needs no data:

```
cd modules/Generator && python3 -m generator.cli demo
```

## 4. Core system: grow the topology

The core CLI mirrors the system facade as JSON in / JSON out:

```
python3 -m cli.cli <verb> '<json-args>'
```

It exposes the gated loop **plus** the growth operators. A worked
sequence — create, train the working state, grow it wider, then
gate the result (numeric target here, so the model self-shapes a
regression head):

```
$ python3 -m cli.cli create_model \
    '{"model_id":"m","holdout":[{"input":{"a":1,"b":2,"c":0},"target":3},
                                 {"input":{"a":2,"b":1,"c":1},"target":3}]}'

$ python3 -m cli.cli study \
    '{"model_id":"m","examples":[...],"steps":300}'
{"loss": 2e-13, "steps": 300, "params": 81}

$ python3 -m cli.cli widen '{"model_id":"m","k":2}'
{"widened": 2, "H": 18, "new_units": [16,17], "params": 91}

$ python3 -m cli.cli commit '{"model_id":"m"}'
{"promoted": true, "version": "v1", "score": 1.0, "live_before": 0.0}
```

Growth and lifecycle verbs (each takes a JSON arg object):

- `study` — train the working state on examples.
- `widen` — add hidden units (wider); `grow` — add capacity at
  unstable sites; `add_feature` — add an input dimension;
  `refound` — rebuild from scratch. All new parameters are
  trainable from step one (nothing is frozen).
- `growth_report` — where the model is unstable (candidate growth
  sites).
- `commit` — gate the working state against held-out reality;
  promotes only if better. `reset` — discard the working state.
- `self_review` / `run_self` — budgeted self-study.
- `infer`, `check_drift`, `discoveries`, `get_versions`,
  `rollback`, `card`, `trajectory` — as in the factory.

### Growth in two directions (and an optional third)

Every scope can grow by **widening** (adding units: rho/omega,
term addition) and by **deepening** (the delta operator: a
zero-initialized residual composition block inside the scope's
body — exact at application, removable, never touching anything
downstream). Direction is chosen by signals, not by fiat: history
prediction first (gain-ledger extrapolation gated by a
predictability certificate), zero-attach probes to verify (curves
read through extrapolated asymptotes, never as raw scores), the
gate adopts. Every algorithmic role is a replaceable part behind a
registry, selected by the policy dict; unknown parts are logged
refusals. See `docs/GROWTH_POLICY.md` for configuration, decision
and ledger record formats, and a runnable example. A THIRD
direction is available as an opt-in option (`loop_enabled`,
default off): the LOOP operator grows a governed directed cycle
whose forward relaxes to a fixed point — iteration as a
computational resource for implicit laws; the model then
iterates at inference on that scope, bounded by `loop_K_max`.

### Substrates: the body is pluggable

Every model instantiates its body through a substrate registry that
enforces a single contract (training step, prediction, growth-site
enumeration, site refinement, height, persistence). The default —
the *reference substrate* — is the recursive multi-scale network of
`modules/ReferenceNet`. The contract is architecture-agnostic, and the
release includes six hosts:

| substrate | body |
|---|---|
| `mlp` (default) | reference recursive multi-scale network |
| `mlp_plus` | reference variant |
| `transformer` | multi-layer transformer encoder (attention + feed-forward blocks), implemented from first principles in NumPy; backpropagation verified against finite differences; deterministic under fixed seed |
| `transformer_plus` | transformer variant |
| `sequence` | causal (autoregressive) variant of the transformer core |
| `growable_attention` | attention host whose heads are grown, not molded (see §7) |

Select at creation via policy, e.g. `{"substrate": "transformer"}`.
On the transformer host, the growth sites are the feed-forward
units of every layer: an unstable unit grows the same
zero-initialized recursive inner network under the same cross-scale
rule, and widening applies at every scale — lifecycle, gate, and
audit are identical across hosts. A domain whose laws require
compositional depth therefore receives it through the substrate
choice rather than through growth; the transformer host's own
attention weights are untouched by growth. Separately, the TYPE
of a node's grown inner body is user-selectable via policy
(`grow_body_type`: the reference body, or an optional small
attention body), governed by a scale-hierarchy guard so a grown
body stays much smaller than its host — see `docs/SPU.md` and
`docs/GROWTH_POLICY.md`. Mechanics are verified by
`tests/substrate_kit/` and the committed growth-audit log
`tests/logs/transformer_growth_events.jsonl`.

## 5. Use it from an LLM (MCP)

The primary use case: a general LLM drives the factory over the
**Model Context Protocol**. The server is a dependency-free stdio
process — **no Docker, nothing to host**. Register it with an MCP
client (e.g. Claude), run from the repo root:

```
claude mcp add growable -- python3 -m mcp.mcp_server
```

The LLM then sees **52 tools in one table**, of two families:

- **softmodel** (the tool's method, 45 tools) — `create_model`,
  `teach`, `study`, `infer`, `add_holdout`, `check_drift`,
  `grow`, `widen`, `add_feature`, `refound`, `run_self`,
  `commit`, `rollback`, `card`, `list_models`, substrate/growth
  reports, and more: the full gated teach/use/adapt loop plus the
  topology-growth operators and self-study.
- **standard** (optional industry mode, 7 tools) — `standard_*`
  (`create`/`train`/`evaluate`/`infer`/`save`/`load`/`list`): a
  fixed-architecture standard transformer/MLP trained from
  scratch, for users who explicitly want conventional methods.
  See `docs/STANDARD_METHODS_USAGE.md`.

**The server teaches the AI itself**: at connect time it hands the
client its operating manual (MCP `instructions`), and every tool
carries when-to-use text plus a next-step `hint` in its response,
so any MCP-capable model can drive it without human coaching. A
typical softmodel loop: read a problem, extract the features the
model's `learned_shape` asks for, `infer`, interpret; over time
`teach` new labeled cases (rows use `target`; `output` is accepted
as an alias) and poll `check_drift` — the gate guaranteeing
teaching never degrades the live model.

**GPU**: the whole tool (both families) runs on CPU or GPU by a
single policy switch — `set_compute_policy("torch", "cuda"|"mps",
"float64")`; float32 is not accuracy-certified and additionally
requires `acknowledge_f32_precision=True` (the precision door).
Models are device-free and serve on CPU by default.
See `docs/COLAB_RUNBOOK.md`.

**SPU (self-processing)**: an optional, off-by-default,
evolution-time mechanism where a freshly grown inner network
refines its own computational state before participating —
enabled per model via policy. See `docs/SPU.md`.

## 6. Python API

```python
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec

f = SoftModelFactory()
f.create(ModelSpec(model_id="risk"))
f.add_holdout("risk", [{"input": {"amount": 120, "night": 0}, "target": "ok"},
                       {"input": {"amount": 870, "night": 1}, "target": "risk"}])
f.teach("risk", [{"input": {"amount": 100, "night": 0}, "target": "ok"},
                 {"input": {"amount": 900, "night": 1}, "target": "risk"},
                 {"input": {"amount": 200, "night": 0}, "target": "ok"},
                 {"input": {"amount": 850, "night": 1}, "target": "risk"}])
print(f.infer("risk", {"amount": 880, "night": 1}))   # -> {"output": "risk", ...}
```

## Parameters

Every method family (growth, SPU, attention) is per-model
tunable through supported surfaces; docs/PARAMETER_REFERENCE.md
is the complete, freshness-tested catalog. Standard users drive
the system through SoftModelSystem (train_converge/model_policy);
this lib's own CLI/MCP/Python surfaces are the advanced direct
tier — both fully supported.

## 7. Structure

- `mcp/`, `cli/` — the entry layer: the MCP server and the
  agent/human command line (both families' tools in one table).
- `core/` — the softmodel method: lifecycle (working state +
  gated commit), growth operators, self-study, and the softmodel
  `facade.py`.
- `core/substrates/` — the substrate registry and the six hosts
  (`mlp`, `mlp_plus`, `transformer`, `transformer_plus`,
  `sequence`, `growable_attention`); one contract,
  architecture-agnostic.
- `standard_methods/` — the optional industry-standard mode
  (fixed-architecture facade; reuses the substrates read-only).
- `modules/Generator/` — the soft-model generator (teaching, gating,
  drift, versions).
- `modules/Engine/` — the shared engine: math primitives,
  numpy/torch backends, loop-block kernel math, and the generic
  self-processing (SPU) machinery.
- `growable_attention` substrate — the attention host whose
  heads are GROWN, not molded: heads can be added and each head's
  width grows independently (function-preserving at every step),
  with per-head instruments, an optional entropy-band
  self-processing discipline, and the governed `grow_attention`
  verb (facade + MCP). See docs/ARCHITECTURE_MAP.md and
  docs/PARAMETER_REFERENCE.md.
- `modules/ReferenceNet/` — the first-generation recursive
  multi-scale network family (net, trainer, curriculum, growth
  policy, its own SPU bindings).
- `trained_models/` — where your trained models live
  (`softmodel/` and `standard/` subfolders; env-overridable).
- `tests/` — unit, integration, capability, backend kit, and the
  one-command acceptance suite.
- `experiments/` — pre-registered experiments (specs, results,
  verdicts).
- `scripts/` — CI guard and utilities.

## 8. Develop / test

```
pip install -r requirements.txt pytest
python3 scripts/ci_guard.py
python3 tests/acceptance/run_acceptance.py
python3 -m pytest tests/unit tests/integration -q
```

## License

AGPL-3.0 (see `LICENSE`). The code is free to use, study, modify,
and share under the AGPL's terms — including that a modified
version offered as a network service must make its source
available. For commercial or closed-source use that the AGPL-3.0
does not permit, a separate commercial license is available;
contact the copyright holder.

## Evaluative learning (reward + structure preference)

The library carries a unified
evaluative-learning surface under the same gate and audit
discipline as teaching:

- **P-loop (policy optimization on the model)** — clipped-
  surrogate policy optimization (PPO/GRPO family) runs directly
  on the growable substrate through the pseudo-target identity
  `y* = h − ∂L/∂h`; advantage estimation, clipping, entropy and
  the optional KL term to a reference policy live outside the
  substrate. Entry point: the `rl_trainer` module's
  `OrganPPORunner`, with `set_kl_reference` (runner) and
  `align_to` (eval provider) for the incumbent-anchored
  reference; every training round appends an `rl_round` audit
  record, persisted to the model's `rl_audit.jsonl`. See
  `docs/RL_TRAINER_RUNBOOK.md`.
- **S-loop (preference over growth moves)** — credited
  advantages per move family with bucketed sufficient
  statistics and five selection rules (production default:
  fixed-scale posterior sampling). Preference only orders which
  candidate is offered; adoption remains the gate's. Entry
  points: `preference_inspect` / `preference_reset` /
  `preference_prior_fold`. See `docs/PREFERENCE_RUNBOOK.md`.

Both loops are verified quasi-statically against official
PyTorch components and authoritative RL libraries
(`tests/logs/QUASISTATIC_FULL_VERIFICATION.md`,
`tests/logs/RL_LIBRARY_COMPARISON.md`).

## Policy-key gate

Every model-policy door (create_model / set_policy, on all
surfaces) refuses unknown top-level keys loudly, naming every
offender — a typo can never be silently stored. Catalog:
docs/PARAMETER_REFERENCE.md.

## Versioning

Current release: `1.0.0` (first public release).
Version history: `CHANGELOG.md`.
