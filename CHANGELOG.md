# Changelog — GrowableSoftModel

## 1.0.0 — first public release (2026-08-10)

A growable soft model: a small neural model that a general LLM
both uses (`infer`) and teaches (`teach`), which shapes itself
from data and grows its own topology under a held-out reality
gate, with every scale trainable for life.

This release contains:

- **Total-plasticity core** --- no parameter and no structural
  decision is ever frozen; stability comes from governance, not
  immobility.
- **Growth operators** --- widen (units and terms), deepen
  (residual composition blocks and whole layers), new input
  features, re-found, and optional governed cycles (the loop
  operator); each exact at application, budgeted, and audited.
- **Held-out reality gate** --- one adjudication rule over both
  parametric and structural change: a candidate is adopted only
  if it scores strictly better on quarantined data; every
  decision appends an audit record; rollback restores the
  previous version byte-for-byte.
- **Six substrate hosts** behind one contract, including a
  transformer encoder, a causal variant, and a growable-attention
  host whose heads are grown rather than molded.
- **Evaluative learning** --- reward-driven policy optimization
  on the growable substrate and preference over growth moves,
  both under the same gate and audit discipline.
- **Governed lifecycle** --- self-shaping birth, working versus
  committed states, store-per-version, drift detection, budgeted
  self-study, plan and trial surfaces.
- **Policy-key gate** --- unknown model-policy keys are refused
  loudly on every surface; a typo can never be silently stored.
- **Three channels** --- an MCP server (52 tools, AI-operated),
  a CLI, and the Python API; plus an optional standard-methods
  mode (fixed-architecture baselines) for comparison.
- **Verification** --- 1,221 tests; the growing model is checked
  quasi-statically against official PyTorch components and
  authoritative reinforcement-learning libraries at every growth
  instant.

Licence: AGPL-3.0 with dual commercial licensing (see LICENSE).
