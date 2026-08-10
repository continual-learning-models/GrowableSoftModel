# The optional industry-standard mode (standard_methods)

The tool's method is the SOFTMODEL (growing, self-processing
models — see the main docs). This page covers the OPTIONAL
secondary mode: a FIXED-architecture standard transformer or MLP
trained from scratch on your data (not pretrained-LLM
fine-tuning).

## Verbs (Python: `import standard_methods as sm`; MCP/CLI:
## the standard_* tools/commands)

- create(name, arch=None, examples=None, mode="numeric",
  **standard_params)
  arch "transformer"|"mlp"|None (None+examples -> auto by data
  form, stated; None alone -> transformer, stated). mode
  numeric|categorical. Hyperparameters (industry vocabulary
  only): transformer n_layers/d_model/n_heads/ffn_hidden/lr;
  mlp hidden/lr. Unknown keys are refused listing the valid
  ones. An existing name is LOADED, never overwritten — the
  response says so.
- train(name, examples, steps=200)
  examples = [{"x": [...], "y": value-or-label}, ...]. The input
  width shapes itself from your first batch. Response: training
  error before/after, held-out score (80/20 split), total steps,
  the model's path, and the next-step hint.
- evaluate(name, examples) -> mse+mae (numeric) or accuracy.
- infer(name, x) -> prediction (+confidence when categorical).
- save(name) / load(name) — the model's own artifact under
  trained_models/standard/<name>/ (override:
  STANDARD_MODELS_ROOT). v1 keeps the LATEST state only — no
  version chain; if you need gated versioning, drift checks and
  rollback, use the softmodel family.
- list_models() — standard-family models only (the softmodel
  list is separate; same name may exist in both families
  without interference).

## GPU

Same switch as everywhere: set_compute_policy("torch",
"cuda"|"mps", "float64") before creating (float32 additionally
requires acknowledge_f32_precision=True — not
accuracy-certified); artifacts stay device-free and serve on
CPU.

## MCP / CLI access

One server serves both families:

    claude mcp add growable -- python3 -m mcp.mcp_server

(run from the repo root; the softmodel tools keep their original
names, the standard mode uses the standard_* tools listed above).
CLI: `python3 cli/cli.py help` lists every verb of both families
with examples; `python3 cli/cli.py standard_create '{"name":
"m1", "arch": "mlp"}'` etc.

Note (name-shadowing caveat): this repo contains a local package
named `mcp`; if you also use the PyPI `mcp` SDK in the SAME
python process with the repo root on sys.path, the local package
shadows it. The server itself is self-contained and runs as a
subprocess, so normal use is unaffected.
