# Generator — the soft-model generator

A factory for evolvable **soft models**: small specialized neural
networks that a general LLM both **uses** (`infer`) and **teaches**
(`teach` / evolve), behind a dual API with an evaluation gate
(evolution only ever improves) and versioned rollback.

The LLM is the brain — it supplies all general capability (language,
context, feature extraction). A soft model carries one domain's
learned judgment in its own weights: no base/foundation model,
trained from scratch on domain data, kilobyte-scale, CPU-trainable
in seconds.

## What it does

- **Self-shaping.** A model is created with only a name; it infers
  its own feature space, output head (numeric regression vs.
  categorical over a learned vocabulary) and capacity from the data
  it is taught — no human-declared types or schemas.
- **Gated evolution.** `teach` trains a candidate, evaluates it on a
  time-stamped held-out stream, and promotes it only if it beats the
  incumbent on the recent slice; contradictory teaching is rejected;
  every version is rollback-able.
- **Drift awareness.** `check_drift` compares recent held-out
  performance against the promotion-time baseline and flags when a
  re-teach is needed.
- **Readable regularities.** Categorically-shaped models mine their
  store into an interpretable decision list surfaced via
  `discoveries`.

## Package layout

- `generator/` — the package:
  - `nets.py` TinyMLP (numeric/categorical heads, scalers);
  - `trainer.py` self-shaping + training; `rules.py` decision-list
    induction; `evolve.py` teach/gate/drift; `evaluator.py`
    recent-slice metric;
  - `registry.py` versions/lineage; `model_manager.py` inference;
  - `factory.py` facade; `api.py` REST; `mcp_server.py`
    dependency-free MCP server; `cli.py`, `config.py`, `data.py`.
- `examples/` — sample datasets.
- `scripts/` — runnable demos.
- `tests/` — unit + regression suites.

## Run

```
pip install -r requirements.txt
python scripts/demo_organ.py          # real learning + generalization
```

Backends: `mlp` (default, the real numpy organ) and `mock` (a
zero-dependency text-lookup stub for exercising the control loop).

License: Apache-2.0 (see `LICENSE`).
