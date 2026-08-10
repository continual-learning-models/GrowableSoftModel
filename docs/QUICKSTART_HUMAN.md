# Quickstart (human, 5 minutes)

The tool is an AI-driven small-model factory. Its method is the
SOFTMODEL (models that grow and self-process for life); an
optional industry-standard fixed-architecture mode is also
available. Your trained models live in `trained_models/`.

## 1. Setup

    git clone <this repo> && cd GrowableSoftModel
    pip install numpy            # torch optional, for GPU

## 2. Train your first model (CLI)

    # softmodel (the tool's method)
    python3 cli/cli.py create_model '{"model_id": "demand"}'
    python3 cli/cli.py teach '{"model_id": "demand",
        "examples": [{"input": {"x1": 1.0, "x2": 2.0}, "target": 3.1}]}'
    python3 cli/cli.py infer '{"model_id": "demand",
        "input_": {"x1": 1.5, "x2": 2.5}}'

    # or the optional standard mode
    python3 cli/cli.py standard_create '{"name": "demand2", "arch": "mlp"}'
    python3 cli/cli.py standard_train '{"name": "demand2",
        "examples": [{"x": [1.0, 2.0], "y": 3.1}]}'
    python3 cli/cli.py standard_infer '{"name": "demand2", "x": [1.5, 2.5]}'

`python3 cli/cli.py help` lists every verb of both families.

## 3. GPU (optional)

    pip install torch
    # in python before creating: from engine.backends import
    # set_compute_policy; set_compute_policy("torch","mps","float64")
    # (float32 additionally requires acknowledge_f32_precision=True
    #  — it is not accuracy-certified)

## 4. Where results live

trained_models/softmodel/<id>/ and trained_models/standard/<name>/
— every create/train response also prints the exact path.



## Tuning any method

All parameters of all method families are per-model controllable
— see docs/PARAMETER_REFERENCE.md for the complete catalog and
worked examples (both access tiers: SoftModelSystem = standard
product gate; this lib's CLI/MCP/Python = advanced direct use).

## Growable attention

`create_model(..., substrate="growable_attention")` gives the
attention host whose head count and widths GROW in service under
the gate; trigger one governed event with
`System.grow_attention(model_id, layer)`. Substrates on offer:
mlp, mlp_plus, transformer, transformer_plus, sequence,
growable_attention (see `list_substrates`).
