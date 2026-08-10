# SelfGrow — the self-growing network core

A recursive multi-scale network whose **structure grows by itself**.
One class, closed under refinement: a network is a one-hidden-layer
MLP whose hidden nodes may each contain an inner network (composite
nodes). Depth is determined by the data, never fixed by code.

## Learning (recursive, one primitive at every depth)

- **Within a scale:** ordinary backprop on that scale's parameters,
  treating composite nodes' inner outputs as pass-through.
- **Across scales:** the enclosing scale hands each composite node a
  target for its output; the inner network takes its own local step
  toward it — and recurses identically for its own composite nodes.
  One primitive, every depth, one step per enclosing iteration.

This makes growth structural (nodes/scales are added where the data
needs them) rather than a fixed architecture with only weights
moving.

## Package layout

- `selfgrow/` — the package:
  - `net.py` the recursive multi-scale network;
  - `trainer.py` training; `curriculum.py` course logic;
  - `instrument.py` instrumentation; `mcp_server.py` MCP surface.
- `tests/` — unit suites.
- `scripts/` — utilities.

## Run

```
python -m pytest tests -q
```

License: Apache-2.0 (see the repository `LICENSE`).
