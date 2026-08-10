"""SoftModel — Multi-Scale Soft Model.

A factory for evolvable soft models: small specialized neural networks
("organs") that a general LLM both *uses* (infer) and *teaches* (teach/evolve),
behind a dual API with an eval gate (evolution only ever improves) and
versioned rollback.

The LLM is the brain (all general capability); a SoftModel model is an organ —
no base/foundation model, trained from scratch on domain data.

Backends:
- backend="mlp"  (default): the real organ — a numpy TinyMLP over LLM-extracted
  features; learns and generalizes; weights serialize per version.
- backend="mock": zero-dep text-lookup stub to exercise the control loop.
"""

__version__ = "0.6.1"
