"""Total-plasticity package: realizes the soft-model axiom — every
scale forever plastic and forever optimizable.

Binding rules (README has the full text):
- NO freezing anywhere: every parameter introduced by any operator is
  trainable from step one; zero-init is an initial value, never a mask.
- Function preservation at every structural operation (mlp exact;
  transformer to eps < 1e-3, the add_class standard).
- modules/Generator and modules/ReferenceNet are additive-only (ci_guard);
  everything here subclasses/wraps, never edits them.
- The commit gate is never bypassed.
"""
