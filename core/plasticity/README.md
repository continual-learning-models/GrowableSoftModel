# core/plasticity — total plasticity operators

Realizes: omega (widen, outward growth at every scale), sigma (input
schema growth), Phi (re-founding), plus the instrumentation (per-scale
locus, saturation, inversion, LP), the experience store (quarantined
from holdouts), and the no-freeze audit that mechanizes the axiom.

Rules (binding):
1. No freezing, ever — in operation or experiments. The no-freeze
   audit (tests/plasticity/test_no_freeze.py) enforces this in CI.
2. Function preservation at every structural operation.
3. modules/ are additive-only (scripts/ci_guard.py baselines); this
   package subclasses/wraps module code, never edits it.
4. The commit gate is never bypassed; worst case = discarded working
   state.
5. Original the frozen predecessor system directory: frozen, never entered.
