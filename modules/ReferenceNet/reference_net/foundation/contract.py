"""P-1, the STRUCTURE CONTRACT (next-dev-docs/52 v1.7
section 3.1) — the first of the two L1 root primitives.

A growable structure is anything that: owns parameters,
computes a forward map, accepts gradients, and serializes.
The contract is DUCK-TYPED — no base class is imposed; this
module documents the member set and provides the conformance
checker. Network and AttentionBody already satisfy it as
written (their de-facto shared surface, formalized).

Behavioral notes (recorded, not checkable by membership):
 - train_from_grad(X, dU, sgd_lr=None): Network's
   implementation requires port ownership (identity scalers);
   callers honor that precondition.
 - __getstate__/__setstate__ mean LOSSLESS, device-free
   serialization; mere member presence (object.__getstate__
   exists on modern Python) is necessary, not sufficient —
   the serialization boxes adjudicate the behavior.

Neutrality (PR-0): this module knows nothing about growth
modes, wiring, positions, timing or birth policy.
"""

REQUIRED = (
    "n_params",          # () -> int
    "predict",           # (X) -> array, the forward map
    "train_from_grad",   # (X, dU, sgd_lr=None), gradient intake
    "__getstate__",      # lossless, device-free state out
    "__setstate__",      # lossless state in
)

OPTIONAL = (
    "instability",       # () -> sequence, instrument hook
    "structure",         # (path) -> rows, instrument hook
    "depth",             # () -> int, instrument hook
)


def conforms(obj):
    """Empty list == conformant; else the missing member names,
    in REQUIRED order (V-A0-1 failure output contract)."""
    return [name for name in REQUIRED
            if not callable(getattr(obj, name, None))]
