"""foundation — L1, the method-agnostic base tier of the
growth machinery (next-dev-docs/52 v1.7 section 2.1).

Contains ONLY neutral meta-mechanisms: the structure contract,
the coupling primitive, the initialization-recipe registry,
the four spec objects and the composition executor. Bound to
NO growth mode and NO parameter values (PR-0): global versus
local scope, read/write designations, insertion position,
growth timing, birth appearance and preservation policy are
all answered ABOVE this package, never inside it.

Dependency rule (52 section 2.2, enforced by the V-A0-2
linter): this package imports numpy / the engine backends /
the standard library ONLY — never hosts, never the method
tier, never the policy package.
"""
from .contract import REQUIRED, OPTIONAL, conforms  # noqa: F401
