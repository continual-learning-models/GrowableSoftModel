"""rl_trainer — Track-B P-loop trainer family (plan 84 D-P2;
design doc 86 N1/§10). L1 WeightAdapter layer: PPO + GRPO
behind the TrainerPlug protocol, PORTED from the verified
blueprint (scripts/verify_vs_authoritative_libs.py, 9/9 vs
SB3/gymnasium/MABWiser — BLUEPRINT-PARITY GATE: with default
keys the update trajectory is BIT-IDENTICAL to the blueprint).

Layering law (doc 86 §3.0): this package is L1/L2 — it may
import the L0 EvaluativeCore (shared math) and numpy; it
never imports preference or SMS types. Substrate contact is
confined to THREE named points (R-R2/R3-2 as-built
precision): the OrganAdapter holds an organ REFERENCE
(public surface only); OrganPPORunner's constructor lazily
imports reference_net.net.Network as an organ-construction
convenience; and instruments.organ_health reads organ
hidden activations through the house private-instrument
pattern (read-only). No other module touches substrate
types; the pseudo-target adapter (§10 scheme (c)) itself
works purely on arrays.
"""
from .defaults import RL_DEFAULTS  # noqa: F401
from . import math          # noqa: F401
from . import buffer        # noqa: F401
from . import trainers      # noqa: F401
from . import pseudo_target  # noqa: F401
from . import organ_adapter  # noqa: F401
from . import worlds        # noqa: F401
from . import records       # noqa: F401
from . import runner        # noqa: F401
from . import instruments   # noqa: F401
from . import eval_provider  # noqa: F401
from . import regime        # noqa: F401

