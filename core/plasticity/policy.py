"""Siting policy (Stage 3c): refine (rho) vs widen (omega).

Principle (theory §6): LOCAL conflict -> refine at the culprit site;
GLOBAL saturation -> widen the saturated container. Thresholds were
derived in E0 and FROZEN there before any experiment uses them.
"""
from __future__ import annotations
from reference_net.growthpolicy import OP_OMEGA, OP_RHO

import numpy as np

from core._modules import reference_net  # noqa: F401
from reference_net.net import Network
from reference_net.trainer import collect_instability
from core.plasticity.metrics import saturation

# Frozen in experiments/E0_instrumentation/REPORT.md — do not tune.
WIDEN_SAT = 0.5          # container saturation to consider widening
UNIFORM_FACTOR = 2.0     # top-unit instability < 2x mean = "uniform"
ESCALATE_DISPOSED = 2    # consecutive gate-disposed refines -> widen


def _containers(organ):
    """[(container_path, instability_vector)] for every network/layer."""
    out = []
    if isinstance(organ, Network):
        def walk(net, path):
            out.append((path, np.asarray(net.instability())))
            for j, inner in net.inner.items():
                walk(inner, f"{path}/{j}" if path != "root" else str(j))
        walk(organ, "root")
    elif hasattr(organ, "_unit_instability"):
        for l in range(organ.L):
            out.append((f"layer{l}", organ._unit_instability(l)))
        for (l, j), net in organ.inner.items():
            def walk(n, path):
                out.append((path, np.asarray(n.instability())))
                for kk, inner in n.inner.items():
                    walk(inner, f"{path}::{kk}")
            walk(net, f"layer{l}/ffn[{j}]")
    else:
        raise TypeError(f"unsupported organ type: {type(organ)!r}")
    return out


def decide(organ, recent_disposed: int = 0,
           widen_sat=WIDEN_SAT, uniform_factor=UNIFORM_FACTOR,
           escalate_disposed=ESCALATE_DISPOSED) -> dict:
    """-> {"action": OP_OMEGA("widen"), "container": path} or
       {"action": OP_RHO("refine"), "site": path}.
    recent_disposed = consecutive gate-disposed refine events reported
    by the caller (escalation rule)."""
    conts = _containers(organ)
    best_path, best_u = max(conts, key=lambda c: saturation(c[1]))
    best_sat = saturation(best_u)
    uniform = (best_u.size > 0 and best_u.mean() > 0
               and best_u.max() < uniform_factor * best_u.mean())
    if recent_disposed >= escalate_disposed:
        return {"action": OP_OMEGA, "container": best_path,
                "reason": f"{recent_disposed} consecutive refines "
                          "gate-disposed while stuck persists"}
    if best_sat >= widen_sat and uniform:
        return {"action": OP_OMEGA, "container": best_path,
                "reason": f"uniform saturation {best_sat:.3f}"}
    if hasattr(organ, "growth_sites"):
        site, score = organ.growth_sites()[0]
    else:                       # bare Network: same ranking, mlp paths
        rows = sorted(collect_instability(organ), key=lambda r: -r[2])
        path, j, score, _ = next(r for r in rows
                                 if r[1] not in r[3].inner)
        site = f"{path}[{j}]"
    return {"action": OP_RHO, "site": site,
            "reason": f"localized conflict (top site {score:.3f}, "
                      f"max container saturation {best_sat:.3f})"}
