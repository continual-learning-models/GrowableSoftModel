"""Read-only instrumentation for total plasticity (Stage 1).

Four metric families, all derivations over existing state/logs — no
training-path changes anywhere:

- per-scale adaptation LOCUS: where (which scale) parameters actually
  moved between two snapshots;
- SATURATION per network/layer: high mean instability with LOW
  dispersion = uniform conflict (widen signal); high dispersion =
  localized culprit (refine signal);
- amplitude-hierarchy INVERSION: fine-scale output contribution
  overtaking the coarse backbone = the foundation is being patched to
  death (Phi trigger);
- LEARNING-PROGRESS / RETENTION curves from evaluate-event streams.

Scale convention: scale 0 = a network's own arrays (the coarse
backbone at that node); scale k = arrays of inner networks k levels
below. For a transformer-host organ, scale 0 = the P dict (attention,
LN, FFN, heads); its inner reference_net Networks start at scale 1.
(Substrate isolation: this module never names concrete substrate
classes — organs are duck-typed on their structure.)
"""
from __future__ import annotations

import numpy as np

from core._modules import reference_net  # noqa: F401
from reference_net.net import Network, gelu


# ---------------- per-scale parameter snapshots / locus ----------------

def _net_arrays_by_scale(net: Network, level: int, out: dict):
    own = [net.W1, net.b1, net.W2, net.c]
    out.setdefault(level, []).extend(a.ravel().copy() for a in own)
    for inner in net.inner.values():
        _net_arrays_by_scale(inner, level + 1, out)
    port = getattr(net, "_port_site", None)
    if port is not None:
        # fullwidth port bodies live one scale below; their
        # assemblies A_g are part of the grown addition
        for slot in port.bodies:
            out.setdefault(level + 1, []).append(
                np.asarray(slot["A"]).ravel().copy())
            _net_arrays_by_scale(slot["body"], level + 1, out)


def scale_snapshot(organ) -> dict:
    """{scale: 1-D concatenated parameter vector}. Works for Network-
    hosted organs (mlp) and for transformer-host organs (P dict = scale 0,
    inner Networks from scale 1)."""
    out: dict = {}
    if isinstance(organ, Network):
        _net_arrays_by_scale(organ, 0, out)
    elif hasattr(organ, "P") and hasattr(organ, "inner"):
        out[0] = [v.ravel().copy() for v in organ.P.values()]
        for net in organ.inner.values():
            _net_arrays_by_scale(net, 1, out)
    else:
        raise TypeError(f"unsupported organ type: {type(organ)!r}")
    return {s: np.concatenate(v) for s, v in out.items()}


def locus(snap0: dict, snap1: dict) -> dict:
    """Per-scale relative displacement D_s = ||th1-th0|| / (||th0||+eps).
    Scales present only in snap1 (grown since snap0) are reported with
    the movement of their full magnitude (born-and-trained content)."""
    eps = 1e-12
    d = {}
    for s, v1 in snap1.items():
        v0 = snap0.get(s)
        if v0 is None or v0.size != v1.size:
            d[s] = float(np.linalg.norm(v1) / (np.linalg.norm(v1) + eps))
        else:
            d[s] = float(np.linalg.norm(v1 - v0)
                         / (np.linalg.norm(v0) + eps))
    return d


# ---------------- saturation (widen-vs-refine signal) ----------------

def saturation(u: np.ndarray) -> float:
    """s = mean(u) * (1 - cv(u)), clipped to [0, 1]. u = per-unit
    instability vector. High mean + low dispersion -> uniform conflict
    -> widen; the score degrades toward 0 as conflict localizes."""
    u = np.asarray(u, dtype=float)
    m = float(u.mean())
    if m <= 0:
        return 0.0
    cv = float(u.std() / (m + 1e-12))
    return float(np.clip(m * (1.0 - cv), 0.0, 1.0))


def saturation_report(organ) -> dict:
    """{location: saturation score} for every network/layer that has an
    instability vector."""
    rep = {}
    def _kids(n):
        kids = list(n.inner.items())
        port = getattr(n, "_port_site", None)
        if port is not None:
            kids += [(s.get("key", g), s["body"])
                     for g, s in enumerate(port.bodies)]
        return kids

    if isinstance(organ, Network):
        def walk(net, path):
            rep[path] = saturation(net.instability())
            for j, inner in _kids(net):
                walk(inner, f"{path}/{j}")
        walk(organ, "root")
    elif hasattr(organ, "_unit_instability"):
        for l in range(organ.L):
            rep[f"layer{l}"] = saturation(organ._unit_instability(l))
        bodies = [((l, j), net)
                  for (l, j), net in organ.inner.items()]
        for l, site in getattr(organ, "_port_sites", {}).items():
            bodies += [(s.get("key", (l, g)), s["body"])
                       for g, s in enumerate(site.bodies)]
        for (l, j), net in bodies:
            def walk(n, path):
                rep[path] = saturation(n.instability())
                for k, inner in _kids(n):
                    walk(inner, f"{path}/{k}")
            walk(net, f"layer{l}/ffn[{j}]")
    else:
        raise TypeError(f"unsupported organ type: {type(organ)!r}")
    return rep


# ---------------- amplitude-hierarchy inversion (Phi trigger) ----------

def inversion(net: Network, X: np.ndarray) -> float:
    """R_inv = ||fine output contribution|| / (||coarse backbone
    contribution|| + eps) on probe X, at this network's top split:
    coarse = pure-gelu backbone through W2; fine = everything produced
    by inner networks (any depth — their predict() recurses).
    Healthy decomposition: R_inv << 1. R_inv approaching/exceeding 1
    means the foundation is being patched to death."""
    if net._x_mu is None:          # untrained: no decomposition yet
        return 0.0
    Xs = net._std_x(X)
    A = Xs @ net.W1.T + net.b1
    coarse_h = gelu(A)
    fine_h = np.zeros_like(coarse_h)
    for j, inner in net.inner.items():
        fine_h[:, j] = inner.predict(Xs)[:, 0]
    port = getattr(net, "_port_site", None)
    if port is not None:
        # fullwidth fine contribution: u_g @ A_g across the width
        for slot in port.bodies:
            fine_h = fine_h + np.asarray(
                slot["body"].predict(Xs)) @ np.asarray(slot["A"])
    coarse_out = coarse_h @ net.W2.T
    fine_out = fine_h @ net.W2.T
    return float(np.linalg.norm(fine_out)
                 / (np.linalg.norm(coarse_out) + 1e-12))


# ---------------- learning progress / retention from events ----------

def lp_curve(evaluate_events: list, suite_idx: int) -> list:
    """Learning-progress deltas for one suite from a chronological list
    of evaluate events (each carrying 'stage_accs'). LP_t = acc_t -
    acc_{t-1}; length = len(events) - 1."""
    accs = [e["stage_accs"][suite_idx] for e in evaluate_events
            if "stage_accs" in e and len(e["stage_accs"]) > suite_idx]
    return [float(b - a) for a, b in zip(accs, accs[1:])]


def retention_trend(evaluate_events: list, suite_idx: int,
                    window: int = 3) -> float:
    """Mean LP over the trailing window for an already-mastered suite:
    negative = retention sagging (proactive-review signal)."""
    lp = lp_curve(evaluate_events, suite_idx)
    if not lp:
        return 0.0
    tail = lp[-window:]
    return float(np.mean(tail))
