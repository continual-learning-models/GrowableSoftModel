"""legacy_compat — QUARANTINED v1 relics (A5, doc 52 v1.7
section 6.4; FR-8): VERBATIM code motion from growth_port.py.
Deprecated-defective, load-only paths that keep pre-reform
artifacts readable and serving on their original code path.
Never a verification reference; refused for new growth.
growth_port re-exports every name so existing imports stay
valid (the facade, doc 52 section 6.3).
"""
from __future__ import annotations

import numpy as np


def legacy_attach(H, X, inner):
    """The v1 scalar attach — VERBATIM code motion from the
    hosts (deprecated-defective; serves LOADED legacy
    structures only, doc 35 R3). Kept in ONE place so no host
    owns a private attach implementation (doc 37 T-17)."""
    for j, net in inner.items():
        H[:, j] = H[:, j] + net.predict(X)[:, 0]
    return H


def legacy_handoff(inner, X, Hact, A, dH, gelu_fn, eta,
                   sgd_lr=None):
    """The v1 eta-handoff training — VERBATIM code motion
    (same caveats as legacy_attach)."""
    for j, net in inner.items():
        t_j = Hact[:, j] - eta * dH[:, j]
        r_j = t_j - gelu_fn(A[:, j])
        net.train_step(X, r_j.reshape(-1, 1), sgd_lr=sgd_lr)


def legacy_attach_layer(H, inner_in, inner, layer, bk, n):
    """v1 attach for per-layer 3-D hosts (growable_attention /
    transformer) — VERBATIM code motion (deprecated-defective,
    load-serving only)."""
    for (ll, j), net in inner.items():
        if ll == layer:
            H[:, :, j] = H[:, :, j] + bk.ingest(
                net.predict(inner_in))[:, 0].reshape(n, -1)
    return H


def legacy_collect_layer(inner, layer, H, dH, pre, eta,
                         gelu_fn):
    """v1 handoff COLLECTION for per-layer 3-D hosts —
    VERBATIM code motion; the host executes the returned
    (net, residual) pairs post-step exactly as before."""
    out = []
    for (ll, j), net in inner.items():
        if ll == layer:
            t_j = H[:, :, j] - eta * dH[:, :, j]
            r_j = (t_j - gelu_fn(pre[:, :, j])).reshape(-1, 1)
            out.append((net, r_j))
    return out


class LegacyScalarPort:
    """Load-only marker for pre-reform artifacts (deprecated-
    defective). Hosts keep serving loaded legacy structures
    through their original code path; this class only names
    the state so audits can see it. It cannot accept new
    bodies."""

    PORT_TYPE = "legacy_scalar"

    def add_body(self, *a, **k):
        raise ValueError(
            "legacy_scalar is load-only (deprecated-defective); "
            "new growth must use the fullwidth port")
