"""Public read-only introspection helper (doc 29 3.4c).

Serves stage-2 circuit research and exam-side verification
(CKA-class analyses): exposes attention maps and hidden
states via the substrate's documented cache-forward path,
with numpy copies and zero mutation. Not a kernel change —
a core-layer wrapper only.
"""
from __future__ import annotations

import numpy as np


def inspect(organ, X):
    """Return {"attention": [layer][head] arrays,
    "pooled": pooled states, "hidden": per-layer states}
    for attention hosts. Read-only; refuses organs without a
    cached forward (loud ValueError)."""
    fwd = getattr(organ, "_forward", None)
    if fwd is None:
        raise ValueError(
            f"organ {type(organ).__name__} exposes no cached "
            "forward; introspection serves attention hosts")
    X = np.asarray(X, float)
    stdx = getattr(organ, "_stdx", None)
    Xs = stdx(X) if stdx is not None else X
    try:
        logits, pooled, caches = fwd(Xs, cache=True)
    except TypeError as e:
        raise ValueError(
            "organ's forward does not support cache=True; "
            "introspection unavailable for this host") from e
    # numpy-copy boundary (the function's contract): hosts on a
    # compute backend expose _bk; the judge path is unchanged
    bk = getattr(organ, "_bk", None)

    def _np(x):
        return np.array(bk.to_numpy(x) if bk is not None else x,
                        copy=True)

    attention, hidden = [], []
    for layer_cache in caches[1:]:
        att = layer_cache[3]
        attention.append([_np(h[3]) for h in att])
        hidden.append(_np(layer_cache[0]))
    return {"attention": attention,
            "hidden": hidden,
            "pooled": _np(pooled),
            "logits": _np(logits)}
