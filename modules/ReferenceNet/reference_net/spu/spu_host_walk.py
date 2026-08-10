"""SPU host walk for attention hosts (DEV_PLAN v2.1, V3).

An attention host's grown inner bodies read LAYER-DEPENDENT
inputs (the layer's normalized token representations), which do
not exist before the host's forward computes them. The host
therefore processes its bodies at the top of its training step
against the PREVIOUS training forward's cached inputs:
processed-before-participation holds; the observation is one
step stale (disclosed); the first step after install has no
cache and is skipped (one step, disclosed). Serving purity:
the cache is written only by training forwards (cache=True),
never by predict.
"""
from ..growthpolicy import GROWTH_MODE_WIDEN_ONLY, get_growth_mode
from .spu_network import _count_skips, spu_scan
from engine.spu.spu_loop import self_process


def spu_host_pre_forward(host, X, y, pol):
    """Process every eligible inner body of an attention host.
    Called by the host's train_step seam; host is duck-typed
    (needs .inner dict keyed (layer, unit), ._t step counter,
    .predict, .mode)."""
    if (not pol["spu_enabled"]
            or get_growth_mode() == GROWTH_MODE_WIDEN_ONLY):
        return
    step = int(host._t)
    # bodies = legacy inner dict AND fullwidth port slots (doc 35
    # D9: the walk sees the same objects whichever coupling
    # carries them); port bodies read the SAME layer-cached input
    bodies = [((l, j), body)
              for (l, j), body in sorted(host.inner.items())]
    for l, site in sorted(
            getattr(host, "_port_sites", {}).items()):
        for g, s in enumerate(site.bodies):
            bodies.append((s.get("key") or (l, g), s["body"]))
    if step % pol["spu_every"] != 0 or not bodies:
        return
    cache = getattr(host, "_spu_input_cache", None)
    if cache is None:                       # first step after install
        _count_skips(host, [{"skip": "no_cached_input"}])
        return
    eligible, skips = [], []
    for (l, j), body in bodies:
        X_body = cache.get(l)
        if X_body is None:
            skips.append({"skip": "no_cached_input"})
            continue
        spu_scan(body, X_body, pol, step, eligible, skips,
                 path=f"L{l}/u{j}", is_root=False)
    _count_skips(host, skips)
    events = []
    if eligible:
        import numpy as np
        mse_before = None
        if host.mode == "numeric":
            ya = host._bk.ingest(y).reshape(-1, 1)
            mse_before = float(((host.predict(X) - ya) ** 2).mean())
        for unit, X_unit, unit_path in eligible:
            ev = self_process(unit, X_unit, pol, step)
            ev["path"] = unit_path
            events.append(ev)
        summary = {"path": "__step__", "processed": len(eligible)}
        if mse_before is not None:
            ya = host._bk.ingest(y).reshape(-1, 1)
            summary["task_mse_before"] = mse_before
            summary["task_mse_after"] = float(
                ((host.predict(X) - ya) ** 2).mean())
        events.append(summary)
    if events:
        if not hasattr(host, "_spu_events"):
            host._spu_events = []
        host._spu_events.extend(events)
