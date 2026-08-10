"""SPUNetwork: integration WITHOUT touching net.py
(DESIGN_SPU v1.3 §4-5; DEV_PLAN S3, architecture ruling).

A subclass of the released Network. Only the ROOT is an
SPUNetwork; the grown end subnets remain plain Network objects
manipulated by the free functions of spu_loop. The override wraps train_step:
an SPU pre-pass walks the inclusion tree (standardizing level by
level exactly as the released input path does) and runs the
bounded solver at every eligible juvenile leaf; then the released
step body executes unchanged via super(). With no policy set, or
spu_enabled=False, behavior is bit-identical to Network by
construction.
"""
from ..growthpolicy import GROWTH_MODE_WIDEN_ONLY, get_growth_mode
from ..net import Network
from engine.spu.spu_loop import self_process, skip_reason
from engine.spu.spu_policy import validate_spu_policy


def spu_scan(scope, X_received, policy, step_index, eligible,
             skips, path="root", is_root=True):
    """Phase 1 (cheap, attribute checks only): walk the tree
    mirroring the released input path — each scope standardizes
    with ITS scaler and passes the result to its children — and
    collect (unit, batch, path) for every eligible end subnet
    plus the
    meaningful skips. No prediction, no processing."""
    reason = skip_reason(scope, len(X_received), policy, step_index,
                         is_root=is_root)
    if reason is None:
        eligible.append((scope, X_received, path))
        return
    # children = legacy inner bodies AND fullwidth port bodies
    # (doc 35 D9: bodies remain reference Networks; the walk sees
    # the same objects whichever coupling carries them)
    kids = list(scope.inner.items())
    port = getattr(scope, "_port_site", None)
    if port is not None:
        kids += [(f"port[{g}]", s["body"])
                 for g, s in enumerate(port.bodies)]
    for l, site in getattr(scope, "_port_sites", {}).items():
        kids += [(f"port[{l}.{g}]", s["body"])
                 for g, s in enumerate(site.bodies)]
    if kids and scope._x_mu is not None:
        Xs = scope._bk.standardize(X_received, scope._x_mu,
                                   scope._x_sd)
        for j, child in kids:
            spu_scan(child, Xs, policy, step_index, eligible,
                     skips, path=f"{path}/{j}", is_root=False)
    elif reason not in ("disabled", "not_this_step"):
        skips.append({"path": path, "skip": reason})


def spu_prepass(scope, X_received, policy, step_index, events,
                path="root", is_root=True):
    """Phase 1 + phase 2: scan, then process the eligible leaves.
    Kept as the single entry point (tests exercise it directly)."""
    eligible, skips = [], []
    spu_scan(scope, X_received, policy, step_index, eligible,
             skips, path=path, is_root=is_root)
    _count_skips(scope, skips)
    for unit, X_unit, unit_path in eligible:
        ev = self_process(unit, X_unit, policy, step_index)
        ev["path"] = unit_path
        events.append(ev)
    return len(eligible)


def _count_skips(holder, skips):
    """Skips are O(1) per-reason counters, not per-step event
    lines: a matured unit must not grow the holder's event list
    forever (the deferred v1.1 catch, landed post-equivalence)."""
    if not skips:
        return
    counts = getattr(holder, "_spu_skip_counts", None)
    if counts is None:
        counts = holder._spu_skip_counts = {}
    for s in skips:
        counts[s["skip"]] = counts.get(s["skip"], 0) + 1


def spu_pre_forward(holder, X, y, pol):
    """The v2.0 holder-level pre-forward walk — the exact v0
    SPUNetwork.train_step body, relocated. Called by the
    Network.train_step seam BEFORE the released body runs, so the
    theory's ordering (process -> participate -> learn) holds at
    every depth, on any host whose root is a Network."""
    if (holder._step_count % pol["spu_every"] == 0
            and get_growth_mode() != GROWTH_MODE_WIDEN_ONLY):
        eligible, skips = [], []
        spu_scan(holder, X, pol, holder._step_count, eligible, skips)
        _count_skips(holder, skips)
        events = []
        if eligible:
            # pay the interference predicts only when work runs
            mse_before = float(((holder.predict(X) - y) ** 2).mean())
            for unit, X_unit, unit_path in eligible:
                ev = self_process(unit, X_unit, pol,
                                  holder._step_count)
                ev["path"] = unit_path
                events.append(ev)
            mse_after = float(((holder.predict(X) - y) ** 2).mean())
            events.append({"path": "__step__",
                           "task_mse_before": mse_before,
                           "task_mse_after": mse_after,
                           "processed": len(eligible)})
        if events:
            if not hasattr(holder, "_spu_events"):
                holder._spu_events = []
            holder._spu_events.extend(events)


def install_spu_policy(obj, policy):
    """Attach a validated SPU policy to any supported holder.
    Dispatch (DESIGN v2.0 §2.2): Network (incl. MSOrgan-family
    hosts, which subclass Network) -> direct; unknown -> refused
    loudly. The transformer host gets its trigger in phase V3."""
    if policy is not None:
        policy = validate_spu_policy(policy)
    if isinstance(obj, Network):
        obj._spu_policy = policy
        return policy
    if (hasattr(obj, "inner") and hasattr(obj, "_t")
            and hasattr(obj, "predict") and hasattr(obj, "mode")):
        # duck-typed attention host (substrate contract; reference_net
        # must not import core, so the check is structural)
        obj._spu_policy = policy
        return policy
    raise TypeError(
        f"install_spu_policy: unsupported holder {type(obj).__name__}"
        " (Network family or attention-host contract required)")


class SPUNetwork(Network):
    """Pure facade (v2.0): the base Network now carries the
    pre-forward walk; this class remains for construction and
    pickle back-compat."""

    def set_spu_policy(self, policy):
        return install_spu_policy(self, policy)

    def get_spu_policy(self):
        return getattr(self, "_spu_policy", None)

    @property
    def spu_events(self):
        return getattr(self, "_spu_events", [])

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "_spu_policy"):
            self._spu_policy = None
        if not hasattr(self, "_spu_events"):
            self._spu_events = []
