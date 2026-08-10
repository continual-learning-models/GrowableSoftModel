"""Role contracts and registries (DESIGN_DEEPEN section 6).

Every algorithmic role is a REPLACEABLE PART behind a named
interface. Selection is an explicit operator act via the policy
dict; the machinery never switches parts on its own (P8). v1 ships
exactly one implementation per role; parts self-register at import.
All parts are deterministic under the seeds they are handed and
return plain dicts so decision logs can store them verbatim.
"""
ROLES = ("extrapolator", "forecastability", "changepoint",
         "backtest", "pricer", "combiner", "preference")

# Growth-mode system control (DESIGN section 12). Named constants —
# no magic strings. The mode is enforced in decide(), ABOVE the
# replaceable-parts layer, so no part swap can bypass it.
GROWTH_MODE_ADAPTIVE = "adaptive"
GROWTH_MODE_WIDEN_ONLY = "widen_only"
VALID_GROWTH_MODES = frozenset(
    {GROWTH_MODE_ADAPTIVE, GROWTH_MODE_WIDEN_ONLY})


def site_widen(scope):
    """Legacy widen siting shared by decide() and combiners:
    u_j argmax over non-composite nodes (rho), omega fallback when
    every node is composite. Returns (site, apply_as, note)."""
    u = scope.instability()
    grown = getattr(scope, "_port_js", set())  # fullwidth grows
    free = [j for j in range(scope.H)
            if j not in scope.inner and j not in grown]
    if free:
        return int(max(free, key=lambda j: u[j])), "rho", None
    return None, "omega", ("all nodes composite: widen applies as "
                           "omega (core wiring next round)")

_REGISTRY = {}


def register(role, name, cls):
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    _REGISTRY.setdefault(role, {})[name] = cls


def list_available(role):
    return sorted(_REGISTRY.get(role, {}))


def get(role, name):
    """Instance of the named part, or a REFUSAL DICT (never raises
    to the caller): unknown parts are a logged refusal, not a crash."""
    cls = _REGISTRY.get(role, {}).get(name)
    if cls is None:
        return {"refusal": f"unknown part {role}/{name}",
                "available": list_available(role)}
    return cls()


class Extrapolator:
    """fit(series) -> {asymptote, ci_low, ci_high, p_useful,
    rel_ci_width, family, params} | {refusal};
    predict_at(fit_result, t) -> float (fitted curve value)."""
    def fit(self, series, seed=0, margin_frac=0.02):
        raise NotImplementedError

    def predict_at(self, fit_result, t):
        raise NotImplementedError


class Forecastability:
    """score(series) -> {score in [0,1], passed} | {refusal}."""
    def score(self, series, threshold=0.4):
        raise NotImplementedError


class ChangepointDetector:
    """detect(series) -> {p_recent_change, location, passed}."""
    def detect(self, series, threshold=0.2):
        raise NotImplementedError


class BacktestEvaluator:
    """skill(extrapolator, series) -> {error, passed}."""
    def skill(self, extrapolator, series, threshold=0.25):
        raise NotImplementedError


class ProbePricer:
    """price(scope, policy) -> {widen_curve, deepen_curve} |
    {refusal}. Probes run on copies; the living scope is never
    touched (fingerprint-asserted)."""
    def price(self, scope, policy):
        raise NotImplementedError


class DecisionCombiner:
    """decide(scope, parts, policy) -> decision dict (arm, site,
    tier_used, reasons, every intermediate value, part names)."""
    def decide(self, scope, parts, policy):
        raise NotImplementedError
