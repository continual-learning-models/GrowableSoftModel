"""EvaluativeCore — the L0 object-agnostic algorithm kernel
(doc 86 v1.23 §3.0: one Evaluative Update Core instantiated by
both loops through L1 adapters).

LAYERING LAW: this module operates on PLAIN NUMBERS/ARRAYS ONLY.
It must never import a substrate, preference, context or trainer
type — the static purity referee (TB-C05) scans this file's
imports. Both L1 adapters (StructureAdapter = preference.py's
M-parts; WeightAdapter = the Track-B trainer family) delegate
their shared arithmetic here.

Normative equations: doc 83 M1 (discounted sufficient-statistics
fold), doc 83 §4.2 (normalization set + clip envelope), doc 83
M3/§4.1 and GAE weighting (one exponential weighting utility),
doc 83 M4 (seeded draw form).
"""
import numpy as np

__all__ = ["advantage", "normalized_multiplier", "credit_weights",
           "credit_fold", "ema_fold", "seeded_draw"]


def advantage(realized, baseline):
    """Elementwise advantage: realized - baseline (both loops'
    ADVANTAGE component; baselines differ per adapter)."""
    return np.asarray(realized, dtype=float) - \
        np.asarray(baseline, dtype=float)


def normalized_multiplier(raw, mu_ref, sd_ref, lo, hi):
    """Reference-distribution normalization + envelope (doc 83
    §4.2 v1.17): the raw score is standardized against the
    OBSERVED ADVANTAGE DISTRIBUTION (mu_ref, sd_ref), never
    against the other candidates:
        z = (raw - mu_ref) / sd_ref
        multiplier = clip(exp(z), lo, hi)
    Magnitude-aware and per-value independent. DEGENERATE EDGE
    (tolerance law, never exact-zero float tests):
    sd_ref <= 1e-9 * max(1, |mu_ref|)  =>  1.0 exactly."""
    sd = float(sd_ref)
    mu = float(mu_ref)
    if not np.isfinite(sd) or sd <= 1e-9 * max(1.0, abs(mu)):
        return 1.0
    z = (float(raw) - mu) / sd
    return float(np.clip(np.exp(z), float(lo), float(hi)))


def credit_weights(spec):
    """ONE weighting utility for both credit-weight families:
      {"kind": "list", "weights": [...]}   -> verbatim array
      {"kind": "exp", "base": b, "n": n}   -> [b^0 .. b^(n-1)]
    (K-window decay weights and (gamma*lambda)^k GAE weights are
    both exponential families from here — doc 86 §3.0)."""
    kind = spec.get("kind")
    if kind == "list":
        return np.asarray(spec["weights"], dtype=float)
    if kind == "exp":
        b, n = float(spec["base"]), int(spec["n"])
        return np.power(b, np.arange(n, dtype=float))
    raise ValueError(f"unknown credit_weights kind: {kind!r}")


def credit_fold(values, weights, normalize):
    """Weighted fold of an array (what gets folded is L1's
    business — doc 86 §3.0 PRECISION):
      normalize=True  -> sum(w*v)/sum(w)   (K-window credited
                                            gain, doc 83 §4.1)
      normalize=False -> sum(w*v)          (GAE-style sum)"""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    s = float(np.dot(w, v))
    return s / float(w.sum()) if normalize else s


def ema_fold(stats, x, decay):
    """The discounted fold (doc 83 M1): semantics defined by the
    sufficient-statistics form
        w' = g*w + 1;  m' = (g*w*m + x)/w'
        v' = (g*w*(v + m^2) + x^2)/w' - m'^2
    IMPLEMENTED (v1.17 normative form) as the algebraically
    IDENTICAL stable West recursion
        d = x - m;  m' = m + d/w';  v' = (g*w*v + d*(x-m'))/w'
    (the naive second-moment form cancels catastrophically at
    |x| >> sd; equivalence refereed by TB-C03b).
    Empty state = (0, 0, 0); first fold => (1, x, 0)."""
    w, m, v = (float(s) for s in stats)
    g, x = float(decay), float(x)
    w2 = g * w + 1.0
    d = x - m
    m2 = m + d / w2
    v2 = max((g * w * v + d * (x - m2)) / w2, 0.0)
    return (w2, m2, v2)


def seeded_draw(rng, mean, se):
    """Seeded posterior draw (doc 83 M4): mean +
    se * rng.standard_normal(). DEGENERATE EDGE: se <= 0 returns
    mean EXACTLY and does NOT consume the generator (the logged
    degenerate draw; stream stays undisturbed)."""
    if se is None or float(se) <= 0.0:
        return float(mean)
    return float(mean) + float(se) * float(rng.standard_normal())
