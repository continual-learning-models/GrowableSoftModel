"""Threshold decision combiner (DEV_PLAN A6): predict first
(Tier 1, on the dense energy series, gated by the predictability
certificate), probe to verify (Tier 2, asymptote-read), the gate
adopts (outside this module, next round). Every branch emits its
reasons; every intermediate value lands in the decision dict."""
from . import OP_DELTA, OP_OMEGA
import numpy as np

from .interfaces import DecisionCombiner, register, site_widen


class ThresholdCombiner(DecisionCombiner):
    NAME = "threshold_policy"

    def decide(self, scope, parts, policy):
        d = {"reasons": [], "certificate": {}, "tier_used": None,
             "arm": None, "site": None, "apply_as": None}
        energy = np.asarray(scope.energy_ring, dtype=float)
        gains = [r["gain"] for r in scope.gain_ledger
                 if r.get("gain") is not None]
        # ---- guards (cold start) ----
        if (len(energy) < policy["min_energy_points"]
                or len(gains) < policy["min_ledger_events"]):
            d["reasons"].append(
                f"cold start: energy={len(energy)} events={len(gains)}")
            if len(scope.window_ring) < policy["min_window_rows"]:
                d.update(tier_used="cold_start_default", arm=OP_OMEGA)
                d["reasons"].append(
                    "window too small for probes: additive default (P1)")
                self._site(scope, d)
                return d
            return self._tier2(scope, parts, policy, d,
                               why="cold start with usable window")
        # ---- predictability certificate ----
        _ml = int(policy.get("instrument_min_len", 64))
        fc = parts["forecastability"].score(
            energy, threshold=policy["forecastability_min"],
            min_len=_ml)
        cp = parts["changepoint"].detect(
            energy, threshold=policy["changepoint_max"],
            min_len=_ml,
            recent=int(policy.get("bocpd_recent", 32)))
        bt = parts["backtest"].skill(
            parts["extrapolator"], energy,
            threshold=policy["backtest_max_err"], min_len=_ml)
        ex = parts["extrapolator"].fit(energy, seed=policy["seed"],
                                       min_len=_ml)
        conf_ok = ("refusal" not in ex
                   and ex["rel_ci_width"] <= policy["max_rel_ci_width"])
        d["certificate"] = {"forecastability": fc, "changepoint": cp,
                            "backtest": bt,
                            "extrapolation": ex, "confidence": conf_ok}
        checks = [fc.get("passed", False), cp.get("passed", False),
                  bt.get("passed", False), conf_ok]
        if not all(checks):
            return self._tier2(scope, parts, policy, d,
                               why=f"certificate failed {checks}")
        # ---- Tier 1 ----
        k, eps = policy["stall_k"], policy["stall_eps"]
        stall = len(gains) >= k and all(g < eps for g in gains[-k:])
        norms = [nc[0] for nc in list(scope._cos_series)[-16:]]
        saturation = (len(norms) >= 8
                      and float(np.median(norms))
                      < policy["saturation_norm"]
                      and (scope.residual_energy() or 0.0)
                      > policy["energy_floor"])
        d["signals"] = {"stall": stall, "saturation": saturation,
                        "recent_gains": gains[-k:],
                        "median_update_norm":
                            (float(np.median(norms)) if norms else None)}
        if not stall and ex["p_useful"] >= 0.5:
            d.update(tier_used="tier1", arm=OP_OMEGA)
            d["reasons"].append(
                f"no stall and additive route pays "
                f"(p_useful={ex['p_useful']:.2f})")
            self._site(scope, d)
            return d
        if stall and saturation:
            d.update(tier_used="tier1", arm=OP_DELTA)
            d["reasons"].append(
                "stall + saturation: class-exhaustion signature")
            return d
        return self._tier2(scope, parts, policy, d,
                           why="tier-1 ambiguous")

    def _tier2(self, scope, parts, policy, d, why):
        d["reasons"].append(f"tier 2: {why}")
        prices = parts["pricer"].price(scope, policy)
        if "refusal" in prices:
            d.update(tier_used="tier2_refused", arm=OP_OMEGA)
            d["reasons"].append(
                f"pricer refusal ({prices['refusal']}): additive "
                "default (P1)")
            self._site(scope, d)
            return d
        E = parts["extrapolator"]
        fw = E.fit(np.asarray(prices["widen_curve"]),
                   seed=policy["seed"])
        fd = E.fit(np.asarray(prices["deepen_curve"]),
                   seed=policy["seed"] + 1)
        d["prices"] = {OP_OMEGA: fw, OP_DELTA: fd,
                       "probe_steps": prices.get("steps"),
                       "probe_cost_total": 2 * prices.get("steps", 0)}
        if "refusal" in fw or "refusal" in fd:
            d.update(tier_used="tier2", arm=OP_OMEGA)
            d["reasons"].append(
                "probe curve unusable (diverged or too short): "
                f"widen={fw.get('refusal', 'ok')} "
                f"deepen={fd.get('refusal', 'ok')} — additive "
                "default (P1), stated explicitly")
            self._site(scope, d)
            return d
        aw = fw.get("asymptote", np.inf)
        ad = fd.get("asymptote", np.inf)
        overlap = not (fw.get("ci_high", np.inf) < fd.get("ci_low",
                                                          -np.inf)
                       or fd.get("ci_high", np.inf) < fw.get("ci_low",
                                                             -np.inf))
        aw_cmp, ad_cmp = aw, ad
        # ---- preference seam (doc 83 §4.3; ~10 lines): consult
        # the preference role, multiply the arm scores, log ----
        from . import preference as _pref
        if _pref.preference_enabled(policy):
            er = np.asarray(scope.energy_ring, dtype=float)
            e_ref = float(er[-1]) if er.size else \
                float(scope.residual_energy() or 0.0)
            adj, _explored = _pref.seam_adjust(
                scope, parts, policy, d, fw, fd, e_ref)
            aw_cmp, ad_cmp = -adj[OP_OMEGA], -adj[OP_DELTA]
        d["tier_used"] = "tier2"
        if overlap or aw_cmp <= ad_cmp:
            d["arm"] = OP_OMEGA
            d["reasons"].append(
                "widen wins or tie within CI (additive default, P1): "
                f"asymptotes w={aw:.4g} d={ad:.4g}")
            self._site(scope, d)
        else:
            d["arm"] = OP_DELTA
            d["reasons"].append(
                f"deepen predicts lower energy: w={aw:.4g} d={ad:.4g}")
        return d

    @staticmethod
    def _site(scope, d):
        site, apply_as, note = site_widen(scope)
        d["site"], d["apply_as"] = site, apply_as
        if note:
            d["reasons"].append(note)


register("combiner", ThresholdCombiner.NAME, ThresholdCombiner)
