"""S9.6 T25: PARAMETER_REFERENCE.md freshness (the T18 doc-
freshness pattern). Every key of every LIVE defaults dict must
appear in the manual; every facade verb must appear; the Tier-1
SMS forms must be documented. The catalog cannot silently rot.
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

DOC = (ROOT / "docs" / "PARAMETER_REFERENCE.md").read_text()


def test_t25a_model_policy_keys():
    from core.lifecycle import DEFAULT_POLICY
    for k in DEFAULT_POLICY:
        assert k in DOC, f"DEFAULT_POLICY key missing: {k}"


def test_t25b_attention_policy_keys():
    from core.substrates.growable_attention import POLICY
    assert len(POLICY) == 13          # census: additions must
                                      # (13 = +att_probe_steps,
                                      # doc 18 two-lane gate)
    for k in POLICY:                  # consciously widen this
        assert k in DOC, f"attention POLICY key missing: {k}"


def test_t25c_spu_policy_keys():
    from engine.spu.spu_policy import DEFAULT_SPU_POLICY
    assert len(DEFAULT_SPU_POLICY) == 14
    for k in DEFAULT_SPU_POLICY:
        assert k in DOC, f"SPU key missing: {k}"
    assert "spu_objective" in DOC     # the optional key too


def test_t25d_growth_policy_keys():
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    # 39 = 38 + eta_target (param-interface batch S2,
    # docs/system/22 item 5 — sanctioned pin update)
    # 41 = 39 + instrument_min_len + bocpd_recent (S5,
    # docs/system/22 items 30-34 — sanctioned pin update)
    # 43 = 41 + grow_port_type + grow_body_out_width (Growth
    # Interface Reform, doc 36 W3(a) — TW-1 sanctioned pin
    # update, keys + PARAMETER_REFERENCE rows same commit)
    assert len(DEFAULT_GROWTH_POLICY) == 43
    for k in DEFAULT_GROWTH_POLICY:
        assert k in DOC, f"growth key missing: {k}"


def test_t25e_every_facade_verb_documented():
    from core.facade import System
    for n, m in inspect.getmembers(System, inspect.isfunction):
        if n.startswith("_"):
            continue
        assert n in DOC, f"facade verb missing: {n}"


def test_t25f_substrate_params_allowed_sets_current():
    """The section-2 table must name every non-birth-derived
    constructor parameter of every registered substrate."""
    from core.substrates import REGISTRY
    for name, cls in REGISTRY.items():
        allowed = set(inspect.signature(
            cls.__init__).parameters) - {
            "self", "d_in", "hidden", "mode", "vocab"}
        for k in allowed:
            assert k in DOC, (name, k)


def test_t25g_tier1_forms_documented():
    assert "model_policy" in DOC
    assert "train_converge" in DOC
    assert "Tier 1" in DOC and "Tier 2" in DOC
    assert "sms-cli" in DOC
