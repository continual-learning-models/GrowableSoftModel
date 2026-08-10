"""SPU S0 tests: policy defaults and load-time validation
(DEV_PLAN_SPU S0; DESIGN_SPU v1.3 §10)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from engine.spu import DEFAULT_SPU_POLICY, validate_spu_policy  # noqa: E402


def test_defaults_present_and_off():
    assert DEFAULT_SPU_POLICY["spu_enabled"] is False
    assert DEFAULT_SPU_POLICY["spu_scope"] == "newborn_inner_nets"
    assert len(DEFAULT_SPU_POLICY) == 14   # no floor (owner 07-10)


def test_none_policy_yields_defaults():
    assert validate_spu_policy(None) == DEFAULT_SPU_POLICY
    assert validate_spu_policy({}) == DEFAULT_SPU_POLICY


def test_valid_custom_accepted_verbatim():
    m = validate_spu_policy({"spu_enabled": True, "spu_K": 6,
                             "spu_scope": "all_inner_nets"})
    assert m["spu_enabled"] is True and m["spu_K"] == 6
    assert m["spu_scope"] == "all_inner_nets"
    assert m["spu_S_max"] == DEFAULT_SPU_POLICY["spu_S_max"]


@pytest.mark.parametrize("key,val", [
    ("spu_p_mask", 0.0), ("spu_p_mask", 1.0), ("spu_p_mask", -0.1),
    ("spu_rho_floor", 0.0), ("spu_rho_floor", 1.0),
    ("spu_S_max", 0), ("spu_K", 1), ("spu_newborn_steps", 0),
    ("spu_n_min", 0), ("spu_every", 0), ("spu_eta", 0.0),
    ("spu_clip", 0.0), ("spu_tau_rel", -1e-9), ("spu_gamma", -0.1),
    ("spu_scope", "root"), ("spu_enabled", "yes"),
    ("spu_warmup_steps", -1), ("spu_warmup_steps", 300),
])
def test_each_violation_refused_with_key_named(key, val):
    with pytest.raises(ValueError) as e:
        validate_spu_policy({key: val})
    assert key in str(e.value)


def test_unknown_key_refused():
    with pytest.raises(ValueError) as e:
        validate_spu_policy({"spu_typo": 1})
    assert "unknown" in str(e.value) and "spu_typo" in str(e.value)


def test_validation_does_not_mutate_input():
    p = {"spu_K": 5}
    validate_spu_policy(p)
    assert p == {"spu_K": 5}


def test_sp1_default_is_ten_percent():
    from engine.spu.spu_policy import DEFAULT_SPU_POLICY
    assert DEFAULT_SPU_POLICY["spu_p_mask"] == 0.10
    # NO node floor exists (owner ruling 2026-07-10): the
    # perturbation is PROBABILISTIC and self-governing; a hard
    # floor would make the behavior deterministic
    assert "spu_min_hidden" not in DEFAULT_SPU_POLICY


def test_sp1_mask_rate_at_ten_percent():
    import numpy as np
    from engine.spu.spu_objective import draw_masks
    m = draw_masks(np.random.default_rng(3), 400, 16, 0.10)
    drop = 1 - m.mean()
    assert abs(drop - 0.10) < 0.02          # rate honest
    perturbed = (m.min(axis=1) < 1).mean()   # >=1 unit masked
    assert perturbed > 0.75                  # H=16 healthy
