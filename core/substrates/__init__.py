"""Substrate registry (SUBSTRATE_ARCHITECTURE v2, Section 5).

Factory and lifecycle instantiate bodies ONLY through this registry.
Artifacts are self-describing; loaders are NEVER removed — every model
ever produced stays loadable.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.substrates.base import Substrate, CONTRACT_V  # noqa: F401
from core.substrates.mlp import MLPSubstrate
from core.substrates.transformer import TransformerSubstrate
from core.substrates.sequence import SequenceSubstrate
from core.substrates.mlp_plus import MLPPlus
from core.substrates.transformer_plus import TransformerPlus
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate

REGISTRY: dict = {"mlp": MLPSubstrate,
                  "transformer": TransformerSubstrate,
                  "sequence": SequenceSubstrate,
                  # total-plasticity substrates (omega operator);
                  # additive entries — existing keys never replaced
                  "mlp_plus": MLPPlus,
                  "transformer_plus": TransformerPlus,
                  # growable-attention host (attention-build S1+)
                  "growable_attention": GrowableAttentionSubstrate}

# machine-readable guidance served by list_substrates (Section 2 guide)
GUIDANCE = {
    "mlp_plus": {
        "data_form": "vector",
        "strengths": "mlp plus the omega widening operator: outward "
                     "growth at every scale (total plasticity); "
                     "behavior identical to mlp until widen() is used",
        "typical_domains": ["as mlp, plus narrow-born models that must "
                            "outgrow their first-batch capacity"],
        "relative_cost": "lowest",
        "status": "experimental (total-plasticity track)",
    },
    "growable_attention": {
        "data_form": "vector",
        "strengths": "attention host whose heads are GROWN, not "
                     "molded: heads can be added and each head's "
                     "width grows independently (unequal sizes), "
                     "function-preserving at every step; per-head "
                     "instruments and an optional entropy-band "
                     "self-processing discipline",
        "typical_domains": ["relation-heavy tables whose relation "
                            "count is unknown up front or drifts "
                            "in service"],
        "relative_cost": "as transformer; capacity follows demand",
        "status": "experimental (attention-build complete: "
                  "training, growth verbs, self-processing, "
                  "instruments, governance; S9.2 adds the "
                  "categorical mode; 60A adds the numeric_dist "
                  "uncertainty head — numeric, categorical and "
                  "numeric_dist all serve)",
    },
    "transformer_plus": {
        "data_form": "vector",
        "strengths": "transformer plus omega (uniform FFN widening + "
                     "inner-scale widening); behavior identical to "
                     "transformer until widen_ffn() is used",
        "typical_domains": ["as transformer, plus capacity-locked "
                            "cases"],
        "relative_cost": "medium",
        "status": "experimental (total-plasticity track)",
    },
    "mlp": {
        "data_form": "vector",
        "strengths": "fixed feature tables with weak inter-feature "
                     "relations; cheapest and fastest; validated core",
        "typical_domains": ["risk scoring", "simple hidden laws",
                            "business judgment"],
        "relative_cost": "lowest",
        "status": "shipped",
    },
    "sequence": {
        "data_form": "sequence",
        "strengths": "timed signals: causal attention over ordered steps "
                     "(no future leakage); trends, regimes, precursors; "
                     "multi-scale growth on FFN units",
        "typical_domains": ["fault-stress time series", "physical "
                            "signals", "campaign curves", "ECG/EEG "
                            "windows"],
        "relative_cost": "medium",
        "status": "shipped",
    },
    "transformer": {
        "data_form": "vector",
        "strengths": "relation-heavy tables: attention lets features "
                     "consult each other (interactions, couplings, "
                     "compositions); multi-scale growth on FFN units",
        "typical_domains": ["math problems", "physics research",
                            "business/marketing/medical/cognitive "
                            "data analysis", "seismic catalogs"],
        "relative_cost": "medium",
        "status": "shipped",
    },
}


def get_substrate(name: str):
    if name not in REGISTRY:
        return None                       # caller turns this into a refusal
    return REGISTRY[name]


def load_artifact(dir_path):
    """Dispatch by the artifact's own self-description."""
    d = Path(dir_path)
    meta = d / "substrate.json"
    name = "mlp"                          # legacy artifacts predate the tag
    if meta.exists():
        name = json.loads(meta.read_text())["substrate"]
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"artifact requires unregistered substrate: {name}")
    return cls.load(d)
