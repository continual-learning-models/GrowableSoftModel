"""The Substrate Contract (SUBSTRATE_ARCHITECTURE v2, Section 4).

Every model body implements this interface; nothing upstream may assume
more. Contract SEMANTICS every substrate must honor:
- oscillation-based instability (local per-site statistics),
- zero-init function-preserving growth at any depth,
- SGD consolidation mode (sgd_lr) that is a no-op at convergence,
- deterministic under seed,
- self-describing artifacts {substrate, contract} via save/load.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

CONTRACT_V = 1


class Substrate(ABC):
    """Abstract model body. DATA_FORM declares the input form served."""

    DATA_FORM: str = "vector"          # vector | sequence | grid | graph
    # 60A L2: output-head whitelist — a substrate DECLARES
    # what it supports; the birth door refuses the rest
    SUPPORTED_HEADS: tuple = ("point",)
    NAME: str = "abstract"

    # ---- learning ----
    @abstractmethod
    def train_step(self, X, y, sgd_lr=None):
        """One coupled learning step (recursive across scales)."""

    # ---- serving ----
    @abstractmethod
    def predict(self, X):
        """Numeric-mode predictions (n, 1)."""

    @abstractmethod
    def predict_proba(self, X):
        """Categorical-mode probabilities (n, n_classes)."""

    @abstractmethod
    def predict_label(self, X):
        """Categorical-mode (labels, confidences)."""

    @abstractmethod
    def add_class(self, label):
        """Vocabulary growth (function-preserving within epsilon)."""

    # ---- numeric uncertainty (GSM-I3; OPTIONAL — deliberately not
    # abstract so hosts without the mode stay valid; additive,
    # CONTRACT_V unchanged) ----
    def predict_dist(self, X):
        """numeric_dist-mode (value, std) arrays."""
        raise NotImplementedError(
            f"substrate '{self.NAME}' has no numeric_dist head")

    # ---- multi-scale growth ----
    @abstractmethod
    def growth_sites(self):
        """Ranked candidates: [(site_path, instability)], ALL depths."""

    @abstractmethod
    def grow_site(self, site_path, hidden=16, body_type=None):
        """Refine one site into an inner network (function-
        preserving). body_type (S9.5): explicit body family where
        the site grows a Network body; None -> policy default."""

    # ---- introspection ----
    @abstractmethod
    def depth(self):
        ...

    @abstractmethod
    def n_params(self):
        ...

    @abstractmethod
    def shape_record(self):
        """{mode, vocab?, depth, params, d_in, hidden, substrate}."""

    # ---- artifact ----
    @abstractmethod
    def save(self, dir_path):
        """Self-describing artifact (substrate name + contract version)."""

    @staticmethod
    @abstractmethod
    def load(dir_path):
        ...
