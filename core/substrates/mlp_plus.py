"""mlp_plus — the mlp substrate with the omega operator (Stage 3a).

Pure-method subclass (no new state): pickle round-trips as MLPPlus;
inherits the full mlp contract behavior unchanged."""
from __future__ import annotations

from core.substrates.mlp import MLPSubstrate
from core.plasticity.net_ops import widen_at


class MLPPlus(MLPSubstrate):
    NAME = "mlp_plus"

    def widen(self, path: str = "root", k: int = 1):
        """omega at any scale of this organ ('root' or 'j/k' inner
        path). Exact function preservation; new units are ordinary
        growth sites afterwards."""
        return widen_at(self, path, k)

    def add_feature(self, default: float = 0.0):
        """sigma at organ level: exact preservation; the new column is
        appended LAST in the organ's input order."""
        from core.plasticity.net_ops import add_feature_net
        return add_feature_net(self, default)
