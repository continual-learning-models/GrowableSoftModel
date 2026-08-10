"""The `mlp` substrate: the shipped MSOrgan, wrapped into the contract.

Composition per R-SYS2: MSOrgan already subclasses the FROZEN Phase-2
recursive Network; this module adds the contract surface (growth_sites /
grow_site / self-describing save) WITHOUT changing MSOrgan behavior —
the full existing test suite remains the behavioral anchor.
"""
from __future__ import annotations

import json
from pathlib import Path

from core._modules import reference_net  # noqa: F401
from reference_net.trainer import collect_instability

from core.substrate import MSOrgan
from core.substrates.base import Substrate, CONTRACT_V


class MLPSubstrate(MSOrgan, Substrate):
    NAME = "mlp"
    DATA_FORM = "vector"
    SUPPORTED_HEADS = ("point", "dist")   # 60A (GSM-I3 head)

    # ---- contract adapters over existing semantics ----
    def growth_sites(self):
        """Ranked refinable sites; already-composite nodes are excluded
        (their own inner nodes appear instead, via the recursion)."""
        rows = sorted(collect_instability(self), key=lambda r: -r[2])
        return [(f"{path}[{j}]", float(score))
                for path, j, score, owner in rows
                if j not in owner.inner
                and j not in getattr(owner, "_port_js", set())]

    def grow_site(self, site_path, hidden=16, body_type=None):
        path, j = site_path.rsplit("[", 1)
        j = int(j.rstrip("]"))
        owner = self
        if path != "root":
            for hop in path.split("/"):
                owner = owner.grown_body(int(hop))
                if owner is None:
                    raise KeyError(int(hop))
        owner.grow(j, hidden=hidden, body_type=body_type)
        return {"grown": site_path, "params": self.n_params(),
                "depth": self.depth()}

    def shape_record(self):
        rec = super().shape_record()
        rec["substrate"] = self.NAME
        return rec

    def perturb(self, rng, sigma):
        """Contract helper for practice attempts: jittered copy."""
        import copy
        p = copy.deepcopy(self)
        # jitter via the 2a round-trip (numpy rng -> identical
        # draws on every backend; GSM-I2)
        W2 = p._bk.to_numpy(p.W2)
        W1 = p._bk.to_numpy(p.W1)
        p.W2 = p._bk.ingest(W2 + rng.normal(0, sigma, W2.shape))
        p.W1 = p._bk.ingest(W1 + rng.normal(0, sigma, W1.shape))
        return p

    # ---- self-describing artifact ----
    def save(self, dir_path):
        super().save(dir_path)
        (Path(dir_path) / "substrate.json").write_text(json.dumps(
            {"substrate": self.NAME, "contract": CONTRACT_V}))

    @staticmethod
    def load(dir_path):
        return MSOrgan.load(dir_path)
