"""Factory wiring (IWP2/S2.1): the Phase-1 factory runs on the MSOrgan
substrate — via subclassing ONLY (frozen modules untouched, R-SYS2).

- SysTrainer._train_organ: builds/trains an MSOrgan (recursive substrate)
  instead of TinyMLP; self-shaping (infer_shape), full-replay stores,
  windowing, rule mining — all reused from the frozen parent verbatim.
- SysModelManager: loads msorgan.pkl artifacts; inference semantics
  identical to Phase-1 (numeric value / label+confidence / rule citation).
- SysFactory: wires the Sys* components; gate, registry, holdout stream,
  drift and all public methods are the frozen parent's own code.
"""
from __future__ import annotations

import json
from typing import Any

from core._modules import generator  # noqa: F401  (shim)
from generator.config import Config
from generator.data import featurize, read_jsonl, write_jsonl, recent_slice
from generator.registry import ModelRegistry
from generator.model_manager import ModelManager
from generator.trainer import Trainer, infer_shape, auto_hidden
from generator.evaluator import Evaluator
from generator.evolve import Evolve
from generator.factory import SoftModelFactory

from core.substrate import MSOrgan  # noqa: F401 (legacy alias)
from core.substrates import get_substrate, load_artifact


class SysTrainer(Trainer):
    def _train_organ(self, model_id, examples, parent, version, window=None):
        import numpy as np
        from generator.rules import induce_rules

        store = read_jsonl(self._store_path(model_id, parent)) + list(examples)
        train_rows = recent_slice(store, window)
        if not train_rows:
            raise ValueError("no examples to learn from")
        # 60AA P2 (D2-1): FAITHFUL class resolution — the
        # model's own policy names the substrate; the legacy
        # substrate_name knob stays as an EXPLICIT override
        pol = {}
        pol_p = self.registry.model_dir(model_id) / "policy.json"
        if pol_p.exists():
            import json as _j
            pol = _j.loads(pol_p.read_text())
        _override = getattr(self, "substrate_name", None)
        _sub = _override or pol.get("substrate") or "mlp"
        cls = get_substrate(_sub) or get_substrate("mlp")
        if cls.DATA_FORM != "vector":            # D2-2 boundary
            raise ValueError(
                f"the teach lane serves vector models; substrate "
                f"'{_sub}' serves '{cls.DATA_FORM}' data — use "
                f"study for {cls.DATA_FORM} models")
        shape = infer_shape(train_rows)
        features = shape["features"]
        if not features:
            raise ValueError("examples carry no feature keys to learn from")

        X = np.array([featurize(ex["input"], features) for ex in train_rows])
        hidden_cfg = (tuple(self.config.hidden_sizes)
                      if self.config.hidden_sizes else None)

        mode = shape["mode"]
        if mode == "numeric" and \
                pol.get("numeric_head") == "dist":   # GSM-I3
            mode = "numeric_dist"
        if mode == "numeric_dist" and "dist" not in getattr(
                cls, "SUPPORTED_HEADS", ("point",)):
            # 60A L2: EVERY birth lane checks the whitelist —
            # this teach/factory lane was a second door without it
            raise ValueError(
                f"substrate '{getattr(cls, 'NAME', cls.__name__)}'"
                f" does not support numeric_head='dist'; supported"
                f" heads: {sorted(getattr(cls, 'SUPPORTED_HEADS',
                                          ('point',)))}")
        # D2-3: the study lane's OWN kwargs filter (in-function
        # import — lifecycle imports this module at module level);
        # empty policy yields exactly {"seed": config seed}, so
        # the default path is argument-identical (D2-4, judged by
        # the equivalence battery)
        from core.lifecycle import _substrate_kwargs
        kw = _substrate_kwargs(pol, cls, self.config.seed)
        if mode in ("numeric", "numeric_dist"):
            y = np.array([[float(ex["target"])] for ex in train_rows])
            h = (hidden_cfg or auto_hidden(len(features), 1, len(train_rows)))[0]
            organ = cls(len(features), h, mode=mode, **kw)
        else:
            vocab = shape["vocab"]
            y = np.array([str(ex["target"]) for ex in train_rows])
            h = (hidden_cfg or auto_hidden(len(features), len(vocab),
                                           len(train_rows)))[0]
            organ = cls(len(features), h, mode="categorical",
                        vocab=vocab, **kw)
        # minibatch schedule mirroring Phase-1 TinyMLP.fit (epochs x batches)
        rng = np.random.default_rng(self.config.seed)
        bs = self.config.batch_size
        for _ in range(self.config.epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), bs):
                idx = order[i:i + bs]
                organ.train_step(X[idx], y[idx])

        out_dir = self.registry.weights_dir(model_id, version)
        organ.save(out_dir)
        shape = {**shape, **{k: v for k, v in organ.shape_record().items()
                             if k in ("depth", "params")}}
        shape["mode"] = organ.mode      # GSM-I3 (identity for numeric)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "shape.json").write_text(json.dumps(shape))

        if shape["mode"] == "categorical":
            rule_list = induce_rules(
                train_rows, features, shape["vocab"],
                max_conditions=self.config.max_rule_conditions,
                min_support=self.config.min_rule_support,
                min_confidence=self.config.min_rule_confidence)
            rule_list.save(out_dir / "rules.json")

        write_jsonl(self._store_path(model_id, version), store)
        self.mm.invalidate_cache(model_id)


class SysModelManager(ModelManager):
    def _load(self, model_id: str, version: str):
        key = (model_id, version)
        if key in self._cache:
            return self._cache[key]
        wdir = self.registry.weights_dir(model_id, version)
        loaded = None
        if (wdir / "msorgan.pkl").exists() and (wdir / "shape.json").exists():
            rules = None
            if (wdir / "rules.json").exists():
                from generator.rules import RuleList
                rules = RuleList.load(wdir / "rules.json")
            loaded = (load_artifact(wdir),
                      json.loads((wdir / "shape.json").read_text()),
                      rules)
        self._cache[key] = loaded
        return loaded

    def predict_dist(self, model_id: str, version: str,
                     input_: Any) -> dict:
        """GSM-I1, substrate arm (mirrors _organ_infer's routing
        exactly — the base-class version serves TinyMLP organs;
        this override serves msorgan.pkl substrates, incl. the
        sequence data form)."""
        import numpy as np
        loaded = (None if version == "v0"
                  else self._load(model_id, version))
        if loaded is None:
            return {"kind": "none", "note": "untrained",
                    "version": version}
        organ, shape, _ = loaded
        if shape.get("data_form") == "sequence":
            x = np.asarray([input_], float)
        else:
            x = np.array([featurize(input_, shape["features"])])
        if shape["mode"] == "numeric_dist":            # GSM-I3
            v, sd = organ.predict_dist(x)
            return {"kind": "numeric_dist",
                    "value": float(v[0]), "std": float(sd[0]),
                    "version": version}
        if shape["mode"] == "numeric":
            return {"kind": "numeric",
                    "value": float(organ.predict(x)[0, 0]),
                    "version": version}
        probs = organ.predict_proba(x)[0]
        return {"kind": "categorical",
                "labels": list(shape["vocab"]),
                "probs": [float(v) for v in probs],
                "version": version}

    def _organ_infer(self, model_id: str, version: str, input_: Any) -> dict:
        import numpy as np
        loaded = None if version == "v0" else self._load(model_id, version)
        if loaded is None:
            return {"output": None, "confidence": 0.0, "note": "untrained"}
        organ, shape, rules = loaded
        if shape.get("data_form") == "sequence":
            x = np.asarray([input_], float)
        else:
            x = np.array([featurize(input_, shape["features"])])
        if shape["mode"] in ("numeric", "numeric_dist"):
            value = float(organ.predict(x)[0, 0])
            if shape.get("integer"):
                value = int(round(value))
            return {"output": value, "confidence": None}
        labels, conf = organ.predict_label(x)
        result = {"output": labels[0], "confidence": round(float(conf[0]), 4)}
        if rules is not None:
            cited = rules.predict(dict(zip(shape["features"], x[0].tolist())))
            if cited["output"] == result["output"] and cited.get("rule"):
                result["rule"] = cited["rule"]
        return result


class SysFactory(SoftModelFactory):
    """The Phase-1 factory surface, running on the multi-scale substrate."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self.registry = ModelRegistry(self.config)
        self.model_manager = SysModelManager(self.config, self.registry)
        self.trainer = SysTrainer(self.config, self.registry,
                                  self.model_manager)
        self.evaluator = Evaluator(self.config, self.model_manager,
                                   self.registry)
        self.evolve_ctl = Evolve(self.config, self.registry,
                                 self.model_manager, self.trainer,
                                 self.evaluator)
