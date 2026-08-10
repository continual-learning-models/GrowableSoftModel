"""MSOrgan — the system's model substrate (IWP1).

Composition per R-SYS2: subclass of the frozen Phase-2 recursive Network
(modules/ReferenceNet/reference_net/net.py, UNTOUCHED). Adds the Phase-1
self-shaping head semantics:
- mode="numeric": delegates entirely to the parent (n_out=1, MSE, scalers)
- mode="categorical": softmax/CE head over a learned vocabulary, with
  confidence; vocabulary can GROW (new logit row, zero weights, bias
  B_NEG) — function-preserving up to a documented epsilon.

Recursive growth, instability stats, scaler/optimizer lessons and the
SGD consolidation mode are INHERITED, not reimplemented. Inner networks
remain the parent class (residuals are numeric), so cross-scale target
handoff recurses exactly as in Phase 2.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from core._modules import reference_net  # noqa: F401  (shim)
from reference_net.net import Network, _Adam, gelu_d

B_NEG = -10.0   # new-class logit bias: e^-10 mass -> old distribution
                # preserved within epsilon < 1e-3 (documented, tested)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MSOrgan(Network):
    """Recursive multi-scale organ with self-shaping heads."""

    def __init__(self, d_in: int, hidden: int, mode: str = "numeric",
                 vocab=None, lr: float = 1e-2, seed: int = 7,
                 new_class_bias=None, nll_clamp=None):
        super().__init__(d_in, hidden, lr=lr, seed=seed)
        if new_class_bias is not None:
            # docs/system/22 item 7: vocabulary-entry logit bias
            # (epsilon leakage e^bias); None -> module B_NEG (-10)
            if isinstance(new_class_bias, bool) or not isinstance(
                    new_class_bias, (int, float)) or new_class_bias > 0:
                raise ValueError(
                    "new_class_bias must be a number <= 0; "
                    f"got {new_class_bias!r}")
            self.new_class_bias = float(new_class_bias)
        if nll_clamp is not None:
            # docs/system/22 item 28: numeric-dist variance clamp;
            # organ-owned, passed per kernel call (singleton safe)
            if isinstance(nll_clamp, bool) or not isinstance(
                    nll_clamp, (int, float)) or nll_clamp <= 0:
                raise ValueError(
                    f"nll_clamp must be a number > 0; got {nll_clamp!r}")
            self.nll_clamp = float(nll_clamp)
        self.mode = mode
        self.vocab = list(vocab) if vocab else []
        if mode == "categorical":
            n_out = max(1, len(self.vocab))
            rng = np.random.default_rng(seed + 1)
            W2 = rng.normal(0, np.sqrt(1.0 / hidden), (n_out, hidden))
            c = np.zeros(n_out)
            # GSM-I2: head params live on the backend like every
            # other param (previously raw numpy — mixed-type
            # crash under torch policies)
            self.W2 = self._bk.ingest(W2)
            self.c = self._bk.ingest(c)
            self.opt = _Adam([self.W1.shape, self.b1.shape,
                              W2.shape, c.shape], lr, self._bk)
        elif mode == "numeric_dist":
            # GSM-I3: heteroscedastic head (mu, log-variance).
            # BOTH rows born ZERO — birth honesty: the newborn
            # predicts the data mean with sigma_std = 1
            W2 = np.zeros((2, hidden))
            c = np.zeros(2)
            self.W2 = self._bk.ingest(W2)
            self.c = self._bk.ingest(c)
            self.opt = _Adam([self.W1.shape, self.b1.shape,
                              W2.shape, c.shape], lr, self._bk)

    # ---------- plasticity guards ----------
    def deepen(self, m=None, position=None, recipe=None,
               recipe_params=None, scope=None, zero_side=None,
               force=False):
        # Phi applies to the numeric readout chain; the categorical
        # and dist heads read Hact directly, so blocks would neither
        # train nor serve — refuse LOUDLY (silent no-op was the bug)
        if self.mode != "numeric":
            raise ValueError(
                f"deepen (Phi) applies to the numeric readout path; "
                f"mode '{self.mode}' heads read the hidden layer "
                f"directly (deepen for head modes is a declared "
                f"follow-up)")
        return super().deepen(m=m, position=position,
                              recipe=recipe,
                              recipe_params=recipe_params,
                              scope=scope, zero_side=zero_side,
                              force=force)

    # ---------- numeric-uncertainty head (GSM-I3) ----------
    def predict(self, X):
        if self.mode != "numeric_dist":
            return super().predict(X)
        value, _ = self.predict_dist(X)
        return value.reshape(-1, 1)

    def predict_dist(self, X):
        assert self.mode == "numeric_dist"
        if self._y_mu is None:
            raise ValueError(
                "untrained: predict_dist needs at least one "
                "training step (the target scalers are unfitted)")
        Xb = self._bk.ingest(np.asarray(X, float))
        _, Hact = self._hidden(self._std_x(Xb))
        mu, v = self._bk.nll_forward(self.W2, self.c, Hact,
                                       nll_clamp=getattr(
                                           self, 'nll_clamp', None))
        mu = self._bk.to_numpy(mu)
        v = self._bk.to_numpy(v)
        value = mu * self._y_sd + self._y_mu
        std = np.exp(v / 2.0) * self._y_sd
        return value, std

    def _train_step_dist(self, X, y, sgd_lr=None):
        # mirror of the categorical step, K16 kernels
        Xnp = np.asarray(X, float)
        ynp = np.asarray(y, float).ravel()
        Xb = self._bk.ingest(Xnp)
        if self._x_mu is None:              # fit scalers once
            self._x_mu, self._x_sd = self._bk.fit_x_stats(Xb)
        if self._y_mu is None:
            self._y_mu, self._y_sd = self._bk.y_stats(
                self._bk.ingest(ynp))
        t = self._bk.ingest((ynp - self._y_mu) / self._y_sd)
        Xs = self._std_x(Xb)
        A, Hact = self._hidden(Xs)
        mu, v = self._bk.nll_forward(self.W2, self.c, Hact,
                                       nll_clamp=getattr(
                                           self, 'nll_clamp', None))
        nll = self._bk.nll_loss(t, mu, v)
        grads, dH = self._bk.nll_backward(
            self.W1, self.b1, self.W2, self.c, Xs, A, Hact,
            mu, v, t,
            nll_clamp=getattr(self, 'nll_clamp', None))
        old_W1 = self._bk.copy(self.W1)
        params = [self.W1, self.b1, self.W2, self.c]
        if sgd_lr is None:
            self.W1, self.b1, self.W2, self.c = self.opt.step(
                params, grads)
        else:
            self.W1, self.b1, self.W2, self.c = \
                self._bk.sgd_step(params, grads, sgd_lr)
        dw = self._bk.to_numpy(self.W1 - old_W1)
        self._ema_dw = 0.95 * self._ema_dw + 0.05 * dw
        self._ema_adw = 0.95 * self._ema_adw + 0.05 * np.abs(dw)
        from reference_net.net import ETA_TARGET
        from reference_net.growthpolicy import \
            DEFAULT_GROWTH_POLICY as _gpe
        _gpe = getattr(self, "_growth_policy", _gpe)  # S9.5 D-N1
        _eta = float(_gpe.get("eta_target", ETA_TARGET))
        from reference_net.growth_port import legacy_handoff
        legacy_handoff(self.inner, Xs, Hact, A, dH,
                       self._bk.gelu, _eta, sgd_lr=sgd_lr)
        port = getattr(self, "_port_site", None)
        if port is not None:
            # TRUE gradient scale (pin P-2; kernel /n convention)
            port.backward_step(dH / max(len(Xs), 1), Xs, self.lr,
                               sgd_lr=sgd_lr)
        return nll

    # ---------- categorical head ----------
    def predict_proba(self, X):
        assert self.mode == "categorical"
        X = np.asarray(X, float)
        if self._x_mu is None:
            n = max(1, len(self.vocab))
            return np.full((len(X), n), 1.0 / n)
        Xb = self._bk.ingest(X)
        _, Hact = self._hidden(self._std_x(Xb))
        probs = self._bk.cat_forward(self.W2, self.c, Hact)
        return self._bk.to_numpy(probs)

    def predict_label(self, X):
        p = self.predict_proba(X)
        idx = p.argmax(axis=1)
        return [self.vocab[i] for i in idx], p[np.arange(len(idx)), idx]

    def add_class(self, label: str):
        """Vocabulary growth: zero-weight logit row, bias B_NEG.
        Old-class distribution preserved within epsilon (tested)."""
        assert self.mode == "categorical" and label not in self.vocab
        self.vocab.append(label)
        # 2a surgery: numpy round-trip + ingest (bitwise-exact
        # on every backend; zero row = silent new class)
        W2 = self._bk.to_numpy(self.W2)
        c = self._bk.to_numpy(self.c)
        self.W2 = self._bk.ingest(
            np.vstack([W2, np.zeros((1, self.H))]))
        self.c = self._bk.ingest(np.concatenate(
            [c, [getattr(self, 'new_class_bias', B_NEG)]]))
        self.opt = _Adam([self.W1.shape, self.b1.shape,
                          self.W2.shape, self.c.shape], self.lr,
                         self._bk)

    # ---------- training ----------
    def train_step(self, X, y, sgd_lr=None):
        if self.mode == "numeric":
            return super().train_step(X, y, sgd_lr=sgd_lr)
        if self.mode == "numeric_dist":            # GSM-I3
            return self._train_step_dist(X, y, sgd_lr=sgd_lr)
        # categorical: y = list/array of labels. GSM-I2: the
        # step routes through backend kernels (numpy bodies are
        # verbatim relocations — judge stays bitwise)
        Xnp = np.asarray(X, float)
        labels = [self.vocab.index(v) for v in np.asarray(y).ravel()]
        n = len(Xnp)
        Xb = self._bk.ingest(Xnp)
        if self._x_mu is None:              # fit input scalers once
            self._x_mu, self._x_sd = self._bk.fit_x_stats(Xb)
        Xs = self._std_x(Xb)
        A, Hact = self._hidden(Xs)
        probs = self._bk.cat_forward(self.W2, self.c, Hact)
        onehot_np = np.zeros((n, len(self.vocab)))
        onehot_np[np.arange(n), labels] = 1.0
        onehot = self._bk.ingest(onehot_np)
        ce = self._bk.cat_ce(probs, onehot)
        grads, dH = self._bk.cat_backward(
            self.W1, self.b1, self.W2, self.c, Xs, A, Hact,
            probs, onehot)
        old_W1 = self._bk.copy(self.W1)
        params = [self.W1, self.b1, self.W2, self.c]
        if sgd_lr is None:
            self.W1, self.b1, self.W2, self.c = self.opt.step(
                params, grads)
        else:
            self.W1, self.b1, self.W2, self.c = \
                self._bk.sgd_step(params, grads, sgd_lr)
        dw = self._bk.to_numpy(self.W1 - old_W1)
        self._ema_dw = 0.95 * self._ema_dw + 0.05 * dw
        self._ema_adw = 0.95 * self._ema_adw + 0.05 * np.abs(dw)
        # cross-scale target handoff — SAME primitive as the parent
        from reference_net.net import ETA_TARGET
        from reference_net.growthpolicy import \
            DEFAULT_GROWTH_POLICY as _gpe
        _gpe = getattr(self, "_growth_policy", _gpe)  # S9.5 D-N1
        _eta = float(_gpe.get("eta_target", ETA_TARGET))
        from reference_net.growth_port import legacy_handoff
        legacy_handoff(self.inner, Xs, Hact, A, dH,
                       self._bk.gelu, _eta, sgd_lr=sgd_lr)
        port = getattr(self, "_port_site", None)
        if port is not None:
            # TRUE gradient scale (pin P-2; kernel /n convention)
            port.backward_step(dH / max(len(Xs), 1), Xs, self.lr,
                               sgd_lr=sgd_lr)
        return ce

    # ---------- artifact (msorgan.pkl inside the version layout) ----------
    # Security note: pickle is used ONLY for locally-produced model
    # artifacts written and read by this system inside its own storage
    # tree (same trust domain; never network-received or user-uploaded).
    # Same pattern as Phase-2 checkpoints. A schema-based serializer for
    # the recursive structure is a deferred-ledger item.
    def save(self, dir_path):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "msorgan.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(dir_path):
        with open(Path(dir_path) / "msorgan.pkl", "rb") as f:
            return pickle.load(f)

    # ---------- structural record ----------
    def shape_record(self):
        return {"mode": self.mode, "vocab": list(self.vocab),
                # 'depth' = height of the inclusion tree (tree height)
                "depth": self.depth(), "params": self.n_params(),
                "d_in": self.d_in, "hidden": self.H}
