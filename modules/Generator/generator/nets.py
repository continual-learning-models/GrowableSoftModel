"""TinyMLP — the soft model's neural core (numpy only, no heavy deps).

A neural network does not come in "types": it learns whatever input->output
mapping its data expresses. Accordingly this net has ONE implementation with
two output heads it chooses between AT TRAINING TIME based on the data it is
taught (the MODEL shapes itself; nothing is declared by humans or imposed by
the factory):

- targets that are all numeric  -> a linear head trained with MSE
  (the model predicts numbers),
- otherwise                     -> a softmax head over the target values it
  has SEEN (the model builds its own output vocabulary from data; the
  vocabulary can grow in later versions as new values appear).

Feature scaling (and, for numeric targets, target scaling) is learned from
the training data. Weights + scalers serialize to one .npz per version — the
weights literally ARE the soft model. No base model, no language ability:
general capability lives in the calling LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TinyMLP:
    def __init__(self, in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int,
                 mode: str = "categorical", seed: int = 7):
        assert mode in ("categorical", "numeric")
        self.in_dim, self.hidden_sizes, self.out_dim = in_dim, tuple(hidden_sizes), out_dim
        self.mode = mode
        rng = np.random.default_rng(seed)
        dims = [in_dim, *hidden_sizes, out_dim]
        # He initialization
        self.W = [rng.normal(0.0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        # scalers (fitted during train)
        self.mu = np.zeros(in_dim)
        self.sigma = np.ones(in_dim)
        self.y_mu = 0.0          # numeric mode: target scaler
        self.y_sigma = 1.0

    # ---------- forward ----------
    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        acts = [X]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = acts[-1] @ W + b
            acts.append(np.maximum(Z, 0.0) if i < last else Z)
        return acts

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.mode == "categorical"
        Xs = (X - self.mu) / self.sigma
        return self._softmax(self._forward(Xs)[-1])

    def predict_value(self, X: np.ndarray) -> np.ndarray:
        assert self.mode == "numeric"
        Xs = (X - self.mu) / self.sigma
        z = self._forward(Xs)[-1][:, 0]
        return z * self.y_sigma + self.y_mu

    # ---------- training (Adam; loss follows the data-chosen head) ----------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-2,
            batch_size: int = 32, seed: int = 7) -> float:
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma < 1e-8] = 1.0
        Xs = (X - self.mu) / self.sigma

        if self.mode == "numeric":
            yf = y.astype(float)
            self.y_mu = float(yf.mean())
            self.y_sigma = float(yf.std()) or 1.0
            ys = (yf - self.y_mu) / self.y_sigma

        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(W) for W in self.W]; vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        loss = 0.0

        for _ in range(epochs):
            idx = rng.permutation(len(Xs))
            for s in range(0, len(Xs), batch_size):
                bi = idx[s:s + batch_size]
                Xb = Xs[bi]
                acts = self._forward(Xb)
                if self.mode == "categorical":
                    yb = y[bi]
                    P = self._softmax(acts[-1])
                    loss = float(-np.log(P[np.arange(len(yb)), yb] + 1e-12).mean())
                    delta = P.copy()
                    delta[np.arange(len(yb)), yb] -= 1.0
                    delta /= len(yb)
                else:
                    yb = ys[bi]
                    pred = acts[-1][:, 0]
                    err = pred - yb
                    loss = float((err ** 2).mean())
                    delta = (2.0 * err / len(yb))[:, None]

                gW = [None] * len(self.W); gb = [None] * len(self.b)
                for i in reversed(range(len(self.W))):
                    gW[i] = acts[i].T @ delta
                    gb[i] = delta.sum(axis=0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (acts[i] > 0)

                t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                    self.W[i] -= lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                    self.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
        return loss

    # ---------- persistence: the weights ARE the soft model ----------
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays |= {f"b{i}": b for i, b in enumerate(self.b)}
        arrays |= {"mu": self.mu, "sigma": self.sigma,
                   "y_scaler": np.array([self.y_mu, self.y_sigma])}
        np.savez(out_dir / "organ.npz", **arrays)
        (out_dir / "meta.json").write_text(json.dumps({
            "in_dim": self.in_dim, "hidden_sizes": list(self.hidden_sizes),
            "out_dim": self.out_dim, "mode": self.mode}))

    @classmethod
    def load(cls, in_dir: Path) -> "TinyMLP":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        net = cls(meta["in_dim"], tuple(meta["hidden_sizes"]), meta["out_dim"],
                  mode=meta.get("mode", "categorical"))
        data = np.load(in_dir / "organ.npz")
        net.W = [data[f"W{i}"] for i in range(len(net.W))]
        net.b = [data[f"b{i}"] for i in range(len(net.b))]
        net.mu, net.sigma = data["mu"], data["sigma"]
        if "y_scaler" in data:
            net.y_mu, net.y_sigma = float(data["y_scaler"][0]), float(data["y_scaler"][1])
        return net
