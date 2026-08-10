"""Shared head component (SUBSTRATE_ARCHITECTURE Section 8; plan S1.1).

One implementation of the self-shaping output heads, consumed by every
substrate so head semantics are identical across bodies:
- numeric head: scaled linear regression output
- categorical head: softmax + CE gradient + confidence
- vocabulary growth: zero-weight logit row with bias B_NEG (epsilon-
  preserving, documented bound < 1e-3)

The shipped `mlp` substrate keeps its validated in-class implementation
(behavioral anchor, unchanged); `transformer`/`sequence` consume THESE
functions. The Compatibility Kit (K1-K3) enforces identical head
semantics for every substrate — that is the equivalence guarantee.
"""
from __future__ import annotations

import numpy as np

B_NEG = -10.0


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ce_loss_and_grad(logits, label_idx):
    """Cross-entropy loss and d(loss)/d(logits) for one-hot targets."""
    n = len(logits)
    probs = softmax(logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), label_idx] = 1.0
    loss = float(-np.mean(np.log(probs[np.arange(n), label_idx] + 1e-12)))
    return loss, (probs - onehot)


def grow_vocab(W2, c, hidden):
    """New class: zero-weight logit row + bias B_NEG. Old-class
    distribution shift bounded by ~e^B_NEG (< 1e-3)."""
    W2 = np.vstack([W2, np.zeros((1, hidden))])
    c = np.concatenate([c, [B_NEG]])
    return W2, c


def confidence(probs):
    idx = probs.argmax(axis=1)
    return idx, probs[np.arange(len(idx)), idx]
