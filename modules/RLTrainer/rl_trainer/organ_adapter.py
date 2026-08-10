"""OrganAdapter — the pseudo-target adapter driving the REAL
growable substrate (doc 86 §10 scheme (c) / §3.35): converts
an output-side gradient into a synthesized target
y* = h - dL/dh and walks the organ's EXISTING train_step path
— zero substrate change, growth operators apply between
updates with the G-1 exactness contract intact (TB-P08 R3).

EXACTNESS LAW (TB-P08 R1, by linearity of the target space):
apply(X, g) with g = h - y_t is BIT-IDENTICAL to
train_step(X, y_t) — the synthesized target IS a target; the
kernel's own scaling chain (standardization, 2/n, ETA_TARGET,
Adam) applies verbatim, so gradients injected here live in
the kernel's dL/dh convention by construction.

Layering: this module holds an ORGAN REFERENCE but only calls
its public serving/training surface (predict / train_step) —
the L1 object-binding adapter of doc 86 §3.0."""
import numpy as np


class OrganAdapter:
    def __init__(self, organ):
        self.organ = organ

    def outputs(self, X):
        """Raw organ outputs (policy organs: logits rows;
        value organs: (n,1) values)."""
        out = np.asarray(self.organ.predict(X), dtype=float)
        if out.ndim == 1:
            out = out[:, None]
        return out

    def probs(self, X):
        """Softmax serving for policy organs (N3 discrete-
        action head: out_width=K organ + this softmax)."""
        z = self.outputs(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def apply(self, X, dL_dout, sgd_lr=None):
        """One organ update driven by the output-side gradient
        dL_dout (same shape as outputs): synthesize
        y* = h - dL_dout and take the EXISTING train step.
        Returns the organ's mse-before-step readout."""
        h = self.outputs(X)
        g = np.asarray(dL_dout, dtype=float)
        if g.shape != h.shape:
            raise ValueError(f"dL_dout shape {g.shape} != "
                             f"outputs shape {h.shape}")
        y_star = h - g
        return self.organ.train_step(X, y_star, sgd_lr=sgd_lr)
