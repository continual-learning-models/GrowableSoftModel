"""The `sequence` substrate (plan SWP5; priority 2).

The CAUSAL VARIANT of the transformer core — one shared implementation,
two registry entries (architecture review finding #3): causal attention
mask (no future leakage), learned temporal positions, last-step head.
Serves timed signals: fault-stress series, physical waveform features,
campaign curves, ECG/EEG windows.

Input form (DATA_FORM = "sequence"): each example's input is an ordered
list of steps, each step a feature vector: shape (T, f), T <= WINDOW
(v0 fixed-window; ragged batching in the deferred ledger). Multi-scale
growth, instability, artifacts, two-timescale inner lr: all inherited.
"""
from __future__ import annotations

from core.substrates.transformer import TransformerSubstrate


class SequenceSubstrate(TransformerSubstrate):
    NAME = "sequence"
    DATA_FORM = "sequence"
    # 60A: EXPLICIT override — this class subclasses the
    # transformer host; without this line it would silently
    # inherit dist support it does not have
    SUPPORTED_HEADS = ("point",)

    def __init__(self, d_in, hidden, mode="numeric",
                 vocab=None, lr=1e-2, seed=7, d_model=32,
                 n_layers=2, n_heads=2, backend=None,
                 window=None, inner_lr_factor=None,
                 new_class_bias=None):
        # 60A L3: enforce THIS class's whitelist — the inherited
        # transformer constructor accepts numeric_dist, this
        # sequence host does not serve it. The signature MIRRORS
        # the parent explicitly (a *args/**kwargs wrapper broke
        # signature introspection: the substrate_params filter
        # and the t25f doc guard read this signature).
        if mode == "numeric_dist":
            raise ValueError(
                "sequence host serves mode='numeric' and "
                "'categorical'; numeric_dist has no sequence "
                "head (SUPPORTED_HEADS)")
        super().__init__(d_in, hidden, mode=mode, vocab=vocab,
                         lr=lr, seed=seed, d_model=d_model,
                         n_layers=n_layers, n_heads=n_heads,
                         backend=backend, window=window,
                         inner_lr_factor=inner_lr_factor,
                         new_class_bias=new_class_bias)
    CAUSAL = True
    # Host-calibrated two-timescale factor (multi-seed A/B on the plateau
    # fixture): 0.3x corrupted post-growth training in 2/3 seeds; 0.1x
    # never corrupted (3/3) and paid in 2/3. Evidence-driven, host-local.
    INNER_LR_FACTOR = 0.1
