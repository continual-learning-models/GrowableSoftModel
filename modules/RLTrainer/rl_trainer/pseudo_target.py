"""Pseudo-target adapter (doc 86 §10, O-1 scheme (c) —
PRIMARY): the substrate's target machinery IS a gradient
interface in disguise — for the quadratic head the gradient
toward pseudo-target y* equals (h - y*), so ANY output-side
gradient dL/dh is injected by synthesizing y* = h - dL/dh.
Pure array algebra (layering law: no substrate import here;
the L2 loop hands arrays in and out). Referee: TB-P05."""
import numpy as np


def pseudo_target_for(h, grad):
    """y* such that the substrate's own quadratic-head
    gradient toward y* reproduces `grad` exactly:
    y* = h - grad (exact inverse: h - y* == grad)."""
    return np.asarray(h, dtype=float) - np.asarray(grad,
                                                   dtype=float)
