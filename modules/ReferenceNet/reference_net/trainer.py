"""Course runner machinery (student-side; PLAN S2.3, R4, R5).

Runs curriculum stages with replay, evaluates on all stage suites,
detects underfit plateaus, executes growth events (top-k most unstable
nodes ANYWHERE in the hierarchy — non-uniform, S6), checkpoints before
each event, and keeps an append-only event log and score matrix.

Pedagogy (when to advance, remediate, alternate study/practice) does NOT
live here — this module only executes and measures (S0 boundary). The
scripted driver in experiments and the LLM Teacher in WP4 make decisions.
"""
from __future__ import annotations

import copy

import numpy as np

from .curriculum import accuracy


def collect_instability(net, path=""):
    """All nodes at all depths: [(path, node_j, score, network_obj)].
    Recurses through legacy inner bodies AND fullwidth port bodies
    (Growth Interface Reform) — port hops keep the grown node's
    key j so deep site addressing stays uniform."""
    rows = [(path or "root", j, s, net)
            for j, s in enumerate(net.instability())]
    for j, inner in net.inner.items():
        rows += collect_instability(inner, f"{path}/{j}" if path else str(j))
    port = getattr(net, "_port_site", None)
    if port is not None:
        for g, slot in enumerate(port.bodies):
            key = slot.get("key", g)
            rows += collect_instability(
                slot["body"], f"{path}/{key}" if path else str(key))
    return rows


class Course:
    def __init__(self, net, stages, eval_every=100, plateau_patience=3,
                 min_gain=0.01, replay_ratio=0.3, seed=7):
        self.net, self.stages = net, stages
        self.eval_every, self.patience = eval_every, plateau_patience
        self.min_gain, self.replay_ratio = min_gain, replay_ratio
        self.rng = np.random.default_rng(seed)
        self.score_matrix = []      # rows: {"t","stage_accs":[...]}
        self.events = []            # growth/rollback/stage events
        self.t = 0

    # ---------- measurement ----------
    def evaluate_all(self):
        accs = [accuracy(self.net.predict(st["X"]["eval"]), st["y"]["eval"])
                for st in self.stages]
        self.score_matrix.append({"t": self.t, "stage_accs": accs})
        return accs

    # ---------- one study block on stage k (cumulative replay) ----------
    # v0 minimal-consistent choice: study on the UNION of all stages so
    # far (full replay, Phase-1 lesson; and S1 literally -- new material
    # contains the old). Deterministic, no sampling noise.
    def study_block(self, k, steps):
        X = np.vstack([self.stages[i]["X"]["study"] for i in range(k + 1)])
        y = np.vstack([self.stages[i]["y"]["study"] for i in range(k + 1)])
        for _ in range(steps):
            self.net.train_step(X, y)
            self.t += 1

    # ---------- plateau detection on current-stage eval acc ----------
    def plateaued(self, k, target):
        """Best-so-far stalled below target (wiggle-proof)."""
        hist = [r["stage_accs"][k] for r in self.score_matrix]
        if len(hist) < self.patience + 1:
            return False
        recent_best = max(hist[-self.patience:])
        prior_best = max(hist[:-self.patience])
        return recent_best < target and recent_best - prior_best < self.min_gain

    # ---------- growth event (R4): checkpoint -> grow top-k anywhere ----------
    def grow_event(self, k_nodes=2, hidden=16, stage=None):
        ckpt = copy.deepcopy(self.net)
        cands = sorted(collect_instability(self.net),
                       key=lambda r: -r[2])
        grown = []
        for path, j, score, owner in cands:
            if len(grown) >= k_nodes:
                break
            if j in owner.inner \
                    or j in getattr(owner, "_port_js", set()):
                continue
            owner.grow(j, hidden=hidden)
            grown.append({"path": f"{path}[{j}]", "instability": round(score, 4)})
        self.events.append({"t": self.t, "event": "grow", "stage": stage,
                            "nodes": grown, "params": self.net.n_params(),
                            "depth": self.net.depth()})
        return ckpt

    def rollback(self, ckpt, stage=None):
        self.net = ckpt
        self.events.append({"t": self.t, "event": "rollback", "stage": stage})

    # ---------- scripted course (fixed-pace driver for experiments) ----------
    def run_scripted(self, targets, max_blocks=40, grow_budget=6,
                     t_post_blocks=4):
        """Per stage: study blocks until target or plateau; on plateau,
        grow (checkpoint; rollback if no gain after t_post_blocks)."""
        grows = 0
        for k, st in enumerate(self.stages):
            self.events.append({"t": self.t, "event": "stage_start",
                                "stage": st["name"]})
            blocks = 0
            while blocks < max_blocks:
                self.study_block(k, self.eval_every)
                accs = self.evaluate_all()
                blocks += 1
                if accs[k] >= targets[k]:
                    break
                if self.plateaued(k, targets[k]) and grows < grow_budget:
                    before = accs[k]
                    ckpt = self.grow_event(stage=st["name"])
                    grows += 1
                    for _ in range(t_post_blocks):
                        self.study_block(k, self.eval_every)
                        accs = self.evaluate_all()
                        blocks += 1
                    if accs[k] < before + self.min_gain:
                        self.rollback(ckpt, stage=st["name"])
        return self.score_matrix, self.events
