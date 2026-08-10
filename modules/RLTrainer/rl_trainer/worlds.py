"""N2 EnvHarness (doc 86 N2; plan 84 D-P3): three seeded
in-house numpy reward worlds — no external dependencies.
Common API: reset(seed)->obs, step(a)->(obs, reward, done),
spec(); reset(seed=None) continues the SAME state stream
(episode boundary without reseeding), so a world seeded once
replays bit-identically (TW-01 determinism contract).

Reward law (all three): contextual episodic task — the world
holds per-stage linear score weights over the state; the
optimal action is argmax_k w_k·s (the world's own oracle);
reward 1.0 for the oracle action, else 0.0. Episode length =
ep_len constructor parameter (census law: no welded module
constants; per-world boundary/arrival_step are likewise
constructor parameters)."""
import numpy as np


class _WorldBase:
    n_actions = 3
    obs_dim = 6
    kind = "base"

    def __init__(self, seed, ep_len=32):
        # every behavioral constant is a constructor parameter
        # (house law: nothing welded; defaults are the spec'd
        # validation-fixture values)
        self.ep_len = int(ep_len)
        self._init_seed = int(seed)
        w_rng = np.random.default_rng(1000 + self._init_seed)
        self._W = self._make_weights(w_rng)   # per-stage list
        self._rng = None
        self._t_global = 0
        self._t_ep = 0
        self._s = None

    # -- per-world hooks --
    def _make_weights(self, rng):
        raise NotImplementedError

    def _stage(self):
        return 0

    def _emit(self, s):
        """Observation the agent SEES for internal state s."""
        return s

    # -- common API --
    def _draw_state(self):
        return self._rng.normal(size=self.obs_dim)

    def reset(self, seed):
        if seed is not None:
            self._rng = np.random.default_rng(2000 + int(seed))
            self._t_global = 0
        self._t_ep = 0
        self._s = self._draw_state()
        return self._emit(self._s)

    def oracle(self, s, stage=None):
        st = self._stage() if stage is None else int(stage)
        W = self._W[min(st, len(self._W) - 1)]
        return int(np.argmax(W @ np.asarray(s)))

    def step(self, a):
        rew = 1.0 if int(a) == self.oracle(self._s) else 0.0
        self._t_global += 1
        self._t_ep += 1
        done = self._t_ep >= self.ep_len
        self._s = self._draw_state()
        return self._emit(self._s), rew, done

    def sample_state(self, i):
        """Deterministic probe state (referee use)."""
        return np.random.default_rng(3000 + int(i)).normal(
            size=self.obs_dim)

    def spec(self):
        return {"type": self.kind, "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "episode_len": self.ep_len,
                "seed": self._init_seed}


class StationaryWorld(_WorldBase):
    """(b) stationary control world — the growth-silence
    regime: one fixed weight set, no drift ever."""
    kind = "stationary"

    def _make_weights(self, rng):
        return [rng.normal(size=(self.n_actions,
                                 self.obs_dim))]


class StagedExpansionWorld(_WorldBase):
    """(a) staged state-space expansion: stage A ignores the
    expansion dims EXACTLY (zero weight columns); at the
    boundary the task starts depending on them STRONGLY —
    arriving structure, the capacity-demand case."""
    kind = "staged_expansion"
    expansion_dims = np.array([4, 5])

    def __init__(self, seed, ep_len=32, boundary=512):
        self.boundary = int(boundary)   # global steps to stage B
        super().__init__(seed, ep_len)

    def _make_weights(self, rng):
        w0 = rng.normal(size=(self.n_actions, self.obs_dim))
        w0[:, self.expansion_dims] = 0.0
        w1 = w0.copy()
        w1[:, self.expansion_dims] = 3.0 * rng.normal(
            size=(self.n_actions, len(self.expansion_dims)))
        return [w0, w1]

    def _stage(self):
        return 0 if self._t_global < self.boundary else 1


class SensorArrivalWorld(_WorldBase):
    """(c) sensor-arrival world (σ): the sensor dims are
    emitted as EXACTLY zero before arrival_step, then the
    true values stream; the score always depends on them, so
    information (not the task) is what arrives."""
    kind = "sensor_arrival"
    sensor_dims = np.array([4, 5])

    def __init__(self, seed, ep_len=32, arrival_step=256):
        self.arrival_step = int(arrival_step)
        super().__init__(seed, ep_len)

    def _make_weights(self, rng):
        w = rng.normal(size=(self.n_actions, self.obs_dim))
        w[:, self.sensor_dims] *= 3.0
        return [w]

    def _emit(self, s):
        if self._t_global < self.arrival_step:
            out = s.copy()
            out[self.sensor_dims] = 0.0
            return out
        return s
