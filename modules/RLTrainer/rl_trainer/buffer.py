"""RolloutBuffer (doc 86 §3.35 RolloutBatch schema; TB-P04
determinism contract: same fills + same batch seed =>
bit-identical minibatches, incl. from a fresh object)."""
import numpy as np


class RolloutBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim))
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity)
        self.dones = np.zeros(capacity, dtype=bool)
        self.logp = np.zeros(capacity)
        self.values = np.zeros(capacity)
        self.n = 0

    def add(self, obs, action, reward, done, logp, value):
        i = self.n
        if i >= self.capacity:
            raise ValueError("rollout buffer full "
                             f"(capacity {self.capacity})")
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.logp[i] = logp
        self.values[i] = value
        self.n = i + 1

    def batch(self):
        """The full RolloutBatch view (doc 86 §3.35)."""
        s = slice(0, self.n)
        return {"obs": self.obs[s], "actions": self.actions[s],
                "rewards": self.rewards[s], "dones": self.dones[s],
                "logp": self.logp[s], "values": self.values[s]}

    def batches(self, batch_size, epochs, seed):
        """Deterministic minibatch stream: a dedicated
        Generator(seed) shuffles indices per epoch — replay
        with the same seed is bit-identical (NFR-1)."""
        rng = np.random.default_rng(seed)
        idx = np.arange(self.n)
        for _ in range(int(epochs)):
            rng.shuffle(idx)
            for s in range(0, self.n, int(batch_size)):
                mb = idx[s:s + int(batch_size)]
                yield {"obs": self.obs[mb],
                       "actions": self.actions[mb],
                       "rewards": self.rewards[mb],
                       "dones": self.dones[mb],
                       "logp": self.logp[mb],
                       "values": self.values[mb],
                       "idx": mb.copy()}
