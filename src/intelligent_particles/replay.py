"""Uniform replay of individual active-particle transitions."""
import numpy as np


class Replay:
    def __init__(self, capacity, features):
        self.capacity = capacity
        self.head = self.size = 0
        self.fields = [np.empty((capacity, n), dtype=np.float32) for n in (features, 1, 1, features, 1)]

    def add(self, obs, action, reward, following, done):
        n = min(len(obs), self.capacity)
        if n == 0:
            return
        slots = (self.head + np.arange(n)) % self.capacity
        for dest, source in zip(self.fields, (obs, action[:, None], reward[:, None], following, done[:, None])):
            dest[slots] = source[-n:]
        self.head = (self.head + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, count, rng):
        slots = rng.integers(self.size, size=count)
        return tuple(field[slots] for field in self.fields)
