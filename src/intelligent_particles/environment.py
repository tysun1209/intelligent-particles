from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .action import turn_angles
from .dynamics import advance
from .observation import local_headings
from .reward import zero_reward
from .state import ParticleState
from .transition import Step


class Environment(Protocol):
    def reset(self, seed: int) -> NDArray: ...
    def step(self, action: NDArray) -> Step: ...


class PeriodicParticles:
    """Replace observe, dynamics, or reward independently using keyword callbacks."""
    def __init__(self, n=64, *, box=10.0, radius=2.0, speed=1.0, dt=0.1,
                 rotational_diffusion=0.05, observe=local_headings,
                 dynamics=advance, reward=zero_reward, action=turn_angles):
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        values = [box, radius, speed, dt, rotational_diffusion]
        if not np.isfinite(values).all() or min(box, dt) <= 0 or min(radius, speed, rotational_diffusion) < 0:
            raise ValueError("invalid physical parameters")
        self.n, self.box, self.radius, self.speed, self.dt = n, box, radius, speed, dt
        self.rotational_diffusion = rotational_diffusion
        self.observe, self.dynamics, self.reward = observe, dynamics, reward
        self.action = action
        self.state = None

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state = ParticleState(self.rng.uniform(0, self.box, (self.n, 2)),
                                   self.rng.uniform(-np.pi, np.pi, self.n))
        return self.observe(self.state, self)

    def step(self, action):
        if self.state is None:
            raise RuntimeError("reset the world before stepping")
        turn = self.action(action, self)
        previous = self.state
        self.state = self.dynamics(previous, turn, self, self.rng)
        return Step(self.observe(self.state, self), self.reward(previous, turn, self.state, self),
                    np.zeros(self.n, dtype=bool))
