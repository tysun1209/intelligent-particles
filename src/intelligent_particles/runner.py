from typing import Iterator

import numpy as np

from .environment import Environment
from .policy import Policy
from .transition import Transition


def rollout(env: Environment, policy: Policy, *, seed: int, steps: int) -> Iterator[Transition]:
    """Run one episode. Arrays have one row per particle, including completed ones."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    obs = np.array(env.reset(seed), copy=True)
    if obs.ndim != 2 or len(obs) == 0:
        raise ValueError("reset must return a nonempty (particles, features) array")
    done = np.zeros(len(obs), dtype=bool)
    for index in range(steps):
        action = np.array(policy(obs.copy(), rng), copy=True)
        if action.ndim == 0 or action.shape[0] != len(obs):
            raise ValueError("policy must return one action per particle")
        result = env.step(action.copy())
        following = np.array(result.observation, copy=True)
        reward = np.array(result.reward, copy=True)
        end = np.array(result.terminated, copy=True)
        if following.shape != obs.shape or reward.shape != done.shape:
            raise ValueError("inconsistent observation or reward shape")
        if end.shape != done.shape or end.dtype != bool or np.any(done & ~end):
            raise ValueError("terminated must be a cumulative boolean mask")
        yield Transition(obs, action, reward, following.copy(), ~done, end.copy(),
                         index + 1 == steps and not end.all())
        if end.all():
            break
        obs, done = following, end
