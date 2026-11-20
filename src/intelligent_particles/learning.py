"""Train the fixed-speed SAC controller on a task with scalar steering actions."""
import numpy as np
import torch

from .runner import rollout
from .models.sac import SAC
from .replay import Replay


def train_turn_sac(env, *, episodes, steps, seed=0, max_turn=1.0, hidden=128,
                   batch_size=128, capacity=100_000, device="cpu"):
    """Return the trained agent and episode returns (sum of active-particle rewards).

    max_turn converts the single normalized action to the environment's units.
    Physical termination disables TD bootstrapping; time limits allow it.
    """
    if min(episodes, steps, batch_size, hidden) < 1 or capacity < batch_size:
        raise ValueError("invalid episode, update or replay size")
    if not np.isfinite(max_turn) or max_turn <= 0:
        raise ValueError("max_turn must be finite and positive")
    replay_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
    torch.manual_seed(seed)
    initial = np.asarray(env.reset(seed))
    if initial.ndim != 2 or min(initial.shape) == 0:
        raise ValueError("expected an observation matrix of shape (N, features)")
    agent = SAC(initial.shape[1], hidden, device)
    replay = Replay(capacity, initial.shape[1])

    def policy(obs, rng):
        if replay.size < batch_size:
            return rng.uniform(-max_turn, max_turn, len(obs))
        return max_turn * agent.normalized_turn(obs)

    returns = []
    for episode in range(episodes):
        total = 0.0
        for tr in rollout(env, policy, seed=seed + episode, steps=steps):
            mask = tr.active
            normalized = tr.action[mask] / max_turn
            replay.add(tr.observation[mask], normalized, tr.reward[mask],
                       tr.next_observation[mask], tr.terminated[mask])
            if replay.size >= batch_size:
                agent.update(replay.sample(batch_size, replay_rng))
            total += float(tr.reward[mask].sum())
        returns.append(total)
    return agent, returns
