import numpy as np


def align_with_neighbors(obs, rng):
    """Turn toward the local mean; keep the heading if the mean vanishes."""
    ego, mean = obs[:, :2], obs[:, 2:4]
    cross = ego[:, 0] * mean[:, 1] - ego[:, 1] * mean[:, 0]
    turn = np.arctan2(cross, (ego * mean).sum(axis=1))
    return np.where(np.linalg.norm(mean, axis=1) > 1e-12, turn, 0.0)
