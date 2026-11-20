import numpy as np


def local_headings(state, world):
    """Ego heading and mean local heading. Minimum-image neighbors exclude self."""
    delta = state.position[:, None] - state.position[None, :]
    delta -= world.box * np.rint(delta / world.box)
    neighbors = (delta ** 2).sum(axis=2) <= world.radius ** 2
    np.fill_diagonal(neighbors, False)
    unit = np.column_stack((np.cos(state.heading), np.sin(state.heading)))
    mean = neighbors @ unit / np.maximum(neighbors.sum(axis=1), 1)[:, None]
    return np.column_stack((unit, mean))
