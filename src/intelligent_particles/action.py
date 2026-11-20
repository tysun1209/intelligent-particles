import numpy as np


def turn_angles(action, world):
    """Validate one turn angle in radians per particle for the periodic dynamics."""
    turn = np.asarray(action, dtype=float)
    if turn.shape != (world.n,) or not np.isfinite(turn).all():
        raise ValueError("expected one finite turn angle (radians) per particle")
    return turn
