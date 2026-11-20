import numpy as np


def zero_reward(previous, action, current, world):
    return np.zeros(world.n)


def alignment_reward(previous, action, current, world):
    unit = np.column_stack((np.cos(current.heading), np.sin(current.heading)))
    return unit @ unit.mean(axis=0)


def eastward_reward(previous, action, current, world):
    return np.cos(current.heading)
