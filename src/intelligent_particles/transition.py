from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Step:
    observation: NDArray
    reward: NDArray
    terminated: NDArray  # cumulative per-particle termination mask


@dataclass
class Transition:
    observation: NDArray
    action: NDArray
    reward: NDArray
    next_observation: NDArray
    active: NDArray
    terminated: NDArray
    truncated: bool  # time limit; distinct from physical termination
