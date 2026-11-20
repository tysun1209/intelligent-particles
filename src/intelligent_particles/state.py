from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleState:
    position: np.ndarray  # (N, 2)
    heading: np.ndarray   # radians, (N,)
