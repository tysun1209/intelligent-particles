from typing import Callable

import numpy as np
from numpy.typing import NDArray


Policy = Callable[[NDArray, np.random.Generator], NDArray]
