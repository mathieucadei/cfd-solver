import numpy as np
from .config import Advection1DConfig

def hat_initial_condition(x: np.ndarray, config: Advection1DConfig) -> np.ndarray:
    """Generates a hat function initial condition based on the provided configuration."""

    u0 = np.full_like(x, config.u_min, dtype=float)
    u0[(x >= config.hat_start) & (x <= config.hat_end)] = config.u_max

    return u0