import numpy as np

def hat_initial_condition(x: np.ndarray, config: object) -> np.ndarray:
    """Generates a hat function initial condition based on the provided configuration."""

    u0 = np.full_like(x, config.u_min, dtype=float)
    u0[(x >= config.hat_start) & (x <= config.hat_end)] = config.u_max

    return u0