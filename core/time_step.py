import numpy as np

def compute_time_step(dx: float, config: object) -> float:
    """Computes the time step based on the provided configuration."""

    return config.sigma * dx**2 / config.viscosity