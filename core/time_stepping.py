import numpy as np

from .grids import compute_dx

def compute_dt(config: object) -> float:
    """Computes the time step based on the provided configuration."""

    dx = compute_dx(config)
    
    return config.sigma * dx**2 / config.viscosity