import numpy as np

def make_1d_grid(config: object) -> np.ndarray:
    """Generates a 1D grid based on the provided configuration."""

    return np.linspace(0.0, config.domain_length, config.num_grid_points)

def make_cole_hopf_1d_grid(config: object) -> np.array:

    return np.linspace(0, 2 * np.pi, config.num_grid_points)

def compute_dx(config: object) -> float:
    """Computes the spacing between grid points."""

    return config.domain_length / (config.num_grid_points - 1)

def compute_cole_hopf_dx(config: object) -> float:
    
    return 2 * np.pi / (config.num_grid_points - 1)