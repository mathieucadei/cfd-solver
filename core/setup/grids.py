"""Grid generation and spacing utilities for 1D numerical solvers."""



import numpy as np



def make_1d_grid(config: object) -> np.ndarray:
    """Generate a uniform 1D grid from the configuration."""

    return np.linspace(0.0, config.domain_length, config.num_grid_points)


def make_cole_hopf_1d_grid(config: object) -> np.array:
    """Generate the 1D periodic grid used for the Cole-Hopf analytical solution."""

    return np.linspace(0, 2 * np.pi, config.num_grid_points)


def compute_dx(config: object) -> float:
    """Compute the uniform 1D grid spacing from the configuration."""

    return config.domain_length / (config.num_grid_points - 1)


def compute_cole_hopf_dx(config: object) -> float:
    """Compute the uniform grid spacing for the Cole-Hopf periodic domain."""
    
    return 2 * np.pi / (config.num_grid_points - 1)