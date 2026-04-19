"""Grid generation and spacing utilities for 1D numerical solvers."""



import numpy as np



def make_x_grid(config: object) -> np.ndarray:
    """Generate a uniform x-coordinate grid from the configuration."""

    return np.linspace(0.0, config.domain_length_x, config.num_grid_points_x)


def make_y_grid(config: object) -> np.ndarray:
    """Generate a uniform y-coordinate grid from the configuration."""

    return np.linspace(0.0, config.domain_length_y, config.num_grid_points_y)


def make_cole_hopf_1d_grid(config: object) -> np.array:
    """Generate the 1D periodic grid used for the Cole-Hopf analytical solution."""

    return np.linspace(0, 2 * np.pi, config.num_grid_points_x)

def compute_dx(config: object) -> float:
    """Compute the uniform grid spacing in the x-direction from a 1D configuration."""
    
    return config.domain_length_x / (config.num_grid_points_x - 1)


def compute_cole_hopf_dx(config: object) -> float:
    """Compute the uniform grid spacing for the Cole-Hopf periodic domain."""

    return 2 * np.pi / (config.num_grid_points_x - 1)


def compute_dx_2d(config: object) -> float:
    """Compute the uniform grid spacing in the x-direction from a 2D configuration."""
    return config.domain_length_x / (config.num_grid_points_x - 1)


def compute_dy_2d(config: object) -> float:
    """Compute the uniform grid spacing in the y-direction from a 2D configuration."""
    return config.domain_length_y / (config.num_grid_points_y - 1)