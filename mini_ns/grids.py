import numpy as np
from .config import Advection1DConfig

def make_1d_grid(conf: Advection1DConfig) -> np.ndarray:
    """Generates a 1D grid based on the provided configuration."""

    return np.linspace(0.0, conf.domain_length, conf.num_grid_points)

def compute_1d_grid_points_spacing(conf: Advection1DConfig) -> float:
    """Computes the spacing between grid points."""

    return conf.domain_length / (conf.num_grid_points - 1)