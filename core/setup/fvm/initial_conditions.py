"""Initial condition utilities for 1D & 2D numerical and analytical solvers."""



import numpy as np

from .mesh import build_mesh



def hat_initial_condition_1d_fvm(hx_array: np.ndarray, config: object) -> np.ndarray:
    """Generate a 1D hat-function initial condition on the provided grid."""

    initial_condition = np.full_like(hx_array, config.u_min, dtype=float)
    initial_condition[int(len(hx_array)*config.hat_start/config.domain_length_x):int(len(hx_array)*config.hat_end/config.domain_length_x)] = config.u_max

    return initial_condition