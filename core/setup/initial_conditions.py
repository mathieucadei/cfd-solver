"""Initial condition utilities for 1D numerical and analytical solvers."""



import numpy as np

from ..analytical.cole_hopf_equation_1d import cole_hopf_1d_ufunc



def hat_initial_condition(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generate a 1D hat-function initial condition on the provided grid."""

    initial_condition = np.full_like(x_array, config.u_min, dtype=float)
    initial_condition[(x_array >= config.hat_start) & (x_array <= config.hat_end)] = config.u_max

    return initial_condition


def cole_hopf_initial_condition(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generate the Cole-Hopf analytical initial condition on the provided grid."""

    initial_condition_func = cole_hopf_1d_ufunc()

    initial_condition = initial_condition_func(0.0, x_array, config.viscosity)

    return initial_condition