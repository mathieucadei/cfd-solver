import numpy as np

from core.config import BurgersEquation1DConfig
from core.setup.time_stepping import compute_cole_hopf_dt
from ..analytical.cole_hopf_equation_1d import cole_hopf_1d_ufunc


def hat_initial_condition(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generates a hat function initial condition based on the provided configuration."""

    initial_condition = np.full_like(x_array, config.u_min, dtype=float)
    initial_condition[(x_array >= config.hat_start) & (x_array <= config.hat_end)] = config.u_max

    return initial_condition


def cole_hopf_initial_condition(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generates an initial condition for the Cole-Hopf transformation based on the provided configuration."""

    time_step = 0

    initial_condition_func = cole_hopf_1d_ufunc()

    initial_condition = initial_condition_func(time_step, x_array, config.viscosity)

    return initial_condition