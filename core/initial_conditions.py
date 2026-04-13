import numpy as np

from core.config import BurgersEquation1DConfig
from .ana_mod.cole_hopf_equation_1d import cole_hopf_1d_ufunc


def hat_initial_condition(x: np.ndarray, config: object) -> np.ndarray:
    """Generates a hat function initial condition based on the provided configuration."""

    u0 = np.full_like(x, config.u_min, dtype=float)
    u0[(x >= config.hat_start) & (x <= config.hat_end)] = config.u_max

    return u0


def cole_hopf_initial_condition(time_step: float, x_array: np.ndarray, config: object) -> np.ndarray:
    """Generates an initial condition for the Cole-Hopf transformation based on the provided configuration."""

    u0 = cole_hopf_1d_ufunc()(time_step, x_array, config.viscosity)

    return u0