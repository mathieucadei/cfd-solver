"""Analytical Cole-Hopf solution utilities for the 1D Burgers' equation."""



import numpy as np

from ..config import BurgersEquation1DConfig

from ..setup.time_stepping import compute_cole_hopf_dt_1d


def cole_hopf_1d_ufunc() -> callable:
    """Return the analytical Cole-Hopf solution as a callable function of time and space."""

    def func(time_step: float, x_array: np.ndarray, viscosity: float) -> np.ndarray:  
        """Evaluate the Cole-Hopf analytical solution at the given time and spatial coordinates."""

        phi = (np.exp(-(x_array - 4 * time_step)**2 / (4 * viscosity * (time_step + 1))) \
            + np.exp(-(x_array - 4 * time_step - 2 * np.pi)**2 / (4 * viscosity * (time_step + 1))))
        
        phiprime = -(-8*time_step + 2*x_array)*np.exp(-(-4*time_step + x_array)**2/(4*viscosity*(time_step + 1)))/(4*viscosity*(time_step + 1)) \
            - (-8*time_step + 2*x_array - 4*np.pi)*np.exp(-(-4*time_step + x_array - 2*np.pi)**2/(4*viscosity*(time_step + 1)))/(4*viscosity*(time_step + 1)) 
        
        return -2 * viscosity * (phiprime / phi) + 4  

    return func


def solve_cole_hopf_1d(
    x_array: np.ndarray,
    config: BurgersEquation1DConfig,
) -> np.ndarray:
    """Evaluate the analytical Cole-Hopf solution over all time steps defined by the configuration."""

    dt = compute_cole_hopf_dt_1d(config)

    history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

    cole_hopf_func = cole_hopf_1d_ufunc()

    for n in range(0, config.max_iterations + 1):

        u = cole_hopf_func(n*dt, x_array, config.viscosity)

        history[n] = u

    return history