import numpy as np

from ..config import BurgersEquation1DConfig

from ..setup.time_stepping import compute_cole_hopf_dt


def cole_hopf_1d_ufunc() -> callable:

    def func(time_step: float, x_array: np.ndarray, viscosity: float) -> np.ndarray:  

        phi = (np.exp(-(x_array - 4 * time_step)**2 / (4 * viscosity * (time_step + 1))) \
            + np.exp(-(x_array - 4 * time_step - 2 * np.pi)**2 / (4 * viscosity * (time_step + 1))))
        
        phiprime = -(-8*time_step + 2*x_array)*np.exp(-(-4*time_step + x_array)**2/(4*viscosity*(time_step + 1)))/(4*viscosity*(time_step + 1)) \
            - (-8*time_step + 2*x_array - 4*np.pi)*np.exp(-(-4*time_step + x_array - 2*np.pi)**2/(4*viscosity*(time_step + 1)))/(4*viscosity*(time_step + 1)) 
        
        return -2 * viscosity * (phiprime / phi) + 4  

    return func


def solve_cole_hopf_1d(
    x_array: np.ndarray,
    initial_condition: np.ndarray, 
    config: BurgersEquation1DConfig,
) -> np.ndarray:
    """Solves the 1D Cole-Hopf equation using the provided initial condition and configuration."""

    dt = compute_cole_hopf_dt(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    f = cole_hopf_1d_ufunc()

    for n in range(0, config.max_iterations + 1):

        u = f(n*dt, x_array, config.viscosity)

        history[n] = u

    return history