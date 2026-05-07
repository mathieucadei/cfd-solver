"""Initial condition utilities for 1D & 2D numerical and analytical solvers."""



import numpy as np

from .grids import compute_dx, compute_dy
from ..analytical.cole_hopf_equation_1d import cole_hopf_1d_ufunc



def hat_initial_condition_1d(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generate a 1D hat-function initial condition on the provided grid."""

    initial_condition = np.full_like(x_array, config.u_min, dtype=float)
    initial_condition[(x_array >= config.hat_start) & (x_array <= config.hat_end)] = config.u_max

    return initial_condition


def cole_hopf_initial_condition_1d(x_array: np.ndarray, config: object) -> np.ndarray:
    """Generate the Cole-Hopf analytical initial condition on the provided grid."""

    initial_condition_func = cole_hopf_1d_ufunc()

    initial_condition = initial_condition_func(0.0, x_array, config.viscosity)

    return initial_condition


def hat_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D hat-function initial condition on the provided grid."""

    dx = compute_dx(config)
    dy = compute_dy(config)

    initial_condition = np.full((config.num_grid_points_y, config.num_grid_points_x), float(config.u_min))

    initial_condition[
        int(config.hat_start_y / dy):int(config.hat_end_y / dy + 1), 
        int(config.hat_start_x / dx):int(config.hat_end_x / dx + 1)
    ] = config.u_max

    return initial_condition


def hat_convective_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D hat-function initial condition on the provided grid for the 2D convection numerical solver."""

    dx = compute_dx(config)
    dy = compute_dy(config)

    u_initial_condition = np.full((config.num_grid_points_y, config.num_grid_points_x), float(config.u_min))
    v_initial_condition = np.full((config.num_grid_points_y, config.num_grid_points_x), float(config.v_min))

    u_initial_condition[
        int(config.hat_start_y / dy):int(config.hat_end_y / dy + 1), 
        int(config.hat_start_x / dx):int(config.hat_end_x / dx + 1)
    ] = config.u_max
    v_initial_condition[
        int(config.hat_start_y / dy):int(config.hat_end_y / dy + 1), 
        int(config.hat_start_x / dx):int(config.hat_end_x / dx + 1)
    ] = config.v_max

    return u_initial_condition, v_initial_condition


def laplace_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D Laplace numerical solver."""

    p = np.zeros((config.num_grid_points_y, config.num_grid_points_x), dtype=float)

    return p

def poisson_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D Poisson numerical solver."""

    p = np.full((config.num_grid_points_y, config.num_grid_points_x), float(config.pressure_init))
    b = p.copy()

    for src in config.source_terms:
            b[int(config.num_grid_points_y * src.y), int(config.num_grid_points_x * src.x)] = src.value



    return p, b


def cavity_flow_initial_condition(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D cavity flow numerical solver."""

    u = np.zeros((config.num_grid_points_y, config.num_grid_points_x))
    v = np.zeros((config.num_grid_points_y, config.num_grid_points_x))
    p = np.zeros((config.num_grid_points_y, config.num_grid_points_x))
    b = np.zeros((config.num_grid_points_y, config.num_grid_points_x))

    return u, v, p, b