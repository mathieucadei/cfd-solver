"""Initial condition utilities for 1D & 2D numerical and analytical solvers."""



import numpy as np

from .mesh import build_mesh, build_h_spacing



def hat_initial_condition_1d(hx_array: np.ndarray, config: object) -> np.ndarray:
    """Generate a 1D hat-function initial condition on the provided grid."""

    initial_condition = np.full_like(hx_array, config.u_min, dtype=float)
    initial_condition[int(len(hx_array)*config.hat_start/config.domain_length_x):int(len(hx_array)*config.hat_end/config.domain_length_x)] = config.u_max

    return initial_condition


def hat_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D hat-function initial condition on the provided grid."""

    hx, hy = build_h_spacing(config)

    initial_condition = np.full((config.num_cells_y, config.num_cells_x), float(config.u_min))

    initial_condition[
        int(len(hy)*config.hat_start_y/config.domain_length_y):
        int(len(hy)*config.hat_end_y/config.domain_length_y),
        int(len(hx)*config.hat_start_x/config.domain_length_x):
        int(len(hx)*config.hat_end_x/config.domain_length_x)
    ] = config.u_max

    return initial_condition


def hat_convective_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D hat-function initial condition on the provided grid."""

    hx, hy = build_h_spacing(config)

    u_initial_condition = np.full((config.num_cells_y, config.num_cells_x), float(config.u_min))
    v_initial_condition = np.full((config.num_cells_y, config.num_cells_x), float(config.v_min))

    u_initial_condition[
        int(len(hy)*config.hat_start_y/config.domain_length_y):
        int(len(hy)*config.hat_end_y/config.domain_length_y),
        int(len(hx)*config.hat_start_x/config.domain_length_x):
        int(len(hx)*config.hat_end_x/config.domain_length_x)
    ] = config.u_max

    v_initial_condition[
        int(len(hy)*config.hat_start_y/config.domain_length_y):
        int(len(hy)*config.hat_end_y/config.domain_length_y),
        int(len(hx)*config.hat_start_x/config.domain_length_x):
        int(len(hx)*config.hat_end_x/config.domain_length_x)
    ] = config.v_max

    return u_initial_condition, v_initial_condition


def laplace_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D Laplace numerical solver."""

    p = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)

    return p


def poisson_initial_condition_2d(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D Poisson numerical solver."""

    p = np.full((config.num_cells_y, config.num_cells_x), float(config.pressure_init))
    b = p.copy()

    for src in config.source_terms:
            b[int(config.num_cells_y * src.y), int(config.num_cells_x * src.x)] = src.value

    return p, b


def cavity_flow_initial_condition(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D cavity flow numerical solver."""

    u = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    v = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    p = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    b = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)

    return u, v, p, b

def channel_flow_initial_condition(config: object) -> np.ndarray:
    """Generate a 2D initial condition on the provided grid for the 2D channel flow numerical solver."""

    u = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    v = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    p = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)
    b = np.zeros((config.num_cells_y, config.num_cells_x), dtype=float)

    return u, v, p, b