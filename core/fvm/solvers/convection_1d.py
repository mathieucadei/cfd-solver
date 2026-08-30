"""FVM numerical solver for the 1D convection equation."""



import numpy as np
import matplotlib.pyplot as plt

from ..operators import compute_convection_1d_term
from ..time_stepping import compute_convection_dt_1d
from ..mesh import build_mesh, build_hx_spacing, build_x_face_positions, build_x_centers
from ..initial_conditions import hat_initial_condition_1d

from dataclasses import dataclass


def solve_convection_1d(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D convection equation with an explicit upwind finite-volume scheme."""

    dt = compute_convection_dt_1d(config)

    hx = build_hx_spacing(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        convection_term = compute_convection_1d_term(un, hx, dt)

        u[1:] = un[1:] - convection_term[1:]

        history[n] = u

    return history