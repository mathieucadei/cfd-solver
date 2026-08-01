"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt

from .operators import compute_advection_2d_term
from .boundary_conditions import apply_advection_boundary_2d
from ...setup.fvm.time_stepping import compute_advective_dt_2d_fvm
from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_x_face_positions, build_x_centers
from ...setup.fvm.initial_conditions import hat_initial_condition_2d_fvm

from dataclasses import dataclass


def solve_advection_2d_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-volume scheme."""

    dt = compute_advective_dt_2d_fvm(config)

    hx, hy = build_h_spacing(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        advection_term = compute_advection_2d_term(un, config.wavespeed, hx, hy, dt)

        u[1:, 1:] = un[1:, 1:] - advection_term[1:, 1:]

        apply_advection_boundary_2d(
            u=un,
            c=config.wavespeed,
            u_min=config.u_min,
            dt=dt,
            hx=hx,
            hy=hy,
        )

        history[n] = u

    return history