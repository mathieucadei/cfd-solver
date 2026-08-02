"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt

from .operators import compute_convection_2d_term
from .boundary_conditions import apply_convection_boundary_2d
from ...setup.fvm.time_stepping import compute_convective_dt_2d_fvm
from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_x_face_positions, build_x_centers, build_face_areas, compute_cell_volumes
from ...setup.fvm.initial_conditions import hat_initial_condition_2d_fvm

from dataclasses import dataclass


def solve_convection_2d_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-volume scheme."""

    dt = compute_convective_dt_2d_fvm(config)

    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)

    u, v = initial_condition[0].copy(), initial_condition[1].copy(),

    u_history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))
    v_history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))

    u_history[0], v_history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        vn = v.copy()

        convection_u_term, convection_v_term = compute_convection_2d_term(
                                                    un, 
                                                    vn, 
                                                    face_areas_x, 
                                                    face_areas_y, 
                                                    cell_volumes, 
                                                    dt
                                                )

        u[1:, 1:] = un[1:, 1:] - convection_u_term[1:, 1:]
        v[1:, 1:] = vn[1:, 1:] - convection_v_term[1:, 1:]

        apply_convection_boundary_2d(
            u=u,
            v=v,
            un=un,
            vn=vn,
            u_min=config.u_min,
            v_min=config.v_min,
            dt=dt,
            face_areas_x=face_areas_x, 
            face_areas_y=face_areas_y, 
            cell_volumes=cell_volumes, 
        )

        u_history[n], v_history[n] = u, v

    return u_history, v_history