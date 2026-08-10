"""Numerical solver for the 2D Laplace equation."""



import numpy as np

from .boundary_conditions import apply_laplace_boundary_2d
from ...setup.fvm.time_stepping import compute_diffusive_dt_2d_fvm
from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes
from ...setup.fvm.initial_conditions import hat_initial_condition_2d_fvm



def solve_laplace_2d_fvm(
    initial_condition: np.ndarray,
    bottom_boundary: float | np.ndarray,
    top_boundary: float | np.ndarray,
    right_boundary: float | np.ndarray,
    left_boundary: float | np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 2D Laplace equation with an explicit central finite-volume scheme."""

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)

    l1norm = 1

    p = initial_condition
    pn = np.empty_like(p)

    history = []

    while l1norm > config.l1_norm_target:

        pn = p.copy()

        f_w = face_areas_x[1:-1, 1:-1] * (pn[1:-1, 1:-1] - pn[1:-1, :-2]) / dist_x[:-1]

        # f_wb_u = face_areas_x[1:, 0] * (p[1:, 0] - left_boundary) / xc[0]

        # f_w = np.hstack((f_wb_u[:,None], f_w_i))

        f_e = face_areas_x[1:-1, 2:] * (pn[1:-1, 2:] - pn[1:-1, 1:-1]) / dist_x[1:]

        f_s = face_areas_y[1:-1, 1:-1] * (pn[1:-1, 1:-1] - pn[:-2, 1:-1]) / dist_y[:-1, None]

        f_n = face_areas_y[2:, 1:-1] * (pn[2:, 1:-1] - pn[1:-1, 1:-1]) / dist_y[1:, None]

        # f_wb =  face_areas_x[1:, 0] * (pn[1:, 1:] - pn[1:, :-1]) / xc

        p[1:-1, 1:-1] = (((f_e - f_w) + (f_n - f_s)) / cell_volumes[1:-1, 1:-1]) / \
                        ((face_areas_x[1:-1, 1:-1] / dist_x[:-1] \
                         + face_areas_x[1:-1, 2:] / dist_x[1:] \
                         + face_areas_y[1:-1, 1:-1] / dist_y[:-1, None] \
                         + face_areas_y[2:, 1:-1] / dist_y[1:, None]) \
                        / cell_volumes[1:-1, 1:-1])

        apply_laplace_boundary_2d(
            p, 
            bottom=bottom_boundary, 
            top=top_boundary, 
            right=right_boundary, 
            left=left_boundary,
            dist_x=dist_x,
            dist_y=dist_y,
            face_areas_x=face_areas_x, 
            face_areas_y=face_areas_y, 
            cell_volumes=cell_volumes,
            lx=config.domain_length_x,
            ly=config.domain_length_y,
            xc=xc,
            yc=yc,
            )


        denominator = np.sum(np.abs(pn))

        if denominator == 0:
            l1norm = np.sum(np.abs(p) - np.abs(pn)) 
        
        else:
            l1norm = (np.sum(np.abs(p) - np.abs(pn))) / denominator
        
        history.append(p.copy())
    
    history_array = np.stack(history, axis=0)

    return history_array
