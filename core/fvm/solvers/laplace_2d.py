"""Numerical solver for the 2D Laplace equation."""



import numpy as np

from .boundary_conditions import apply_laplace_boundary_2d
from ..time_stepping import compute_diffusive_dt_2d
from ..mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes
from ..initial_conditions import hat_initial_condition_2d



def solve_laplace_2d(
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

    a_w = face_areas_x[1:-1, 1:-1] / dist_x[:-1]
    a_e = face_areas_x[1:-1, 2:] / dist_x[1:]
    a_s = face_areas_y[1:-1, 1:-1] / dist_y[:-1, None]
    a_n = face_areas_y[2:, 1:-1] / dist_y[1:, None]


    while l1norm > config.l1_norm_target:

        pn = p.copy()

        f_w = a_w * pn[1:-1, :-2]
        f_e = a_e * pn[1:-1, 2:]
        f_s = a_s * pn[:-2, 1:-1]
        f_n = a_n * pn[2:, 1:-1]

        p[1:-1, 1:-1] =(f_e + f_w + f_n + f_s) / (a_w + a_e + a_s + a_n)

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
            l1norm = np.sum(np.abs(p - pn))
        
        else:
            l1norm = np.sum(np.abs(p - pn)) / denominator

        history.append(p.copy())        
    
    history_array = np.stack(history, axis=0)

    return history_array
