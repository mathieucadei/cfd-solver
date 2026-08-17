"""Numerical solver for the 2D diffusion equation."""



import numpy as np

# from .boundary_conditions import apply_poisson_boundary_2d

from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes



def solve_poisson_2d_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 2D Laplace equation with an explicit central finite-difference scheme."""

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)

    p, b = initial_condition
    pn = np.empty_like(p)

    history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))

    history[0] = initial_condition[1]

    a_w = face_areas_x[1:-1, 1:-1] / dist_x[:-1]
    a_e = face_areas_x[1:-1, 2:] / dist_x[1:]
    a_s = face_areas_y[1:-1, 1:-1] / dist_y[:-1, None]
    a_n = face_areas_y[2:, 1:-1] / dist_y[1:, None]

    fb_w = a_w * b[1:-1, :-2]
    fb_e = a_e * b[1:-1, 2:]
    fb_s = a_s * b[:-2, 1:-1]
    fb_n = a_n * b[2:, 1:-1]

    for n in range(1, config.max_iterations + 1):

        pn = p.copy()

        f_w = a_w * pn[1:-1, :-2]
        f_e = a_e * pn[1:-1, 2:]
        f_s = a_s * pn[:-2, 1:-1]
        f_n = a_n * pn[2:, 1:-1]

        p[1:-1, 1:-1] =(f_e + f_w + f_n + f_s - b[1:-1, 1:-1]) / (a_w + a_e + a_s + a_n)

        # apply_poisson_boundary_2d(p)
        
        history[n] = p
    
    return history
