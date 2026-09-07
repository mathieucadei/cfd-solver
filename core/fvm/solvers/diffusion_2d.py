"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from ..operators import compute_diffusion_2d_term
from ..boundary_conditions import apply_diffusion_boundary_2d
from ..time_stepping import compute_diffusive_dt_2d
from ..mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes
from ..initial_conditions import hat_initial_condition_2d


def solve_diffusion_2d(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D diffusion equation with an explicit central finite-difference scheme."""

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)
    dt = compute_diffusive_dt_2d(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        diffusion_term = compute_diffusion_2d_term(
                            un,
                            dist_x,
                            dist_y,
                            face_areas_x, 
                            face_areas_y, 
                            cell_volumes,                             
                            dt, 
                            config.viscosity
                        )
        
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + diffusion_term[1:-1, 1:-1]

        # apply_diffusion_boundary_2d(
        #     u=u,
        #     un=un,
        #     dt=dt,
        #     dist_x=dist_x,
        #     dist_y=dist_y,
        #     face_areas_x=face_areas_x, 
        #     face_areas_y=face_areas_y, 
        #     cell_volumes=cell_volumes,
        #     xc=xc,
        #     yc=yc,
        #     lx=config.domain_length_x,
        #     ly=config.domain_length_y,
        #     nu=config.viscosity,
        # )

        history[n] = u
    
    return history