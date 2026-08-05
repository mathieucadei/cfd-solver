"""Numerical solver for the 1D Burgers' equation."""



import numpy as np

from .operators import compute_convection_2d_term, compute_diffusion_2d_term
# from .boundary_conditions import apply_burgers_boundary_2d_fvm

from ...config_fvm import BurgersEquation1DFVMConfig
from ...setup.fvm.mesh import (
    build_mesh, 
    build_h_spacing, 
    build_face_areas,
    compute_cell_volumes, 
    build_centers, 
    build_dist,
)
from ...setup.fvm.time_stepping import compute_diffusive_dt_2d_fvm


def solve_burgers_equation_2d_fvm(
    initial_condition: np.ndarray,
    config: BurgersEquation1DFVMConfig,
) -> np.ndarray:
    """Solve the 1D Burgers' equation with an explicit finite-difference scheme."""

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)
    dt = compute_diffusive_dt_2d_fvm(config)

    u, v = initial_condition[0].copy(), initial_condition[1].copy()

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
        
        diffusion_u_term = compute_diffusion_2d_term(
                            un,
                            dist_x,
                            dist_y,
                            face_areas_x, 
                            face_areas_y, 
                            cell_volumes,                             
                            dt, 
                            config.viscosity
                        )

        diffusion_v_term = compute_diffusion_2d_term(
                            vn,
                            dist_x,
                            dist_y,
                            face_areas_x, 
                            face_areas_y, 
                            cell_volumes,                             
                            dt, 
                            config.viscosity
                        )
        
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - convection_u_term[1:-1, 1:-1] + diffusion_u_term[1:-1, 1:-1]
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - convection_v_term[1:-1, 1:-1] + diffusion_v_term[1:-1, 1:-1]
        
        # apply_burgers_boundary_2d_fvm(
        #     u=u,
        #     un=un,
        #     dt=dt,
        #     hx=hx,
        #     dist_x=dist_x,
        #     xc=xc,
        #     lx=config.domain_length_x,
        #     nu=config.viscosity,
        # )
        
        u_history[n] = u
        v_history[n] = v
    
    return u_history, v_history