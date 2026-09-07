"""Numerical solver for the 1D diffusion equation."""



import numpy as np

from ..operators import compute_diffusion_1d_term
from ..boundary_conditions import apply_diffusion_boundary_1d

from ..mesh import build_mesh, build_hx_spacing, build_x_face_positions, build_x_centers, build_dist_x
from ..time_stepping import compute_diffusive_dt_1d


def solve_diffusion_1d(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D diffusion equation with an explicit central finite-difference scheme."""

    hx = build_hx_spacing(config)
    dist_x = build_dist_x(config)
    xc = build_x_centers(config)
    dt = compute_diffusive_dt_1d(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        diffusion_term = compute_diffusion_1d_term(un, hx, dist_x, dt, config.viscosity)
        
        u[1:-1] = un[1:-1] + diffusion_term[1:-1]

        apply_diffusion_boundary_1d(
            u=u,
            un=un,
            dt=dt,
            hx=hx,
            dist_x=dist_x,
            xc=xc,
            lx=config.domain_length_x,
            nu=config.viscosity,
        )

        history[n] = u
    
    return history