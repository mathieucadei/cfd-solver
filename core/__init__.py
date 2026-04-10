from .config import Advection1DConfig, Convection1DConfig, Diffusion1DConfig
from .grids import make_1d_grid, compute_1d_grid_points_spacing
from .time_step import compute_time_step
from .initial_conditions import hat_initial_condition
from .models.advection_1d import solve_advection_1d
from .models.convection_1d import solve_convection_1d
from .models.diffusion_1d import solve_diffusion_1d
from .plotting import plot_snapshots, plot_animation