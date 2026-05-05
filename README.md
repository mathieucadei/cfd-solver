# CFD Solver

Small Python CFD learning project evolving toward a Navier-Stokes solver, currently focused on 1D and 2D finite-difference models.

## Current status

Currently implemented:

* 1D and 2D linear advection
* 1D and 2D nonlinear convection
* 1D and 2D diffusion
* 1D and 2D Burgers equation
* 2D Laplace equation
* 2D Poisson equation with configurable source terms
* uniform grid generation
* hat-function and Cole-Hopf initial conditions
* explicit finite-difference time-marching solvers
* iterative pressure/potential solves using L1 convergence or fixed iteration limits
* solution plots and animations

Planned next steps:

* validate 2D Laplace and Poisson solvers
* use the Poisson solver as a pressure-projection building block
* add tests for boundary conditions, source placement, and convergence behavior
* continue toward incompressible Navier-Stokes / cavity flow

## Project structure

```text
core/config.py              simulation dataclasses
core/setup/                 grids, time steps, initial conditions
core/numerics/              finite-difference solvers
core/analytical/            analytical reference solutions
post_processing/            plotting and animation helpers
run_*.py                    executable examples
```

## Example visualizations

### 2D diffusion

![2D diffusion](docs/images/diffusion_2d.gif)

### Validation: 1D diffusion vs heat equation

![1D diffusion vs heat equation](docs/images/diffusion_1d_vs_heat_solution.png)

### 2D Poisson equation

![2D Poisson equation](docs/images/poisson_2d_solution.png)

## Implemented models

The current solvers model:

* the 1D linear advection equation:

  du/dt + c du/dx = 0

  using an explicit upwind finite-difference scheme.

* the 2D linear advection equation:

  du/dt + c du/dx + c du/dy = 0

  using an explicit upwind finite-difference scheme.

* the 1D nonlinear convection equation:

  du/dt + u du/dx = 0

  using an explicit upwind finite-difference scheme.

* the 2D nonlinear convection equations:

  du/dt + u du/dx + v du/dy = 0

  dv/dt + u dv/dx + v dv/dy = 0

  using an explicit upwind finite-difference scheme.

* the 1D diffusion equation:

  du/dt = ν d²u/dx²

  using an explicit central finite-difference scheme.

* the 2D diffusion equation:

  du/dt = ν (d²u/dx² + d²u/dy²)

  using an explicit central finite-difference scheme.

* the 1D Burgers equation:

  du/dt + u du/dx = ν d²u/dx²

  using an explicit upwind scheme for the convective term and a central scheme for the diffusive term.

* the 2D Burgers equations:

  du/dt + u du/dx + v du/dy = ν (d²u/dx² + d²u/dy²)

  dv/dt + u dv/dx + v dv/dy = ν (d²u/dx² + d²u/dy²)

  using an explicit upwind scheme for the convective terms and a central scheme for the diffusive terms.

* the 2D Laplace equation:

  d²p/dx² + d²p/dy² = 0

  solved iteratively with finite differences until an L1 target is reached.

* the 2D Poisson equation:

  d²p/dx² + d²p/dy² = b

  solved iteratively with configurable positive and negative source terms.

## Validation

This project includes validation workflows that compare numerical finite-difference solutions with analytical reference solutions.

For the 1D diffusion equation, the numerical solver is validated against a Fourier-based analytical solution of the heat equation.

For the 1D Burgers equation, the numerical solver is validated against the analytical Cole-Hopf solution.

These comparisons are used to assess solver correctness and visualize agreement between numerical and analytical results.

## Run

```bash
python run_advection_1d.py
python run_advection_2d.py
python run_convection_1d.py
python run_convection_2d.py
python run_diffusion_1d.py
python run_diffusion_1d_vs_heat.py
python run_diffusion_2d.py
python run_burgers_equation_1d.py
python run_burgers_equation_1d_vs_cole_hopf.py
python run_burgers_equation_2d.py
python run_laplace_2d.py
python run_poisson_2d.py
```
