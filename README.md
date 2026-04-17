# CFD Solver

Small Python CFD learning project evolving toward a Navier-Stokes solver, currently focused on 1D finite-difference models.

## Current status

Currently implemented:

* 1D linear advection
* 1D nonlinear convection
* 1D diffusion
* 1D Burgers equation
* uniform 1D grid generation
* hat function initial condition
* Cole-Hopf initial condition
* explicit finite-difference solvers
* solution plots and animations

Planned next steps:

* 2D equations
* incompressible Navier-Stokes components

## Implemented models

The current solvers advance:

* the 1D linear advection equation:

  du/dt + c du/dx = 0

  using a simple explicit upwind finite-difference scheme.

* the 1D nonlinear convection equation:

  du/dt + u du/dx = 0

  using a simple explicit upwind finite-difference scheme.

* the 1D diffusion equation:

  du/dt = ν d²u/dx²

  using a simple explicit central finite-difference scheme.

* the 1D Burgers equation:

  du/dt + u du/dx = ν d²u/dx²

  using an explicit upwind scheme for the convective term and a central scheme for the diffusive term.

## Validation

This project includes validation workflows that compare numerical finite-difference solutions with analytical reference solutions.

For the 1D diffusion equation, the numerical solver is validated against a Fourier-based analytical solution of the heat equation.

For the 1D Burgers equation, the numerical solver is validated against the analytical Cole-Hopf solution.

These comparisons are used to assess solver correctness and visualize agreement between numerical and analytical results.

## Run

```bash
python run_advection_1d.py
python run_convection_1d.py
python run_diffusion_1d.py
python run_diffusion_1d_vs_heat.py
python run_burgers_equation_1d.py
python run_burgers_equation_1d_vs_cole_hopf.py
```
