# CFD Solver

Small Python CFD learning project evolving toward a Navier-Stokes solver.

## Current status

Currently implemented:

* 1D linear advection
* 1D nonlinear convection
* 1D diffusion
* uniform 1D grid generation
* hat function initial condition
* explicit finite-difference solvers
* solution snapshots and animation

Planned next steps:

* 1D Burgers equation
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

## Run

```bash
python run_advection_1d.py
python run_convection_1d.py
python run_diffusion_1d.py
```
