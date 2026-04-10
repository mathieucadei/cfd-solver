# Mini Navier-Stokes Solver

Small Python CFD learning project intended to grow progressively toward a Navier-Stokes solver.

## Current status

Currently implemented:
- 1D linear advection equation
- uniform 1D grid generation
- hat function initial condition
- explicit finite-difference time marching
- solution snapshots and animation

Planned next steps:
- 1D convection
- 1D diffusion
- 1D Burgers equation
- 2D equations
- incompressible Navier-Stokes components

## Current example

The current solver advances the 1D linear advection equation:

du/dt + c du/dx = 0

using a simple explicit upwind finite-difference scheme.

## Run

```bash
python run_advection_1d.py
```