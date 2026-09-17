import pandas as pd

from core import fdm

from pathlib import Path


def test_cavity_matches_ghia():

    domain_length_x = 1.0
    u_lid = 1.0
    reynolds_number = 100

    config = fdm.CavityFlowConfig(
        domain_length_x=domain_length_x,
        domain_length_y=1.0,
        num_grid_points_x=40,
        num_grid_points_y=40,
        max_iterations=10000,
        max_pseudo_iterations=50,
        time_step=0.001,
        u_lid=u_lid,
        density=1.0,
        viscosity=u_lid*domain_length_x/reynolds_number,
    )

    u_numerical = fdm.solve_cavity_flow(
        fdm.cavity_flow_initial_condition(config), config=config
    )[0][-1]

    DATA = Path(__file__).resolve().parents[2] / 'data'
    ghia_table_1 = pd.read_csv(DATA / 'ghia_table_1.csv')

    validation_u_values=ghia_table_1['100']

    peak = u_numerical.min()
    exact = validation_u_values.min()

    assert abs(peak - exact) < 0.05