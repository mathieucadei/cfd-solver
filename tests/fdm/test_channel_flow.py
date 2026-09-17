from core import analytical, fdm


def test_channel_matches_poiseuille():
    config = fdm.ChannelFlowConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_grid_points_x=40,
        num_grid_points_y=40,
        max_pseudo_iterations=50,
        time_step=0.001,
        source=1.0,
        density=1.0,
        viscosity=0.1,
        u_l1_norm_target=1e-6,
    )

    u_numerical = fdm.solve_channel_flow(
        fdm.channel_flow_initial_condition(config), config=config
    )[0][-1]

    y_array = fdm.make_y_grid(config)

    u_analytical = analytical.compute_poiseuille_flow(
        y_array=y_array,
        config=config,
    )

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.01