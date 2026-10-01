from core import analytical, fvm


def test_channel_matches_poiseuille():
    config = fvm.ChannelFlowConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_cells_x=40,
        num_cells_y=40,
        expansion_ratio_x=0.0,
        expansion_ratio_y=0.0,
        max_pseudo_iterations=50,
        time_step=0.001,
        source=1.0,
        density=1.0,
        viscosity=0.1,
        u_l1_norm_target=1e-6,
    )

    u_numerical = fvm.solve_channel_flow(
        fvm.channel_flow_initial_condition(config), config=config
    )[0][-1]

    yc_array = fvm.build_centers(config)[1]

    u_analytical = analytical.compute_poiseuille_flow(
        y_array=yc_array,
        config=config,
    )

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.01