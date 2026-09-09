"""Poiseuille Flow solution utilities for the channel flow."""

import numpy as np

def compute_poiseuille_flow(
        y_array: np.ndarray,
        config: object,
):

    u = config.source / (2 * config.viscosity) * y_array * (config.domain_length_y - y_array)

    return u