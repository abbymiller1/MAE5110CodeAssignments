import numpy as np


def explicit_euler(dynamics_fn, t, state, params, timestep):
    """Advance state by one step using Explicit (Forward) Euler.

    Args:
        dynamics_fn: callable(t, state, params) -> state_derivative
        t: current time (float)
        state: current state, shape (n,)
        params: dict of model parameters
        timestep: step size dt

    Returns:
        next_state: state at t + timestep, shape (n,)
    """
    state_derivative = dynamics_fn(t, state, params)
    next_state = state + timestep * state_derivative
    return next_state
