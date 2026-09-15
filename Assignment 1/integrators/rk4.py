import numpy as np


def rk4(dynamics_fn, t, state, params, timestep):
    """Advance state by one step using classic 4th-order Runge-Kutta.

    Args:
        dynamics_fn: callable(t, state, params) -> state_derivative
        t: current time (float)
        state: current state, shape (n,)
        params: dict of model parameters
        timestep: step size dt

    Returns:
        next_state: state at t + timestep, shape (n,)
    """
    dt = timestep

    k1 = dynamics_fn(t, state, params)
    k2 = dynamics_fn(t + dt / 2, state + dt / 2 * k1, params)
    k3 = dynamics_fn(t + dt / 2, state + dt / 2 * k2, params)
    k4 = dynamics_fn(t + dt, state + dt * k3, params)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state