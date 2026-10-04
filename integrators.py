def explicit_euler(dynamics_fn, t, state, timestep, params):
    """Advance the state using Explicit Euler."""
    state_derivative = dynamics_fn(t, state, params)
    next_state = state + timestep * state_derivative
    return next_state


def rk4(dynamics_fn, t, state, timestep, params):
    """Advance the state using fourth-order Runge-Kutta."""
    dt = timestep

    k1 = dynamics_fn(t, state, params)
    k2 = dynamics_fn(t + dt / 2, state + dt / 2 * k1, params)
    k3 = dynamics_fn(t + dt / 2, state + dt / 2 * k2, params)
    k4 = dynamics_fn(t + dt, state + dt * k3, params)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state