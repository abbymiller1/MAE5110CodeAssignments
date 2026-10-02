
import numpy as np

from integrators import rk4
from models import pendulum


def simulate_pendulum(params, initial_state, n_steps=100, dt=0.01):
    """Simulate the pendulum for n_steps using RK4."""
    states = [initial_state.copy()]

    for i in range(n_steps):
        next_state = rk4(
            pendulum.dynamics,
            i * dt,
            states[-1],
            dt,
            params,
        )
        states.append(next_state)

    return np.array(states).T


def total_energy(states, params):
    """Calculate total mechanical energy."""
    kinetic, potential = pendulum.calculate_energy(states, params)
    return kinetic + potential


def test_energy_conservation():
    """Energy should remain constant without torque or damping."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    states = simulate_pendulum(params, np.array([0.25, 0.5]))
    energy = total_energy(states, params)

    assert np.all(np.isclose(energy, energy[0], rtol=1e-6, atol=1e-8))


def test_damping_removes_energy():
    """Positive damping should remove mechanical energy."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0

    states = simulate_pendulum(params, np.array([0.25, 0.5]))
    energy = total_energy(states, params)

    assert np.all(np.diff(energy) <= 1e-8)
    assert energy[-1] < energy[0]


def test_torque_work():
    """With no damping, energy change should equal applied torque work."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0

    states = simulate_pendulum(params, np.array([0.25, 0.5]))
    energy = total_energy(states, params)

    delta_energy = energy[-1] - energy[0]
    work = params["torque"] * (states[0, -1] - states[0, 0])

    assert np.isclose(delta_energy, work, rtol=1e-6, atol=1e-8)
