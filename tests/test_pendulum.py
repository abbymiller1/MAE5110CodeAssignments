import inspect

import numpy as np
import pytest

from integrators import rk4
from models import pendulum


def simulate_pendulum(params, initial_state, n_steps=100, timestep=0.01):
    states = [initial_state]
    """Simulate the pendulum with RK4 and return states with shape (2, n_steps + 1)."""

    for step in range(n_steps):
        states.append(rk4(pendulum.dynamics, step * timestep, states[-1], timestep, params))

    return np.array(states).T


def find_total_energy(states, params):
    """Return kinetic plus potential energy at every time step in states."""
    kinetic_energy, potential_energy = pendulum.calculate_energy(states, params)
    return kinetic_energy + potential_energy


def test_energy_conservation():
    """With no damping and no torque, total energy should stay constant."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0  # Ensure no damping for energy conservation test
    params["torque"] = 0.0  # Ensure no external torque for energy conservation test
    states = simulate_pendulum(params, np.array([0.25, 0.5]))
    
    total_energy = find_total_energy(states, params)

    # Check that the total energy is approximately constant
    assert np.allclose(total_energy, total_energy[0], rtol=1e-6), "Total energy is not conserved"


def test_damping_removes_energy():
    """Damping removes energy, so total energy should decrease."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5  # Introduce damping
    params["torque"] = 0.0  # Ensure no external torque for energy conservation test
    states = simulate_pendulum(params, np.array([0.25, 0.5]))
    
    total_energy = find_total_energy(states, params)

    # Check that the total energy decreases over time due to damping
    energy_change = np.diff(total_energy)
    assert np.all(energy_change <= 1e-9), "Energy increased at some step with damping"


def test_torque():
    """With no damping, the energy gained should equal the torque's work:  tau * delta_theta"""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0
    initial_state = np.array([0.25, 0.5])
 
    states = simulate_pendulum(params, initial_state)

    total_energy = find_total_energy(states, params)
    energy_change = total_energy[-1] - total_energy[0]
    work = params["torque"] * (states[0, -1] - states[0, 0])
    assert np.isclose(energy_change, work, rtol=1e-6), "Energy change does not match work done by torque"