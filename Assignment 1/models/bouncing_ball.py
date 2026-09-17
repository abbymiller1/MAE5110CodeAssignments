import numpy as np


def dynamics(t, state, params):
    """Continuous dynamics for free-fall.

    state: [position (m), velocity (m/s)]
    returns: [velocity (m/s), acceleration (m/s^2)]
    """
    g = params.get("gravity", 9.81)
    
    pos_dot = state[1]
    vel_dot = -g
    
    return np.array([pos_dot, vel_dot])


def handle_collision(state, params):
    """Detect ground collision and update state discretely."""
    pos, vel = state[0], state[1]
    restitution = params.get("restitution", 0.8)  # 1.0 = elastic, <1.0 = inelastic

    # If ball is at or below ground AND moving downward, bounce
    if pos <= 0.0 and vel < 0.0:
        pos = 0.0
        vel = -restitution * vel

    return np.array([pos, vel])


def calculate_energy(state_traj, params):
    """Calculate potential, kinetic, and total mechanical energy over time."""
    g = params.get("gravity", 9.81)
    m = params.get("mass", 0.1) 

    positions = state_traj[0, :]
    velocities = state_traj[1, :]

    potential_energy = m * g * np.maximum(0, positions)
    kinetic_energy = 0.5 * m * (velocities ** 2)

    return potential_energy, kinetic_energy