import numpy as np

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "n_spokes": 8, # number of evenly spaced spokes
        "gamma": 0.08 # slope incline angle
        #"damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
    }
    return params


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        gravity * np.sin(angle)
    ) / (length)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_impact (state, params):
    """
    Impact-event guard: 
    Returns 0 exactly at the foot strike (collision when angle = slope_angle + half_slope_angle). 
    If the sign changes from negative to positive that means a strike occured.
    """
    alpha = np.pi / params["n_spokes"]
    angle = state[0]
    return angle - (params["gamma"] + alpha)
    

def reset_state(state, params):
    """
    Instantaneous plastic collision at impact:
      1. Coordinate shift: the new spoke becomes the reference, so
         the angle jumps back by 2*half_spoke_angle.
      2. Velocity reset from conservation of angular momentum about
         the new contact point: angular_velocity+ = angular_velocity- *
         cos(2*half_spoke_angle).
    """
    alpha = np.pi / params["n_spokes"]
    angle = state[0]
    angular_velocity = state[1]

    new_angle = angle - 2 * alpha
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)
    return np.array([new_angle, new_angular_velocity])


def calculate_energy(state, params):
    """
    Compute energies for a state ``(2,)`` or trajectory ``(2, N)``.
    Mass sits above the pivot so potential energy maximized at angle = 0, which is an unstable equilibrium. 
    """
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy
