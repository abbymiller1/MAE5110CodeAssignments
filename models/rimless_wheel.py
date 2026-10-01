import numpy as np


def dynamics(t, state, params):
    """Single-stance continuous dynamics: identical to an inverted pendulum 
    pivoting at the ground-contact point. The slope doesn't enter here -- 
    it only affects the impact geometry below."""
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (gravity / length) * np.sin(angle) #basically just and undamped pendulum

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_impact(t, state, params):
    #Event detection, for next spoke hitting the ground
    #output zero ONLY when next spoke is making contact, in real code check for if this
    #is less than zero
    alpha = np.pi / params["num_spokes"]
    slope = params["slope"]
    angle = state[0]
    return angle - (slope + alpha)


# For scipy.integrate.solve_ivp: stop integration at the event, and only
# trigger on the forward (theta increasing) crossing.
detect_impact.terminal = True
detect_impact.direction = 1


def reset_state(state, params):
    """Switch stance to the next spoke (coordinate shift by 2*alpha) 
    and reduce angular velocity per conservation of angular momentum 
    about the new contact point."""
    alpha = np.pi / params["num_spokes"]
    angle = state[0]
    angular_velocity = state[1]

    new_angle = angle - 2 * alpha
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)

    return np.array([new_angle, new_angular_velocity])


def generate_params():
    params = {
        "gravity": 9.81,   # gravity (m/s^2)
        "length": 1.0,     # spoke length l (m)
        "mass": 1.0,       # point mass at hub (kg)
        "num_spokes": 8,   # number of spokes N
        "slope": 0.08,     # downhill ground inclination gamma (rad)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state (2,) or trajectory (2, N).
    Energy should be constant *within* a stance phase (no damping) and
    drop at each impact -- that drop is exactly your sanity check."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2 #recall v = omega*l
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy


def generate_initial_condition():
    return np.array([0.0, 0.0])