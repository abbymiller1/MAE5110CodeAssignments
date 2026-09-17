import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4

params = model.generate_params()

initial_state = np.array([params["gamma"] - np.pi / params["n_spokes"], 3.0])

timestep = 1e-4
sim_time = 10.0
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


for step, time in enumerate(time_traj[:-1]):
    current_state = state_traj[:, step]
    next_state = rk4(model.dynamics, time, current_state, params, timestep)
    previous_guard = model.detect_impact(current_state, params)
    next_guard = model.detect_impact(next_state, params)

    # Check whether an impact occurred during this timestep
    if previous_guard < 0 and next_guard >= 0:
        # Estimate where the impact occurred
        fraction = (-previous_guard / (next_guard - previous_guard))
        impact_state = (current_state + fraction * (next_state - current_state))
        # Apply impact reset
        next_state = model.reset_state(impact_state, params)

    state_traj[:, step + 1] = next_state


# Sanity Checks

# Check 1: Impact reset
alpha = np.pi / params["n_spokes"]
impact_state = np.array([params["gamma"] + alpha, 3.0])
new_state = model.reset_state(impact_state, params)
KE_before, PE_before = model.calculate_energy(impact_state, params) 
KE_after, PE_after = model.calculate_energy(new_state, params)

expected_ratio = np.cos(2 * alpha) ** 2
actual_ratio = KE_after / KE_before
print("\nCheck 1: Impact Energy Loss")
print("Expected KE ratio:", expected_ratio)
print("Actual KE ratio:", actual_ratio)

if np.isclose(actual_ratio, expected_ratio):
    print("PASS")
else:
    print("FAIL")


# Check 2: Flat ground at [0, 0]
flat_params = params.copy()
flat_params["gamma"] = 0.0

state = np.array([0.0, 0.0])

derivative = model.dynamics(0.0, state, flat_params)

print("\nCheck 2: Flat Ground Dynamics:")
print("Dynamics:", derivative)
if np.allclose(derivative, [0.0, 0.0]):
    print("PASS")
else:
    print("FAIL")


# Check 3: N -> infinity
print("\nCheck 3:")

for n_spokes in [8, 20, 50, 100, 1000]:

    alpha = np.pi / n_spokes
    energy_ratio = np.cos(2 * alpha) ** 2

    print("N =", n_spokes, "alpha =", alpha, "KE ratio =", energy_ratio)

print("As N increases, alpha -> 0 and KE ratio -> 1.")


#Plot State Trajectory
plt.figure()

plt.plot(time_traj, state_traj[0, :], label="Angle")

plt.plot(time_traj, state_traj[1, :], label="Angular velocity")

plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid()

plt.show()

def simulate_impacts(initial_state, params, n_impacts=20, timestep=1e-3, max_steps = 100000):

    state = initial_state.copy()
    impact_states = []
    time = 0.0

    for impact in range(n_impacts):
        previous_state = state.copy()
        previous_guard = model.detect_impact(previous_state, params)

        for step in range(max_steps):
            next_state = rk4(model.dynamics, time, state, params, timestep)
            next_guard = model.detect_impact(next_state, params)

            # Impact occurred
            if previous_guard < 0 and next_guard >= 0:
                fraction = (-previous_guard / (next_guard - previous_guard))
                impact_state = (state + fraction * (next_state - state))

                # Save the pre-impact state
                impact_states.append(impact_state.copy())

                # Apply reset
                state = model.reset_state(impact_state, params)
                time += timestep * fraction
                break

            state = next_state
            previous_guard = next_guard
            previous_state = state.copy()

            time += timestep

        else:
            break

    return np.array(impact_states)

# Start just after an impact
alpha = np.pi / params["n_spokes"]

post_impact_state = np.array([params["gamma"] - alpha, 1.0])

# Poincare Return Map
impact_states = simulate_impacts(initial_state, params, n_impacts=100, timestep=1e-3)

if len(impact_states) >= 2:

    angular_velocity_k = impact_states[:-1, 1]
    angular_velocity_next = impact_states[1:, 1]

    #Plot Poincare Return Map
    plt.figure()

    plt.scatter(angular_velocity_k, angular_velocity_next, s=10, alpha=0.5)

    minimum = min(angular_velocity_k.min(), angular_velocity_next.min())
    maximum = max(angular_velocity_k.max(), angular_velocity_next.max())

    plt.plot([minimum, maximum], [minimum, maximum], "k--", label="Identity")

    plt.xlabel(r"$\dot{\theta}_k^+$")
    plt.ylabel(r"$\dot{\theta}_{k+1}^+$")
    plt.title("Poincare Return Map")
    plt.legend()
    plt.grid()

    plt.show()


# Return Map
def return_map(angular_velocity, params):
    alpha = np.pi / params["n_spokes"]
    gravity = params["gravity"]
    length = params["length"]
    gamma = params["gamma"]

    velocity_before = np.sqrt(
        angular_velocity**2
        + (4 * gravity / length) * np.sin(gamma) * np.sin(alpha)
    )

    return velocity_before * np.cos(2 * alpha)

def first_impact_velocity(initial_state, params):
    g = params["gravity"]
    l = params["length"]
    alpha = np.pi / params["n_spokes"]
    theta = initial_state[0]
    omega = initial_state[1]
    impact_angle = params["gamma"] + alpha

    if theta >= impact_angle:
        return np.nan

    energy = 0.5 * l**2 * omega**2 + g * l * np.cos(theta)
    impact_energy = g * l * np.cos(impact_angle)

    if energy <= impact_energy:
        return np.nan

    if omega <= 0 and theta <= 0:
        return np.nan

    return np.sqrt(2 * (energy - impact_energy) / l**2)

# Find Fixed Point
angular_velocity_min = 0.1
angular_velocity_max = 3.0
n_points = 200

angular_velocity_values = np.linspace(angular_velocity_min, angular_velocity_max, n_points)

return_values = []

for angular_velocity in angular_velocity_values:
    next_velocity = return_map(angular_velocity, params)
    return_values.append(next_velocity)

return_values = np.array(return_values)

# Find intersection with identity line
difference = (return_values - angular_velocity_values)

valid = np.isfinite(difference)

angular_velocity_star = angular_velocity_values[valid][np.argmin(np.abs(difference[valid]))]
print("\nFixed point =", angular_velocity_star, "rad/s")

def find_fixed_point(params, angular_velocity_min=0.01, angular_velocity_max=10.0, n_points=200, timestep=1e-3):
    angular_velocity_values = np.linspace(angular_velocity_min, angular_velocity_max, n_points)
    differences = np.zeros(len(angular_velocity_values))

    for i, angular_velocity in enumerate(angular_velocity_values):
        next_angular_velocity = return_map(angular_velocity, params)
        differences[i] = next_angular_velocity - angular_velocity

    for i in range(len(angular_velocity_values) - 1):
        if differences[i] * differences[i + 1] <= 0:
            angular_velocity_1 = angular_velocity_values[i]
            angular_velocity_2 = angular_velocity_values[i + 1]

            difference_1 = differences[i]
            difference_2 = differences[i + 1]

            # Linear interpolation
            angular_velocity_star = angular_velocity_1 - difference_1 * (angular_velocity_2 - angular_velocity_1) / (difference_2 - difference_1)

            return angular_velocity_star

    return np.nan

#Plot Poincare return map
plt.figure()
plt.plot(angular_velocity_values, return_values, label="Return map")
plt.plot(angular_velocity_values, angular_velocity_values, "k--", label="Identity")
plt.plot(angular_velocity_star, angular_velocity_star, "o", label="Fixed point")

plt.xlabel("Angular velocity at step k (rad/s)")
plt.ylabel("Angular velocity at step k+1 (rad/s)")
plt.title("Poincare Return Map")

plt.legend()
plt.grid()
plt.savefig("Poincare.png", dpi=300, bbox_inches="tight")
plt.show()

# Floquet Multiplier
def floquet_multiplier(params, angular_velocity_star, delta=1e-4):

    angular_velocity_plus = return_map(angular_velocity_star + delta, params)
    angular_velocity_minus = return_map(angular_velocity_star - delta, params)

    multiplier = (angular_velocity_plus - angular_velocity_minus) / (2 * delta)

    return multiplier


# Region of Attraction
def classify_trajectory(initial_state, params):
    velocity = first_impact_velocity(initial_state, params)

    if not np.isfinite(velocity):
        return 0

    alpha = np.pi / params["n_spokes"]
    velocity *= np.cos(2 * alpha)

    return int(velocity > 0)


def calculate_roa(params, angle_values, angular_velocity_values):

    roa = np.zeros((len(angular_velocity_values), len(angle_values)), dtype=int)

    for i, angular_velocity in enumerate(angular_velocity_values):
        for j, angle in enumerate(angle_values):
            initial_state = np.array([angle, angular_velocity])
            roa[i, j] = classify_trajectory(initial_state, params)

    return roa


angle_values = np.linspace(-1.5, 1.5, 30)
angular_velocity_values = np.linspace(-5, 5, 30)

roa = calculate_roa(params, angle_values, angular_velocity_values)


plt.figure(figsize=(8, 6))

plt.imshow(
    roa,
    origin="lower",
    extent=[angle_values[0], angle_values[-1], angular_velocity_values[0], angular_velocity_values[-1]],
    aspect="auto",
    cmap="viridis"
)
plt.scatter(
    angular_velocity_star,
    angular_velocity_star,
    s=50,
    label="Fixed point"
)

#Plot Region of Attraction
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
plt.title("Region of Attraction")

plt.colorbar(label="Attractor")
plt.savefig("RoA.png", dpi=300, bbox_inches="tight")
plt.show()


# Gamma Sweep
gammas = np.linspace(0.01, 0.25, 20)

gamma_multipliers = []
gamma_roa_fraction = []

for gamma in gammas:

    print("Gamma =", gamma)

    params_i = params.copy()
    params_i["gamma"] = gamma

    # Find fixed point
    angular_velocity_star_i = find_fixed_point(params_i)

    if np.isfinite(angular_velocity_star_i):

        # Calculate Floquet multiplier
        multiplier_i = floquet_multiplier(params_i, angular_velocity_star_i)

        # Calculate RoA
        roa_i = calculate_roa(params_i, angle_values, angular_velocity_values)

        fraction_i = np.mean(roa_i == 1)

    else:

        multiplier_i = np.nan
        fraction_i = 0.0

    gamma_multipliers.append(multiplier_i)
    gamma_roa_fraction.append(fraction_i)

#Plot Floquet Multiplier vs Slope
plt.figure()

plt.plot(gammas, gamma_multipliers, "o-")

plt.axhline(1.0, color="k", linestyle="--")

plt.xlabel(r"Slope $\gamma$")
plt.ylabel("Floquet multiplier")
plt.title("Floquet Multiplier vs Slope")
plt.grid()
plt.savefig("gamma_floquet.png", dpi=300, bbox_inches="tight")
plt.show()

#Plot RoA  vs Slope
plt.figure()

plt.plot(gammas, gamma_roa_fraction, "o-")

plt.xlabel(r"Slope $\gamma$")
plt.ylabel("Fraction of state space in RoA")
plt.title("Region of Attraction vs Slope")
plt.grid()
plt.savefig("gamma_RoA.png", dpi=300, bbox_inches="tight")
plt.show()


# Number of Spokes Sweep
spoke_values = range(6, 13)

spoke_multipliers = []
spoke_roa_fraction = []

for n_spokes in spoke_values:

    print("N =", n_spokes)

    params_i = params.copy()
    params_i["n_spokes"] = n_spokes

    # Find fixed point
    fixed_point_i = find_fixed_point(params_i)
    delta = 1e-4

    velocity_plus = (fixed_point_i + delta)
    velocity_minus = (fixed_point_i - delta)
    return_plus = return_map(velocity_plus, params_i)
    return_minus = return_map(velocity_minus, params_i) 
    multiplier = (return_plus - return_minus) / (2 * delta)

    spoke_multipliers.append(multiplier)

    # Estimate RoA size
    count = 0
    total = 0

    for angular_velocity in angular_velocity_values:

        for angle in angle_values:

            initial_state = np.array([angle, angular_velocity])

            count += classify_trajectory(initial_state, params_i)

            total += 1

    spoke_roa_fraction.append(count / total)

#Plot Floquet Multiplier vs Number of Spokes
plt.figure()

plt.plot(list(spoke_values), spoke_multipliers, "o-")

plt.axhline(1.0, color="k", linestyle="--")

plt.xlabel("Number of spokes N")
plt.ylabel("Floquet multiplier")
plt.title("Floquet Multiplier vs Number of Spokes")
plt.grid()
plt.savefig("spokes_floquet.png", dpi=300, bbox_inches="tight")
plt.show()

#Plot RoA  vs Number of Spokes
plt.figure()

plt.plot(list(spoke_values), spoke_roa_fraction, "o-")

plt.xlabel("Number of spokes N")
plt.ylabel("Fraction of state space in RoA")
plt.title("Region of Attraction vs Number of Spokes")
plt.grid()
plt.savefig("spokes_RoA.png", dpi=300, bbox_inches="tight")
plt.show()
