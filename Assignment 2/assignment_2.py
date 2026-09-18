"""Assignment 2: stabilize and walk the inverted-pendulum walker down a slope.

Two controllers are combined:
  - a continuous ankle-torque balance law, active only once the state has
    reached its (grid-verified) region of attraction (RoA);
  - a discrete step-to-step controller that chooses the angle of attack once
    per stance phase, from a lookup table built off a gridded theta = 0
    Poincare return map, to steer the walker into that RoA in as few steps
    as possible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators.rk4 import rk4 as integrate
from models import inverted_pendulum_walker as model

#Parameters
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

# Continuous ankle-balance controller gains
Kp = 8.0
Kd = 4.0

# Torque and angle-of-attack bounds
min_torque = -0.1 * params["mass"] * params["gravity"] * params["length"]
max_torque = 0.05 * params["mass"] * params["gravity"] * params["length"]
min_angle_of_attack = np.pi / 8
max_angle_of_attack = np.pi / 7
froude_2_velocity = np.sqrt(2 * params["gravity"] / params["length"])


def compute_ankle_balance_torque(state, params):
    """Feedback-linearizing PD law that stabilizes theta = 0.
    theta_ddot = g/l*sin(theta) + tau/(m*l^2)
    u = -Kp*theta - Kd*theta_dot placing the poles.
    """
    gravity, length, mass = params["gravity"], params["length"], params["mass"]
    theta, theta_dot = state
    stabilizing_input = -Kp * theta - Kd * theta_dot
    torque = mass * length**2 * stabilizing_input - mass * gravity * length * np.sin(theta)
    return np.clip(torque, min_torque, max_torque)


def balance_converges(theta0, theta_dot0, params, timestep, sim_time, tolerance):
    """Simulate the balance-only closed loop (no footsteps) and report
    whether it settles at the upright equilibrium instead of swinging far
    enough to trigger a footswitch."""
    state = np.array([theta0, theta_dot0])
    strike_angle = params["incline"] + params["angle_of_attack"]
    local_params = dict(params)
    for step in range(round(sim_time / timestep)):
        t = step * timestep
        local_params["ankle_torque"] = compute_ankle_balance_torque(state, local_params)
        next_state = integrate(model.dynamics, t, state, local_params, timestep)
        if abs(next_state[0]) >= strike_angle:
            return False
        state = next_state
    return abs(state[0]) <= tolerance and abs(state[1]) <= tolerance


def compute_roa_grid(theta_values, velocity_values, params, timestep, sim_time, tolerance):
    """Boolean grid, True where the balance controller converges."""
    roa_grid = np.zeros((len(velocity_values), len(theta_values)), dtype=bool)
    for i, theta_dot0 in enumerate(velocity_values):
        for j, theta0 in enumerate(theta_values):
            roa_grid[i, j] = balance_converges(theta0, theta_dot0, params, timestep, sim_time, tolerance)
    return roa_grid


def roa_contains(theta_values, velocity_values, roa_grid, state):
    """Nearest-grid-point membership test for the region of attraction."""
    theta, theta_dot = state
    if not (theta_values[0] <= theta <= theta_values[-1]):
        return False
    if not (velocity_values[0] <= theta_dot <= velocity_values[-1]):
        return False
    j = np.argmin(np.abs(theta_values - theta))
    i = np.argmin(np.abs(velocity_values - theta_dot))
    return bool(roa_grid[i, j])


def compute_ankle_torque(state, params, theta_values, velocity_values, roa_grid):
    """Ankle torque command: off while stepping, on once inside the RoA."""
    if not roa_contains(theta_values, velocity_values, roa_grid, state):
        return 0.0
    return compute_ankle_balance_torque(state, params)


# ---------------------------------------------------------------------------
# Discrete step-to-step controller. Poincare section: theta = 0, which is
# crossed exactly once per stance phase no matter the chosen angle of
# attack (post-impact theta = incline - angle_of_attack < 0, footswitch
# theta = incline + angle_of_attack > 0), unlike the footswitch itself.
# ---------------------------------------------------------------------------
def advance_to_next_section(theta_dot, angle_of_attack, params, timestep, max_time=2.0):
    """One passive (ankle torque off) step of the return map: from velocity
    `theta_dot` at the section, integrate through the footswitch chosen by
    `angle_of_attack` and return the velocity at the next theta = 0
    crossing, or None if the walker stalls before completing the step."""
    local_params = dict(params)
    local_params["angle_of_attack"] = angle_of_attack
    local_params["ankle_torque"] = 0.0

    state = np.array([0.0, theta_dot])
    struck = False
    for step in range(round(max_time / timestep)):
        t = step * timestep
        next_state = integrate(model.dynamics, t, state, local_params, timestep)

        if not struck and model.event_guard(state, next_state, local_params):
            next_state = model.event_dynamics(next_state, local_params)
            struck = True
        elif struck and state[0] < 0.0 <= next_state[0]:
            crossing_fraction = -state[0] / (next_state[0] - state[0])
            return state[1] + crossing_fraction * (next_state[1] - state[1])

        if next_state[1] <= 0.0:
            return None  # stalled: swung back before completing the step
        state = next_state

    return None  # never crossed the section within max_time


def build_step_to_step_map(theta_dot_grid, angle_of_attack_grid, params, timestep):
    """Grid the return map: next_theta_dot[i, j] from theta_dot_grid[i]
    under angle_of_attack_grid[j], or NaN where that step stalls."""
    next_theta_dot = np.full((theta_dot_grid.size, angle_of_attack_grid.size), np.nan)
    for i, theta_dot in enumerate(theta_dot_grid):
        for j, angle_of_attack in enumerate(angle_of_attack_grid):
            result = advance_to_next_section(theta_dot, angle_of_attack, params, timestep)
            if result is not None:
                next_theta_dot[i, j] = result
    return next_theta_dot


def compute_steps_to_standstill(theta_dot_grid, next_theta_dot, theta_values, velocity_values, roa_grid):
    """Backward induction over the gridded return map: steps_to_stop[i] is
    the fewest footsteps needed to bring theta_dot_grid[i] into the RoA;
    policy[i] indexes the angle of attack that achieves it. Both are
    inf/-1 where no footstep sequence reaches the RoA on this grid."""
    n_states = theta_dot_grid.size
    already_stopped = np.array([
        roa_contains(theta_values, velocity_values, roa_grid, (0.0, theta_dot))
        for theta_dot in theta_dot_grid
    ])
    steps_to_stop = np.where(already_stopped, 0.0, np.inf)
    policy = np.full(n_states, -1)

    updated = True
    while updated:
        updated = False
        for i in range(n_states):
            if steps_to_stop[i] == 0:
                continue
            for j, next_value in enumerate(next_theta_dot[i]):
                if np.isnan(next_value):
                    continue
                nearest = np.argmin(np.abs(theta_dot_grid - next_value))
                candidate_steps = steps_to_stop[nearest] + 1
                if candidate_steps < steps_to_stop[i]:
                    steps_to_stop[i] = candidate_steps
                    policy[i] = j
                    updated = True
    return steps_to_stop, policy


def compute_max_steps_to_standstill(theta_dot_grid, next_theta_dot, steps_to_stop):
    """Longest footstep sequence that still reaches the RoA. Exact rather
    than a heuristic: each footswitch collision loses energy
    (cos(2*angle_of_attack) < 1) and this incline is too shallow to make
    that up in one swing, so theta_dot strictly decreases every step and
    the return map has no cycles."""
    n_states = theta_dot_grid.size
    already_stopped = steps_to_stop == 0
    max_steps = np.where(already_stopped, 0.0, -np.inf)
    policy = np.full(n_states, -1)

    updated = True
    while updated:
        updated = False
        for i in range(n_states):
            if already_stopped[i] or not np.isfinite(steps_to_stop[i]):
                continue
            for j, next_value in enumerate(next_theta_dot[i]):
                if np.isnan(next_value):
                    continue
                nearest = np.argmin(np.abs(theta_dot_grid - next_value))
                if not np.isfinite(max_steps[nearest]):
                    continue
                candidate_steps = max_steps[nearest] + 1
                if candidate_steps > max_steps[i]:
                    max_steps[i] = candidate_steps
                    policy[i] = j
                    updated = True
    max_steps[np.isneginf(max_steps)] = np.inf  # can't reach the RoA at all
    return max_steps, policy


def report_grid_resolution_check(candidate_sizes, angle_of_attack_grid, params, timestep,
                                  theta_values, velocity_values, roa_grid, theta_dot_of_interest):
    """Print how the step count for `theta_dot_of_interest` changes with
    state-grid density, to justify the chosen resolution (coarser grids
    that disagree are not fine enough). Returns the (theta_dot_grid,
    next_theta_dot) built for the finest candidate size."""
    print(f"{'n_theta_dot':>11} | {'steps@' + f'{theta_dot_of_interest:.2f}':>10} | unreachable states")
    theta_dot_grid = next_theta_dot = None
    for n_theta_dot in candidate_sizes:
        theta_dot_grid = np.linspace(0.0, froude_2_velocity, n_theta_dot)
        next_theta_dot = build_step_to_step_map(theta_dot_grid, angle_of_attack_grid, params, timestep)
        steps_to_stop, _ = compute_steps_to_standstill(
            theta_dot_grid, next_theta_dot, theta_values, velocity_values, roa_grid
        )
        nearest = np.argmin(np.abs(theta_dot_grid - theta_dot_of_interest))
        n_unreachable = int(np.sum(np.isinf(steps_to_stop)))
        print(f"{n_theta_dot:>11} | {steps_to_stop[nearest]:>10.0f} | {n_unreachable}/{n_theta_dot}")
    return theta_dot_grid, next_theta_dot


def choose_from_policy(theta_dot, theta_dot_grid, angle_of_attack_grid, policy):
    """Look up one control value: the angle of attack the policy assigns to
    the grid point nearest `theta_dot`."""
    nearest = np.argmin(np.abs(theta_dot_grid - abs(theta_dot)))
    chosen = policy[nearest]
    return angle_of_attack_grid[chosen] if chosen >= 0 else angle_of_attack_grid[0]


def compute_new_stance_position(pre_impact_state, stance_position, params):
    """World-frame (x, y) of the swing foot at the instant of touchdown,
    which becomes the new stance foot once the legs swap.

    Must be called with the state at the strike angle -- i.e. *before*
    model.event_dynamics() resets theta -- together with the angle of
    attack that was in effect for that step, since both feet's positions
    are only fixed relative to each other at that instant.
    """
    theta = pre_impact_state[0]
    alpha = params["angle_of_attack"]
    length = params["length"]
    stance_position = np.asarray(stance_position, dtype=float)

    hub = stance_position + length * np.array([np.sin(theta), np.cos(theta)])
    swing_angle = theta - 2 * alpha
    swing_foot = hub - length * np.array([np.sin(swing_angle), np.cos(swing_angle)])
    return swing_foot


def simulate_walk(initial_state, params, theta_values, velocity_values, roa_grid,
                   theta_dot_grid, angle_of_attack_grid, policy, timestep, sim_time):
    """Advance the full hybrid system: discrete step-to-step control (angle
    of attack chosen once per stance phase from the lookup table) until the
    state enters the ankle-balance RoA, then continuous ankle-torque
    control.

    Also tracks the stance foot's world position, advancing it by the
    footswitch geometry at every impact, so the walker can be drawn
    actually walking down the slope instead of pivoting in place.
    """
    n_steps = round(sim_time / timestep)
    time_traj = np.arange(n_steps + 1) * timestep
    state_traj = np.zeros((2, n_steps + 1))
    state_traj[:, 0] = initial_state
    stance_position_traj = np.zeros((2, n_steps + 1))
    stance_position = np.array([0.0, 0.0])

    local_params = dict(params)
    state = np.array(initial_state, dtype=float)
    completed_steps = 0
    balancing = roa_contains(theta_values, velocity_values, roa_grid, state)
    if not balancing:
        local_params["angle_of_attack"] = choose_from_policy(state[1], theta_dot_grid, angle_of_attack_grid, policy)

    for step in range(n_steps):
        t = time_traj[step]
        local_params["ankle_torque"] = compute_ankle_torque(state, local_params, theta_values, velocity_values, roa_grid)
        next_state = integrate(model.dynamics, t, state, local_params, timestep)

        if not balancing and model.event_guard(state, next_state, local_params):
            # Advance the stance foot using the pre-impact state (the exact
            # strike angle) and the angle of attack that produced it, before
            # event_dynamics() resets theta for the new stance leg.
            stance_position = compute_new_stance_position(next_state, stance_position, local_params)
            next_state = model.event_dynamics(next_state, local_params)
            completed_steps += 1
        elif not balancing and state[0] < 0.0 <= next_state[0]:
            # Crossed the Poincare section: choose the angle of attack for
            # the upcoming stance phase from the lookup table.
            local_params["angle_of_attack"] = choose_from_policy(
                next_state[1], theta_dot_grid, angle_of_attack_grid, policy
            )

        state = next_state
        state_traj[:, step + 1] = state
        stance_position_traj[:, step + 1] = stance_position

        if not balancing and roa_contains(theta_values, velocity_values, roa_grid, state):
            balancing = True

    return time_traj, state_traj, stance_position_traj, completed_steps


def draw_frame(index, state_traj, stance_position_traj, time_traj, params, ax, view_limits=None):
    # The stance foot has moved to wherever the last footswitch put it, so
    # the walker actually advances down the slope instead of resetting to
    # the origin every frame.
    model.visualize(
        state_traj[:, index],
        params,
        ax=ax,
        stance_position=stance_position_traj[:, index],
        view_limits=view_limits,
    )
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# ---------------------------------------------------------------------------
# Build the region of attraction and the step-to-step lookup table.
# ---------------------------------------------------------------------------
print("Determining the ankle-balance controller's region of attraction...")
theta_values = np.linspace(-0.35, 0.35, 41)
velocity_values = np.linspace(-0.5, 0.5, 41)
roa_grid = compute_roa_grid(theta_values, velocity_values, params, timestep=2e-3, sim_time=3.0, tolerance=0.02)
print(f"RoA grid: {roa_grid.sum()} / {roa_grid.size} states converge.")

print("\nChecking step-to-step lookup table resolution (theta_dot grid size):")
angle_of_attack_grid = np.linspace(min_angle_of_attack, max_angle_of_attack, 21)
step_timestep = 2e-4
# 21 -> 41 grid points already agree on the step count for theta_dot = 3.0,
# so 41 points (the finest candidate below) is fine enough; 11 is not.
theta_dot_grid, next_theta_dot = report_grid_resolution_check(
    candidate_sizes=(11, 21, 41),
    angle_of_attack_grid=angle_of_attack_grid,
    params=params,
    timestep=step_timestep,
    theta_values=theta_values,
    velocity_values=velocity_values,
    roa_grid=roa_grid,
    theta_dot_of_interest=3.0,
)
steps_to_stop, fastest_policy = compute_steps_to_standstill(
    theta_dot_grid, next_theta_dot, theta_values, velocity_values, roa_grid
)
max_steps_to_stop, slowest_policy = compute_max_steps_to_standstill(theta_dot_grid, next_theta_dot, steps_to_stop)

# ---------------------------------------------------------------------------
# Simulate the walker from an initial condition that needs several steps.
# ---------------------------------------------------------------------------
initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 6.0

time_traj, state_traj, stance_position_traj, completed_steps = simulate_walk(
    initial_state, params, theta_values, velocity_values, roa_grid,
    theta_dot_grid, angle_of_attack_grid, fastest_policy, timestep, sim_time,
)
_, slow_state_traj, slow_stance_position_traj, slow_completed_steps = simulate_walk(
    initial_state, params, theta_values, velocity_values, roa_grid,
    theta_dot_grid, angle_of_attack_grid, slowest_policy, timestep, sim_time,
)

nearest_initial = np.argmin(np.abs(theta_dot_grid - initial_state[1]))
print(f"\nFrom theta_dot = {initial_state[1]} rad/s "
      f"(lookup predicts {steps_to_stop[nearest_initial]:.0f}-{max_steps_to_stop[nearest_initial]:.0f} footsteps):")
print(f"  fastest policy reaches the RoA in {completed_steps} footsteps.")
print(f"  longest-walk policy reaches the RoA in {slow_completed_steps} footsteps.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

fig_roa, ax_roa = plt.subplots(figsize=(7, 5), layout="constrained")
ax_roa.pcolormesh(theta_values, velocity_values, roa_grid, shading="auto")
ax_roa.set_xlabel(r"$\theta$ (rad)")
ax_roa.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax_roa.set_title("Ankle-balance controller: region of attraction")
fig_roa.savefig(output / "roa.png", dpi=150)
plt.savefig("roa.png", dpi=300, bbox_inches="tight")
fig_steps, ax_steps = plt.subplots(figsize=(7, 5), layout="constrained")
finite = np.isfinite(steps_to_stop)
ax_steps.plot(theta_dot_grid[finite], steps_to_stop[finite], "o-", label="reaches RoA")
ax_steps.plot(
    theta_dot_grid[~finite],
    np.zeros(np.sum(~finite)),
    "rx",
    label="never reaches RoA on this grid",
)
ax_steps.set_xlabel(r"$\dot{\theta}_k$ at $\theta = 0$ (rad/s)")
ax_steps.set_ylabel("footsteps to standstill")
ax_steps.set_title("Steps to standstill vs. initial condition")
ax_steps.legend()
fig_steps.savefig(output / "steps_to_standstill.png", dpi=150)
plt.savefig("steps_to_standstill.png", dpi=300, bbox_inches="tight")
fig_state, ax_state = plt.subplots(figsize=(7, 5), layout="constrained")
ax_state.plot(state_traj[0], state_traj[1], label=f"fastest ({completed_steps} steps)")
ax_state.plot(
    slow_state_traj[0],
    slow_state_traj[1],
    "--",
    label=f"slowest ({slow_completed_steps} steps)",
)
ax_state.scatter(state_traj[0, 0], state_traj[1, 0], color="k", zorder=5, label="Initial state")
ax_state.set_xlabel(r"$\theta$ (rad)")
ax_state.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax_state.set_title("State-Space Trajectory")
ax_state.legend()
fig_state.savefig(output / "state_space_trajectory.png", dpi=150)
plt.savefig("state_space_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

# Fix the camera to the whole walk instead of the default per-frame
# "recenter on the current stance foot" view: that default follow-cam
# would keep every frame looking visually identical (foot always in the
# same place on screen), which is what made the walker look like it was
# resetting in place. With a single wide, static window sized to the
# full excursion of the stance foot, the walker visibly travels across
# the frame as it steps down the slope.
margin = 2.5 * params["length"]
x_min = stance_position_traj[0, : frame_indices[-1] + 1].min() - margin
x_max = stance_position_traj[0, : frame_indices[-1] + 1].max() + margin
y_min = stance_position_traj[1, : frame_indices[-1] + 1].min() - margin
y_max = stance_position_traj[1, : frame_indices[-1] + 1].max() + margin
view_limits = (x_min, x_max, y_min, y_max)

animation = FuncAnimation(
    fig,
    lambda index: draw_frame(index, state_traj, stance_position_traj, time_traj, params, ax, view_limits),
    frames=frame_indices,
    interval=1000 / fps,
    repeat=False,
)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
print(f"\nSaved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()