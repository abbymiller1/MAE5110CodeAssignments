import numpy as np
import matplotlib.pyplot as plt

from models import bouncing_ball as model

from integrators import rk4 as integrator

# Basic simulation of the bouncing ball

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "mass": 0.2,  # point mass at end of rod (kg)
    "restitution_coeff": 1.0,  # coefficient of restitution
}


# some set-up
initial_state = np.array([5.0, 0.0]) #start at 5 meters above the ground with zero initial velocity

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(model.dynamics, t, state_traj[:, step], timestep, params)

    if state_traj[0, step + 1] <= 0: #if the position is less than or equal to zero, we have hit the ground
        state_traj[1, step + 1] = -params["restitution_coeff"] * state_traj[1, step + 1]
        state_traj[0, step + 1] = 0

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

potential_energy, kinetic_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Ball energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
