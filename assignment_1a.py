import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model

from integrators import rk4 as integrator

# Basic simulation of the rimless wheel

params = model.generate_params()

# some set-up: start near the reset angle (slope - alpha), rolling downhill
alpha = np.pi / params["num_spokes"]
initial_state = np.array([params["slope"] - alpha + 0.05, 1])

timestep = 1e-3
sim_time = 8.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

impact_times = []

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    current_state = state_traj[:, step]
    next_state = integrator(model.dynamics, t, current_state, timestep, params)

    guard_before = model.detect_impact(t, current_state, params)
    guard_after = model.detect_impact(t + timestep, next_state, params)
    #check if we are about to hit the ground in the next time step
    
    if guard_before < 0 and guard_after >= 0:
        next_state = model.reset_state(next_state, params)
        impact_times.append(time_traj[step + 1])

    state_traj[:, step + 1] = next_state

#check the energies
#it should be preserved through each swing then drop at every collision

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = kinetic_energy + potential_energy

print(f"Number of impacts: {len(impact_times)}")
print(f"Energy at t=0: {total_energy[0]:.4f} J")
print(f"Energy at t=end: {total_energy[-1]:.4f} J")

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, total_energy, label="Total energy")

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Rimless wheel energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot