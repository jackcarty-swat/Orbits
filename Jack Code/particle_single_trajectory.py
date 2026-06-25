import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Particle
from WHAMField import WHAMField
from Maxwellian import generate_maxwellian

# ----------------------------
# Simulation parameters
# ----------------------------

B = WHAMField()

q = 1.0
m = 1.0

B0 = 1.0
T = 1.0
scale = 1.0

dt = 0.1/(2*np.pi)
nsteps = 200000

radius_max = 2.0*(scale/0.000102)*np.sqrt(m*T)/(q*B0)
z_max      = 2.5*(scale/0.000102)*np.sqrt(m*T)/(q*B0)
z_min      =-2.5*(scale/0.000102)*np.sqrt(m*T)/(q*B0)

# ----------------------------
# Generate one Maxwellian particle
# ----------------------------

velocity = generate_maxwellian(
    n_particles=1,
    vth=1.0,
    seed=42
)[0]

print("Initial velocity:", velocity)

p = Particle.particle(
    init_position=[0.0,0.0,0.0],
    init_velocity=velocity,
    dt=dt,
    q=q,
    m=m,
    write_data=False,
    silent=True
)

p.set_boundaries(
    radius_max=radius_max,
    z_max=z_max,
    z_min=z_min
)

print("Running particle...")

p.step(B.field, nsteps)

r = p.get_r()

print("Escaped:", p.outOfBounds)
print("Final position:", r[-1])

# ----------------------------
# 3D trajectory
# ----------------------------

skip = 20

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection="3d")

ax.plot(
    r[::skip,0],
    r[::skip,1],
    r[::skip,2],
    linewidth=1.0
)

# Mark start and end
ax.scatter(
    r[0,0], r[0,1], r[0,2],
    color="green",
    s=40,
    label="Start"
)

ax.scatter(
    r[-1,0], r[-1,1], r[-1,2],
    color="red",
    s=40,
    label="End"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title("Single Particle Trajectory")

ax.legend()
ax.view_init(elev=20, azim=35)

plt.tight_layout()
plt.savefig("single_particle_trajectory.png", dpi=300)

print("Saved single_particle_trajectory.png")