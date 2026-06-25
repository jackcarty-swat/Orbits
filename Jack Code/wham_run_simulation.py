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
nsteps = 2000

n_particles = 100
vth = 1.0

# ----------------------------
# Simulation boundaries
# ----------------------------

radius_max = 2.0*(scale/0.000102)*np.sqrt(m*T)/(q*B0)
z_max      = 2.5*(scale/0.000102)*np.sqrt(m*T)/(q*B0)
z_min      =-2.5*(scale/0.000102)*np.sqrt(m*T)/(q*B0)

# ----------------------------
# Initial conditions
# ----------------------------

positions = np.zeros((n_particles,3))

velocities = generate_maxwellian(
    n_particles=n_particles,
    vth=vth,
    seed=42
)

print("Velocities generated.")

# ----------------------------
# Storage arrays
# ----------------------------

escaped = np.zeros(n_particles, dtype=bool)
max_z = np.zeros(n_particles)
max_r = np.zeros(n_particles)
final_z = np.zeros(n_particles)

# ----------------------------
# Run particles
# ----------------------------

for i in range(n_particles):

    p = Particle.particle(
        init_position=positions[i],
        init_velocity=velocities[i],
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

    p.step(B.field, nsteps)

    r = p.get_r()

    radius = np.sqrt(r[:,0]**2 + r[:,1]**2)

    escaped[i] = p.outOfBounds
    max_z[i] = np.max(r[:,2])
    max_r[i] = np.max(radius)
    final_z[i] = r[-1,2]

    print(f"Particle {i} completed.")

# ----------------------------
# Summary
# ----------------------------

print(f"Particles run: {n_particles}")
print(f"Escaped: {escaped.sum()}")
print(f"Confined: {n_particles-escaped.sum()}")

# ----------------------------
# Diagnostic plots
# ----------------------------

# Final z position
plt.figure(figsize=(7,5))

plt.hist(final_z, bins=50)

plt.xlabel("Final z Position")
plt.ylabel("Number of Particles")
plt.title("Distribution of Final z Positions")

plt.grid(True)

plt.savefig("final_z_distribution.png", dpi=300)


# Maximum radius reached
plt.figure(figsize=(7,5))

plt.hist(max_r, bins=50)

plt.xlabel("Maximum Radius")
plt.ylabel("Number of Particles")
plt.title("Distribution of Maximum Radius")

plt.grid(True)

plt.savefig("max_r_distribution.png", dpi=300)


# Initial parallel velocity vs final z
plt.figure(figsize=(7,5))

plt.scatter(
    velocities[:,2],
    final_z,
    s=8,
    alpha=0.6
)

plt.xlabel(r"Initial $v_{\parallel}$")
plt.ylabel("Final z Position")

plt.title("Final Position vs Initial Parallel Velocity")

plt.grid(True)

plt.savefig("vpar_vs_final_z.png", dpi=300)


print("Saved:")
print("  final_z_distribution.png")
print("  max_r_distribution.png")
print("  vpar_vs_final_z.png")