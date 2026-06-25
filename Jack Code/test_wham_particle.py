import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Particle
from WHAMField import WHAMField

B = WHAMField()

p = Particle.particle(
    init_position=[0.05, 0, 0],
    init_velocity=[0.0, 0.1, 1.0],
    dt=1e-4,
    write_data=False,
    silent=True
)

p.set_boundaries(radius_max=2.0, z_max=2.5, z_min=-2.5)
p.step(B.field, 20000)

r = p.get_r()
v = p.get_v()

KE = 0.5 * np.sum(v**2, axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(r[:, 0], r[:, 1], r[:, 2])
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Test particle in WHAM field")
plt.savefig("wham_particle_orbit.png", dpi=300, bbox_inches="tight")

plt.figure()
plt.plot(KE)
plt.xlabel("Step")
plt.ylabel("Kinetic energy")
plt.title("WHAM energy conservation check")
plt.savefig("wham_energy_check.png", dpi=300, bbox_inches="tight")

print("Iterations:", p.iter)
print("Escaped:", p.outOfBounds)
print("Initial position:", r[0])
print("Final position:", r[-1])
print("Initial KE:", KE[0])
print("Final KE:", KE[-1])
print("Relative KE change:", (KE[-1] - KE[0]) / KE[0])
print("Saved wham_particle_orbit.png and wham_energy_check.png")