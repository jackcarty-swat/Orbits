import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Particle
from WHAMField import WHAMField

B = WHAMField()

q = 1.0
m = 1.0
B0 = 1.0
T = 1.0
scale = 1.0

dt = 0.5 / (2 * np.pi)
nsteps = 10000

radius_max = 2.0 * (scale / 0.000102) * np.sqrt(m*T) / (q*B0)
z_max = 2.5 * (scale / 0.000102) * np.sqrt(m*T) / (q*B0)
z_min = -2.5 * (scale / 0.000102) * np.sqrt(m*T) / (q*B0)

z_start_m = 0.70
init_z = z_start_m / 0.000102

vperp_values = np.linspace(0.01, 5.0, 100)
vpar_values = np.linspace(9.0, 110.0, 100)


turned_map = np.zeros((len(vperp_values), len(vpar_values)))
max_z_map = np.zeros_like(turned_map)
final_z_map = np.zeros_like(turned_map)

for i, vperp in enumerate(vperp_values):
    for j, vpar in enumerate(vpar_values):

        p = Particle.particle(
            init_position=[0.0, 0.0, init_z],
            init_velocity=[vperp, 0.0, vpar],
            dt=dt,
            q=q,
            m=m,
            write_data=False,
            silent=True
        )

        p.set_boundaries(radius_max=radius_max, z_max=z_max, z_min=z_min)
        t0 = time.perf_counter()
        p.step(B.field, nsteps)
        t1 = time.perf_counter()

        print(f"runtime = {t1-t0:.2f} s, steps/s = {nsteps/(t1-t0):.0f}")

        r = p.get_r()
        z = r[:, 2]
        radius = np.sqrt(r[:, 0]**2 + r[:, 1]**2)

        max_z = z.max()
        final_z = z[-1]
        max_r = radius.max()

        turned = (not p.outOfBounds) and (final_z < 0.98 * max_z)

        turned_map[i, j] = 1 if turned else 0
        max_z_map[i, j] = max_z
        final_z_map[i, j] = final_z

        print(
            f"vperp={vperp:.3f}, vpar={vpar:.3f} | "
            f"escaped={p.outOfBounds} | "
            f"max_z={max_z:.2f} | final_z={final_z:.2f} | "
            f"max_r={max_r:.2f} | turned={turned}"
        )

plt.figure(figsize=(7, 5))
plt.imshow(
    turned_map,
    origin="lower",
    aspect="auto",
    extent=[
        vpar_values.min(), vpar_values.max(),
        vperp_values.min(), vperp_values.max()
    ]
)
plt.colorbar(label="Turned around? 1=yes, 0=no")
plt.xlabel(r"$v_\parallel$")
plt.ylabel(r"$v_\perp$")
plt.title("Turnaround Map")
plt.savefig("turnaround_map.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(7, 5))
plt.imshow(
    max_z_map,
    origin="lower",
    aspect="auto",
    extent=[
        vpar_values.min(), vpar_values.max(),
        vperp_values.min(), vperp_values.max()
    ]
)
plt.colorbar(label="Maximum z")
plt.xlabel(r"$v_\parallel$")
plt.ylabel(r"$v_\perp$")
plt.title("Maximum z in Velocity Space")
plt.savefig("max_z_map.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(7, 5))
plt.imshow(
    final_z_map,
    origin="lower",
    aspect="auto",
    extent=[
        vpar_values.min(), vpar_values.max(),
        vperp_values.min(), vperp_values.max()
    ]
)
plt.colorbar(label="Final z")
plt.xlabel(r"$v_\parallel$")
plt.ylabel(r"$v_\perp$")
plt.title("Final z in Velocity Space")
plt.savefig("final_z_map.png", dpi=300, bbox_inches="tight")

print("\nSaved:")
print("  turnaround_map.png")
print("  max_z_map.png")
print("  final_z_map.png")