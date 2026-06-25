import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

br_data = np.loadtxt("./data/Field_WHAM_R_Br_6kA_17500A.txt", comments="%")
bz_data = np.loadtxt("./data/Field_WHAM_R_Bz_6kA_17500A.txt", comments="%")

br_points = br_data[:, 0:2]   # r, z
bz_points = bz_data[:, 0:2]   # r, z

Br_values = br_data[:, 2]
Bz_values = bz_data[:, 2]

print("Br points:", br_points.shape, "Br values:", Br_values.shape)
print("Bz points:", bz_points.shape, "Bz values:", Bz_values.shape)

Br_interp = LinearNDInterpolator(br_points, Br_values)
Bz_interp = LinearNDInterpolator(bz_points, Bz_values)

r_grid = np.linspace(0, 2.0, 300)
z_grid = np.linspace(-2.5, 2.5, 500)
Z, R = np.meshgrid(z_grid, r_grid)

Bz_plot = Bz_interp(R, Z)

plt.figure()
plt.pcolormesh(Z, R, Bz_plot, shading="auto")
plt.colorbar(label="Bz (T)")
plt.xlabel("z (m)")
plt.ylabel("r (m)")
plt.title("WHAM Bz(r,z)")
plt.savefig("wham_Bz_heatmap.png", dpi=300, bbox_inches="tight")

print("Saved wham_Bz_heatmap.png")

for r, z in [(0, 0), (0, 0.9), (0.1, 0), (0.5, 0)]:
    print(f"r={r}, z={z}: Br={Br_interp(r,z)}, Bz={Bz_interp(r,z)}")

print("Center:")
print("Br =", Br_interp(0,0))
print("Bz =", Bz_interp(0,0))

print("Mirror throat:")
print("Br =", Br_interp(0,0.9))
print("Bz =", Bz_interp(0,0.9))