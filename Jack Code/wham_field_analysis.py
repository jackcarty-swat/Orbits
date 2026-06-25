import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt(
    "./data/Field_WHAM_R_Baxial_6kA_17500A.txt",
    comments="%"
)

z = data[:, 0]
B = data[:, 1]

plt.figure()
plt.plot(z, B)
plt.xlabel("z (m)")
plt.ylabel("|B| (T)")
plt.title("WHAM on-axis magnetic field")
plt.grid(True)
plt.savefig("wham_B_on_axis.png", dpi=300, bbox_inches="tight")

print("z range:", z.min(), z.max())
print("B range:", B.min(), B.max())
print("Saved wham_B_on_axis.png")