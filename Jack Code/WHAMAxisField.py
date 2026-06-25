import numpy as np
from scipy.interpolate import interp1d

class WHAMAxisField:
    def __init__(self, filename="./data/Field_WHAM_R_Baxial_6kA_17500A.txt"):
        data = np.loadtxt(filename, comments="%")
        z = data[:, 0]
        B = data[:, 1]

        self.B_interp = interp1d(
            z, B,
            bounds_error=False,
            fill_value=0.0
        )

    def field(self, pos, t=0):
        x, y, z = pos
        Bz = float(self.B_interp(z))
        return np.array([0.0, 0.0, Bz])