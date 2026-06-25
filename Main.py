from Github.Orbits import Initializers
from Github.Orbits import Fields
from Github.Orbits import Visualizers
from Github.Orbits import Workers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



initial = Initializers.InitialFromBody(1, [[180,200],[180,200],[-10,10]], 1, 1, 0, 1)
#initial = pd.DataFrame(np.array([[1000, 1000], [1000, 1000], [10, 10], [10, -10], [0, 0], [-1, -1], [1, 1], [0, 0]]).T, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
print("Initialized")
fR = r"C:\Users\Student\Desktop\Particle Orbits\6kA_17500A\Field_WHAM_R_Br_6kA_17500A.txt"
fZ = r"C:\Users\Student\Desktop\Particle Orbits\6kA_17500A\Field_WHAM_R_Bz_6kA_17500A.txt"
output, steps = Workers.BetterBorisIterator(2*3.14*10, 1, initial, Fields.FieldFromFile, 6.5, 1, True, Fields.FileInitializer, fR, fZ)
print("Computed")
Visualizers.VisualizeTrajectories(steps, 1)

