from WHAMField import WHAMField
import numpy as np

B = WHAMField()

for pos in [
    [0, 0, 0],
    [0, 0, 0.9],
    [0.1, 0, 0],
    [0.1, 0, 0.9],
]:
    Bvec = B.field(pos, 0)
    print(pos, Bvec, "|B| =", np.linalg.norm(Bvec))