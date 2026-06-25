import numpy as np
from numba import njit
import pandas as pd

@njit
def ScalarBorisPush(state, B, dt):
    x = state[:3]
    v = state[3:6]
    t =0.5*(B*dt) * state[6] # half of the magnetic rotation during time step
    t_mag2 =np.dot(t,t) # magnitude squared of t vector
    s = (2*t)/(1+t_mag2) # converts t into full corrected rotation
    v_minus=v
    v_prime= v_minus + np.cross(v_minus,t) # rotate velocity
    v_plus=v_minus + np.cross(v_prime,s) # final rotated velocity
    v_new=v_plus
    x_new=x + v_new*dt 
    return np.concatenate((x_new, v_new, np.array([state[6]]), np.array([state[7] + dt])))


@njit
def BorisPush(state, B, dt):
    x = state[:, :3]
    v = state[:, 3:6]
    qm = state[:, 6]

    t = 0.5 * dt * B * qm[:, np.newaxis]        # shape (N, 3)
    t_mag2 = np.sum(t * t, axis=1)              # shape (N,), per-particle |t|^2
    s = (2 * t) / (1 + t_mag2)[:, np.newaxis]   # shape (N, 3)

    v_minus = v
    v_prime = v_minus + np.cross(v_minus, t)    # cross along last axis, per row
    v_plus = v_minus + np.cross(v_prime, s)
    v_new = v_plus

    x_new = x + v_new * dt

    return np.column_stack((x_new, v_new, qm, state[:, 7] + dt))


def BorisIterator(N, dt, df, B_func, B_0=1, keepSteps=False):
    if keepSteps:
        steps = []
    for i, p in df.iterrows():
        state = np.array(p)
        if keepSteps:
            temp = []
        c = 0
        while c < N:
            c += 1
            B = B_func(*state[:3], B_0)
            state = ScalarBorisPush(state, B, dt)
            if keepSteps:
                temp.append(state)
        df.iloc[i] = p
        if keepSteps:
            tempDf = pd.DataFrame(temp, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
            steps.append(tempDf)
    if keepSteps:
        return df, steps
    else:
        return df


def BetterBorisIterator(N, dt, df, B_func, B_0, nkeep=0, fieldFromFile=False, FileInit=None, fnameR="", fnameZ=""):
    state = np.array(df)
    if nkeep != 0:
        steps = [state[:nkeep]]
    if fieldFromFile:
        tree, Br, Bz = FileInit(fnameR, fnameZ)
    c = 0
    while c < N:
        c += 1
        if not fieldFromFile:
            B = B_func(*state[:,:3].T, B_0)
        else:
            B = B_func(*state[:,:3].T, tree, Br, Bz)
        state = BorisPush(state, B, dt)
        if nkeep != 0:
            steps.append(state[:nkeep])
    df = pd.DataFrame(state, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
    if nkeep != 0:
        return df, np.array(steps)
    else:
        return df