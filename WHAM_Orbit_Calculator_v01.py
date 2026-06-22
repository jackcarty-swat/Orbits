import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import maxwell
from scipy.stats import uniform
from scipy.stats import uniform_direction
from scipy.integrate import RK45
from numba import njit
import time

# Defining parameters of the magnetic field
B_0 = 6.5
Z_m = 1
gamma = 0.125

@njit
def CartesianToCylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z

@njit
def CylindricalToCartesian(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

@njit
def RadialMagneticField(R, theta, Z):
    # Calculates the radial magnetic field at a given position in cylindrical coordinates
    B = R*B_0/(np.pi*gamma**3) * ((Z-Z_m)/((1+((Z-Z_m)/gamma)**2)**2) + (Z+Z_m)/((1+((Z+Z_m)/gamma)**2)**2))
    return B

@njit
def AxialMagneticField(R, theta, Z):
    # Calculates the axial magnetic field at a given position in cylindrical coordinates
    B = B_0/(np.pi*gamma) * (1/(1+((Z-Z_m)/gamma)**2) + 1/(1+((Z+Z_m)/gamma)**2))
    return B

@njit
def CartesianMagneticField(x, y, z):
    """
    Calculates the magnetic field of the mirrors in cartesian coordinates. First, converts cartesian coordinates
    to cylindrical coordinates. Then, finds magnetic field in cylindrical coordinates. Finally, converts magnetic
    field back to cartesian coordinates.
    input:
        x, y, z: integers representing the position at which to calculate the magnetic field.
    output:
        B_x, B_y, B_z: integers representing the strength of the magnetic field in each direction.
    """
    r, theta, z = CartesianToCylindrical(x, y, z)
    B_r = RadialMagneticField(r, theta, z)
    B_z = AxialMagneticField(r, theta, z)
    B_x, B_y, B_z = CylindricalToCartesian(B_r, theta, B_z)
    return np.stack((B_x, B_y, B_z), axis=-1)

@njit
def ScalarCartesianMagneticField(x, y, z):
    """
    Calculates the magnetic field of the mirrors in cartesian coordinates. First, converts cartesian coordinates
    to cylindrical coordinates. Then, finds magnetic field in cylindrical coordinates. Finally, converts magnetic
    field back to cartesian coordinates.
    input:
        x, y, z: integers representing the position at which to calculate the magnetic field.
    output:
        B_x, B_y, B_z: integers representing the strength of the magnetic field in each direction.
    """
    r, theta, z = CartesianToCylindrical(x, y, z)
    B_r = RadialMagneticField(r, theta, z)
    B_z = AxialMagneticField(r, theta, z)
    B_x, B_y, B_z = CylindricalToCartesian(B_r, theta, B_z)
    return np.array((B_x, B_y, B_z))


def MakeInitialMaxwellian(N, X, q, T_perp, T_para, m, t_0=0, X_range=[0,0,0]):
    """
    Creates an array of initial values for the particles.
    Input:
        N: int, number of particles.
        X: array of shape (3), starting location of the particles in cylindrical coordinates.
        q: int, charge of the particles (Coulombs)
        T_perp: int, temperature of the particles in the direction perpendicular to the axis (eV)
        T_para: int, temperature of the particles in the direction parallel to the axis (eV)
        m: int, mass of the particles
        t_0: int, optional, time at which particles are in given position.
        X_range: array of shpae (3), optional, range of variation of inital x position in cyliindrical coordinates. 
            If given, creates a uniform distribution of particles with set range and center specified by X.
    
    Output:
        Tuple with 4 columns, N rows, where each row represents a particle. The first column gives the particle's
        position, the second gives its velocity, the third its charge-mass ratio, and the fourth gives time.
    """
    scaleParamPerp = 4*10**(-10) * np.sqrt(T_perp/m)
    scaleParamPara = 4*10**(-10) * np.sqrt(T_para/m)
    v_perp = maxwell.rvs(loc=0, scale=scaleParamPerp, size=N)
    v_para = maxwell.rvs(loc=0, scale=scaleParamPara, size=N)
    V = np.array([v_perp, np.zeros(N), v_para]).T
    
    x_rad = uniform.rvs(loc=(X[0]-X_range[0]/2), scale=(X_range[0]), size=N)
    x_theta = uniform.rvs(loc=(X[1]-X_range[1]/2), scale=(X_range[1]), size=N)
    x_ax = uniform.rvs(loc=(X[2]-X_range[2]/2), scale=(X_range[2]), size=N)
    X = np.array([x_rad, x_theta, x_ax]).T
    
    qm = np.full(N, q / m)
    t = np.full(N, t_0)
    
    df = pd.DataFrame({'position': X, 'velocity': V, 'qom': qm, 'time': t})
    
    return df

def InitialFromSource(N, X, V, q, m, t, T, xRange=[0,0,0]):
    """
    Creates the initial states of N particles originating from a point source or a source with small volume,
    where the direction of the particles' initial velocity is defined.
    Input: 
        N: int, number of particles.
        X: array of shape (3), starting location of the particles in cartesian coordinates.
        V: array of shpae (3), unit vector in the direction of the particles' initial velocity.
        q: int, charge of the particles (Coulombs)
        m: int, mass of the particles (kilograms)
        t: int, time at which particles are in given position.
        T: int, temperature of the particles (eV)
        X_range: array of shpae (3), optional, range of variation of inital x position in cyliindrical coordinates. 
            If given, creates a uniform distribution of particles with set range and center specified by X.
    
    Output:
        df: pandas dataframe with 9 columns and N rows. Each row represents the properties of a single particle.
            Columns 1-3 give the position of the particle in cartesian coordinates. Columns 4-6 give its velocity
            in cartesian coordinates. Column 7 gives the charge/mass ratio, column 8 gives time, and column 9
            is a boolean representing if the particle is confined between the magnetic mirrors.
    """
    scaleParam = 4*10**(-10) * np.sqrt(T/m)
    vMag = maxwell.rvs(loc=0, scale=scaleParam, size=N)
    
    x = uniform.rvs(loc=(X[0] - xRange[0]/2), scale=xRange[0], size=N)
    y = uniform.rvs(loc=(X[1] - xRange[1]/2), scale=xRange[1], size=N)
    z = uniform.rvs(loc=(X[2] - xRange[2]/2), scale=xRange[2], size=N)
    df = pd.DataFrame({
        'x': x, 'y': y, 'z': z, 
        'vX': (V[0] * vMag), 'vY': (V[1] * vMag), 'vZ': (V[2] * vMag),
        'qom': np.full(N, (q/m)), 't': np.full(N, t, dtype=float)
        })
    
    return df

def InitialFromBody(N, xRange, q, m, t, T, T_perp=-1):
    """
    Creates the initial states of N particles in a defined area, where the direction of the particles' velocities
    are not known.
    Input: 
        N: int, number of particles.
        xRange: array of shpae (3,2), the bounds of the region where the particles begin, in cartesian coordinates.
        q: int, charge of the particles (Coulombs)
        m: int, mass of the particles (kilograms)
        t: int, time at which particles are in given position.
        T: int, temperature of the particles (eV)
        T_perp: int, optional, temperature of the particles in the direction perpendicular to the axis (eV).
            If given, T should be the temperature of the particles in the direction parallel to the axis.
    
    Output:
        df: pandas dataframe with 9 columns and N rows. Each row represents the properties of a single particle.
            Columns 1-3 give the position of the particle in cartesian coordinates. Columns 4-6 give its velocity
            in cartesian coordinates. Column 7 gives the charge/mass ratio, column 8 gives time, and column 9
            is a boolean representing if the particle is confined between the magnetic mirrors.
    """
    if T_perp == -1:
        scaleParam = 4*10**(-10) * np.sqrt(T/m)
        vMag = maxwell.rvs(loc=0, scale=scaleParam, size=N)
        V = uniform_direction.rvs(dim=3, size=N)
        vX = V[:, 0] * vMag
        vY = V[:, 1] * vMag
        vZ = V[:, 2] * vMag
    else:
        scaleParamPerp = 4*10**(-10) * np.sqrt(T_perp/m)
        scaleParamPara = 4*10**(-10) * np.sqrt(T/m)
        vMagPerp = maxwell.rvs(loc=0, scale=scaleParamPerp, size=N)
        V = uniform_direction.rvs(dim=2, size=N)
        vX = V[:, 0] * vMagPerp
        vY = V[:, 1] * vMagPerp
        vZ = maxwell.rvs(loc=0, scale=scaleParamPara, size=N)
    
    x = uniform.rvs(loc=xRange[0][0], scale=(xRange[0][1] - xRange[0][0]), size=N)
    y = uniform.rvs(loc=xRange[1][0], scale=(xRange[1][1] - xRange[1][0]), size=N)
    z = uniform.rvs(loc=xRange[2][0], scale=(xRange[2][1] - xRange[2][0]), size=N)
    
    df = pd.DataFrame({
        'x': x, 'y': y, 'z': z,
        'vX': vX, 'vY': vY, 'vZ': vZ,
        'qom': np.full(N, (q/m)), 't': np.full(N, t, dtype=float)
        })
    
    return df

# using Boris method for updating position and velocity after time step dt
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
    


def BorisIterator(N, dt, df, B_func, keepSteps=False):
    if keepSteps:
        steps = []
    for i, p in df.iterrows():
        state = np.array(p)
        if keepSteps:
            temp = []
        c = 0
        while c < N and np.abs(state[2]) < Z_m:
            c += 1
            B = ScalarCartesianMagneticField(*state[:3])
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


def BetterBorisIterator(N, dt, df, B_func, keepSteps=False):
    state = np.array(df)
    if keepSteps:
        steps = [state]
    c = 0
    while c < N:
        c += 1
        B = CartesianMagneticField(*state[:,:3].T)
        state = BorisPush(state, B, dt)
        if keepSteps:
            steps.append(state)
    df = pd.DataFrame(state, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
    if keepSteps:
        return df, np.array(steps)
    else:
        return df


def ScipyFunc(t, state):
    x = state[:3]
    v = state[3:6]
    B = CartesianMagneticField(*x)
    dv = state[6] * np.cross(v, B)
    dx = v
    return np.concatenate([dx, dv, [state[6]]])

def ScipyIterator(N, dt, df, B_func, keepSteps=False):
    if keepSteps:
        steps = []
    for i, p in df.iterrows():
        if keepSteps:
            tempY = []
            tempT = []
        solver = RK45(ScipyFunc, p['t'], np.array(p[['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom']]), (N*dt), max_step=dt)
        #while solver.status == 'running' and np.abs(solver.y[2]) < Z_m:
        C = 0
        while C < N and np.abs(solver.y[2]) < Z_m:
            C += 1
            solver.step()
            if keepSteps:
                tempY.append(solver.y)
                tempT.append(solver.t)
        p[['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom']] = solver.y
        df.iloc[i] = p
        if keepSteps:
            tempDf = pd.DataFrame(np.column_stack([tempY, tempT]), columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
            steps.append(tempDf)
    if keepSteps:
        return df, steps
    else:
        return df
        
def Show3d(x, y, z):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    # 3. Plot the continuous trajectory line
    ax.plot(x, y, z, label='3D Trajectory', color='blue', linewidth=2)

    # 4. Highlight the start and end points
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')

    # 5. Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Trajectory Map')
    ax.legend()

    # 6. Display the plot
    plt.show()

def VisualizeTrajectories(steps, N):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    for i in range(N):
        if np.abs(steps[:,i,2,][-1]) <= Z_m:
            try:
                ax.plot(*steps[:,i,:3].T, label=f'Particle {i}', linewidth=1)
            except:
                print("Failed to display particle ", i)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Trajectory Map')
    plt.savefig("Particle_trajectories.png")
    plt.show()
    
#    for i in range(N):
#        if np.abs(steps[:,i,2,][-1]) <= Z_m:
#            try:
#                plt.plot(steps[:,i,7], steps[:,i,2])
#            except:
#                print("Failed to display particle ", i)
#    plt.title("Z vs Time")
#    plt.xlabel("Time")
#    plt.ylabel("Z position")
#    plt.show()
    
def GetStatistics(data):
    numLost = len(data[np.abs(data['z'])>Z_m])
    numTot = len(data)
    return numLost / numTot



"""
print("Testing scipy energy conservation")
particles = pd.DataFrame(np.array([[0.1], [0.1], [0], [-419000], [0], [1000000], [-1.76*10**11], [0]]).T, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])

startingE = 0.5 * 9.1*10**(-31) * (particles.loc[0, 'vX']**2 + particles.loc[0, 'vY']**2 + particles.loc[0, 'vZ']**2)

particles, steps = ScipyIterator(500000, 10**(-12), particles, CartesianMagneticField, True)

Show3d(steps)
print(steps[0][['x', 'y', 'z', 'vX', 'vY', 'vZ']])
finalE = 0.5 * 9.1*10**(-31) * (particles.loc[0, 'vX']**2 + particles.loc[0, 'vY']**2 + particles.loc[0, 'vZ']**2)
print('Starting energy: ', startingE)
print("Final energy: ", finalE)


print("Old Boris:")
particles = pd.DataFrame(np.array([[0.1, -0.1], [0.1, -0.1], [0, 0], [-419000, 419000], [0, 0], [5000000, 5000000], [-1.76*10**11, -1.76*10**11], [0, 0]]).T, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])

particles, steps = BorisIterator(1000000, 10**(-11), particles, CartesianMagneticField, True)
Show3d(steps[0]['x'].to_numpy(), steps[0]['y'].to_numpy(), steps[0]['z'].to_numpy())
Show3d(steps[1]['x'].to_numpy(), steps[1]['y'].to_numpy(), steps[1]['z'].to_numpy())

plt.plot(steps[0]['z'], steps[0]['t'])
plt.plot(steps[1]['z'], steps[1]['t'])
plt.show()
"""

print("New Boris:")

#initial = pd.DataFrame(np.array([[0.1, -0.1], [0.1, -0.1], [0, 0], [-419000, 419000], [0, 0], [10000000, 10000000], [-1.76*10**11, -1.76*10**11], [0, 0]]).T, columns=['x', 'y', 'z', 'vX', 'vY', 'vZ', 'qom', 't'])
#initial = InitialFromSource(50, [0,0,0], [0,0,1], 1.6*10**(-19), 9.1*10**(-31), 0, 1)
t1 = time.perf_counter()
initial = InitialFromBody(5000, [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]], 1.6*10**(-19), 1.67*10**(-27), 0, 5000)
t2 = time.perf_counter()
output = BetterBorisIterator(2000000, 10**(-11), initial, CartesianMagneticField, False)
t3 = time.perf_counter()
#VisualizeTrajectories(steps, 10)
t4 = time.perf_counter()
pLost = GetStatistics(output)
t5 = time.perf_counter()


t1 = t2 - t1
t2 = t3 - t2
t3 = t4 - t3
t4 = t5 - t4

text = f"Time 1: {t1}. Time 2: {t2}. Time 3: {t3}. Time 4: {t4}\nPercentage lost: {pLost}"

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(text)

def VisualizeMagneticField(bFunc, cuttingAxis, cutval, ax1Range, ax2Range):
    n1 = int(100*(ax1Range[1] - ax1Range[0])+1)
    n2 = int(100*(ax2Range[1] - ax2Range[0])+1)
    ax1 = np.linspace(*ax1Range, num=n1)
    ax2 = np.linspace(*ax2Range, num=n2)
    b1 = np.zeros((n1, n2))
    b2 = np.zeros((n1, n2))
    b3 = np.zeros((n1, n2))
    
    if cuttingAxis == 'x':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = ScalarCartesianMagneticField(cutval, a1, a2)
                b1[i, j] = temp[0]
                b2[i, j] = temp[1]
                b3[i, j] = temp[2]
    elif cuttingAxis == 'y':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = ScalarCartesianMagneticField(a1, cutval, a2)
                b1[i, j] = temp[0]
                b2[i, j] = temp[1]
                b3[i, j] = temp[2]
    elif cuttingAxis == 'z':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = ScalarCartesianMagneticField(a1, a2, cutval)
                b1[i, j] = temp[0]
                b2[i, j] = temp[1]
                b3[i, j] = temp[2]
    else:
        print("Failed. Please enter a string either containing x, y, or z as cuttingAxis")
        return
    
    b1Df = pd.DataFrame(b1, index=ax1, columns=ax2)
    b2Df = pd.DataFrame(b2, index=ax1, columns=ax2)
    b3Df = pd.DataFrame(b3, index=ax1, columns=ax2)

    ax = sns.heatmap(b1Df)
    ax.invert_yaxis()
    plt.title('Magnetic field along axis 1')
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.show()

    ax = sns.heatmap(b2Df)
    ax.invert_yaxis()
    plt.title('Magnetic field along axis 2')
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.show()

    ax = sns.heatmap(b3Df)
    ax.invert_yaxis()
    plt.title('Magnetic field along axis 3')
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.show()
"""
VisualizeMagneticField(ScalarCartesianMagneticField, 'z', 1.5, [-0.5, 0.5], [-0.5, 0.5])


# Build a 2D grid of sample points in the z-x plane
z = np.linspace(-5, 5, num=500)
x = np.linspace(-2, 2, num=200)
Z, X = np.meshgrid(z, x)  # each shape (100, 200)

# Evaluate the field at every grid point
BX = np.zeros_like(X)
BZ = np.zeros_like(Z)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        temp = ScalarCartesianMagneticField(X[i, j], 0, Z[i, j])
        BX[i, j] = temp[0]
        BZ[i, j] = temp[2]

# Plot field lines
fig, ax = plt.subplots(figsize=(10, 5))

strm = ax.streamplot(
    Z, X, BZ, BX,          # horizontal axis is z, vertical axis is x
    density=2,              # controls how many field lines are drawn
    color=np.sqrt(BX**2 + BZ**2),  # color lines by field strength
    cmap='plasma',
    linewidth=1,
    arrowsize=1.5
)

fig.colorbar(strm.lines, ax=ax, label='|B| (T)')
ax.set_xlabel('z (m)')
ax.set_ylabel('x (m)')
ax.set_title('Magnetic Field Lines (y=0 cross-section)')
plt.tight_layout()
plt.show()

"""



# NEXT STEPS: MODIFY THE CODE SO THAT IT RUNS THE BORIS PUSH ON ALL PARTICLES AT ONCE RATHER THAN ONE AT A TIME
# OR ALTERNATIVELY FIGURE OUT A BETTER WAY TO IMPLEMENT IT.


    