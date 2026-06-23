import pandas as pd
import numpy as np
from scipy.stats import uniform
from scipy.stats import uniform_direction
from scipy.stats import maxwell

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
    scaleParamPerp = np.sqrt(T_perp/m)
    scaleParamPara = np.sqrt(T_para/m)
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
    scaleParam = np.sqrt(T/m)
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
        scaleParam = np.sqrt(T/m)
        vMag = maxwell.rvs(loc=0, scale=scaleParam, size=N)
        V = uniform_direction.rvs(dim=3, size=N)
        vX = V[:, 0] * vMag
        vY = V[:, 1] * vMag
        vZ = V[:, 2] * vMag
    else:
        scaleParamPerp = np.sqrt(T_perp/m)
        scaleParamPara = np.sqrt(T/m)
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