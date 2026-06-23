import numpy as np
from numba import njit

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
def RadialAxisymmetricMagneticField(R, theta, Z, B_0):
    # Calculates the radial magnetic field at a given position in cylindrical coordinates
    B = R*0.0001022*B_0/(np.pi*gamma**3) * ((Z*0.0001022-Z_m)/((1+((Z*0.0001022-Z_m)/gamma)**2)**2) + (Z*0.0001022+Z_m)/((1+((Z*0.0001022+Z_m)/gamma)**2)**2))
    return B

@njit
def AxialAxisymmetricMagneticField(R, theta, Z, B_0):
    # Calculates the axial magnetic field at a given position in cylindrical coordinates
    B = B_0/(np.pi*gamma) * (1/(1+((Z*0.0001022-Z_m)/gamma)**2) + 1/(1+((Z*0.0001022+Z_m)/gamma)**2))
    return B

@njit
def AxisymmetricMagneticField(x, y, z, B_0):
    """
    Calculates the magnetic field of the mirrors in cartesian coordinates. First, converts cartesian coordinates
    to cylindrical coordinates. Then, finds magnetic field in cylindrical coordinates. Finally, converts magnetic
    field back to cartesian coordinates.
    input:
        x, y, z: vectors of floats representing the positions at which to calculate the magnetic field.
        B_0: float representing the magnitude of the magnetic field.
    output:
        B_x, B_y, B_z: vectors of floats representing the strength of the magnetic field in each direction.
    """
    r, theta, z = CartesianToCylindrical(x, y, z)
    B_r = RadialAxisymmetricMagneticField(r, theta, z, B_0)
    B_z = AxialAxisymmetricMagneticField(r, theta, z, B_0)
    B_x, B_y, B_z = CylindricalToCartesian(B_r, theta, B_z)
    return np.stack((B_x, B_y, B_z), axis=-1)

@njit
def ScalarAxisymmetricMagneticField(x, y, z, B_0):
    """
    Calculates the magnetic field of the mirrors in cartesian coordinates. First, converts cartesian coordinates
    to cylindrical coordinates. Then, finds magnetic field in cylindrical coordinates. Finally, converts magnetic
    field back to cartesian coordinates.
    input:
        x, y, z: floats representing the position at which to calculate the magnetic field.
        B_0: float representing the magnitude of the magnetic field.
    output:
        B_x, B_y, B_z: floats representing the strength of the magnetic field in each direction.
    """
    r, theta, z = CartesianToCylindrical(x, y, z)
    B_r = RadialAxisymmetricMagneticField(r, theta, z, B_0)
    B_z = AxialAxisymmetricMagneticField(r, theta, z, B_0)
    B_x, B_y, B_z = CylindricalToCartesian(B_r, theta, B_z)
    return np.array((B_x, B_y, B_z))

def UniformField(x, y, z, B_0):
    """
    Returns a magnetic field of strength B_0 in the z-direction.
    
    Parameters
    ----------
    x : Vector (n) of floats
        X-positions
    y : Vector (n) of floats
        Y-positions
    z : Vector (n) of floats
        Z-positions
    B_0 : Float
        Magnetic field strength

    Returns
    -------
    Array of integers with shape (n, 3)
        Magnetic field strength in each coordinate direction.

    """
    Bx = np.zeros_like(x)
    By = np.zeros_like(y)
    Bz = np.full_like(z, B_0)
    return np.stack((Bx, By, Bz), axis=-1)
