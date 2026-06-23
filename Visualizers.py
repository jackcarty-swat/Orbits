import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def VisualizeTrajectories(steps, N, cutoff=9787):
    """
    Plots trajectories of particles in magnetic fields

    Parameters
    ----------
    steps : Numpy Array
        Output from Boris Iterator function
    N : Integer
        Number of particles to display
    cuttoff : float
        Maximum z value at which particles should be displayed, excludes particles that end above this value.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    for i in range(N):
        if np.abs(steps[:,i,2,][-1]) <= cutoff:
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
    
    for i in range(N):
        if np.abs(steps[:,i,2][-1]) <= cutoff:
            try:
                plt.plot(steps[:,i,7], steps[:,i,1])
            except:
                print("Failed to display particle ", i)
    plt.title("Z vs Time")
    plt.xlabel("Time")
    plt.ylabel("Z position")
    plt.show()


def VisualizeMagneticField(BFunc, cuttingAxis, cutval, axRange):
    """
    Tool for visualizing magnetic fields. Produces heatmaps of the magnetic field in each of the cartesian coordinate
    directions on a specified cutting plane.

    Parameters
    ----------
    BFunc : fun. callable
        Magnetic field to produce plot of
    cuttingAxis : string, options: 'x', 'y', 'z'
        The axis perpendicular to which the cut will be made
    cutval : float
        Value along specified axis at which plots will be made
    axRange : vector of floats of shape (2,)
        Boundaries of the plots along the axes perpendicular to the cutting axis.

    Returns
    -------
    None.

    """
    ax1 = np.linspace(*axRange, num=500)
    ax2 = np.linspace(*axRange, num=500)
    b1 = np.zeros((500, 500))
    b2 = np.zeros((500, 500))
    b3 = np.zeros((500, 500))
    
    if cuttingAxis == 'x':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = BFunc(cutval, a1, a2)
                b1[i, j] = temp[0]
                b2[i, j] = temp[1]
                b3[i, j] = temp[2]
    elif cuttingAxis == 'y':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = BFunc(a1, cutval, a2)
                b1[i, j] = temp[0]
                b2[i, j] = temp[1]
                b3[i, j] = temp[2]
    elif cuttingAxis == 'z':
        for i, a1 in enumerate(ax1):
            for j, a2 in enumerate(ax2):
                temp = BFunc(a1, a2, cutval)
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


def AxisymmetricStreamPlot(BFunc, zMin, zMax, rMin, rMax):
    """
    Produces a stremplot of an axisymmetric magnetic field

    Parameters
    ----------
    BFunc : fun. callable
        axisymmetric magnetic field funciton
    zMin : float
        minimum axial value for the plot
    zMax : float
        maximum axial value for the plot
    rMin : float
        minimum radial value for the plot
    rMax : float
        maximum radial value for the plot

    Returns
    -------
    None.

    """
    # Build a 2D grid of sample points in the z-x plane
    z = np.linspace(zMin, zMax, num=500)
    x = np.linspace(rMin, rMax, num=500)
    Z, X = np.meshgrid(z, x)  # each shape (100, 200)

    # Evaluate the field at every grid point
    BX = np.zeros_like(X)
    BZ = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp = BFunc(X[i, j], 0, Z[i, j])
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

