import numpy as np
import matplotlib.pyplot as plt
from orbit_functions import *
from matplotlib.animation import FuncAnimation

def Bellan_field(x,B0=10,L0=1.0):  # Bellan field formula in Cylindrical coordinates : Br = (-B0rz)/L0^2, Bz =B0(1+z^2/L0^2) (Bz shows the magnetic field strength in z)
    X,Y,Z =x # we can convert into cartesian formula by expressing the formula all x,y and z components using r =sqrt(x^2 + y^2)
    Bx =(-B0*X*Z)/L0**2
    By=(-B0*Z*Y)/L0**2
    Bz=B0*(1+(Z**2/L0**2))
    return np.array([Bx,By,Bz])


x_initial = np.array([0.5,0,0])
v_initial = np.array([0,0.3,2.0])
dt = 0.001 # corresponds to 100 time steps per gyration
nsteps = 100000 # many gyrations


new_positions =[]
new_velocities=[]
x = x_initial.copy()
v = v_initial.copy()
plt.figure()
for i in range(nsteps):
    B=Bellan_field(x,B0=10,L0=1)
    new_positions.append(x.copy()) # append adds on to the list as the position changes
    new_velocities.append(v.copy())
    x,v = boris_push(x,v, B, dt)
new_positions= np.array(new_positions)
new_velocities=np.array(new_velocities)

z_pos = new_positions[:,2]
x_pos = new_positions[:,0]
y_pos=new_positions[:,1]

plt.plot(x_pos,z_pos,lw=1)
plt.xlabel("x")
plt.xlim(0,1)
plt.ylabel("z")
plt.ylim(-3,3)
plt.title("Bellan field mirror trajectory: side-view(x-z)")

plt.savefig("Bellan_field.png",dpi=200)


# plotting field lines of Bellan field model

z_vals = np.linspace(-7,7,300)
x_vals = np.linspace(-1.5,1.5,150)

Z,X =np.meshgrid(z_vals, x_vals)
Bz_grid=np.zeros_like(Z)
Bx_grid=np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos=np.array([X[i,j],0.0,Z[i,j]]) # [x,y,z]
        B=Bellan_field(pos,B0=1,L0=1.0)

        Bx_grid[i,j]=B[0]
        Bz_grid[i,j]=B[2]

plt.figure(figsize=(12,4))
plt.streamplot(
    Z,X, #coordinates: horizontal z, vertical x
    Bz_grid, # horizontal field component
    Bx_grid, # vertical field component
    density=0.7,
    linewidth=1.2,
    arrowsize=0.5
)

plt.xlabel("z")
plt.ylabel("x")
plt.title("Bellan mirror field lines")
plt.xlim(-7,7)
plt.ylim(-1.2,1.2)
plt.show()


L0 =1.0
y0_particle = 0.5
z_anyl=np.linspace(-7,7,1000)
y0_values=np.linspace(-1.2,1.2,8)

fig,ax = plt.subplots(figsize=(12,6))
for y0 in y0_values:
    y=y0/np.sqrt(1 + (z_anyl**2/L0**2))
    ax.plot(z_anyl,y, color="red",lw=0.8)

ax.axhline(0,color="red",lw=0.8)
particle, = ax.plot([],[],marker='o', markersize=4, color="black",linestyle='None',zorder=5)
trail, =ax.plot([],[], color="black",lw=0.8,solid_capstyle="round",zorder=4)
ax.set_xlabel("z")
ax.set_ylabel("y")
ax.set_title("Guiding-center mirror bounce on Bellan field lines")
ax.set_xlim(-7,7)
ax.set_ylim(-1.2,1.2)

z_history=[]
y_history=[]
z_state=0.0
vz_state=2.0
mu=0.005
B0=10
dt_gc = 0.03
tail_length=30
gyro_phase =0.0
def update(frame):
    global z_state,vz_state,gyro_phase
    B=B0*(1 + z_state**2/L0**2) # guiding-center magnetic field strength
    dBdz = 2*B0*z_state/L0**2

    az =-mu*dBdz # mirror force
    vz_state += az*dt_gc
    z_state += vz_state *dt_gc
    x_gc = 0.0
    y_gc=y0_particle/np.sqrt(1+ z_state**2/L0**2) # guiding-center field line
    r_gc=np.array([x_gc,y_gc,z_state])

    # local magnetic field
    Bvec = Bellan_field(r_gc, B0=B0,L0=L0)
    Bmag=np.linalg.norm(Bvec)
    bhat = Bvec/Bmag

    a = np.array([1.0,0.0,0.0]) # choose one vector not parallel to B

    # make e1 perpendicular to B
    e1=a-np.dot(a,bhat)*bhat
    e1=e1/np.linalg.norm(e1)

    # e2 also perpendicular to B
    e2=np.cross(bhat,e1)
    
    #magnetic moment conservation
    v_perp =np.sqrt(2*mu*Bmag) 
    rho=v_perp/Bmag
    gyro_phase += Bmag *dt_gc

    r_particle = r_gc +rho*(np.cos(gyro_phase)*e1 +np.sin(gyro_phase)*e2)
    z_plot = r_particle[2]
    y_plot =r_particle[1]
    z_history.append(z_plot)
    y_history.append(y_plot)
    if len(z_history)>tail_length:
        z_history.pop(0)
        y_history.pop(0)
    trail.set_data(z_history,y_history)
    particle.set_data([z_plot],[y_plot])
    return particle,trail

ani = FuncAnimation(fig,update, frames =400, interval =30, blit=True)
ani.save(
    "particle_in_Bellan.mp4",
    writer="ffmpeg",
    fps=60,
    dpi=100
)
plt.show()

# Checking if the particle mirrors in between the strong B fields:

plt.figure()
plt.plot(np.arange(nsteps)*dt, z_pos)
plt.xlabel("time")
plt.ylabel("z")
plt.title("Mirror bounce")
plt.show()


# showing the gyration motion of the particle in all three directions
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,6))
ax =fig.add_subplot(111, projection='3d')

ax.plot(
    new_positions[:,0],
    new_positions[:,1],
    new_positions[:,2]
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

