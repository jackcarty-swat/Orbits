import numpy as np
import matplotlib.pyplot as plt
from orbit_functions import *
from matplotlib.animation import FuncAnimation


r_star = 1.0
r_start=5.0
v_perp=0.0005 # small velocity perpendicular to B
v_para = 0.002 # along the field line (-z at equator)

B_local = (r_star/r_start)**3 # local B field at the equator
omega_C=B_local  # since q/m =1
r_L = v_perp/B_local
t_gyro = 2*np.pi/omega_C

x_initial = np.array([r_start,0,0])
v_initial = np.array([0,v_perp,v_para])
dt = t_gyro/100 # corresponds to 100 time steps per gyration
nsteps = int(500*t_gyro/dt) # many gyrations

def Bmag_dipole(x,r_star):
    return np.linalg.norm(dipole_field(x,r_star)) # this helps us get the magnitude of B field which can be later used for defining the gradB in dipole field.

def gradB_dipole(x,r_star):  # gradB is the gradient of dipole magnetic field:
    h=1e-5 # gradB is calculated by finding the change over small change in x,y,or z respectively
    x = np.array(x,dtype=float)
    dx=np.array([h,0.0,0.0])
    dy=np.array([0.0,h,0.0])
    dz=np.array([0.0,0.0,h])

    dBdx=(Bmag_dipole(x+dx,r_star)-Bmag_dipole(x-dx,r_star))/(2*h)
    dBdy=(Bmag_dipole(x+dy,r_star)-Bmag_dipole(x-dy,r_star))/(2*h)
    dBdz=(Bmag_dipole(x+dz, r_star)-Bmag_dipole(x-dz,r_star))/(2*h)
    return np.array([dBdx,dBdy,dBdz])
    
    

# defining curvature drift equation: v_c = v(para)^2/B*(b x K)
def b_hat_dipole(x,r_star): # defining unit vector for B which is later used while calculating the curvature drift velocity
    B=dipole_field(x,r_star)
    Bmag=np.linalg.norm(B)

    if Bmag<1e-20:
        return np.array([0.0,0.0,0.0])
    return B/Bmag

def curvature_dipole(x,r_star):
    h=1e-5 # gradB is calculated by finding the change over small change in x,y,or z respectively
    x=np.array(x,dtype=float)
    dx=np.array([h,0.0,0.0])
    dy=np.array([0.0,h,0.0])
    dz=np.array([0.0,0.0,h])

    b=b_hat_dipole(x,r_star)
    dBdx=(b_hat_dipole(x+dx,r_star)-b_hat_dipole(x-dx,r_star))/(2*h)
    dBdy=(b_hat_dipole(x+dy,r_star)-b_hat_dipole(x-dy,r_star))/(2*h)
    dBdz=(b_hat_dipole(x+dz, r_star)-b_hat_dipole(x-dz,r_star))/(2*h)

    kappa = b[0]*dBdx + b[1]*dBdy + b[2]*dBdz

    return kappa

new_positions =[]
new_velocities=[]
v_gradB_list=[]
v_curvB_list=[]
x = x_initial.copy()
v = v_initial.copy()
plt.figure()
for i in range(nsteps):
    B= dipole_field(x,r_star)
    Bmag = np.linalg.norm(B)
    b_hat = B/Bmag
    v_parallel=np.dot(v,b_hat)
    v_perp_vec =v - v_parallel * b_hat
    v_perp2 = np.dot(v_perp_vec,v_perp_vec)
    gradB= gradB_dipole(x,r_star)
    kappa=curvature_dipole(x,r_star)
    v_curv=(v_parallel**2/Bmag)*np.cross(b_hat, kappa)
    v_gradB=(v_perp2/(2*Bmag**3))*np.cross(B,gradB)
    v_gradB_list.append(v_gradB.copy())
    v_curvB_list.append(v_curv.copy())
    new_positions.append(x.copy()) # append adds on to the list as the position changes
    new_velocities.append(v.copy())
    x,v = boris_push(x,v, B, dt)
new_positions= np.array(new_positions)
new_velocities=np.array(new_velocities)
v_grad_list = np.array(v_gradB_list)
v_curvB_list=np.array(v_curvB_list)
v_total_drift=v_grad_list + v_curvB_list

z_pos = new_positions[:,2]
x_pos = new_positions[:,0]
y_pos=new_positions[:,1]

plt.plot(x_pos,z_pos,lw=1)
plt.xlabel("x")
plt.xlim(-5,5)
plt.ylabel("z")
plt.ylim(-2,2)
plt.title("Side view (x-z plane)")
plt.legend()
plt.show()


# Animation of the trapped particle simulation
skip = 100
frames = range(0,nsteps,skip)
fig = plt.figure()
line, = plt.plot([],[],lw=1)
dot,=plt.plot([],[],'go')

plt.xlabel("x")
plt.ylabel("z")
plt.ylim(-2,2)
plt.xlim(-5,5)
plt.title("Particle in dipole field")
plt.grid(True)

# defining frame update
def update(frame):
    line.set_data(x_pos[:frame],z_pos[:frame])
    dot.set_data([x_pos[frame]],[z_pos[frame]])
    return line, dot


ani = FuncAnimation(
    fig,update,frames=frames, interval=5,blit=True, repeat=False
)
ani.save(
    "dipole_trapped_particle.mp4",
    writer="ffmpeg",
    fps=60,
    dpi=100
)
plt.show()


# plotting the effects of Grad B and curvature drifts on particle motion in dipole magnetic field.

# this plot shows the velocity drift components in x,y and z
time=np.arange(nsteps)*dt
plt.figure()
plt.plot(time,v_total_drift[:,0], label="total drift in vx")
plt.plot(time, v_total_drift[:,1], label="total drift in vy")
plt.plot(time, v_total_drift[:,2], label="total drift in vz")

plt.xlabel("time")
plt.ylabel("total drift velocity")
plt.title("total guiding-center gradB and curvature drift velocity")
plt.legend()
plt.grid(True)
plt.show()

# plot comparing total drift velocity magnitude vs. time

v_total_mag = np.linalg.norm(v_total_drift, axis=1)

plt.figure()
plt.plot(time, v_total_mag, label = "total drift-v")
plt.xlabel("time")
plt.ylabel("total v-drift magnitude")
plt.title("total magnitude of velocity drift vs. time")
plt.legend()
plt.grid(True)
plt.show()