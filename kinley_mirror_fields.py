import numpy as np
import matplotlib.pyplot as plt
from orbit_functions import *
from matplotlib.animation import FuncAnimation

R=1.0
z0=8.0

def B_loop(z):
    return(
        R**2/(R**2 +(z-z0/2)**2)**(3/2) + R**2/(R**2+(z+z0/2)**2)**(3/2)
        )
    

Bmin=B_loop(0.0)
Bmax=B_loop(z0/2)

mirror_ratio=Bmax/Bminhow 
alpha_loss = np.arcsin(np.sqrt(Bmin/Bmax))

print("Bmin =", Bmin)
print("Bmax =",Bmax)
print("Mirror ratio=", mirror_ratio)
print("Loss cone angle=", np.degrees(alpha_loss),"degrees")


def dBdz(z):
    h=1e-4
    return (B_loop(z+h) - B_loop(z-h))/(2*h)

def mirror_field(x):
    X,Y,Z=x

    Bz=B_loop(Z)
    Bx=-0.5*X*dBdz(Z) # near-axis divergence-free approximation:
    By=-0.5*Y*dBdz(Z)

    return np.array([Bx,By,Bz])


#field-line plot in the x-z plane:

x = np.linspace(-1.2,1.2,200)
z=np.linspace(-6,6,400)
X,Z=np.meshgrid(x,z)

Bx=np.zeros_like(X)
Bz=np.zeros_like(X)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        B=mirror_field([X[i,j],0,Z[i,j]])
        Bx[i,j]=B[0]
        Bz[i,j]=B[2]

plt.streamplot(z,x,Bz.T,Bx.T,density=1.5)
plt.xlabel("z")
plt.ylabel("x")
plt.axis("equal")
plt.show()

# Tracking particle motion in mirror field, using Boris_push to enforce particle acceleration

dt=0.01
N=5000
x_particle=np.array([0.2,0.0,0.0]) # starting near center
v_particle=np.array([0.0,1.0,0.2]) # perpendicular + parallel velocities
xs=np.zeros((N,3))
B=mirror_field(x_particle)
for n in range(N):
    xs[n]=x_particle
    x_particle,v_particle=boris_push(x_particle,v_particle,B,dt)
# animating in z-x plane:

fig,ax=plt.subplots()
ax.streamplot(z,x,Bz.T,Bx.T, density=1.2)
ax.set_xlabel("z")
ax.set_ylabel("x")
ax.set_xlim(-6,6)
ax.set_ylim(-1.5,1.5)

particle_dot,=ax.plot([],[],"ko",markersize=5)
trail,=ax.plot([],[],"k-",lw=1)

def init():
    particle_dot.set_data([],[])
    trail.set_data([],[])
    return particle_dot,trail

def update(frame):
    particle_dot.set_data([xs[frame,2]],[xs[frame,0]])

    start=max(0,frame-300)
    trail.set_data(xs[start:frame,2],xs[start:frame,0])

    return particle_dot, trail
ani=FuncAnimation(fig, update, frames=N, init_func=init, interval=10, blit=True)
plt.show()